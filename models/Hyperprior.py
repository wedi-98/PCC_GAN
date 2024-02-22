import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from utils import maths
from utils.compression import hyperprior_model, prior_model

"""
Hyperprior as proposed in [1]. 

[1] Ballé et. al., "Variational image compression with a scale hyperprior", 
    arXiv:1802.01436 (2018).
    
这篇论文提出的模型是在这里实现，总的计算定义在类Hyperprior或者HyperpriorDLMM中
"""

MIN_SCALE = 0.11
LOG_SCALES_MIN = -3.
MIN_LIKELIHOOD = 1e-9
MAX_LIKELIHOOD = 1e3
SMALL_HYPERLATENT_FILTERS = 192
LARGE_HYPERLATENT_FILTERS = 320

HyperInfo = namedtuple(
    "HyperInfo",
    "decoded "
    "latent_nbpp hyperlatent_nbpp total_nbpp latent_qbpp hyperlatent_qbpp total_qbpp",
)

CompressionOutput = namedtuple("CompressionOutput",
                               ["hyperlatents_encoded",
                                "latents_encoded",
                                "hyperlatent_spatial_shape",
                                "batch_shape",
                                "spatial_shape",
                                "hyper_coding_shape",
                                "latent_coding_shape",
                                "hyperlatent_bits",
                                "latent_bits",
                                "total_bits",
                                "hyperlatent_bpp",
                                "latent_bpp",
                                "total_bpp"]
                               )


def get_num_mixtures(K_agg, C, params=['mu', 'scale', 'mix']):
    print("get_num_mixtures")
    return K_agg // (len(params) * C)


def unpack_likelihood_params(x, conv_out, log_scales_min):
    print("unpack_likelihood_params")
    N, C, H, W = x.shape
    K_agg = conv_out.shape[1]

    K = get_num_mixtures(K_agg, C)

    # For each channel: K pi / K mu / K sigma
    conv_out = conv_out.reshape(N, 3, C, K, H, W)
    logit_pis = conv_out[:, 0, ...]
    means = conv_out[:, 1, ...]
    log_scales = conv_out[:, 2, ...]
    log_scales = lower_bound_toward(log_scales, log_scales_min)
    x = x.reshape(N, C, 1, H, W)

    return x, (logit_pis, means, log_scales), K


def get_num_DLMM_channels(C, K=4, params=['mu', 'scale', 'mix']):
    """
    C:  Channels of latent representation (L3C uses 5).
    K:  Number of mixture coefficients.
    """
    print("get_num_DLMM_channels")
    return C * K * len(params)


class HyperpriorAnalysis(nn.Module):
    def __init__(self, C=50, N=320, activation="relu"):
        print("HyperpriorAnalysis.init")
        super(HyperpriorAnalysis, self).__init__()
        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2, padding_mode="reflect")
        self.activation = getattr(F, activation)
        self.n_down_sampling_layers = 2
        # TODO 后续改一下参数
        self.conv1 = nn.Conv1d(C, N, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(N, N, **cnn_kwargs)
        self.conv3 = nn.Conv1d(N, N, **cnn_kwargs)

    def forward(self, x):
        print("HyperpriorAnalysis.forward")
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)

        return x


class HyperpriorSynthesis(nn.Module):
    def __init__(self, C=50, N=320, activation='relu', final_activation=None):
        super(HyperpriorSynthesis, self).__init__()
        print("HyperpriorSynthesis.init")
        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation = getattr(F, activation)
        self.final_activation = final_activation

        self.conv1 = nn.ConvTranspose1d(N, N, **cnn_kwargs)
        self.conv2 = nn.ConvTranspose1d(N, N, **cnn_kwargs)
        self.conv3 = nn.ConvTranspose1d(N, C, kernel_size=3, stride=1, padding=1)

        if self.final_activation is not None:
            self.final_activation = getattr(F, final_activation)

    def forward(self, x):
        print("HyperpriorSynthesis.forward")
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)

        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


class HyperpriorSynthesisDLMM(nn.Module):
    """
    Outputs distribution parameters of input latents, conditional on
    hyperlatents, assuming a discrete logistic mixture model.

    C:  Number of output channels 后面调一下
    """

    def __init__(self, C=64, N=320, activation='relu', final_activation=None):
        super(HyperpriorSynthesisDLMM, self).__init__()
        print("HyperpriorSynthesisDLMM.init")
        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation = getattr(F, activation)
        self.final_activation = final_activation

        self.conv1 = nn.ConvTranspose1d(N, N, **cnn_kwargs)
        self.conv2 = nn.ConvTranspose1d(N, N, **cnn_kwargs)
        self.conv3 = nn.ConvTranspose1d(N, C, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(C, get_num_DLMM_channels(C), kernel_size=1, stride=1)

        if self.final_activation is not None:
            self.final_activation = getattr(F, final_activation)

    def forward(self, x):
        print("HyperpriorSynthesisDLMM.forward")
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        x = self.conv_out(x)

        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


lower_bound_identity = maths.LowerBoundIdentity.apply
lower_bound_toward = maths.LowerBoundToward.apply


class CodingModel(nn.Module):
    """
    Probability model for estimation of (cross)-entropies in the context
    of data compression. TODO: Add tensor -> string compression and
    decompression functionality.
    """

    def __init__(self, n_channels, min_likelihood=MIN_LIKELIHOOD, max_likelihood=MAX_LIKELIHOOD):
        super(CodingModel, self).__init__()
        print("CodingModel.__init__")
        self.n_channels = n_channels
        self.min_likelihood = float(min_likelihood)
        self.max_likelihood = float(max_likelihood)

    def _quantize(self, x, mode='noise', means=None):
        """
        mode:       If 'noise', returns continuous relaxation of hard
                    quantization through additive uniform noise channel.
                    Otherwise, perform actual quantization (through rounding).
        """
        print("CodingModel._quantize")
        if mode == 'noise':
            quantization_noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
            x = x + quantization_noise

        elif mode == 'quantize':
            if means is not None:
                x = x - means
                x = torch.floor(x + 0.5)
                x = x + means
            else:
                x = torch.floor(x + 0.5)
        else:
            raise NotImplementedError

        return x

    def _estimate_entropy(self, likelihood, spatial_shape):
        print("CodingModel._estimate_entropy")
        EPS = 1e-9
        quotient = -np.log(2.)
        batch_size = likelihood.size()[0]

        assert len(spatial_shape) == 2, 'Mispecified spatial dims'
        n_pixels = np.prod(spatial_shape)

        log_likelihood = torch.log(likelihood + EPS)
        n_bits = torch.sum(log_likelihood) / (batch_size * quotient)
        bpp = n_bits / n_pixels

        return n_bits, bpp

    def _estimate_entropy_log(self, log_likelihood, spatial_shape):
        print("CodingModel._estimate_entropy_log")
        quotient = -np.log(2.)
        batch_size = log_likelihood.size()[0]

        assert len(spatial_shape) == 2, 'Mispecified spatial dims'
        n_pixels = np.prod(spatial_shape)

        n_bits = torch.sum(log_likelihood) / (batch_size * quotient)
        bpp = n_bits / n_pixels

        return n_bits, bpp

    def quantize_latents_st(self, inputs, means=None):
        # Latents rounded instead of additive uniform noise
        # Ignore rounding in backward pass
        values = inputs
        print("CodingModel.quantize_latents_st")
        if means is not None:
            values = values - means

        delta = (torch.floor(values + 0.5) - values).detach()
        values = values + delta

        if means is not None:
            values = values + means

        return values

    def latent_likelihood(self, x, mean, scale):
        print("CodingModel.latent_likelihood")
        # Assumes 1 - CDF(x) = CDF(-x)
        x = x - mean
        x = torch.abs(x)
        cdf_upper = self.standardized_CDF((0.5 - x) / scale)
        cdf_lower = self.standardized_CDF(-(0.5 + x) / scale)

        # Naive
        # cdf_upper = self.standardized_CDF( (x + 0.5) / scale )
        # cdf_lower = self.standardized_CDF( (x - 0.5) / scale )

        likelihood_ = cdf_upper - cdf_lower
        likelihood_ = lower_bound_toward(likelihood_, self.min_likelihood)

        return likelihood_


class Hyperprior(CodingModel):

    def __init__(self, bottleneck_capacity=220, hyperlatent_filters=LARGE_HYPERLATENT_FILTERS,
                 mode='large', likelihood_type='gaussian', scale_lower_bound=MIN_SCALE, entropy_code=False,
                 vectorize_encoding=True, block_encode=True):
        print("Hyperprior.__init__")
        """
        Introduces probabilistic model over latents of
        latents.

        The hyperprior over the standard latents is modelled as
        a non-parametric, fully factorized density.
        """

        super(Hyperprior, self).__init__(n_channels=bottleneck_capacity)

        self.bottleneck_capacity = bottleneck_capacity
        self.scale_lower_bound = scale_lower_bound

        analysis_net = HyperpriorAnalysis
        synthesis_net = HyperpriorSynthesis

        if mode == 'small':
            hyperlatent_filters = SMALL_HYPERLATENT_FILTERS

        self.analysis_net = analysis_net(C=bottleneck_capacity, N=hyperlatent_filters)

        self.synthesis_mu = synthesis_net(C=bottleneck_capacity, N=hyperlatent_filters)
        self.synthesis_std = synthesis_net(C=bottleneck_capacity, N=hyperlatent_filters)

        self.amortization_models = [self.analysis_net, self.synthesis_mu, self.synthesis_std]

        self.hyperlatent_likelihood = hyperprior_model.HyperpriorDensity(n_channels=hyperlatent_filters)

        if likelihood_type == 'gaussian':
            self.standardized_CDF = maths.standardized_CDF_gaussian
        elif likelihood_type == 'logistic':
            self.standardized_CDF = maths.standardized_CDF_logistic
        else:
            raise ValueError('Unknown likelihood model: {}'.format(likelihood_type))

        if entropy_code is True:
            print('Building prior probability tables...')
            self.hyperprior_entropy_model = hyperprior_model.HyperpriorEntropyModel(
                distribution=self.hyperlatent_likelihood)
            self.prior_density = prior_model.PriorDensity(n_channels=bottleneck_capacity,
                                                          scale_lower_bound=self.scale_lower_bound,
                                                          likelihood_type=likelihood_type)
            self.prior_entropy_model = prior_model.PriorEntropyModel(
                distribution=self.prior_density, min_scale=self.scale_lower_bound)
            self.index_tables = self.prior_entropy_model.scale_table_tensor
            self.vectorize_encoding = vectorize_encoding
            self.block_encode = block_encode

    def compress_forward(self, latents, spatial_shape, **kwargs):
        print("Hyperprior.compress_forward")
        # Obtain hyperlatents from hyperencoder
        hyperlatents = self.analysis_net(latents)
        hyperlatent_spatial_shape = hyperlatents.size()[2:]
        batch_shape = latents.size(0)

        # Estimate Shannon entropies for hyperlatents
        hyp_agg = self.hyperprior_entropy_model._estimate_compression_bits(
            hyperlatents, spatial_shape)
        hyperlatent_bits, hyperlatent_bpp, hyperlatent_bpi = hyp_agg

        # Compress, then decompress hyperlatents
        hyperlatents_encoded, hyper_coding_shape, _ = self.hyperprior_entropy_model.compress(hyperlatents,
                                                                                             vectorize=self.vectorize_encoding,
                                                                                             block_encode=self.block_encode)
        hyperlatents_decoded, _ = self.hyperprior_entropy_model.decompress(hyperlatents_encoded,
                                                                           batch_shape=batch_shape,
                                                                           broadcast_shape=hyperlatent_spatial_shape,
                                                                           coding_shape=hyper_coding_shape,
                                                                           vectorize=self.vectorize_encoding,
                                                                           block_decode=self.block_encode)
        hyperlatents_decoded = hyperlatents_decoded.to(latents)

        # Recover latent statistics from compressed hyperlatents
        latent_means = self.synthesis_mu(hyperlatents_decoded)
        latent_scales = self.synthesis_std(hyperlatents_decoded)
        latent_scales = lower_bound_toward(latent_scales, self.scale_lower_bound)

        # Use latent statistics to build indexed probability tables, and compress latents
        latents_encoded, latent_coding_shape, _ = self.prior_entropy_model.compress(latents, means=latent_means,
                                                                                    scales=latent_scales,
                                                                                    vectorize=self.vectorize_encoding,
                                                                                    block_encode=self.block_encode)

        # Estimate Shannon entropies for latents
        latent_agg = self.prior_entropy_model._estimate_compression_bits(latents,
                                                                         means=latent_means, scales=latent_scales,
                                                                         spatial_shape=spatial_shape)
        latent_bits, latent_bpp, latent_bpi = latent_agg

        # What the decoder needs for reconstruction
        compression_output = CompressionOutput(
            hyperlatents_encoded=hyperlatents_encoded,
            latents_encoded=latents_encoded,
            hyperlatent_spatial_shape=hyperlatent_spatial_shape,  # 2D
            spatial_shape=spatial_shape,  # 2D
            hyper_coding_shape=hyper_coding_shape,  # C,H,W
            latent_coding_shape=latent_coding_shape,  # C,H,W
            batch_shape=batch_shape,
            hyperlatent_bits=hyperlatent_bits.item(),  # for reporting
            latent_bits=latent_bits.item(),
            total_bits=(hyperlatent_bits + latent_bits).item(),
            hyperlatent_bpp=hyperlatent_bpp.item(),
            latent_bpp=latent_bpp.item(),
            total_bpp=(hyperlatent_bpp + latent_bpp).item(),
        )

        return compression_output

    def decompress_forward(self, compression_output, device):
        print("Hyperprior.decompress_forward")
        hyperlatents_encoded = compression_output.hyperlatents_encoded
        latents_encoded = compression_output.latents_encoded
        hyperlatent_spatial_shape = compression_output.hyperlatent_spatial_shape
        batch_shape = compression_output.batch_shape

        # Decompress hyperlatents
        hyperlatents_decoded, _ = self.hyperprior_entropy_model.decompress(hyperlatents_encoded,
                                                                           batch_shape=batch_shape,
                                                                           broadcast_shape=hyperlatent_spatial_shape,
                                                                           coding_shape=compression_output.hyper_coding_shape,
                                                                           vectorize=self.vectorize_encoding,
                                                                           block_decode=self.block_encode)
        hyperlatents_decoded = hyperlatents_decoded.to(device)

        # Recover latent statistics from compressed hyperlatents
        latent_means = self.synthesis_mu(hyperlatents_decoded)
        latent_scales = self.synthesis_std(hyperlatents_decoded)
        latent_scales = lower_bound_toward(latent_scales, self.scale_lower_bound)
        latent_spatial_shape = latent_scales.size()[2:]

        # Use latent statistics to build indexed probability tables, and decompress latents
        latents_decoded, _ = self.prior_entropy_model.decompress(latents_encoded, means=latent_means,
                                                                 scales=latent_scales,
                                                                 broadcast_shape=latent_spatial_shape,
                                                                 coding_shape=compression_output.latent_coding_shape,
                                                                 vectorize=self.vectorize_encoding,
                                                                 block_decode=self.block_encode)

        return latents_decoded.to(device)

    def forward(self, latents, spatial_shape, **kwargs):
        print("Hyperprior.forward")
        hyperlatents = self.analysis_net(latents)

        # Mismatch b/w continuous and discrete cases?
        # Differential entropy, hyperlatents
        noisy_hyperlatents = self._quantize(hyperlatents, mode='noise')
        noisy_hyperlatent_likelihood = self.hyperlatent_likelihood(noisy_hyperlatents)
        noisy_hyperlatent_bits, noisy_hyperlatent_bpp = self._estimate_entropy(
            noisy_hyperlatent_likelihood, spatial_shape)

        # Discrete entropy, hyperlatents
        quantized_hyperlatents = self._quantize(hyperlatents, mode='quantize')
        quantized_hyperlatent_likelihood = self.hyperlatent_likelihood(quantized_hyperlatents)
        quantized_hyperlatent_bits, quantized_hyperlatent_bpp = self._estimate_entropy(
            quantized_hyperlatent_likelihood, spatial_shape)

        if self.training is True:
            hyperlatents_decoded = noisy_hyperlatents
        else:
            hyperlatents_decoded = quantized_hyperlatents

        latent_means = self.synthesis_mu(hyperlatents_decoded)
        latent_scales = self.synthesis_std(hyperlatents_decoded)
        # latent_scales = F.softplus(latent_scales)
        latent_scales = lower_bound_toward(latent_scales, self.scale_lower_bound)

        # Differential entropy, latents
        noisy_latents = self._quantize(latents, mode='noise', means=latent_means)
        noisy_latent_likelihood = self.latent_likelihood(noisy_latents, mean=latent_means,
                                                         scale=latent_scales)
        noisy_latent_bits, noisy_latent_bpp = self._estimate_entropy(
            noisy_latent_likelihood, spatial_shape)

        # Discrete entropy, latents
        quantized_latents = self._quantize(latents, mode='quantize', means=latent_means)
        quantized_latent_likelihood = self.latent_likelihood(quantized_latents, mean=latent_means,
                                                             scale=latent_scales)
        quantized_latent_bits, quantized_latent_bpp = self._estimate_entropy(
            quantized_latent_likelihood, spatial_shape)

        latents_decoded = self.quantize_latents_st(latents, latent_means)

        info = HyperInfo(
            decoded=latents_decoded,
            latent_nbpp=noisy_latent_bpp,
            hyperlatent_nbpp=noisy_hyperlatent_bpp,
            total_nbpp=noisy_latent_bpp + noisy_hyperlatent_bpp,
            latent_qbpp=quantized_latent_bpp,
            hyperlatent_qbpp=quantized_hyperlatent_bpp,
            total_qbpp=quantized_latent_bpp + quantized_hyperlatent_bpp,
        )

        return info


"""
========
Discretized logistic mixture model.
========
"""


class HyperpriorDLMM(CodingModel):
    def __init__(self, bottleneck_capacity=64, hyperlatent_filters=LARGE_HYPERLATENT_FILTERS, mode='large',
                 likelihood_type='gaussian', scale_lower_bound=MIN_SCALE, mixture_components=4,
                 entropy_code=False):
        """
        Introduces probabilistic model over latents of
        latents.

        The hyperprior over the standard latents is modelled as
        a non-parametric, fully factorized density.
        """
        super(HyperpriorDLMM, self).__init__(n_channels=bottleneck_capacity)
        print("HyperpriorDLMM.__init__")
        assert bottleneck_capacity <= 128, 'Will probably run out of memory!'
        self.bottleneck_capacity = bottleneck_capacity
        self.scale_lower_bound = scale_lower_bound
        self.mixture_components = mixture_components

        analysis_net = HyperpriorAnalysis
        synthesis_net = HyperpriorSynthesisDLMM

        if mode == 'small':
            hyperlatent_filters = SMALL_HYPERLATENT_FILTERS

        self.analysis_net = analysis_net(C=bottleneck_capacity, N=hyperlatent_filters)

        # TODO: Combine scale, loc into single network
        self.synthesis_DLMM_params = synthesis_net(C=bottleneck_capacity, N=hyperlatent_filters)

        self.amortization_models = [self.analysis_net, self.synthesis_DLMM_params]

        self.hyperlatent_likelihood = hyperprior_model.HyperpriorDensity(n_channels=hyperlatent_filters)

        if likelihood_type == 'gaussian':
            self.standardized_CDF = maths.standardized_CDF_gaussian
        elif likelihood_type == 'logistic':
            self.standardized_CDF = maths.standardized_CDF_logistic
        else:
            raise ValueError('Unknown likelihood model: {}'.format(likelihood_type))

    def latent_log_likelihood_DLMM(self, x, DLMM_params):
        print("HyperpriorDLMM.latent_log_likelihood_DLMM")
        # (B C K H W)
        x, (logit_pis, means, log_scales), K = unpack_likelihood_params(x, DLMM_params, LOG_SCALES_MIN)

        # Assumes 1 - CDF(x) = CDF(-x) symmetry
        # Numerical stability, do subtraction in left tail

        x_centered = x - means
        x_centered = torch.abs(x_centered)
        inv_stds = torch.exp(-log_scales)
        cdf_upper = self.standardized_CDF(inv_stds * (0.5 - x_centered))
        cdf_lower = self.standardized_CDF(inv_stds * (- 0.5 - x_centered))
        pmf_mixture_component = lower_bound_toward(cdf_upper - cdf_lower, MIN_LIKELIHOOD)
        log_pmf_mixture_component = torch.log(pmf_mixture_component)

        # Non-negativity + normalization via softmax
        lse_in = F.log_softmax(logit_pis, dim=2) + log_pmf_mixture_component
        log_DLMM = torch.logsumexp(lse_in, dim=2)

        return log_DLMM

    def forward(self, latents, spatial_shape, **kwargs):
        print("HyperpriorDLMM.forward")
        hyperlatents = self.analysis_net(latents)

        # Mismatch b/w continuous and discrete cases?
        # Differential entropy, hyperlatents
        noisy_hyperlatents = self._quantize(hyperlatents, mode='noise')
        noisy_hyperlatent_likelihood = self.hyperlatent_likelihood(noisy_hyperlatents)
        noisy_hyperlatent_bits, noisy_hyperlatent_bpp = self._estimate_entropy(
            noisy_hyperlatent_likelihood, spatial_shape)

        # Discrete entropy, hyperlatents
        quantized_hyperlatents = self._quantize(hyperlatents, mode='quantize')
        quantized_hyperlatent_likelihood = self.hyperlatent_likelihood(quantized_hyperlatents)
        quantized_hyperlatent_bits, quantized_hyperlatent_bpp = self._estimate_entropy(
            quantized_hyperlatent_likelihood, spatial_shape)

        if self.training is True:
            hyperlatents_decoded = noisy_hyperlatents
        else:
            hyperlatents_decoded = quantized_hyperlatents

        latent_DLMM_params = self.synthesis_DLMM_params(hyperlatents_decoded)

        # Differential entropy, latents
        noisy_latents = self._quantize(latents, mode='noise')
        noisy_latent_log_likelihood = self.latent_log_likelihood_DLMM(noisy_latents,
                                                                      DLMM_params=latent_DLMM_params)
        noisy_latent_bits, noisy_latent_bpp = self._estimate_entropy_log(
            noisy_latent_log_likelihood, spatial_shape)

        # Discrete entropy, latents
        quantized_latents = self._quantize(latents, mode='quantize')
        quantized_latent_log_likelihood = self.latent_log_likelihood_DLMM(quantized_latents,
                                                                          DLMM_params=latent_DLMM_params)
        quantized_latent_bits, quantized_latent_bpp = self._estimate_entropy_log(
            quantized_latent_log_likelihood, spatial_shape)

        if self.training is True:
            latents_decoded = self.quantize_latents_st(latents)
        else:
            latents_decoded = quantized_latents

        info = HyperInfo(
            decoded=latents_decoded,
            latent_nbpp=noisy_latent_bpp,
            hyperlatent_nbpp=noisy_hyperlatent_bpp,
            total_nbpp=noisy_latent_bpp + noisy_hyperlatent_bpp,
            latent_qbpp=quantized_latent_bpp,
            hyperlatent_qbpp=quantized_hyperlatent_bpp,
            total_qbpp=quantized_latent_bpp + quantized_hyperlatent_bpp,
        )

        return info

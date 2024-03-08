import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VarianceSchedule(nn.Module):
    def __init__(self, num_steps, beta_1, beta_T, mode="linear"):
        super(VarianceSchedule, self).__init__()
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode
        betas = []
        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=self.num_steps)


        betas = torch.cat([torch.zeros([1]), betas], dim=0)

        alphas = 1 - betas
        log_alphas = torch.log(alphas)

        for i in range(1, log_alphas.size(0)):
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        # register_buffer将张量注册到模型中，但不会在训练时更新，但可以一起转移到GPU或者方便模型的存储和加载。
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas_flex", sigmas_flex)
        self.register_buffer("sigmas_inflex", sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
    
    
# 这个类将t、context嵌入到点云数据中，然后进行噪声估计
class PointWiseNet(nn.Module):
    def __init__(self, context_dim, residual):
        super(PointWiseNet, self).__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.context_dim = context_dim
        self.W_q = nn.Linear(3, 50, bias=False)
        self.W_k = nn.Linear(50, 50, bias=False)
        self.W_v = nn.Linear(50, 50, bias=False)
        self.cross_attention = nn.MultiheadAttention(embed_dim=50, num_heads=1)
        self.layers = nn.ModuleList([
            nn.Linear(50, 128, bias=False),
            nn.Linear(128, 256, bias=False),
            nn.Linear(256, 512, bias=False),
            nn.Linear(512, 256, bias=False),
            nn.Linear(256, 128, bias=False),
            nn.Linear(128, 3, bias=False)
        ])

    def forward(self, x:torch.Tensor, beta:torch.Tensor, context:torch.Tensor):
        batch_size = x.size(0)
        beta = torch.full((batch_size, 1, 50), beta)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=1)
        ctx_emb = torch.cat([context, time_emb], dim=1)

        Q = self.W_q(x).permute(1, 0, 2)
        K = self.W_k(ctx_emb).permute(1, 0, 2)
        V = self.W_v(ctx_emb).permute(1, 0, 2)

        out, weight = self.cross_attention(query=Q, key=K, value=V)
        out = out.permute(1, 0, 2)
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.act(out)
        print("out ", out.shape)
        if self.residual:
            return x + out
        return out


class DiffusionPoint(nn.Module):
    def __init__(self, net:PointWiseNet, var_sche:VarianceSchedule):
        super(DiffusionPoint, self).__init__()
        self.net = net
        self.var_sche = var_sche

    def forward(self, x_0, context, t=None):
        batch_size, _, point_dim = x_0.size()
        if t is None:
            t = self.var_sche.uniform_sample_t(batch_size)
        alpha_bar = self.var_sche.alpha_bars[t]
        beta = self.var_sche.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)

        loss = F.mse_loss(e_theta.reshape(-1, point_dim), e_rand.reshape(-1, point_dim), reduction='mean')
        return loss

    def sample(self, num_points, context, point_dim=3, flexibility=0.0, ret_traj=False):
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        traj = {self.var_sche.num_steps: x_T}
        for t in range(self.var_sche.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sche.alphas[t]
            alpha_bar = self.var_sche.alpha_bars[t]
            sigma = self.var_sche.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sche.betas[[t] * batch_size]
            e_theta = self.net(x_t, beta=beta, context=context)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t - 1] = x_next.detach()  # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()  # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj
        else:
            return traj[0]
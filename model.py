import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.optim import Adam

# =========================
# 2. Beta-CVAE
# =========================
class BetaCVAE(nn.Module):
    """
    Beta Conditional Variational Autoencoder (Beta-CVAE) for binary tasks:
      - Encoder processes (x, y).
      - Decoder processes (z, y).
      - Beta > 1 emphasizes the KL divergence to encourage a more spread-out latent space.
    """

    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, beta=4.0):
        super(BetaCVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder layers
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc3_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fc4 = nn.Linear(latent_dim + 1, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, y):
        xy = torch.cat([x, y], dim=1)
        h = F.relu(self.fc1(xy))
        h = F.relu(self.fc2(h))
        mean = self.fc3_mean(h)
        logvar = self.fc3_logvar(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        h = F.relu(self.fc4(zy))
        h = F.relu(self.fc5(h))
        x_recon = self.fc6(h)
        return x_recon

    def forward(self, x, y):
        mean, logvar = self.encode(x, y)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mean, logvar


# =========================
# 3. Transformer Detector
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :].to(x.device)


class TransformerDetector(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerDetector, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(1)


# =========================
# 4. Mixture of Experts
# =========================
class MixtureOfExperts(nn.Module):
    def __init__(self, input_size, num_experts, d_model=128, nhead=8, num_layers=2,
                 dim_feedforward=256, dropout=0.1, gating_hidden_size=64):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            TransformerDetector(input_size=input_size, d_model=d_model,
                                nhead=nhead, num_layers=num_layers,
                                dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_experts)
        ])

        self.gating_network = nn.Sequential(
            nn.Linear(input_size, gating_hidden_size),
            nn.ReLU(),
            nn.Linear(gating_hidden_size, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        gating_weights = self.gating_network(x)
        expert_outputs = torch.cat([expert(x).unsqueeze(1) for expert in self.experts], dim=1)
        output = torch.sum(expert_outputs * gating_weights, dim=1)
        return output


# =========================
# 5. PPO Components
# =========================
class PolicyNetwork(nn.Module):
    """
    Gaussian policy that outputs (mu, log_std) for continuous actions.
    Action = delta vector in input space, applied as x_adv = x_orig + delta.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_log_std = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        log_std = torch.clamp(self.fc_log_std(h), -5, 2)
        return mu, log_std

    def sample_action(self, x):
        """Reparameterized sample with log_prob and entropy."""
        mu, log_std = self(x)
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        action = mu + std * eps
        log_prob = (-0.5 * (((action - mu) / (std + 1e-8)) ** 2
                    + 2 * log_std + math.log(2 * math.pi))).sum(dim=-1, keepdim=True)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + log_std).mean(dim=-1, keepdim=True)
        return action, log_prob, entropy

    def log_prob_of(self, x, action):
        """Compute log_prob of a given action under the current policy."""
        mu, log_std = self(x)
        std = torch.exp(log_std)
        log_prob = (-0.5 * (((action - mu) / (std + 1e-8)) ** 2
                    + 2 * log_std + math.log(2 * math.pi))).sum(dim=-1, keepdim=True)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + log_std).mean(dim=-1, keepdim=True)
        return log_prob, entropy


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPOTrainer:
    """
    PPO trainer for the cooperative evolution between generator and detector.
    
    The policy generates delta vectors in input space. The reward is designed
    to balance diversity (entropy) and adversarial quality (deceiving detector),
    following the DELTA reward formulation with gamma decay.
    """
    def __init__(
            self,
            policy_net: PolicyNetwork,
            value_net: ValueNetwork,
            policy_lr=1e-4,
            value_lr=1e-4,
            gamma=0.99,
            clip_epsilon=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
            device="cpu"
    ):
        self.device = device
        self.policy_net = policy_net.to(device)
        self.value_net = value_net.to(device)
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_optimizer = Adam(self.value_net.parameters(), lr=value_lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient

    def ppo_update(self, states, actions, old_log_probs, returns, advantages, n_epochs=4):
        """
        PPO clipped objective update.
        
        Args:
            states: (N, input_dim) — the x_orig used to generate each sample
            actions: (N, output_dim) — the delta vectors that were applied
            old_log_probs: (N, 1) — log_prob at the time of generation
            returns: (N, 1) — reward (used as return in single-step setting)
            advantages: (N, 1) — advantage estimates
            n_epochs: number of PPO update epochs per call
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        for _ in range(n_epochs):
            # Recompute log_prob of the SAME actions under current policy
            new_log_probs, entropy = self.policy_net.log_prob_of(states, actions)

            log_ratio = torch.clamp(new_log_probs - old_log_probs.detach(), -20, 20)
            ratio = torch.exp(log_ratio)

            adv = advantages.detach()
            # Normalize advantages for stability
            if adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            obj1 = ratio * adv
            obj2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                               1.0 + self.clip_epsilon) * adv
            policy_loss = -torch.mean(torch.min(obj1, obj2))

            # Value loss
            values_pred = self.value_net(states)
            value_loss = F.mse_loss(values_pred, returns)

            # Entropy bonus
            entropy_bonus = entropy.mean()

            total_loss = (policy_loss
                          + self.value_coefficient * value_loss
                          - self.entropy_coefficient * entropy_bonus)

            # Skip step if loss is non-finite
            if not torch.isfinite(total_loss):
                continue

            # Snapshot parameters for potential rollback
            policy_snapshot = {k: v.clone() for k, v in self.policy_net.state_dict().items()}
            value_snapshot = {k: v.clone() for k, v in self.value_net.state_dict().items()}

            # Update policy
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
            self.policy_optimizer.step()
            self.value_optimizer.step()

            # Rollback if parameters became non-finite
            if not (all(torch.isfinite(p).all() for p in self.policy_net.parameters()) and
                    all(torch.isfinite(p).all() for p in self.value_net.parameters())):
                self.policy_net.load_state_dict(policy_snapshot)
                self.value_net.load_state_dict(value_snapshot)

        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy": entropy_bonus.item()}


# =========================
# 6. Alternative Detector Architectures (LSTM, CNN) for arch ablation
# =========================
class LSTMDetector(nn.Module):
    """
    BiLSTM detector. Accepts flat input (N, input_size) and reshapes
    internally to (N, window_size, num_features) where
    window_size = input_size // num_features.
    Output: (N,) logit per sample, matching TransformerDetector.
    """
    def __init__(self, input_size, num_features=9, hidden_size=128,
                 num_layers=2, dropout=0.1, bidirectional=True):
        super(LSTMDetector, self).__init__()
        assert input_size % num_features == 0, (
            f"input_size ({input_size}) must be divisible by num_features ({num_features})"
        )
        self.num_features = num_features
        self.window_size = input_size // num_features

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), self.window_size, self.num_features)
        elif x.dim() == 3 and x.size(-1) != self.num_features:
            x = x.view(x.size(0), self.window_size, self.num_features)
        # cuDNN LSTM cannot backward in eval() mode (needed by One_Step_To_Feasible_Action
        # which calls detector.eval() then propagates gradient through it). Fall back to
        # the native (non-cuDNN) LSTM only in that case; normal train/inference keeps cuDNN.
        if (not self.training) and torch.is_grad_enabled() and x.requires_grad:
            with torch.backends.cudnn.flags(enabled=False):
                out, _ = self.lstm(x)
        else:
            out, _ = self.lstm(x)
        h = out.mean(dim=1)
        return self.fc(h).squeeze(1)


class CNNDetector(nn.Module):
    """
    1D-CNN detector. Accepts flat input (N, input_size) and reshapes
    internally to (N, num_features, window_size) for Conv1d.
    Output: (N,) logit per sample, matching TransformerDetector.
    """
    def __init__(self, input_size, num_features=9, channels=(64, 128, 128),
                 kernel_size=3, dropout=0.1):
        super(CNNDetector, self).__init__()
        assert input_size % num_features == 0, (
            f"input_size ({input_size}) must be divisible by num_features ({num_features})"
        )
        self.num_features = num_features
        self.window_size = input_size // num_features

        layers = []
        in_ch = num_features
        for ch in channels:
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=kernel_size,
                                    padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), self.window_size, self.num_features).transpose(1, 2)
        elif x.dim() == 3 and x.size(-1) == self.num_features:
            x = x.transpose(1, 2)
        elif x.dim() == 3 and x.size(-1) != self.window_size:
            x = x.view(x.size(0), self.window_size, self.num_features).transpose(1, 2)
        h = self.conv(x)
        h = self.pool(h).squeeze(-1)
        return self.fc(h).squeeze(1)


# =========================
# 7. GDN (Graph Deviation Network) Detector
#    Faithful-ish re-impl of Deng & Hooi, AAAI 2021, adapted to binary
#    classification head. Per-channel learnable embeddings induce a top-k
#    cosine-similarity adjacency over sensor channels; per-channel temporal
#    features are aggregated by graph attention; flattened nodes feed an MLP.
# =========================
class GDNDetector(nn.Module):
    def __init__(self, input_size, num_features=9, embed_dim=64,
                 hidden_dim=128, top_k=3, kernel_size=5, dropout=0.1):
        super(GDNDetector, self).__init__()
        assert input_size % num_features == 0
        self.num_features = num_features
        self.window_size = input_size // num_features
        self.top_k = min(top_k, num_features - 1)
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.sensor_embed = nn.Embedding(num_features, embed_dim)
        self.temporal = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.attn = nn.Linear(hidden_dim + embed_dim, 1)
        self.W_node = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), self.window_size, self.num_features)
        elif x.dim() == 3 and x.size(-1) != self.num_features:
            x = x.view(x.size(0), self.window_size, self.num_features)
        B, T, C = x.shape

        x_ch = x.permute(0, 2, 1).contiguous().view(B * C, 1, T)
        h_ch = self.temporal(x_ch).squeeze(-1)
        h = h_ch.view(B, C, self.hidden_dim)

        emb = self.sensor_embed.weight
        emb_n = F.normalize(emb, dim=-1)
        sim = emb_n @ emb_n.t()
        sim_mask = sim.clone()
        sim_mask.fill_diagonal_(float('-inf'))
        topk_idx = sim_mask.topk(self.top_k, dim=-1).indices

        h_w = self.W_node(h)
        h_neigh = h_w[:, topk_idx]
        emb_self = emb.unsqueeze(1).expand(-1, self.top_k, -1)
        emb_self_b = emb_self.unsqueeze(0).expand(B, -1, -1, -1)
        attn_in = torch.cat([h_neigh, emb_self_b], dim=-1)
        attn_logits = self.attn(attn_in).squeeze(-1)
        attn_w = F.softmax(attn_logits, dim=-1).unsqueeze(-1)
        h_agg = (attn_w * h_neigh).sum(dim=2)

        h_out = F.relu(h + h_agg)
        return self.head(h_out.flatten(1)).squeeze(-1)


# =========================
# 8. Mamba Detector (pure-PyTorch selective-scan, mamba-minimal style)
#    Re-implements the S6 block from Gu & Dao 2024 with a sequential Python
#    scan. Slower than the fused CUDA kernel but math-identical, and
#    adequate for our short windows (T=60). Used as the detector backbone
#    in our architecture ablation.
# =========================
def _selective_scan(u, dt, A, B, C, D):
    # Sequential selective scan (math-identical to Mamba S6).
    # Stable: avoids the cumulative-product overflow that breaks log-space
    # parallel scans for typical Mamba init (sum_t dt*A can reach -1000 over
    # T=60 with d_state=16, making exp(-log_P) overflow). Sequential keeps
    # state bounded. For T=60 the Python loop is ~170ms/iter at B=128 which
    # is acceptable for the architecture-ablation runs we need.
    Bsz, T, N = u.shape
    deltaA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    deltaB_u = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
    state = torch.zeros(Bsz, N, A.shape[1], device=u.device, dtype=u.dtype)
    ys = []
    for t in range(T):
        state = deltaA[:, t] * state + deltaB_u[:, t]
        y_t = (state * C[:, t].unsqueeze(1)).sum(dim=-1)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)
    return y + u * D


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = max(1, d_model // 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner,
                                kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner,
                                self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = x_in.transpose(1, 2)
        x_in = self.conv1d(x_in)[:, :, :T]
        x_in = x_in.transpose(1, 2)
        x_in = F.silu(x_in)

        x_dbl = self.x_proj(x_in)
        dt, Bmat, Cmat = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)

        y = _selective_scan(x_in, dt, A, Bmat, Cmat, self.D)
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaDetector(nn.Module):
    def __init__(self, input_size, num_features=9, d_model=64, n_layers=2,
                 d_state=16, d_conv=4, expand=2, dropout=0.1):
        super(MambaDetector, self).__init__()
        assert input_size % num_features == 0
        self.num_features = num_features
        self.window_size = input_size // num_features
        self.embed = nn.Linear(num_features, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), self.window_size, self.num_features)
        elif x.dim() == 3 and x.size(-1) != self.num_features:
            x = x.view(x.size(0), self.window_size, self.num_features)
        h = self.embed(x)
        for blk, norm in zip(self.layers, self.norms):
            h = h + self.dropout(blk(norm(h)))
        h = h.mean(dim=1)
        return self.head(h).squeeze(-1)


# =========================
# 9. TimesNet Detector (Wu et al., ICLR 2023)
#    Multi-period 2D-Inception backbone. FFT picks top-k periods; each
#    period folds the 1D series into a (T/p, p) 2D map, an Inception
#    block with multiple kernel sizes processes it, and per-period
#    outputs are aggregated with amplitude-derived softmax weights.
# =========================
class _InceptionV1(nn.Module):
    def __init__(self, in_ch, out_ch, num_kernels=4):
        super(_InceptionV1, self).__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])

    def forward(self, x):
        outs = [k(x) for k in self.kernels]
        return torch.stack(outs, dim=-1).mean(-1)


def _fft_top_k_periods(x, k):
    # x: (B, T, D). Returns (periods (k,), weights (B, k)).
    B, T, _ = x.shape
    xf = torch.fft.rfft(x, dim=1)
    amp = xf.abs().mean(dim=-1)               # (B, T//2+1)
    amp_global = amp.mean(dim=0)              # (T//2+1,)
    amp_global = amp_global.clone()
    amp_global[0] = 0                         # ignore DC
    _, top_idx = torch.topk(amp_global, k)    # (k,)
    periods = T // top_idx.clamp(min=1)       # (k,)
    periods = periods.clamp(min=1, max=T)
    weights = amp[:, top_idx]                 # (B, k)
    return periods, weights


class TimesBlock(nn.Module):
    def __init__(self, d_model, d_ff, k=3, num_kernels=4):
        super(TimesBlock, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            _InceptionV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            _InceptionV1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        periods, weights = _fft_top_k_periods(x, self.k)
        outs = []
        for p in periods.tolist():
            pad = (p - T % p) % p
            if pad:
                x_p = F.pad(x, (0, 0, 0, pad))
            else:
                x_p = x
            t_p = x_p.shape[1] // p
            x2d = x_p.view(B, t_p, p, D).permute(0, 3, 1, 2).contiguous()  # (B,D,t_p,p)
            x2d = self.conv(x2d)
            x_back = x2d.permute(0, 2, 3, 1).reshape(B, t_p * p, D)
            if pad:
                x_back = x_back[:, :T, :]
            outs.append(x_back)
        outs = torch.stack(outs, dim=-1)              # (B, T, D, k)
        w = F.softmax(weights, dim=-1)                # (B, k)
        # Broadcast: (B, k) -> (B, 1, 1, k)
        agg = (outs * w.view(B, 1, 1, self.k)).sum(dim=-1)
        return agg


class TimesNetDetector(nn.Module):
    def __init__(self, input_size, num_features=9, d_model=64, d_ff=128,
                 n_layers=2, k=3, num_kernels=4, dropout=0.1):
        super(TimesNetDetector, self).__init__()
        assert input_size % num_features == 0
        self.num_features = num_features
        self.window_size = input_size // num_features
        self.embed = nn.Linear(num_features, d_model)
        self.blocks = nn.ModuleList([
            TimesBlock(d_model, d_ff, k=k, num_kernels=num_kernels)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), self.window_size, self.num_features)
        elif x.dim() == 3 and x.size(-1) != self.num_features:
            x = x.view(x.size(0), self.window_size, self.num_features)
        h = self.embed(x)
        for blk, norm in zip(self.blocks, self.norms):
            h = h + self.dropout(blk(norm(h)))
        h = h.mean(dim=1)
        return self.head(h).squeeze(-1)


# =========================
# 10. LSTM-AD Detector (Malhotra et al., 2015)
#     Stacked unidirectional LSTM; classification head reads the last
#     hidden state of the top layer. (The original LSTM-AD uses
#     next-step prediction error as the anomaly score; we adapt the
#     architectural choices to a binary head so the detector slots into
#     the DELTA pipeline without changes to training utilities.)
# =========================
class LSTMADDetector(nn.Module):
    def __init__(self, input_size, num_features=9, hidden_size=128,
                 num_layers=2, dropout=0.1):
        super(LSTMADDetector, self).__init__()
        assert input_size % num_features == 0
        self.num_features = num_features
        self.window_size = input_size // num_features

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), self.window_size, self.num_features)
        elif x.dim() == 3 and x.size(-1) != self.num_features:
            x = x.view(x.size(0), self.window_size, self.num_features)
        # cuDNN LSTM cannot backward in eval() mode; the warm-up
        # One_Step_To_Feasible_Action path propagates gradients through
        # the detector in eval mode, so fall back to native LSTM there.
        if (not self.training) and torch.is_grad_enabled() and x.requires_grad:
            with torch.backends.cudnn.flags(enabled=False):
                _, (h, _) = self.lstm(x)
        else:
            _, (h, _) = self.lstm(x)
        h_last = h[-1]                       # (B, hidden_size)
        return self.fc(h_last).squeeze(-1)


# =========================
# 11. DLinear Detector (Zeng et al., AAAI 2023)
#     Series decomposition (trend = moving-average, seasonal = residual)
#     followed by channel-independent linear projections per part. We
#     read out a binary logit from a small MLP head on the flattened
#     reconstructed signal. No CUDA / RNN dependence.
# =========================
class _SeriesDecomp(nn.Module):
    """Moving-average decomposition with edge-replicate padding."""
    def __init__(self, kernel_size):
        super(_SeriesDecomp, self).__init__()
        # kernel_size must be odd for symmetric padding
        if kernel_size % 2 == 0:
            kernel_size = max(3, kernel_size - 1)
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: (B, T, C)
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x_pad = torch.cat([front, x, end], dim=1)           # (B, T+2p, C)
        trend = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, C)
        seasonal = x - trend
        return seasonal, trend


class DLinearDetector(nn.Module):
    def __init__(self, input_size, num_features=9, kernel_size=25, dropout=0.1):
        super(DLinearDetector, self).__init__()
        assert input_size % num_features == 0
        self.num_features = num_features
        self.window_size = input_size // num_features

        # Clamp kernel to <= window and odd.
        ks = min(kernel_size, self.window_size - 1)
        if ks % 2 == 0:
            ks -= 1
        if ks < 3:
            ks = 3
        self.decomp = _SeriesDecomp(ks)
        # Channel-independent linear projections (one pair per channel).
        self.linear_seasonal = nn.ModuleList(
            [nn.Linear(self.window_size, self.window_size) for _ in range(num_features)])
        self.linear_trend = nn.ModuleList(
            [nn.Linear(self.window_size, self.window_size) for _ in range(num_features)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(self.window_size * num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), self.window_size, self.num_features)
        elif x.dim() == 3 and x.size(-1) != self.num_features:
            x = x.view(x.size(0), self.window_size, self.num_features)
        seasonal, trend = self.decomp(x)                # both (B, T, C)
        outs = []
        for c in range(self.num_features):
            s = self.linear_seasonal[c](seasonal[:, :, c])  # (B, T)
            t = self.linear_trend[c](trend[:, :, c])         # (B, T)
            outs.append(s + t)
        y = torch.stack(outs, dim=-1)                       # (B, T, C)
        flat = self.dropout(y.flatten(1))                   # (B, T*C)
        return self.head(flat).squeeze(-1)


# =========================
# 12. Dilated TCN Detector (Bai et al., 2018)
#     Stack of dilated causal Conv1d residual blocks. Dilations
#     1,2,4,8 with kernel=3 give a receptive field of 61 -> covers
#     the T=60 window. Pure-Conv, no recurrence -> no cuDNN-eval
#     shim required for the One_Step_To_Feasible_Action path.
# =========================
class _TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super(_TCNBlock, self).__init__()
        pad = (kernel_size - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size,
                               padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def _chomp(self, x):
        return x[:, :, :-self.pad] if self.pad > 0 else x

    def forward(self, x):
        h = self.drop(F.relu(self._chomp(self.conv1(x))))
        h = self.drop(F.relu(self._chomp(self.conv2(h))))
        r = x if self.res is None else self.res(x)
        return F.relu(h + r)


class DilatedTCNDetector(nn.Module):
    def __init__(self, input_size, num_features=9, channels=(64, 64, 64, 64),
                 kernel_size=3, dropout=0.1):
        super(DilatedTCNDetector, self).__init__()
        assert input_size % num_features == 0, (
            f"input_size ({input_size}) must be divisible by num_features ({num_features})"
        )
        self.num_features = num_features
        self.window_size = input_size // num_features
        in_ch = num_features
        blocks = []
        for i, ch in enumerate(channels):
            blocks.append(_TCNBlock(in_ch, ch, kernel_size,
                                    dilation=2 ** i, dropout=dropout))
            in_ch = ch
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), self.window_size, self.num_features).transpose(1, 2)
        elif x.dim() == 3 and x.size(-1) == self.num_features:
            x = x.transpose(1, 2)
        elif x.dim() == 3 and x.size(-1) != self.window_size:
            x = x.view(x.size(0), self.window_size, self.num_features).transpose(1, 2)
        h = self.tcn(x)
        h = self.pool(h).squeeze(-1)
        return self.head(h).squeeze(1)


# =========================
# 13. GAT-GRU Detector (channel-axis GAT + temporal GRU)
#     MTAD-GAT-inspired simplification: a Conv1d-encoder lifts each
#     channel's T-trace to a node embedding, one GAT layer mixes
#     channels, and a GRU rolls the per-step features along time.
#     Same cuDNN-eval shim as LSTMDetector so the warm-up
#     One_Step_To_Feasible_Action path can backprop through
#     detector.eval().
# =========================
class _ChannelGAT(nn.Module):
    """Single multi-head GAT layer (Velickovic et al., 2018) over a small
    fully connected channel graph."""
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1):
        super(_ChannelGAT, self).__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h):
        B, C, _ = h.shape
        Wh = self.W(h).view(B, C, self.n_heads, self.head_dim)
        e_src = (Wh * self.a_src).sum(dim=-1)
        e_dst = (Wh * self.a_dst).sum(dim=-1)
        e = e_src.permute(0, 2, 1).unsqueeze(-1) + e_dst.permute(0, 2, 1).unsqueeze(-2)
        e = self.leaky(e)
        attn = F.softmax(e, dim=-1)
        attn = self.dropout(attn)
        Wh_h = Wh.permute(0, 2, 1, 3)
        out = torch.matmul(attn, Wh_h)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, C, -1)
        return F.elu(out)


class GATGRUDetector(nn.Module):
    def __init__(self, input_size, num_features=9, d_node=64, gat_heads=4,
                 gru_hidden=128, gru_layers=1, dropout=0.1):
        super(GATGRUDetector, self).__init__()
        assert input_size % num_features == 0, (
            f"input_size ({input_size}) must be divisible by num_features ({num_features})"
        )
        self.num_features = num_features
        self.window_size = input_size // num_features
        self.d_node = d_node

        self.ch_encoder = nn.Sequential(
            nn.Conv1d(1, d_node, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_node, d_node, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.gat = _ChannelGAT(d_node, d_node, n_heads=gat_heads, dropout=dropout)
        self.ch_proj = nn.Linear(d_node * num_features, d_node)
        self.step_lin = nn.Linear(num_features, d_node)

        self.gru = nn.GRU(
            input_size=d_node,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), self.window_size, self.num_features)
        elif x.dim() == 3 and x.size(-1) != self.num_features:
            x = x.view(x.size(0), self.window_size, self.num_features)
        B, T, C = x.shape

        x_ch = x.permute(0, 2, 1).contiguous().view(B * C, 1, T)
        h_ch = self.ch_encoder(x_ch).squeeze(-1).view(B, C, self.d_node)

        h_gat = self.gat(h_ch)
        ch_summary = h_gat.reshape(B, -1)
        ch_bias = self.ch_proj(ch_summary).unsqueeze(1)

        step_feat = self.step_lin(x)
        seq = step_feat + ch_bias

        if (not self.training) and torch.is_grad_enabled() and seq.requires_grad:
            with torch.backends.cudnn.flags(enabled=False):
                out, _ = self.gru(seq)
        else:
            out, _ = self.gru(seq)
        h = out.mean(dim=1)
        return self.head(h).squeeze(-1)

# training/algorithms/ppo_lstm/ppo_lstm_policy.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Robust import for your MaskedCategorical (works whether flat or packaged)
try:
    from training.algorithms.ppo_lstm.masked_categorical import MaskedCategorical
except Exception:
    from masked_categorical import MaskedCategorical  # fallback for flat layout


# ----------------------------
# Small utility modules
# ----------------------------

def _act(name: str) -> nn.Module:
    name = (name or "gelu").lower()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    return nn.GELU()


def _norm(norm: str, num_channels: int) -> nn.Module:
    norm = (norm or "layernorm").lower()
    if norm == "batchnorm":
        return nn.BatchNorm2d(num_channels)
    if norm == "groupnorm":
        return nn.GroupNorm(num_groups=8, num_channels=num_channels)
    # Default: LayerNorm over (C,H,W) via channels_last
    return _ChannelwiseLayerNorm(num_channels)


class _ChannelwiseLayerNorm(nn.Module):
    """LayerNorm that normalizes across C with channels_last tricks for (N,C,H,W)."""
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,C,H,W) -> (N,H,W,C) for LN -> back
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation (channel attention)."""
    def __init__(self, channels: int, reduction: int = 16, act="silu"):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            _act(act),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class ResidualBlock(nn.Module):
    def __init__(self, c: int, dilation: int = 1, norm="layernorm", act="gelu", se: bool = True, preact: bool = True):
        super().__init__()
        self.preact = preact
        self.n1 = _norm(norm, c)
        self.a1 = _act(act)
        self.c1 = nn.Conv2d(c, c, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.n2 = _norm(norm, c)
        self.a2 = _act(act)
        self.c2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.se = SEBlock(c) if se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        y = x
        if self.preact:
            y = self.n1(y)
            y = self.a1(y)
        y = self.c1(y)
        y = self.n2(y)
        y = self.a2(y)
        y = self.c2(y)
        y = self.se(y)
        return r + y


class Positional2D(nn.Module):
    """Learned 2D positional bias for (H,W) grid, broadcast to token dim."""
    def __init__(self, H: int, W: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, H * W, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, B: int) -> torch.Tensor:
        # (1, HW, D) -> (B, HW, D)
        return self.pos.expand(B, -1, -1)


class TransformerGridEncoder(nn.Module):
    """
    Lightweight Transformer over grid tokens (and optional [CLS] + hand tokens).
    To avoid blowing up LSTM input, we PROJECT BACK to conv channels and add to the feature map (residual).
    """
    def __init__(
        self,
        conv_channels: int,
        H: int,
        W: int,
        *,
        enabled: bool = True,
        d_model: int = 128,
        n_heads: int = 8,
        ffn_hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.0,
        pre_ln: bool = True,
        use_cls: bool = True,
        positional: str = "relative",
    ):
        super().__init__()
        self.enabled = enabled
        if not enabled:
            # stub
            self.proj_in = nn.Identity()
            self.encoder = nn.Identity()
            self.proj_out = nn.Identity()
            self.H, self.W = H, W
            self.use_cls = use_cls
            self.pos = None
            return

        self.H, self.W = H, W
        self.use_cls = use_cls

        self.proj_in = nn.Linear(conv_channels, d_model, bias=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=pre_ln,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.proj_out = nn.Linear(d_model, conv_channels, bias=True)

        # learned absolute 2D pos is plenty here; no extra relative impl needed
        self.pos = Positional2D(H, W, d_model)

        # [CLS] token (optional)
        if self.use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)
        else:
            self.cls = None

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, C, H, W)  ->  returns residual-enhanced features (same shape)
        """
        if not self.enabled:
            return feat

        B, C, H, W = feat.shape
        assert H == self.H and W == self.W, "TransformerGridEncoder H/W mismatch"

        # (B,C,H,W) -> (B, HW, C)
        x = feat.flatten(2).transpose(1, 2)  # B, HW, C
        x = self.proj_in(x)                  # B, HW, D
        x = x + self.pos(B)                  # add absolute pos

        if self.use_cls:
            cls = self.cls.expand(B, -1, -1)     # (B,1,D)
            x = torch.cat([cls, x], dim=1)       # (B, 1+HW, D)

        x = self.encoder(x)                   # B, T, D

        # drop CLS before projection back to grid channels
        if self.use_cls:
            x = x[:, 1:, :]                   # B, HW, D

        x = self.proj_out(x)                  # B, HW, C
        x = x.transpose(1, 2).reshape(B, C, H, W)  # B,C,H,W

        # Residual enhancement
        return feat + x


# ----------------------------
# Main PPO Recurrent Policy
# ----------------------------

class PPORecurrentPolicy(nn.Module):
    """
    CNN -> (optional ResNet tower) -> (optional Transformer grid encoder) -> LSTM -> heads
    - Keeps LSTM input size the SAME (we only modify feature maps and project back), so the 1.7× bump
      is controlled by widening LSTM hidden (128 -> 216).
    """

    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        action_dim: int,
        conv_channels,
        lstm_hidden: int,
        lstm_layers: int = 2,
        device: torch.device | str = "cpu",
        value_tanh_bound: float = 5.0,
        *,
        model_cfg: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.obs_shape = obs_shape
        self.action_dim = int(action_dim)
        self.value_bound = float(value_tanh_bound)
        self.deterministic = bool(deterministic)

        C_in, H, W = int(obs_shape[0]), int(obs_shape[1]), int(obs_shape[2])
        model_cfg = model_cfg or {}
        act_name = model_cfg.get("activation", "gelu")
        norm_name = model_cfg.get("norm", "layernorm")

        # -------- CNN stem (kept from your current config) --------
        chs = [C_in] + list(conv_channels)
        conv_layers = []
        for idx in range(1, len(chs)):
            conv_layers += [
                nn.Conv2d(chs[idx - 1], chs[idx], kernel_size=3, padding=1, bias=False),
                _norm(norm_name, chs[idx]),
                _act(act_name),
            ]
        self.conv = nn.Sequential(*conv_layers)
        conv_out_ch = chs[-1]

        # -------- Optional ResNet tower (same channel width) --------
        bb_cfg = model_cfg.get("backbone", {})
        use_resnet = bool(bb_cfg.get("enabled", False))
        blocks = int(bb_cfg.get("blocks", 0 if not use_resnet else 8))
        se = bool(bb_cfg.get("use_se", True))
        preact = bool(bb_cfg.get("preact", True))
        dil_sched = list(bb_cfg.get("dilation_schedule", [1, 1, 2, 1, 1, 2, 1, 1]))

        res_layers = []
        for i in range(blocks):
            d = dil_sched[i % len(dil_sched)]
            res_layers.append(ResidualBlock(conv_out_ch, dilation=d, norm=norm_name, act=act_name, se=se, preact=preact))
        self.resnet = nn.Sequential(*res_layers) if blocks > 0 else nn.Identity()

        # -------- Transformer grid encoder (residual to feature map) --------
        tr_cfg = model_cfg.get("transformer", {})
        self.tr = TransformerGridEncoder(
            conv_channels=conv_out_ch,
            H=H, W=W,
            enabled=bool(tr_cfg.get("enabled", False)),
            d_model=int(tr_cfg.get("d_model", 128)),
            n_heads=int(tr_cfg.get("n_heads", 8)),
            ffn_hidden=int(tr_cfg.get("ffn_hidden", 256)),
            layers=int(tr_cfg.get("layers", 2)),
            dropout=float(tr_cfg.get("dropout", 0.0)),
            pre_ln=bool(tr_cfg.get("pre_ln", True)),
            use_cls=bool(tr_cfg.get("tokens", {}).get("global_cls", True)),
            positional=str(tr_cfg.get("positional", "relative")),
        )

        # -------- Flatten -> LSTM --------
        self.spatial_dim = H * W * conv_out_ch
        self.flatten = nn.Flatten(start_dim=1)
        self.lstm = nn.LSTM(
            input_size=self.spatial_dim,
            hidden_size=int(lstm_hidden),
            num_layers=int(lstm_layers),
            batch_first=False,  # we use (seq=1, batch, feat)
        )

        # -------- Heads --------
        # Policy: MLP(lstm_hidden -> 256 -> action_dim)
        self.pi = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            _act(act_name),
            nn.Linear(256, self.action_dim),
        )
        # Value: MLP(lstm_hidden -> 128 -> 1)
        self.v_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            _act(act_name),
            nn.Linear(128, 1),
        )

        # Inits
        self.apply(self._init_weights)

        self.to(self.device)

    # ---------------- API expected by your learner/train loop ----------------

    def get_initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=self.device)
        c = torch.zeros_like(h)
        return h, c

    @torch.no_grad()
    def select_action(
        self,
        legal_mask: Optional[torch.Tensor],
        ctx: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        ctx expects:
          - "obs": (N,C,H,W)
          - "h0":  (L,N,H)
          - "c0":  (L,N,H)
        Returns:
          action:   (N,)
          log_prob: (N,)
          value:    (N,)
          h, c:     (L,N,H)
        """
        obs = ctx["obs"].to(self.device, dtype=torch.float32)
        h0  = ctx["h0"].to(self.device)
        c0  = ctx["c0"].to(self.device)

        logits, value, (hn, cn) = self._forward_core(obs, h0, c0)

        mask = None if legal_mask is None else torch.as_tensor(legal_mask, device=self.device, dtype=torch.float32)
        dist = MaskedCategorical.from_logits_and_mask(logits, mask)

        if self.deterministic:
            action = torch.argmax(dist.logits, dim=-1)
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return {
            "action": action,
            "log_prob": log_prob,
            "value": value.squeeze(-1),
            "h": hn,
            "c": cn,
        }

    def forward(
        self,
        obs: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor,
        # accept common mask kw names so generic callers succeed
        legal_mask: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generic forward used for bootstrapping AND (as a fallback) PPO evaluation.
        Returns:
          {
            "value":  (N,),
            "logits": (N, A),
            "dist":   MaskedCategorical  # built from logits and the provided mask (if any)
          }
        """
        obs = obs.to(self.device, dtype=torch.float32)

        # unify mask arg
        mask = legal_mask
        if mask is None:
            mask = masks
        if mask is None:
            mask = action_mask
        if mask is not None:
            mask = mask.to(self.device, dtype=torch.float32)

        logits, value, _ = self._forward_core(obs, h0.to(self.device), c0.to(self.device))
        dist = MaskedCategorical.from_logits_and_mask(logits, mask)

        return {
            "value": value.squeeze(-1),
            "logits": logits,
            "dist": dist,
        }


    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        legal_mask: Optional[torch.Tensor],
        h0: torch.Tensor,
        c0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For PPO update:
          returns (log_probs [B], entropy [], values [B])
        """
        logits, value, _ = self._forward_core(
            obs.to(self.device, dtype=torch.float32),
            h0.to(self.device), c0.to(self.device)
        )
        mask = None if legal_mask is None else legal_mask.to(self.device, dtype=torch.float32)

        dist = MaskedCategorical.from_logits_and_mask(logits, mask)
        log_probs = dist.log_prob(actions.to(self.device))
        entropy = dist.entropy().mean()
        return log_probs, entropy, value.squeeze(-1)

    # ---------------- internal core ----------------

    def _forward_core(
        self,
        obs: torch.Tensor,   # (N,C,H,W)
        h0: torch.Tensor,    # (L,N,H)
        c0: torch.Tensor,    # (L,N,H)
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # CNN -> (ResNet) -> (Transformer grid) -> flatten
        x = self.conv(obs)
        x = self.resnet(x)
        x = self.tr(x)                       # residual enhancement keeps channels fixed
        x = self.flatten(x)                  # (N, spatial_dim)

        # LSTM (seq len = 1)
        x_seq = x.unsqueeze(0)               # (1, N, spatial_dim)
        y, (hn, cn) = self.lstm(x_seq, (h0, c0))  # y: (1,N,H)
        y = y.squeeze(0)                     # (N, H)

        # Heads
        logits = self.pi(y)                  # (N, A)
        value  = self.v_head(y)              # (N, 1)
        # bound value with tanh if requested
        if math.isfinite(self.value_bound) and self.value_bound > 0:
            value = torch.tanh(value) * self.value_bound

        return logits, value, (hn, cn)

    # ---------------- inits ----------------

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

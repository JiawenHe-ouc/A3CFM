from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict 
try:
    from transformers import CLIPVisionModel, WavLMConfig, WavLMModel
except Exception:
    CLIPVisionModel = None
    WavLMConfig = None
    WavLMModel = None
    print("Warning: transformers not available; FaceEncoder will fall back to ConvNet and WavLM audio will be unavailable.")
try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: [B, D, T]
        w = torch.softmax(self.attn(h), dim=-1)
        pooled = (h * w).sum(dim=-1)
        return pooled, w


class PhysioEncoder(nn.Module):
    """Physiological encoder returning h_p(t) and pooled z_p."""

    def __init__(self, in_channels: int, embedding_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.out_channels = hidden_dim * 2
        self.features = nn.Sequential(
            ConvBlock1D(in_channels, hidden_dim, 9, 2, dropout),
            ConvBlock1D(hidden_dim, hidden_dim, 7, 2, dropout),
            ConvBlock1D(hidden_dim, self.out_channels, 5, 2, dropout),
            ConvBlock1D(self.out_channels, self.out_channels, 5, 2, dropout),
        )
        self.pool = TemporalAttentionPooling(self.out_channels, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.out_channels, embedding_dim),
        )
        self.temporal_proj = nn.Sequential(
            nn.Linear(self.out_channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def encode_temporal(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, T]
        h = self.features(x)
        pooled_raw, _ = self.pool(h)
        z = F.normalize(self.proj(pooled_raw), dim=-1)
        h_seq = self.temporal_proj(h.transpose(1, 2))
        h_seq = F.normalize(h_seq, dim=-1)
        return h_seq, z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, z = self.encode_temporal(x)
        return z


class AudioEncoder(nn.Module):
   
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        use_mel: bool = False,          # retained for backward-compatible config files
        n_mels: int = 64,               # retained for backward-compatible config files
        sample_rate: int = 16000,
        wavlm_config: Optional[object] = None,
        freeze_feature_encoder: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        del hidden_dim, use_mel, n_mels
        if WavLMModel is None or WavLMConfig is None:
            raise ImportError(
                "WavLM audio encoding requires transformers. Install a compatible "
                "version with: pip install 'transformers>=4.40'."
            )
        if int(sample_rate) != 16000:
            raise ValueError(
                f"WavLM expects 16 kHz waveform input, but sample_rate={sample_rate}. "
                "Resample audio to 16000 Hz in AudioEmotionDataset."
            )

        if wavlm_config is None:
            wavlm_config = WavLMConfig()
        elif isinstance(wavlm_config, dict):
            wavlm_config = WavLMConfig(**wavlm_config)
        self.wavlm = WavLMModel(wavlm_config)
        self.wavlm_hidden_size = int(self.wavlm.config.hidden_size)
        self.sample_rate = 16000
        self.freeze_feature_encoder = bool(freeze_feature_encoder)

        if self.freeze_feature_encoder:
            freeze_fn = getattr(self.wavlm.feature_extractor, "_freeze_parameters", None)
            if callable(freeze_fn):
                freeze_fn()
            else:
                for p in self.wavlm.feature_extractor.parameters():
                    p.requires_grad = False

        if gradient_checkpointing:
            enable_gc = getattr(self.wavlm, "gradient_checkpointing_enable", None)
            if callable(enable_gc):
                enable_gc()

        self.temporal_proj = nn.Sequential(
            nn.LayerNorm(self.wavlm_hidden_size),
            nn.Linear(self.wavlm_hidden_size, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool_score = nn.Sequential(
            nn.Linear(embedding_dim, max(64, embedding_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(64, embedding_dim // 2), 1),
        )
        self.pooled_proj = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    @staticmethod
    def _prepare_waveform(wav: torch.Tensor) -> torch.Tensor:
        """Return finite, RMS-normalized mono waveform shaped [B, T]."""
        if wav.dim() == 3:
            if wav.shape[1] == 1:
                wav = wav[:, 0]
            else:
                wav = wav.mean(dim=1)
        if wav.dim() != 2:
            raise ValueError(f"Expected audio [B,T] or [B,1,T], got {tuple(wav.shape)}")
        wav = torch.nan_to_num(wav.float())
        wav = wav - wav.mean(dim=-1, keepdim=True)
        rms = wav.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-5)
        return wav / rms

    @staticmethod
    def _sample_attention_mask(wav: torch.Tensor) -> torch.Tensor:
        nonzero = wav.abs() > 1e-7
        lengths = nonzero.long().sum(dim=-1).clamp_min(1)
        idx = torch.arange(wav.shape[-1], device=wav.device).unsqueeze(0)
        return (idx < lengths.unsqueeze(1)).long()

    def encode_temporal(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        wav = self._prepare_waveform(wav)
        attention_mask = self._sample_attention_mask(wav)
        outputs = self.wavlm(
            input_values=wav,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        h_wavlm = outputs.last_hidden_state                 # [B, T', H_wavlm]
        h_seq = self.temporal_proj(h_wavlm)                 # [B, T', D]
        h_seq = F.normalize(h_seq, dim=-1)

        feature_mask = None
        get_mask = getattr(self.wavlm, "_get_feature_vector_attention_mask", None)
        if callable(get_mask):
            feature_mask = get_mask(h_seq.shape[1], attention_mask).to(h_seq.device)
        scores = self.pool_score(h_seq).squeeze(-1)         # [B, T']
        if feature_mask is not None:
            scores = scores.masked_fill(~feature_mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (h_seq * weights).sum(dim=1)
        z = F.normalize(self.pooled_proj(pooled) + pooled, dim=-1)
        return h_seq, z

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        _, z = self.encode_temporal(wav)
        return z


class FaceEncoderV1(nn.Module):

    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.out_channels = hidden_dim * 2
        self.features = nn.Sequential(
            ConvBlock2D(3, hidden_dim // 2, 5, 2, dropout),
            ConvBlock2D(hidden_dim // 2, hidden_dim, 3, 2, dropout),
            ConvBlock2D(hidden_dim, hidden_dim, 3, 2, dropout),
            ConvBlock2D(hidden_dim, self.out_channels, 3, 2, dropout),
        )
        self.proj = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.out_channels, embedding_dim),
        )
        self.temporal_proj = nn.Sequential(
            nn.Linear(self.out_channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool_score = nn.Linear(embedding_dim, 1)

    def encode_temporal(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # image: [B,3,H,W], values roughly normalized.
        h = self.features(image)  # [B,C,H',W']
        b, c, hh, ww = h.shape
        tokens_raw = h.flatten(2).transpose(1, 2)  # [B,T,C]
        tokens = F.normalize(self.temporal_proj(tokens_raw), dim=-1)
        attn = torch.softmax(self.pool_score(tokens), dim=1)
        pooled_raw = (tokens_raw * attn).sum(dim=1)
        z = F.normalize(self.proj(pooled_raw), dim=-1)
        return tokens, z

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _, z = self.encode_temporal(image)
        return z

class FaceEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 256, lora_rank: int = 8, dropout: float = 0.1):
        super().__init__()
        self.use_clip = CLIPVisionModel is not None
        self.clip = None
        self.fallback = None

        if self.use_clip:
            try:
                # Try to load CLIP vision model; if it fails, fall back.
                self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
                for p in self.clip.parameters():
                    p.requires_grad = False
            except Exception:
                self.clip = None
                self.use_clip = False

        # If CLIP is available and PEFT is present, attempt LoRA injection.
        if self.clip is not None and LoraConfig is not None and get_peft_model is not None:
            try:
                lora_cfg = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=dropout,
                    bias="none",
                )
                self.clip = get_peft_model(self.clip, lora_cfg)
            except Exception:
                # Fail gracefully and continue with plain CLIP if PEFT injection fails.
                pass

        if self.clip is not None:
            # Typical ViT-L hidden dim; if the model exposes a different size use it.
            clip_hidden = getattr(self.clip.config, "hidden_size", 1024)
            self.cls_proj = nn.Sequential(nn.LayerNorm(clip_hidden), nn.Linear(clip_hidden, embedding_dim))
            self.patch_proj = nn.Sequential(nn.LayerNorm(clip_hidden), nn.Linear(clip_hidden, embedding_dim))
        else:
            # Fallback: use the lightweight convolutional encoder defined above.
            self.fallback = FaceEncoderV1(embedding_dim=embedding_dim, hidden_dim=max(128, embedding_dim // 2), dropout=dropout)
            # Identity projections since fallback already returns desired shapes
            self.cls_proj = nn.Identity()
            self.patch_proj = nn.Identity()

    def encode_temporal(self, face: torch.Tensor):
        """
        Inputs: face [B, C, H, W]
        Returns: (h_seq [B, T, D], z_raw [B, D])
        """
        if self.clip is not None:
            outputs = self.clip.vision_model(pixel_values=face)
            cls_token = outputs.last_hidden_state[:, 0, :]
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            z_raw = self.cls_proj(cls_token)
            h_seq = self.patch_proj(patch_tokens)
            return h_seq, z_raw
        else:
            # FaceEncoderV1 returns (tokens, z)
            tokens, z = self.fallback.encode_temporal(face)
            return tokens, z
    
class TemporalAffectProjector(nn.Module):
    """Projects modality-specific h_m(t) into continuous shared affect latent z_m(t)."""

    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.frame_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1),
        )
        self.pool_score = nn.Linear(embedding_dim, 1)

    def forward(self, h_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_seq: [B, T, D]
        x = self.frame_mlp(h_seq)
        x = x + self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        x = F.normalize(x, dim=-1)
        attn = torch.softmax(self.pool_score(x), dim=1)
        pooled = F.normalize((x * attn).sum(dim=1), dim=-1)
        return x, pooled


class AURegionAttention(nn.Module):

    AU_REGION_CENTERS = {
        # 在 14×14 patch 网格上的行列范围（row_start, row_end, col_start, col_end）
        "brow":   (1, 4,  2, 12),
        "eye":    (4, 6,  2, 12),
        "nose":   (6, 9,  4, 10),
        "mouth":  (9, 13, 3, 11),
        "cheek":  (5, 10, 0, 3),   # 左右合并
    }

    def __init__(self, embedding_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.regions = list(self.AU_REGION_CENTERS.keys())
        self.num_regions = len(self.regions)
        
        # 为每个区域学习一个可学习查询向量
        self.region_queries = nn.Parameter(
            torch.randn(self.num_regions, embedding_dim) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.region_agg = nn.Sequential(
            nn.Linear(self.num_regions * embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _get_region_patches(self, patch_tokens: torch.Tensor, region: str) -> torch.Tensor:
        """Extract an AU region from any square ViT patch grid without changing B."""
        import math

        if patch_tokens.dim() != 3:
            raise ValueError(f"Expected [B,N,D] face tokens, got {tuple(patch_tokens.shape)}")
        b, n, d = patch_tokens.shape
        side = int(math.isqrt(n))
        if side * side != n:
            # Some encoders may accidentally include a CLS token.
            side_minus = int(math.isqrt(max(0, n - 1)))
            if side_minus * side_minus == n - 1:
                patch_tokens = patch_tokens[:, 1:, :]
                side = side_minus
            else:
                side = int(math.ceil(math.sqrt(n)))
                pad_n = side * side - n
                pad = patch_tokens[:, -1:, :].expand(b, pad_n, d)
                patch_tokens = torch.cat([patch_tokens, pad], dim=1)

        # Convert the original 14x14 boxes to the actual grid resolution.
        r0, r1, c0, c1 = self.AU_REGION_CENTERS[region]
        scale = side / 14.0
        rr0 = max(0, min(side - 1, int(math.floor(r0 * scale))))
        rr1 = max(rr0 + 1, min(side, int(math.ceil(r1 * scale))))
        cc0 = max(0, min(side - 1, int(math.floor(c0 * scale))))
        cc1 = max(cc0 + 1, min(side, int(math.ceil(c1 * scale))))
        grid = patch_tokens.reshape(b, side, side, d)
        return grid[:, rr0:rr1, cc0:cc1, :].reshape(b, -1, d)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        B = patch_tokens.shape[0]
        queries = self.region_queries.unsqueeze(0).expand(B, -1, -1)  # [B, R, D]
        
        region_feats = []
        for i, region_name in enumerate(self.regions):
            kv = self._get_region_patches(patch_tokens, region_name)  # [B, n, D]
            q = queries[:, i:i+1, :]                                   # [B, 1, D]
            feat, _ = self.cross_attn(q, kv, kv)                       # [B, 1, D]
            region_feats.append(feat)
        
        region_feats = torch.cat(region_feats, dim=1)  # [B, R, D]
        # 拼接所有区域特征后聚合
        z_local = self.region_agg(region_feats.reshape(B, -1))  # [B, D]
        return z_local


class RegionAwareAffectProjector(nn.Module):

    def __init__(self, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        # 全局路径
        self.global_mlp = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        # 局部路径
        self.local_au = AURegionAttention(embedding_dim, dropout=dropout)
        self.local_norm = nn.LayerNorm(embedding_dim)
        
        # 门控融合：学习每个样本的全局/局部权重
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, h_seq: torch.Tensor, z_raw: torch.Tensor):
        """
        h_seq: [B, 196, D] patch tokens
        z_raw: [B, D]       CLS token
        返回:
            z_seq    [B, 196, D]  时序特征（保留给上层 HierarchicalAffectFusion）
            z_affect [B, D]       最终情感嵌入
        """
        z_global = self.global_mlp(z_raw)          # [B, D]
        z_local = self.local_norm(self.local_au(h_seq))  # [B, D]
        
        # 门控：拼接两路算权重
        gate_w = self.gate(torch.cat([z_global, z_local], dim=-1))  # [B, 2]
        z_affect = gate_w[:, 0:1] * z_global + gate_w[:, 1:2] * z_local  # [B, D]
        
        return h_seq, z_affect  # z_seq 保持原 patch tokens 供后续使用
    
class HierarchicalAffectFusion(nn.Module):
 

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, z_raw: torch.Tensor, z_affect: torch.Tensor, z_seq: torch.Tensor) -> torch.Tensor:
        seq_mean = z_seq.mean(dim=1)
        seq_max = z_seq.max(dim=1).values
        x = torch.cat([z_raw, z_affect, seq_mean, seq_max], dim=-1)
        return F.normalize(self.net(x) + z_affect, dim=-1)


class PromptAwareAffectConditioner(nn.Module):

    def __init__(self, embedding_dim: int = 256, vocab_size: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.prompt_embed = nn.Embedding(self.vocab_size, embedding_dim)
        self.film = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
        )

    def forward(self, z: torch.Tensor, prompt_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if prompt_ids is None:
            return z
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(device=z.device, dtype=torch.long).clamp(0, self.vocab_size - 1)
        if prompt_ids.shape[0] == 1 and z.shape[0] > 1:
            prompt_ids = prompt_ids.expand(z.shape[0], -1)
        mask = (prompt_ids != 0).to(z.dtype).unsqueeze(-1)
        emb = self.prompt_embed(prompt_ids) * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        p = emb.sum(dim=1) / denom
        gamma, beta = self.film(p).chunk(2, dim=-1)
        zc = z * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)
        return F.normalize(zc, dim=-1)

class SharedAffectManifold(nn.Module):

    def __init__(self, num_classes: int, embedding_dim: int = 256, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        return z @ p.t() / self.temperature


SharedPrototypeHead = SharedAffectManifold

class PoincareBall(nn.Module):

    def __init__(self, curvature: float = 1.0):
        super().__init__()
        # 曲率作为可学习参数，让模型自适应调整空间弯曲程度
        self.c = nn.Parameter(torch.tensor([curvature]))

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        """将切空间向量 v 映射到 Poincaré Ball（指数映射，从原点出发）。"""
        c = self.c.abs().clamp(min=1e-6)
        sqrt_c = c.sqrt()
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        tanh_term = torch.tanh(sqrt_c * v_norm)
        return tanh_term * v / (sqrt_c * v_norm)

    def logmap0(self, y: torch.Tensor) -> torch.Tensor:
        """将 Poincaré Ball 上的点 y 映射回切空间（对数映射）。"""
        c = self.c.abs().clamp(min=1e-6)
        sqrt_c = c.sqrt()
        y_norm = y.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        atanh_term = torch.atanh((sqrt_c * y_norm).clamp(max=1 - 1e-5))
        return atanh_term * y / (sqrt_c * y_norm)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Möbius 加法：双曲空间上的平移操作。"""
        c = self.c.abs().clamp(min=1e-6)
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = (1 + 2 * c * xy + c**2 * x2 * y2).clamp(min=1e-10)
        return num / denom

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算双曲空间中两点的测地距离。"""
        c = self.c.abs().clamp(min=1e-6)
        sqrt_c = c.sqrt()
        diff = self.mobius_add(-x, y)
        diff_norm = diff.norm(dim=-1).clamp(min=1e-10, max=1 - 1e-5)
        return (2 / sqrt_c) * torch.atanh(sqrt_c * diff_norm)


class HyperbolicEmotionManifold(nn.Module):
    # 情感到极性（粗粒度）的映射
    EMOTION_TO_VALENCE = {
        "angry": "negative", "disgusted": "negative",
        "fearful": "negative", "sad": "negative",
        "happy": "positive",
        "neutral": "neutral", "surprised": "neutral",
    }
    VALENCES = ["positive", "negative", "neutral"]

    def __init__(self, num_classes: int, embedding_dim: int, temperature: float = 0.07, curvature: float = 1.0):
        super().__init__()
        self.ball = PoincareBall(curvature)
        self.temperature = temperature
        self.embedding_dim = embedding_dim

        # 细粒度情感原型（子节点）
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, embedding_dim) * 0.02
        )
        # 粗粒度极性原型（父节点）
        self.valence_prototypes = nn.Parameter(
            torch.randn(len(self.VALENCES), embedding_dim) * 0.02
        )
        # 欧氏空间 → 双曲切空间的投影
        self.to_hyperbolic = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),   # 限制范数，避免投影后超出 Poincaré Ball 边界
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, D] 欧氏空间的情感嵌入
        返回: [B, num_classes] 基于测地距离的分类 logits
        """
        # 映射到双曲空间
        z_h = self.ball.expmap0(self.to_hyperbolic(z))           # [B, D]
        proto_h = self.ball.expmap0(self.prototypes)              # [C, D]
        
        # 计算每个样本到每个原型的测地距离
        z_expanded = z_h.unsqueeze(1).expand(-1, proto_h.shape[0], -1)   # [B, C, D]
        p_expanded = proto_h.unsqueeze(0).expand(z_h.shape[0], -1, -1)   # [B, C, D]
        dists = self.ball.dist(z_expanded, p_expanded)                     # [B, C]
        
        # 距离越小相似度越高，取负距离再除温度
        return -dists / self.temperature

    def hierarchy_consistency_loss(self, label_names: list) -> torch.Tensor:
        """
        层级一致性损失：子原型应比父原型（极性）更靠近自己所属的父节点。
        
        具体约束：
        d(child, own_parent) + margin < d(child, other_parent)
        """
        proto_h = self.ball.expmap0(self.prototypes)
        valence_h = self.ball.expmap0(self.valence_prototypes)
        valence_idx = {v: i for i, v in enumerate(self.VALENCES)}
        
        loss = torch.tensor(0.0, device=self.prototypes.device)
        margin = 0.5
        count = 0
        
        for cls_idx, emotion in enumerate(label_names):
            valence = self.EMOTION_TO_VALENCE.get(emotion)
            if valence is None:
                continue
            own_idx = valence_idx[valence]
            own_dist = self.ball.dist(
                proto_h[cls_idx].unsqueeze(0),
                valence_h[own_idx].unsqueeze(0)
            )
            for other_valence, other_idx in valence_idx.items():
                if other_valence == valence:
                    continue
                other_dist = self.ball.dist(
                    proto_h[cls_idx].unsqueeze(0),
                    valence_h[other_idx].unsqueeze(0)
                )
                loss = loss + torch.clamp(own_dist - other_dist + margin, min=0.0)
                count += 1
        
        return loss / max(count, 1)

class AffectLoss(nn.Module):
    """
    联合损失 = CE + Supervised Contrastive + Hyperbolic Hierarchy
    
    """
    def __init__(self, temperature: float = 0.07, ce_w=1.0, supcon_w=0.5, hier_w=0.1):
        super().__init__()
        self.temperature = temperature
        self.ce_w = ce_w
        self.supcon_w = supcon_w
        self.hier_w = hier_w

    def supervised_contrastive(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """SupConLoss：同类对作正样本，跨类对作负样本。"""
        z_norm = nn.functional.normalize(z, dim=-1)
        sim = torch.matmul(z_norm, z_norm.T) / self.temperature  # [B, B]
        # 去掉对角线（自身）
        B = z.shape[0]
        mask_self = ~torch.eye(B, dtype=torch.bool, device=z.device)
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & mask_self  # 同类 mask
        
        # 对每个样本：log(正样本相似度之和 / 所有非自身相似度之和)
        exp_sim = torch.exp(sim) * mask_self
        log_prob = sim - torch.log(exp_sim.sum(dim=-1, keepdim=True).clamp(min=1e-9))
        loss = -(log_prob * mask_pos).sum(dim=-1) / mask_pos.sum(dim=-1).clamp(min=1)
        return loss.mean()

    def forward(self, out: dict, labels: torch.Tensor, label_names: list, manifold: HyperbolicEmotionManifold):
        ce = nn.functional.cross_entropy(out["face_logits"], labels)
        proto_ce = nn.functional.cross_entropy(out["face_proto_logits"], labels)
        supcon = self.supervised_contrastive(out["face_embedding"], labels)
        if self.hier_w <= 0.0 or manifold is None:
            hier = torch.zeros_like(ce)
        else:
            hier = manifold.hierarchy_consistency_loss(label_names)
        total = (self.ce_w * ce
                 + 0.2 * proto_ce
                 + self.supcon_w * supcon
                 + self.hier_w * hier)
        return total, {"ce": ce, "proto_ce": proto_ce, "supcon": supcon, "hier": hier}
    
class ReliabilityEstimator(nn.Module):

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + 3, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _entropy_confidence(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(logits, dim=-1)
        conf = probs.max(dim=-1, keepdim=True).values
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
        entropy = entropy / torch.log(torch.tensor(float(probs.shape[-1]), device=logits.device, dtype=logits.dtype))
        certainty = 1.0 - entropy.clamp(0.0, 1.0)
        return conf, certainty

    def forward(self, z: torch.Tensor, logits: torch.Tensor, signal_quality: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = z.shape[0]
        conf, certainty = self._entropy_confidence(logits)
        if signal_quality is None:
            signal_quality = torch.ones(b, 1, device=z.device, dtype=z.dtype)
        if signal_quality.dim() == 1:
            signal_quality = signal_quality[:, None]
        x = torch.cat([z, conf, certainty, signal_quality.to(z.dtype)], dim=-1)
        return self.net(x)


class TriViewDynamicFusion(nn.Module):

    MODS = ("audio", "physio", "face")

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.1, num_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.experts = nn.ModuleDict({
            "audio": nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(embedding_dim, embedding_dim)),
            "physio": nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(embedding_dim, embedding_dim)),
            "face": nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(embedding_dim, embedding_dim)),
            "joint": nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(embedding_dim, embedding_dim)),
        })
        self.router = nn.Sequential(nn.Linear(embedding_dim + 6, embedding_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(embedding_dim, 4))

    def forward(
        self,
        z_audio: Optional[torch.Tensor] = None,
        z_physio: Optional[torch.Tensor] = None,
        z_face: Optional[torch.Tensor] = None,
        rel_audio: Optional[torch.Tensor] = None,
        rel_physio: Optional[torch.Tensor] = None,
        rel_face: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        z_by_mod = {"audio": z_audio, "physio": z_physio, "face": z_face}
        rel_by_mod = {"audio": rel_audio, "physio": rel_physio, "face": rel_face}
        present = [m for m in self.MODS if z_by_mod[m] is not None]
        if not present:
            raise ValueError("At least one modality must be provided")
        device = z_by_mod[present[0]].device
        dtype = z_by_mod[present[0]].dtype
        b = z_by_mod[present[0]].shape[0]

        batch_sizes = {m: int(z_by_mod[m].shape[0]) for m in present}
        if len(set(batch_sizes.values())) != 1:
            raise ValueError(
                "All modalities passed to TriViewDynamicFusion must have the same batch size. "
                f"Got batch sizes: {batch_sizes}. Build same-label pseudo pairs/triples before fusion."
            )

        rel_values = []
        mask_values = []
        for m in self.MODS:
            if z_by_mod[m] is None:
                rel_values.append(torch.zeros(b, 1, device=device, dtype=dtype))
                mask_values.append(torch.zeros(b, 1, device=device, dtype=dtype))
            else:
                r = rel_by_mod[m]
                if r is None:
                    r = torch.ones(b, 1, device=device, dtype=dtype)
                if r.dim() == 1:
                    r = r[:, None]
                rel_values.append(r.to(device=device, dtype=dtype))
                mask_values.append(torch.ones(b, 1, device=device, dtype=dtype))
        rel_tensor = torch.cat(rel_values, dim=1)  # [B,3]
        mask_tensor = torch.cat(mask_values, dim=1)  # [B,3]
        rel_masked = rel_tensor * mask_tensor
        modality_weights = rel_masked / rel_masked.sum(dim=1, keepdim=True).clamp_min(1e-6)

        tokens = torch.stack([z_by_mod[m] for m in present], dim=1)
        attended, attn = self.cross_attn(tokens, tokens, tokens, need_weights=True)
        joint = attended.mean(dim=1)
        per_mod_context: Dict[str, torch.Tensor] = {}
        for i, m in enumerate(present):
            per_mod_context[m] = attended[:, i]

        expert_outputs = []
        for m in self.MODS:
            if m in per_mod_context:
                expert_outputs.append(self.experts[m](per_mod_context[m]))
            else:
                expert_outputs.append(torch.zeros(b, joint.shape[-1], device=device, dtype=dtype))
        expert_outputs.append(self.experts["joint"](joint))
        expert_stack = torch.stack(expert_outputs, dim=1)  # [B,4,D]

        router_in = torch.cat([joint, rel_tensor, mask_tensor], dim=-1)
        router_logits = self.router(router_in)
        # Disable absent modality experts by large negative logits.
        expert_mask = torch.cat([mask_tensor, torch.ones(b, 1, device=device, dtype=dtype)], dim=1)
        router_logits = router_logits.masked_fill(expert_mask <= 0, -1e4)
        router_weights = torch.softmax(router_logits, dim=-1)
        z_moe = (router_weights[:, :, None] * expert_stack).sum(dim=1)

        late = torch.zeros_like(z_moe)
        for j, m in enumerate(self.MODS):
            if z_by_mod[m] is not None:
                late = late + modality_weights[:, j:j+1] * z_by_mod[m]
        z = F.normalize(0.55 * z_moe + 0.45 * late, dim=-1)
        return z, modality_weights, {
            "router_weights": router_weights,
            "modality_weights": modality_weights,
            "modality_mask": mask_tensor,
            "cross_attention": attn,
            "present_modalities": present,
        }


# Backward-compatible alias.
DynamicMoEFusion = TriViewDynamicFusion
GatedFusion = TriViewDynamicFusion


class MultimodalEmotionModel(nn.Module):

    def __init__(
        self,
        num_classes: int,
        physio_channels: int,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        temperature: float = 0.07,
        fusion_heads: int = 4,
        use_face: bool = True,
        audio_use_mel: bool = False,
        audio_n_mels: int = 64,
        audio_sample_rate: int = 16000,
        lora_rank: int = 8  # FaceEncoder V2 中的新参数
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.use_face = use_face
        self.physio_encoder = PhysioEncoder(physio_channels, embedding_dim, hidden_dim, dropout)
        self.audio_encoder = AudioEncoder(
            embedding_dim, hidden_dim, dropout, use_mel=audio_use_mel, n_mels=audio_n_mels, sample_rate=audio_sample_rate
        )
        self.face_encoder = FaceEncoder(embedding_dim, lora_rank=lora_rank, dropout=dropout)
        self.audio_affect_projector = TemporalAffectProjector(embedding_dim, max(embedding_dim, hidden_dim), dropout)
        self.physio_affect_projector = TemporalAffectProjector(embedding_dim, max(embedding_dim, hidden_dim), dropout)
        self.face_affect_projector = RegionAwareAffectProjector(embedding_dim, dropout) # V2
        self.audio_hierarchical_affect_fusion = HierarchicalAffectFusion(embedding_dim, dropout)
        self.physio_hierarchical_affect_fusion = HierarchicalAffectFusion(embedding_dim, dropout)
        self.face_hierarchical_affect_fusion = HierarchicalAffectFusion(embedding_dim, dropout)
        self.prompt_conditioner = PromptAwareAffectConditioner(embedding_dim, vocab_size=4096, dropout=dropout)
        self.prototype_head = HyperbolicEmotionManifold(num_classes, embedding_dim, temperature) # V2
        self.classifier = nn.Linear(embedding_dim, num_classes)
        try:
            nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0.0)
        except Exception:
            pass
        self.reliability = ReliabilityEstimator(embedding_dim, dropout)
        self.fusion = TriViewDynamicFusion(embedding_dim, dropout, num_heads=fusion_heads)

    @staticmethod
    def prompt_ids_from_texts(texts, max_length: int = 64, vocab_size: int = 4096, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert text prompts to stable hashed ids without external tokenizers.

        ID 0 is reserved for padding.  The method accepts a string or a list of
        strings and returns [B, max_length].
        """
        import hashlib
        import re

        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            toks = re.findall(r"[\w\u4e00-\u9fff]+", str(text).lower())[:max_length]
            ids = []
            for tok in toks:
                h = hashlib.md5(tok.encode("utf-8")).hexdigest()
                ids.append(int(h[:8], 16) % (vocab_size - 1) + 1)
            ids += [0] * (max_length - len(ids))
            rows.append(ids)
        return torch.tensor(rows, dtype=torch.long, device=device)

    def encode_audio(self, audio: torch.Tensor, return_temporal: bool = False, prompt_ids: Optional[torch.Tensor] = None):
        h_seq, z_raw = self.audio_encoder.encode_temporal(audio)
        z_seq, z_affect = self.audio_affect_projector(h_seq)
        z_affect = self.audio_hierarchical_affect_fusion(z_raw, z_affect, z_seq)
        z_affect = self.prompt_conditioner(z_affect, prompt_ids)
        if return_temporal:
            return z_seq, z_affect, z_raw
        return z_affect

    def encode_physio(self, physio: torch.Tensor, return_temporal: bool = False, prompt_ids: Optional[torch.Tensor] = None):
        h_seq, z_raw = self.physio_encoder.encode_temporal(physio)
        z_seq, z_affect = self.physio_affect_projector(h_seq)
        z_affect = self.physio_hierarchical_affect_fusion(z_raw, z_affect, z_seq)
        z_affect = self.prompt_conditioner(z_affect, prompt_ids)
        if return_temporal:
            return z_seq, z_affect, z_raw
        return z_affect

    def encode_face(self, face: torch.Tensor, return_temporal: bool = False, prompt_ids: Optional[torch.Tensor] = None):
        h_seq, z_raw = self.face_encoder.encode_temporal(face)
        # face_affect_projector expects both the patch-token sequence and the
        # CLS/global token (`z_raw`). Pass both to avoid TypeError.
        z_seq, z_affect = self.face_affect_projector(h_seq, z_raw)
        z_affect = self.face_hierarchical_affect_fusion(z_raw, z_affect, z_seq)
        z_affect = self.prompt_conditioner(z_affect, prompt_ids)
        if return_temporal:
            return z_seq, z_affect, z_raw
        return z_affect

    def classify_embedding(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "logits": self.classifier(z),
            "proto_logits": self.prototype_head(z),
            "embedding": z,
        }

    @staticmethod
    def _signal_quality_1d(x: Optional[torch.Tensor], batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if x is None:
            return torch.zeros(batch_size, 1, device=device, dtype=dtype)
        flat = x.float().reshape(x.shape[0], -1)
        finite = torch.isfinite(flat).float().mean(dim=1, keepdim=True)
        std = flat.nan_to_num().std(dim=1, keepdim=True)
        clipped = (flat.abs() > 0.999).float().mean(dim=1, keepdim=True)
        quality = finite * torch.tanh(std).clamp(0.0, 1.0) * (1.0 - 0.5 * clipped).clamp(0.0, 1.0)
        return quality.to(device=device, dtype=dtype)

    @staticmethod
    def _signal_quality_image(x: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        flat = x.float().reshape(x.shape[0], -1)
        finite = torch.isfinite(flat).float().mean(dim=1, keepdim=True)
        contrast = flat.nan_to_num().std(dim=1, keepdim=True)
        # Simple no-reference image quality proxy: finite ratio times contrast.
        quality = finite * torch.tanh(2.0 * contrast).clamp(0.0, 1.0)
        return quality.to(device=device, dtype=dtype)

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        physio: Optional[torch.Tensor] = None,
        face: Optional[torch.Tensor] = None,
        prompt_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        z_audio = z_physio = z_face = None
        rel_a = rel_p = rel_v = None

        if audio is not None:
            seq_audio, z_audio, _ = self.encode_audio(audio, return_temporal=True, prompt_ids=prompt_ids)
            audio_out = self.classify_embedding(z_audio)
            out.update({f"audio_{k}": v for k, v in audio_out.items()})
            out["audio_affect_seq"] = seq_audio
            q_a = self._signal_quality_1d(audio, audio.shape[0], z_audio.device, z_audio.dtype)
            rel_a = self.reliability(z_audio, out["audio_logits"], q_a)
            out["audio_reliability"] = rel_a

        if physio is not None:
            seq_physio, z_physio, _ = self.encode_physio(physio, return_temporal=True, prompt_ids=prompt_ids)
            physio_out = self.classify_embedding(z_physio)
            out.update({f"physio_{k}": v for k, v in physio_out.items()})
            out["physio_affect_seq"] = seq_physio
            q_p = self._signal_quality_1d(physio, physio.shape[0], z_physio.device, z_physio.dtype)
            rel_p = self.reliability(z_physio, out["physio_logits"], q_p)
            out["physio_reliability"] = rel_p

        if face is not None:
            seq_face, z_face, _ = self.encode_face(face, return_temporal=True, prompt_ids=prompt_ids)
            face_out = self.classify_embedding(z_face)
            out.update({f"face_{k}": v for k, v in face_out.items()})
            out["face_affect_seq"] = seq_face
            q_v = self._signal_quality_image(face, z_face.device, z_face.dtype)
            rel_v = self.reliability(z_face, out["face_logits"], q_v)
            out["face_reliability"] = rel_v

        if z_audio is not None or z_physio is not None or z_face is not None:
            z_fused, modality_weights, aux = self.fusion(z_audio, z_physio, z_face, rel_a, rel_p, rel_v)
            z_fused = self.prompt_conditioner(z_fused, prompt_ids)
            fused_out = self.classify_embedding(z_fused)
            out.update({f"fused_{k}": v for k, v in fused_out.items()})
            out["z_affect"] = z_fused
            out["fusion_modality_weights"] = modality_weights
            out["fusion_modality_mask"] = aux.get("modality_mask")
            # Backward compatibility for older UI code.
            out["fusion_gate_audio"] = modality_weights[:, 0:1]
            out["fusion_gate_physio"] = modality_weights[:, 1:2]
            out["fusion_gate_face"] = modality_weights[:, 2:3]
            out["fusion_router_weights"] = aux["router_weights"]
            if "cross_attention" in aux:
                out["fusion_cross_attention"] = aux["cross_attention"]
        return out

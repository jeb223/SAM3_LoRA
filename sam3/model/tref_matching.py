import math
import re
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.train.loss.loss_fns import CORE_LOSS_KEY


_COLOR_WORDS = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "cyan",
    "black",
    "white",
    "gray",
    "grey",
    "brown",
]

_KEYWORD_FEATURES = [
    ("left", ("left", "leftmost")),
    ("right", ("right", "rightmost")),
    ("center", ("center", "middle")),
    ("upper", ("upper", "top", "above")),
    ("lower", ("lower", "bottom", "below")),
    ("small", ("small", "tiny")),
    ("large", ("large", "big")),
    ("thin", ("thin", "slender", "narrow")),
    ("long", ("long", "tall")),
    ("horizontal", ("horizontal",)),
    ("vertical", ("vertical",)),
    ("near", ("near", "next", "beside", "adjacent")),
]


def text_attribute_feature_dim(hash_dim: int = 32) -> int:
    return len(_KEYWORD_FEATURES) + len(_COLOR_WORDS) + int(hash_dim)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def extract_text_attribute_features(
    texts: Iterable[str],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    hash_dim: int = 32,
) -> torch.Tensor:
    """Build lightweight text features for candidate ranking.

    The SAM3 query features already contain the deep text-image interaction. These
    features add explicit position/size/color cues so the matching head has a
    stable signal for common referring expressions and hard negatives.
    """

    hash_dim = int(hash_dim)
    dim = text_attribute_feature_dim(hash_dim)
    rows = []
    for text in texts:
        tokens = _tokenize(text)
        token_set = set(tokens)
        values = []
        for _, words in _KEYWORD_FEATURES:
            values.append(1.0 if any(word in token_set for word in words) else 0.0)
        for color in _COLOR_WORDS:
            values.append(1.0 if color in token_set else 0.0)

        hashed = [0.0 for _ in range(hash_dim)]
        if hash_dim > 0:
            for token in tokens:
                # Python hash is randomized per process; use a stable simple hash.
                h = 2166136261
                for ch in token:
                    h ^= ord(ch)
                    h *= 16777619
                    h &= 0xFFFFFFFF
                idx = h % hash_dim
                sign = 1.0 if (h >> 8) & 1 else -1.0
                hashed[idx] += sign / math.sqrt(max(len(tokens), 1))
        values.extend(hashed)
        rows.append(values)

    if not rows:
        return torch.zeros(0, dim, device=device, dtype=dtype)
    return torch.tensor(rows, device=device, dtype=dtype)


def candidate_geometry_features(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    boxes = outputs["pred_boxes"]
    if boxes.dim() != 3 or boxes.size(-1) != 4:
        raise ValueError("pred_boxes must have shape [B, Q, 4]")

    cx, cy, w, h = boxes.unbind(dim=-1)
    w = w.clamp(min=1e-4)
    h = h.clamp(min=1e-4)
    area = (w * h).clamp(max=1.0)
    aspect = torch.log(w / h).clamp(min=-6.0, max=6.0) / 6.0

    if "pred_logits" in outputs:
        score = outputs["pred_logits"].detach().sigmoid().squeeze(-1)
    else:
        score = torch.zeros_like(cx)

    return torch.stack([cx, cy, w, h, area, aspect, score], dim=-1)


class TextCandidateMatchingHead(nn.Module):
    """Candidate-aware RES scoring head for TRef-SAM3.

    It scores each SAM3 candidate query for the current expression. The input
    query feature is already conditioned by SAM3's text-image transformer; explicit
    text attributes and candidate geometry are added for referring disambiguation.
    """

    def __init__(
        self,
        query_dim: int,
        text_feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(query_dim)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.geo_text_proj = nn.Sequential(
            nn.Linear(7 + text_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        queries = outputs.get("queries")
        if queries is None:
            raise KeyError("outputs must contain text-conditioned candidate 'queries'")
        if queries.dim() != 3:
            raise ValueError("queries must have shape [B, Q, C]")
        if text_features.dim() != 2 or text_features.size(0) != queries.size(0):
            raise ValueError(
                "text_features must have shape [B, C_text] aligned with output batch"
            )

        geom = candidate_geometry_features(outputs).to(dtype=queries.dtype)
        text_features = text_features.to(device=queries.device, dtype=queries.dtype)
        text_features = text_features[:, None, :].expand(-1, queries.size(1), -1)
        geo_text = torch.cat([geom, text_features], dim=-1)

        hidden = self.query_proj(self.query_norm(queries))
        hidden = hidden + self.geo_text_proj(geo_text)
        return self.scorer(hidden)


class TRefSelectionLoss(nn.Module):
    """Selection and hard-negative ranking loss for candidate-aware RES."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        rank_weight: float = 0.5,
        margin: float = 0.3,
        hard_negatives: int = 16,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.rank_weight = float(rank_weight)
        self.margin = float(margin)
        self.hard_negatives = int(hard_negatives)
        self.focal_gamma = float(focal_gamma)

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        logits = outputs["ref_match_logits"].squeeze(-1)
        labels = torch.zeros_like(logits)
        indices = outputs.get("indices")
        if indices is None:
            raise KeyError("outputs must contain matcher indices before TRef loss")

        batch_idx, src_idx = indices[0], indices[1]
        if batch_idx.numel() > 0:
            labels[(batch_idx.to(logits.device), src_idx.to(logits.device))] = 1.0

        valid_rows = targets["num_boxes"].to(logits.device) > 0
        if not valid_rows.any():
            zero = logits.sum() * 0.0
            return {
                CORE_LOSS_KEY: zero,
                "loss_tref_select": zero,
                "loss_tref_rank": zero,
            }

        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        if self.focal_gamma > 0:
            prob = logits.sigmoid()
            pt = prob * labels + (1.0 - prob) * (1.0 - labels)
            bce = bce * (1.0 - pt).pow(self.focal_gamma)
        bce = bce[valid_rows].mean()

        rank_loss = self._ranking_loss(logits, labels, valid_rows)
        total = self.bce_weight * bce + self.rank_weight * rank_loss
        return {
            CORE_LOSS_KEY: total,
            "loss_tref_select": bce.detach(),
            "loss_tref_rank": rank_loss.detach(),
        }

    def _ranking_loss(self, logits, labels, valid_rows):
        losses = []
        for row in torch.nonzero(valid_rows, as_tuple=False).flatten().tolist():
            pos = logits[row][labels[row] > 0.5]
            neg = logits[row][labels[row] <= 0.5]
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            if self.hard_negatives > 0 and neg.numel() > self.hard_negatives:
                neg = torch.topk(neg, k=self.hard_negatives, largest=True).values
            losses.append(F.relu(self.margin - pos[:, None] + neg[None, :]).mean())
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()

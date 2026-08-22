from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss for same-label alignment."""
    if z.numel() == 0 or z.shape[0] < 2:
        return z.new_tensor(0.0)
    z = F.normalize(z, dim=-1)
    labels = labels.view(-1, 1)
    sim = z @ z.t() / temperature
    logits_mask = torch.ones_like(sim) - torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
    pos_mask = (labels == labels.t()).to(sim.dtype) * logits_mask
    exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values.detach()) * logits_mask
    log_prob = sim - sim.max(dim=1, keepdim=True).values.detach() - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if not valid.any():
        return z.new_tensor(0.0)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count.clamp_min(1.0)
    return -mean_log_prob_pos[valid].mean()


def gaussian_mmd_loss(x: torch.Tensor, y: torch.Tensor, sigmas=(1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
    """Distribution match loss: MMD between unpaired modality embeddings."""
    if x.numel() == 0 or y.numel() == 0:
        return (x.sum() + y.sum()) * 0.0
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d2 = torch.cdist(a, b).pow(2)
        k = 0.0
        for s in sigmas:
            k = k + torch.exp(-d2 / (2.0 * float(s) ** 2))
        return k

    return kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean()


def sinkhorn_ot_loss(x: torch.Tensor, y: torch.Tensor, epsilon: float = 0.05, n_iters: int = 40) -> torch.Tensor:

    if x.numel() == 0 or y.numel() == 0:
        return (x.sum() + y.sum()) * 0.0
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    n, m = x.shape[0], y.shape[0]
    cost = torch.cdist(x, y, p=2).pow(2)
    log_k = -cost / max(float(epsilon), 1e-6)
    log_u = x.new_zeros(n)
    log_v = y.new_zeros(m)
    log_a = x.new_full((n,), -torch.log(torch.tensor(float(n), device=x.device, dtype=x.dtype)))
    log_b = y.new_full((m,), -torch.log(torch.tensor(float(m), device=y.device, dtype=y.dtype)))
    for _ in range(int(n_iters)):
        log_u = log_a - torch.logsumexp(log_k + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_k + log_u[:, None], dim=0)
    plan = torch.exp(log_k + log_u[:, None] + log_v[None, :])
    return (plan * cost).sum()


def temporal_smoothness_loss(z_seq: torch.Tensor | None) -> torch.Tensor:
    """Dynamics consistency: encourages smooth z_affect(t)."""
    if z_seq is None or z_seq.numel() == 0 or z_seq.shape[1] < 2:
        if z_seq is None:
            return torch.tensor(0.0)
        return z_seq.sum() * 0.0
    return (z_seq[:, 1:, :] - z_seq[:, :-1, :]).pow(2).mean()


def cross_modal_retrieval_stats(z_audio: torch.Tensor, y_audio: torch.Tensor, z_physio: torch.Tensor, y_physio: torch.Tensor) -> dict[str, float]:
    if z_audio.numel() == 0 or z_physio.numel() == 0:
        return {"same_label_cosine": 0.0, "diff_label_cosine": 0.0, "recall_at_1": 0.0}
    with torch.no_grad():
        za = F.normalize(z_audio, dim=-1)
        zp = F.normalize(z_physio, dim=-1)
        sim = za @ zp.t()
        same = y_audio[:, None] == y_physio[None, :]
        diff = ~same
        same_val = sim[same].mean().item() if same.any() else 0.0
        diff_val = sim[diff].mean().item() if diff.any() else 0.0
        nn_idx = sim.argmax(dim=1)
        r1 = (y_physio[nn_idx] == y_audio).float().mean().item()
        return {"same_label_cosine": same_val, "diff_label_cosine": diff_val, "recall_at_1": r1}


def cross_entropy_with_optional_empty(logits: torch.Tensor | None, labels: torch.Tensor | None) -> torch.Tensor:
    if logits is None or labels is None or labels.numel() == 0:
        if logits is not None:
            return logits.sum() * 0.0
        if labels is not None:
            return labels.float().sum() * 0.0
        return torch.tensor(0.0)
    return F.cross_entropy(logits, labels)


def class_confidence(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1).max(dim=-1).values


def pairwise_multiview_ot_loss(embeddings: dict[str, torch.Tensor], epsilon: float = 0.05, n_iters: int = 40) -> torch.Tensor:
    items = [(k, v) for k, v in embeddings.items() if v is not None and v.numel() > 0]
    if len(items) < 2:
        if items:
            return items[0][1].sum() * 0.0
        return torch.tensor(0.0)
    losses = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            losses.append(sinkhorn_ot_loss(items[i][1], items[j][1], epsilon=epsilon, n_iters=n_iters))
    return torch.stack(losses).mean()


def pairwise_multiview_mmd_loss(embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
    items = [(k, v) for k, v in embeddings.items() if v is not None and v.numel() > 0]
    if len(items) < 2:
        if items:
            return items[0][1].sum() * 0.0
        return torch.tensor(0.0)
    losses = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            losses.append(gaussian_mmd_loss(items[i][1], items[j][1]))
    return torch.stack(losses).mean()


def reliability_consistency_loss(reliabilities: dict[str, torch.Tensor]) -> torch.Tensor:
    vals = [v for v in reliabilities.values() if v is not None and v.numel() > 0]
    if not vals:
        return torch.tensor(0.0)
    r = torch.cat(vals, dim=0)
    # Encourage usable but not overconfident reliability before strong calibration.
    target = torch.full_like(r, 0.7)
    return F.mse_loss(r, target)


def _active_class_tensor(active_classes, device: torch.device) -> torch.Tensor | None:
    if active_classes is None:
        return None
    if isinstance(active_classes, torch.Tensor):
        ids = active_classes.to(device=device, dtype=torch.long).view(-1)
    else:
        ids = torch.tensor(list(active_classes), device=device, dtype=torch.long).view(-1)
    if ids.numel() == 0:
        return None
    return ids


def mask_logits_to_active_classes(logits: torch.Tensor, active_classes=None, mask_value: float = -1.0e4) -> torch.Tensor:
    ids = _active_class_tensor(active_classes, logits.device)
    if ids is None or ids.numel() >= logits.shape[-1]:
        return logits
    keep = torch.zeros(logits.shape[-1], device=logits.device, dtype=torch.bool)
    keep[ids.clamp(0, logits.shape[-1] - 1)] = True
    return logits.masked_fill(~keep.view(1, -1), mask_value)


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    active_classes=None,
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    logits = mask_logits_to_active_classes(logits, active_classes)
    if weight is not None:
        weight = weight.to(device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits, labels, weight=weight, label_smoothing=float(label_smoothing)) 


def masked_cross_entropy_per_sample(
    logits: torch.Tensor,
    labels: torch.Tensor,
    active_classes=None,
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    logits = mask_logits_to_active_classes(logits, active_classes)
    if weight is not None:
        weight = weight.to(device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits, labels, weight=weight, label_smoothing=float(label_smoothing), reduction="none")


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    active_classes=None,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Focal loss computed on masked logits. Returns scalar mean loss.

    gamma: focusing parameter. gamma=0 reduces to CE.
    """
    per = masked_cross_entropy_per_sample(logits, labels, active_classes=active_classes, weight=weight, label_smoothing=label_smoothing)
    p_t = torch.exp(-per)
    loss = ((1.0 - p_t) ** float(gamma)) * per
    return loss.mean()


def masked_argmax(logits: torch.Tensor, active_classes=None) -> torch.Tensor:
    return mask_logits_to_active_classes(logits, active_classes).argmax(dim=-1)


def class_balanced_weight_from_labels(
    labels: torch.Tensor,
    num_classes: int,
    active_classes=None,
    max_weight: float = 5.0,
) -> torch.Tensor:
    """Return a full-length CE weight vector capped for stability."""
    labels = labels.detach().cpu().long().view(-1)
    ids = _active_class_tensor(active_classes, torch.device("cpu"))
    if ids is None:
        ids = torch.unique(labels)
    counts = torch.bincount(labels, minlength=int(num_classes)).float()
    weights = torch.ones(int(num_classes), dtype=torch.float32)
    active = ids.tolist()
    if active:
        active_counts = counts[ids].clamp_min(1.0)
        # inverse-frequency normalized to mean 1 over active classes
        w = active_counts.sum() / (len(active) * active_counts)
        w = w.clamp(max=float(max_weight))
        weights[:] = 0.0
        weights[ids] = w
    return weights

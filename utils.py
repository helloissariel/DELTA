import torch
import torch.nn.functional as F
from sklearn.metrics import (classification_report, roc_auc_score,
                              average_precision_score, precision_recall_curve)
import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from affiliation_metrics import affiliation_metrics_from_binary_vectors

# =========================
# 1. Load & Utility Functions
# =========================


@dataclass
class EvalResult:
    """
    Backward-compatible evaluation payload.

    Existing call sites can still unpack it as:
        auroc, aupr, best_f1, report = evaluate_full(...)

    Additional metrics, such as affiliation-based precision/recall/F1, remain
    available as attributes on the returned object.
    """

    auroc: float
    aupr: float
    best_f1: float
    report: str
    best_threshold: float
    affiliation_precision: float
    affiliation_recall: float
    affiliation_f1: float

    def __iter__(self):
        yield self.auroc
        yield self.aupr
        yield self.best_f1
        yield self.report

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return (self.auroc, self.aupr, self.best_f1, self.report)[index]

def load_adbench_data(dataset_path):
    """
    Load dataset from a .npz file.
    Assumes the file contains:
    - 'X': Feature matrix (N, d)
    - 'y': Labels (N,)
    """
    data = np.load(dataset_path)
    X = data['X']
    y = data['y']
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def evaluate_with_classification_report_and_auc(model, test_loader, device, threshold=0.5):
    """
    Evaluate a model using classification report and AUC-ROC metric.
    Legacy interface — calls evaluate_full internally.
    """
    auroc, aupr, best_f1, report = evaluate_full(model, test_loader, device)
    return report, auroc


def evaluate_full(model, test_loader, device):
    """
    Full evaluation with AUC-ROC, AUPR, Best point-wise F1, and affiliation F1.
    
    Follows the evaluation protocol used by GenIAS, CARLA, and other TSAD papers:
    - AUC-ROC: Area under ROC curve
    - AUPR: Area under Precision-Recall curve (important for imbalanced data)
    - Best F1: Maximum F1 score across all possible thresholds
    
    Affiliation metrics are computed on the binary prediction vector induced by
    the point-wise best-F1 threshold.

    Returns:
        EvalResult, which is iterable as (auroc, aupr, best_f1, report)
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).squeeze()
            y_pred = torch.sigmoid(y_pred)
            all_preds.append(y_pred.cpu())
            all_labels.append(y_batch.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    if len(set(labels)) <= 1:
        print("Evaluation: Only one class present in labels — metrics undefined.")
        return None, None, None, None

    # 1. AUC-ROC
    auroc = roc_auc_score(labels, preds)

    # 2. AUPR (Average Precision Score)
    aupr = average_precision_score(labels, preds)

    # 3. Best F1 (optimal threshold search)
    precisions, recalls, thresholds = precision_recall_curve(labels, preds)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_f1_idx = f1_scores.argmax()
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5

    # Classification report at best threshold
    binary_preds = (preds > best_threshold).astype(int)
    point_report = classification_report(labels, binary_preds, target_names=['Normal', 'Anomaly'])
    affiliation = affiliation_metrics_from_binary_vectors(
        y_pred=binary_preds.tolist(),
        y_true=labels.astype(int).tolist(),
    )
    affiliation_precision = affiliation["precision"]
    affiliation_recall = affiliation["recall"]
    affiliation_f1 = affiliation["f1"]
    affiliation_report = (
        f"Affiliation Precision: {affiliation_precision:.4f}\n"
        f"Affiliation Recall:    {affiliation_recall:.4f}\n"
        f"Affiliation F1:        {affiliation_f1:.4f} "
        f"(threshold={best_threshold:.4f})"
    )
    report = point_report.rstrip() + "\n" + affiliation_report + "\n"

    print(f"AUC-ROC:  {auroc:.4f}")
    print(f"AUPR:     {aupr:.4f}")
    print(f"Best F1:  {best_f1:.4f} (threshold={best_threshold:.4f})")
    print(point_report)
    print(affiliation_report)

    return EvalResult(
        auroc=auroc,
        aupr=aupr,
        best_f1=best_f1,
        report=report,
        best_threshold=best_threshold,
        affiliation_precision=affiliation_precision,
        affiliation_recall=affiliation_recall,
        affiliation_f1=affiliation_f1,
    )

def log_to_file(file_path, message):
    """Append a log message to the specified file."""
    with open(file_path, "a") as file:
        file.write(message + "\n")


# =========================
# 1.5 Distribution & Calibration Metrics (MMD, ECE)
# =========================

def gaussian_kernel(x, y, sigma=1.0):
    """
    Compute Gaussian (RBF) kernel between x and y.
    K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

    Args:
        x: (n, d) tensor
        y: (m, d) tensor
        sigma: kernel bandwidth
    Returns:
        kernel matrix (n, m)
    """
    x = x.unsqueeze(1)  # (n, 1, d)
    y = y.unsqueeze(0)  # (1, m, d)
    dist = torch.sum((x - y) ** 2, dim=2)  # (n, m)
    return torch.exp(-dist / (2 * sigma ** 2))


def compute_mmd(x_gen, x_real, sigma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions.

    MMD is a classic two-sample test statistic measuring the difference between
    two distributions in a reproducing kernel Hilbert space (RKHS).

    Formula: MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

    Used to measure the distributional shift between generated anomalies (x_gen)
    and real anomalies (x_real).

    Args:
        x_gen: Generated samples (n_gen, d)
        x_real: Real reference samples (n_real, d)
        sigma: RBF kernel bandwidth
    Returns:
        MMD value (scalar)
    """
    n_gen = x_gen.shape[0]
    n_real = x_real.shape[0]

    # K(x_gen, x_gen) - self-similarity of generated samples
    K_gen_gen = gaussian_kernel(x_gen, x_gen, sigma)
    # Exclude diagonal (don't count self-similarity)
    mmd_gen = (K_gen_gen.sum() - torch.trace(K_gen_gen)) / (n_gen * (n_gen - 1) + 1e-8)

    # K(x_real, x_real) - self-similarity of real samples
    K_real_real = gaussian_kernel(x_real, x_real, sigma)
    mmd_real = (K_real_real.sum() - torch.trace(K_real_real)) / (n_real * (n_real - 1) + 1e-8)

    # K(x_gen, x_real) - cross-similarity
    K_gen_real = gaussian_kernel(x_gen, x_real, sigma)
    mmd_cross = K_gen_real.mean()

    # MMD^2 = E[k(x,x)] + E[k(y,y)] - 2*E[k(x,y)]
    mmd_squared = mmd_gen + mmd_real - 2 * mmd_cross

    # Ensure non-negative (numerical stability)
    mmd_squared = torch.clamp(mmd_squared, min=0.0)

    return torch.sqrt(mmd_squared)


def get_detector_probs(detector, x_samples, device="cpu", batch_size=512):
    """
    Compute detector anomaly probabilities in batches and return them on CPU.
    """
    detector.eval()
    x_samples = x_samples.to(torch.float32)
    probs = []

    with torch.no_grad():
        for start in range(0, len(x_samples), batch_size):
            batch = x_samples[start:start + batch_size].to(device)
            logits = detector(batch)
            batch_probs = torch.sigmoid(logits).view(-1).detach().to("cpu")
            probs.append(batch_probs)

    if not probs:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(probs, dim=0)


def compute_ece(detector, x_samples, y_true, n_bins=10,
                focus_hard=True, hard_threshold=0.4, device="cpu",
                batch_size=512):
    """
    Compute Expected Calibration Error (ECE) for the detector.

    ECE measures the consistency between model confidence and empirical accuracy.
    It partitions predictions into bins and computes the weighted average of
    |confidence - accuracy| across bins.

    When focus_hard=True, ECE is computed on the subset of x_samples whose
    detector outputs lie near the decision boundary. For the paper-aligned
    boundary metric, callers should first build an explicit boundary-focused
    evaluation pool (hard held-out anomalies plus nearby held-out normals) and
    then call this function with focus_hard=False.

    Args:
        detector: Trained detector model
        x_samples: Input samples (N, d)
        y_true: Ground truth labels (N,)
        n_bins: Number of confidence bins
        focus_hard: If True, only compute ECE on hard samples near boundary
        hard_threshold: Threshold for defining hard samples (confidence within
                       [0.5 - threshold, 0.5 + threshold])
        device: Computation device
    Returns:
        ECE value (scalar), fraction of hard samples
    """
    detector.eval()
    y_true = y_true.detach().to("cpu").float().view(-1)
    probs = get_detector_probs(detector, x_samples, device=device, batch_size=batch_size)
    confidences = torch.where(y_true == 1, probs, 1 - probs)
    predictions = (probs > 0.5).float()
    correct = (predictions == y_true).float()

    # Focus on hard samples near decision boundary
    if focus_hard:
        hard_mask = (probs >= (0.5 - hard_threshold)) & (probs <= (0.5 + hard_threshold))
        if hard_mask.sum() < n_bins:
            # Too few hard samples, compute ECE on all
            hard_mask = torch.ones_like(probs, dtype=torch.bool)
    else:
        hard_mask = torch.ones_like(probs, dtype=torch.bool)

    hard_confidences = confidences[hard_mask]
    hard_correct = correct[hard_mask]
    n_hard = hard_mask.sum().item()

    if n_hard == 0:
        return 0.0, 0.0

    # Bin boundaries
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (hard_confidences > bin_boundaries[i]) & (hard_confidences <= bin_boundaries[i + 1])
        bin_size = in_bin.sum().item()

        if bin_size > 0:
            avg_confidence = hard_confidences[in_bin].mean()
            avg_accuracy = hard_correct[in_bin].mean()
            ece += (bin_size / n_hard) * torch.abs(avg_confidence - avg_accuracy)

    return ece.item(), n_hard / len(probs)


def build_boundary_evaluation_pool(
    detector,
    anomaly_samples,
    normal_samples,
    device="cpu",
    batch_size=512,
    boundary_margin=0.1,
    min_boundary_anomalies=32,
    max_boundary_anomalies=256,
    normals_per_anomaly=3,
    max_normal_candidates=2048,
    normal_candidate_multiplier=4,
):
    """
    Construct a boundary-focused evaluation pool from held-out real anomalies
    and nearby held-out normals.

    The pool is built in two stages:
    1. Select the held-out anomalies closest to the detector decision boundary.
    2. For each selected anomaly, retrieve nearby held-out normals from a
       boundary-biased normal candidate set.

    This matches the paper's ECE definition more closely than applying ECE to
    an anomaly-only pool.
    """
    anomaly_samples = anomaly_samples.detach().to("cpu", dtype=torch.float32)
    normal_samples = normal_samples.detach().to("cpu", dtype=torch.float32)

    if len(anomaly_samples) == 0:
        raise ValueError("No anomaly samples provided for boundary evaluation.")
    if len(normal_samples) == 0:
        raise ValueError("No normal samples provided for boundary evaluation.")

    anomaly_probs = get_detector_probs(
        detector, anomaly_samples, device=device, batch_size=batch_size
    )
    normal_probs = get_detector_probs(
        detector, normal_samples, device=device, batch_size=batch_size
    )

    anomaly_distance = torch.abs(anomaly_probs - 0.5)
    boundary_mask = anomaly_distance <= boundary_margin
    n_boundary_mask = int(boundary_mask.sum().item())

    target_boundary_anomalies = max(n_boundary_mask, min_boundary_anomalies)
    target_boundary_anomalies = min(
        target_boundary_anomalies,
        len(anomaly_samples),
        max_boundary_anomalies,
    )
    anomaly_order = torch.argsort(anomaly_distance)
    boundary_idx = anomaly_order[:target_boundary_anomalies]
    boundary_anomalies = anomaly_samples[boundary_idx]

    normal_distance = torch.abs(normal_probs - 0.5)
    target_normal_candidates = max(
        min_boundary_anomalies * normals_per_anomaly,
        len(boundary_idx) * normals_per_anomaly * normal_candidate_multiplier,
    )
    target_normal_candidates = min(
        target_normal_candidates,
        len(normal_samples),
        max_normal_candidates,
    )
    normal_candidate_idx = torch.argsort(normal_distance)[:target_normal_candidates]
    normal_candidates = normal_samples[normal_candidate_idx]

    if len(normal_candidates) == 0:
        raise ValueError("No candidate normal samples available for boundary evaluation.")

    # Use CPU for pairwise distances to avoid unnecessary GPU memory spikes.
    boundary_cpu = boundary_anomalies.detach().to("cpu", dtype=torch.float32)
    normal_candidates_cpu = normal_candidates.detach().to("cpu", dtype=torch.float32)
    neighbor_k = min(normals_per_anomaly, len(normal_candidates))
    pairwise = torch.cdist(boundary_cpu, normal_candidates_cpu, p=2)
    nearest_local_idx = torch.topk(pairwise, k=neighbor_k, dim=1, largest=False).indices
    nearest_local_idx = torch.unique(nearest_local_idx.reshape(-1))

    selected_normal_idx = normal_candidate_idx[nearest_local_idx.to(normal_candidate_idx.device)]
    boundary_normals = normal_samples[selected_normal_idx]

    x_pool = torch.cat([boundary_anomalies, boundary_normals], dim=0)
    y_pool = torch.cat([
        torch.ones(len(boundary_anomalies)),
        torch.zeros(len(boundary_normals)),
    ], dim=0)

    stats = {
        "n_boundary_anomalies": int(len(boundary_anomalies)),
        "n_boundary_normals": int(len(boundary_normals)),
        "boundary_anomaly_fraction": float(len(boundary_anomalies) / len(anomaly_samples)),
        "boundary_mask_fraction": float(n_boundary_mask / len(anomaly_samples)),
        "n_normal_candidates": int(len(normal_candidates)),
        "pool_size": int(len(x_pool)),
    }
    return x_pool, y_pool, stats


def compute_hard_mmd_ece(x_gen, x_real, detector, device="cpu",
                          sigma=1.0, n_bins=10, hard_threshold=0.4):
    """
    Legacy helper for anomaly-only MMD/ECE computation.

    This function evaluates the quality of generated anomalies by:
    1. MMD: Measuring distributional distance between generated and real anomalies
    2. ECE: Measuring calibration quality on anomaly-only hard samples

    Note:
        This helper does not construct the held-out boundary pool used by the
        paper-aligned evaluation protocol. Prefer build_boundary_evaluation_pool
        + compute_ece(..., focus_hard=False) for that setting.

    Low MMD + Low ECE on hard samples indicates:
    - Generated anomalies are distributionally similar to real ones
    - The detector's decision boundary is well-calibrated in ambiguous regions

    Args:
        x_gen: Generated anomaly samples
        x_real: Real anomaly samples
        detector: Trained detector model
        device: Computation device
        sigma: RBF kernel bandwidth for MMD
        n_bins: Number of bins for ECE
        hard_threshold: Threshold for defining hard samples
    Returns:
        dict with 'mmd', 'ece', 'hard_fraction' metrics
    """
    x_gen = x_gen.to(device)
    x_real = x_real.to(device)

    # Compute MMD between generated and real anomalies
    mmd_value = compute_mmd(x_gen, x_real, sigma=sigma)

    # Combine samples for ECE computation
    x_combined = torch.cat([x_gen, x_real], dim=0)
    y_combined = torch.cat([
        torch.ones(len(x_gen), device=device),
        torch.ones(len(x_real), device=device)
    ], dim=0)

    # Compute ECE focused on hard samples
    ece_value, hard_frac = compute_ece(
        detector, x_combined, y_combined,
        n_bins=n_bins, focus_hard=True,
        hard_threshold=hard_threshold, device=device
    )

    return {
        'mmd': mmd_value.item(),
        'ece': ece_value,
        'hard_fraction': hard_frac
    }


# =========================
# 2. Loss Functions
# =========================

class FocalLoss(torch.nn.Module):
    """
    Focal Loss for binary classification (with logits input).
    Prevents detector over-confidence by down-weighting easy examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.75, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        targets = targets.float()
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * ce).mean()


def beta_cvae_loss_fn(x, x_recon, mean, logvar, beta=4.0, sigma_prior=0.5):
    """
    Compute Beta-CVAE loss with Enhanced KL Divergence (inspired by GenIAS).
    
    Using sigma_prior < 1.0 enforces tighter latent representations of normal
    samples, improving separation between normal and anomalous data.
    Standard KL corresponds to sigma_prior=1.0.
    
    Args:
        x: Original input data.
        x_recon: Reconstructed data.
        mean: Mean of latent space distribution.
        logvar: Log variance of latent space distribution.
        beta: Weight for KL divergence.
        sigma_prior: Prior standard deviation (< 1.0 for tighter latent space).
    Returns:
        Total loss (scalar).
    """
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # Enhanced KL with tunable prior variance
    sigma_prior_sq = sigma_prior ** 2
    kl_loss = -0.5 * torch.sum(
        1 + logvar
        - (mean ** 2) / sigma_prior_sq
        - logvar.exp() / sigma_prior_sq
        + 2 * torch.log(torch.tensor(sigma_prior, device=x.device))
    )
    return recon_loss + beta * kl_loss


# =========================
# 3. Training Functions
# =========================

def train_beta_cvae(model, data_loader, optimizer, device, sigma_prior=0.5):
    """
    Train Beta-CVAE model for one epoch.
    """
    model.train()
    total_loss = 0
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        x_recon, mean, logvar = model(x_batch, y_batch)
        loss = beta_cvae_loss_fn(x_batch, x_recon, mean, logvar,
                                  beta=model.beta, sigma_prior=sigma_prior)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

def train_detector(model, train_loader, optimizer, criterion, device):
    """
    Train a detector model for one epoch.
    Returns (avg_loss, avg_grad_norm).
    """
    model.train()
    total_loss = 0
    total_grad_norm = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        # Compute grad norm without modifying gradients (read-only)
        with torch.no_grad():
            grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        optimizer.step()

        total_loss += loss.item()
        total_grad_norm += grad_norm
    n = len(train_loader)
    return total_loss / n, total_grad_norm / n

def make_balanced_loader(D_train, y_train, batch_size=64, seed=0, sample_labels=None):
    """
    Create a DataLoader with balanced sampling (equal normal/anomaly per batch).
    Uses WeightedRandomSampler to handle class imbalance.

    `sample_labels` can be provided when the training targets are soft labels
    but the sampler should still balance by hard class labels.
    """
    if sample_labels is None:
        sample_labels = y_train

    sample_labels = sample_labels.detach().view(-1).long().cpu()
    if len(sample_labels) != len(y_train):
        raise ValueError("sample_labels must have the same length as y_train")

    classes, class_counts = torch.unique(sample_labels, return_counts=True)
    weights = torch.zeros(int(sample_labels.max().item()) + 1, dtype=torch.float32)
    weights[classes] = 1.0 / class_counts.float()
    sample_weights = weights[sample_labels]

    g = torch.Generator()
    g.manual_seed(seed)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train), replacement=True, generator=g)
    dataset = TensorDataset(D_train, y_train)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


# =========================
# 4. Reward Functions
# =========================

def set_seed(seed=0):
    """Set all random seeds for full reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_entropy(D_train, x_syn, n_bins=50):
    """
    Estimate entropy increase after adding x_syn to D_train.
    Uses histogram-based approximation over feature dimensions.
    
    Args:
        D_train: Current training data (N, d).
        x_syn: New synthetic sample (1, d) or (d,).
    Returns:
        Entropy estimate (scalar tensor).
    """
    if x_syn.dim() == 1:
        x_syn = x_syn.unsqueeze(0)
    
    combined = torch.cat([D_train, x_syn], dim=0)
    entropy = 0.0
    for d in range(combined.shape[1]):
        col = combined[:, d]
        col = col[torch.isfinite(col)]
        if col.numel() == 0 or col.min() == col.max():
            continue
        hist = torch.histc(col, bins=n_bins)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy += -(probs * torch.log(probs)).sum()

    return entropy / combined.shape[1]  # Average entropy across dimensions

def compute_reward(x_syn, detector, D_train, gamma_decay, episode, device):
    """
    Compute reward following the DELTA reward formulation:
        R = gamma^episode * H(D_train ∪ x_syn) - log(W(x_syn))
    
    Early episodes: gamma^episode ≈ 1, emphasis on diversity (entropy).
    Later episodes: gamma^episode → 0, emphasis on deceiving detector.
    
    Args:
        x_syn: Generated synthetic samples (N, d).
        detector: Trained detector model.
        D_train: Current training data.
        gamma_decay: Decay factor for entropy weight (0 < gamma < 1).
        episode: Current episode number.
        device: Computation device.
    Returns:
        Reward tensor (N,).
    """
    detector.eval()
    x_syn = torch.nan_to_num(x_syn, nan=0.0, posinf=3.0, neginf=-3.0).clamp(-3.0, 3.0)
    x_syn = x_syn.to(device)

    with torch.no_grad():
        detect_prob = torch.sigmoid(detector(x_syn)).view(-1)
        detect_prob = torch.nan_to_num(detect_prob, nan=0.5).clamp(1e-8, 1.0)
    
    # Entropy term (batched approximation)
    D_train_dev = D_train.to(device)
    entropy_rewards = []
    for i in range(x_syn.shape[0]):
        ent = compute_entropy(D_train_dev, x_syn[i])
        entropy_rewards.append(ent)
    entropy_term = torch.stack(entropy_rewards).to(device)
    
    # Combined reward with gamma decay
    gamma_weight = gamma_decay ** episode
    reward = gamma_weight * entropy_term - torch.log(detect_prob + 1e-8)
    
    return reward


# =========================
# 5. Utility Functions
# =========================

def to_tensor(x, device="cpu", dtype=torch.float32):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device=device, dtype=dtype)


def One_Step_To_Feasible_Action(
        beta_cvae,
        detector,
        x_orig,
        device,
        previously_generated=None,
        alpha=1.0,
        lambda_div=0.1,
        lr=0.001,
        steps=50,
        log_file=None
):
    """
    Generate adversarial samples by gradient descent in latent space.
    Used during warmup episodes before PPO takes over.
    
    Optimizes: min prob_class1 + lambda_div * similarity_to_previous
    (i.e., generate samples that fool detector AND are diverse)
    """
    beta_cvae.eval()
    detector.eval()

    if previously_generated is None:
        previously_generated = []

    x_orig = x_orig.to(device).unsqueeze(0)
    y_class1 = torch.full((1, 1), 1.0, device=device)  # Use 1.0 for consistent conditioning

    # Encode input data into latent space
    with torch.no_grad():
        mean, logvar = beta_cvae.encode(x_orig, y_class1)
        z = beta_cvae.reparameterize(mean, logvar).detach().clone()
    
    z.requires_grad_(True)

    # Optimize latent space representation
    optimizer_z = torch.optim.Adam([z], lr=lr)
    for step in range(steps):
        optimizer_z.zero_grad()

        x_synthetic = beta_cvae.decode(z, y_class1)
        prob_class1 = torch.sigmoid(detector(x_synthetic))

        # Diversity term
        if previously_generated:
            x_old_cat = torch.stack(previously_generated, dim=0).to(device)
            dist = torch.norm(x_synthetic - x_old_cat, p=2, dim=1)
            diversity_term = torch.exp(-alpha * dist).sum()
        else:
            diversity_term = torch.tensor(0.0, device=device)

        # Loss = detection probability + similarity penalty (minimize both)
        loss = prob_class1.mean() + lambda_div * diversity_term
        loss.backward()
        optimizer_z.step()

    deceive_reward = 1.0 / (prob_class1.item() + 1e-4)
    div_val = diversity_term.item() if torch.is_tensor(diversity_term) else diversity_term
    div_reward = 1.0 / (div_val + 1e-4)
    print(f"Deceiving Detector Reward: {deceive_reward:.4f}",
          f"Diversity reward: {div_reward:.4f}",
          f"Loss: {loss.item():.4f}")

    if log_file:
        log_to_file(log_file, f"deceive={deceive_reward:.4f} div={div_reward:.4f} loss={loss.item():.4f}")

    with torch.no_grad():
        x_adv = beta_cvae.decode(z, y_class1).detach().cpu().squeeze(0)
    return x_adv

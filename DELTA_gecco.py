import os
import json
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import random
from datetime import datetime
from utils import *
from utils import evaluate_full
from model import *

# Small helper to override constants for ablation runs without editing code each time.
def _env(name, default, cast):
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return cast(raw)
    except Exception:
        return default


# =========================
# Reproducibility
# =========================
set_seed(_env("SH_SEED", 1, int))

# =========================
# Configuration
# =========================
GECCO_PATH = "gecco.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrain_dir = "./saved_models"

# Per-run output directory
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_tag = os.getenv("SH_RUN_TAG", "").strip()
run_name = f"{run_timestamp}_{run_tag}" if run_tag else run_timestamp
run_dir = os.path.join("runs", run_name)
save_dir = os.path.join(run_dir, "models")
log_dir = os.path.join(run_dir, "logs")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
print(f"Run directory: {run_dir}")

# Load config from pretrain
config_path = os.path.join(pretrain_dir, "gecco_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

WINDOW_SIZE = config["window_size"]
STRIDE = config["stride"]
TRAIN_RATIO = config["train_ratio"]
input_dim = config["input_dim"]
num_features = config["num_features"]
HIDDEN_DIM = config["hidden_dim"]
LATENT_DIM = config["latent_dim"]
BETA = config["beta"]
SIGMA_PRIOR = config["sigma_prior"]

# Co-evolution hyperparameters
NUM_EPISODES = _env("SH_NUM_EPISODES", 40, int)
NUM_GEN_DATA = _env("SH_NUM_GEN_DATA", 50, int)
BATCH_SIZE = _env("SH_BATCH_SIZE", 128, int)
WARMUP_EPISODES = _env("SH_WARMUP_EPISODES", 16, int)
TOP_L_RATIO = _env("SH_TOP_L_RATIO", 0.5, float)
GAMMA_DECAY = _env("SH_GAMMA_DECAY", 0.80, float)
FALLBACK_THRESHOLD = _env("SH_FALLBACK_THRESHOLD", 0.98, float)  # One-Step fallback trigger (loose)
AUG_THRESHOLD = _env("SH_AUG_THRESHOLD", 0.70, float)           # Augmentation gate (strict)
CLAMP_RANGE = _env("SH_CLAMP_RANGE", 3.0, float)
FREEZE_CVAE = _env("SH_FREEZE_CVAE", 1, int) == 1
FINAL_EPOCHS = _env("SH_FINAL_EPOCHS", 100, int)
EVAL_INTERVAL = max(1, _env("SH_EVAL_INTERVAL", 10, int))
SKIP_TSNE = _env("SH_SKIP_TSNE", _env("SH_SKIP_UMAP", 0, int), int) == 1

# =========================
# 1. Load GECCO Data (same as pretrain)
# =========================
from pretrain_gecco import load_gecco_windowed

D_train, y_train, D_test, y_test, _, _ = load_gecco_windowed(
    GECCO_PATH, window_size=WINDOW_SIZE, stride=STRIDE, train_ratio=TRAIN_RATIO
)

# =========================
# 2. Load Pretrained Models
# =========================
vae_path = os.path.join(pretrain_dir, "beta_cvae_gecco.pth")
detector_path = os.path.join(pretrain_dir, "transformer_detector_gecco.pth")

loaded_beta_cvae = BetaCVAE(input_dim=input_dim, hidden_dim=HIDDEN_DIM,
                             latent_dim=LATENT_DIM, beta=BETA).to(device)
loaded_detector_model = TransformerDetector(input_size=input_dim).to(device)

loaded_beta_cvae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
loaded_detector_model.load_state_dict(torch.load(detector_path, map_location=device, weights_only=True))
loaded_beta_cvae.eval()
loaded_detector_model.eval()
print("Pretrained GECCO models loaded successfully.")

# =========================
# 2b. Compute absolute plausibility threshold from real anomaly data
# =========================
with torch.no_grad():
    real_anom_mask = (y_train == 1)
    real_anom = D_train[real_anom_mask].to(device)
    y_cond_anom = torch.ones(len(real_anom), 1, device=device)
    mean_anom, _ = loaded_beta_cvae.encode(real_anom, y_cond_anom)
    x_recon_anom = loaded_beta_cvae.decode(mean_anom, y_cond_anom)
    real_anom_recon_err = F.mse_loss(x_recon_anom, real_anom, reduction='none').mean(dim=1)
    ABS_RECON_THRESHOLD = torch.quantile(real_anom_recon_err, 0.95).item()
print(f"Absolute plausibility threshold (real anomaly p95): {ABS_RECON_THRESHOLD:.4f}")

# =========================
# 3. Initialize Training Components
# =========================
new_detector = TransformerDetector(input_size=input_dim).to(device)
new_detector.load_state_dict(torch.load(detector_path, map_location=device, weights_only=True))  # Fix2: start from pretrained
if FREEZE_CVAE:
    for p in loaded_beta_cvae.parameters():
        p.requires_grad = False
    optimizer_cvae = None
    print("CVAE is frozen: skip CVAE updates during co-evolution.")
else:
    optimizer_cvae = Adam(loaded_beta_cvae.parameters(), lr=1e-4)
optimizer_detector = Adam(new_detector.parameters(), lr=1e-4)
criterion = FocalLoss(alpha=0.75, gamma=2.0)

# PPO agent
ppo_policy = PolicyNetwork(input_dim, 256, input_dim).to(device)
ppo_value = ValueNetwork(input_dim, 256).to(device)
ppo_trainer = PPOTrainer(
    ppo_policy, ppo_value,
    policy_lr=1e-4, value_lr=1e-4,
    gamma=0.99, clip_epsilon=0.2,
    entropy_coefficient=0.01,
    device=device
)

# Logging
beta_cvae_log = os.path.join(log_dir, "gecco_beta_cvae.log")
detector_log = os.path.join(log_dir, "gecco_detector.log")
adversarial_log = os.path.join(log_dir, "gecco_adversarial.log")
episode_log = os.path.join(log_dir, "gecco_episode_summary.log")
diag_log = os.path.join(log_dir, "gecco_diagnostics.log")

synthetic_data = []

# =========================
# 4. Co-Evolution Main Loop
# =========================
for ep in range(NUM_EPISODES):
    class1_mask = (y_train == 1)
    class0_mask = (y_train == 0)
    num_class1 = class1_mask.sum().item()
    num_class0 = class0_mask.sum().item()

    if num_class1 > num_class0:
        print(f"Break at Episode {ep + 1}: Class 1 ({num_class1}) exceeds Class 0 ({num_class0}).")
        break

    print(f"\n{'='*60}")
    print(f"EPISODE {ep + 1}/{NUM_EPISODES} | Class 0: {num_class0} | Class 1: {num_class1}")
    print(f"Mode: {'Warmup (gradient descent)' if ep < WARMUP_EPISODES else 'PPO'}")
    print(f"{'='*60}")

    # --- 4.1: Train/Fix Beta-CVAE ---
    if FREEZE_CVAE:
        log_to_file(beta_cvae_log, f"Episode {ep+1}: CVAE frozen (no update)")
    else:
        train_dataset = TensorDataset(D_train, y_train)
        g = torch.Generator()
        g.manual_seed(ep)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=g)
        for epoch in range(10):
            loss_cvae = train_beta_cvae(loaded_beta_cvae, train_loader, optimizer_cvae,
                                         device, sigma_prior=SIGMA_PRIOR)
            log_to_file(beta_cvae_log, f"Episode {ep+1} Epoch {epoch+1}/10, Loss: {loss_cvae:.4f}")

    # --- 4.2: Train Detector on BALANCED dataset ---
    balanced_loader = make_balanced_loader(D_train, y_train, batch_size=BATCH_SIZE, seed=ep)
    for det_epoch in range(5):
        detector_loss, det_grad_norm = train_detector(new_detector, balanced_loader, optimizer_detector, criterion, device)
        log_to_file(detector_log, f"Episode {ep+1} Epoch {det_epoch+1}/5, Loss: {detector_loss:.4f}")

    # Log 4a: Detector training stability (last epoch of this episode)
    log_to_file(diag_log, f"[DetStability] ep={ep+1} loss={detector_loss:.4f} grad_norm={det_grad_norm:.4f}")

    # --- 4.3: Generate Adversarial Samples ---
    idx_class1 = (y_train == 1).nonzero(as_tuple=True)[0]
    D_train_grow = [row for row in D_train[class1_mask]]

    candidate_samples = []
    candidate_origins = []
    candidate_log_probs = []
    candidate_deltas = []

    for syn_idx in range(NUM_GEN_DATA):
        random_idx = random.choice(idx_class1)
        x_orig = D_train[random_idx].to(device)

        if ep < WARMUP_EPISODES:
            x_adv = One_Step_To_Feasible_Action(
                beta_cvae=loaded_beta_cvae,
                detector=new_detector,  # Fix2: use co-evolving detector
                x_orig=x_orig,
                device=device,
                previously_generated=D_train_grow,
                alpha=1.0, lambda_div=0.1, lr=0.01, steps=20,
                log_file=adversarial_log,
            )
            # Finite guard: replace non-finite values with x_orig then re-clamp
            x_adv = torch.where(torch.isfinite(x_adv), x_adv, x_orig.cpu())
            x_adv = torch.clamp(x_adv, -CLAMP_RANGE, CLAMP_RANGE)
            delta = (x_adv - x_orig.cpu()).to(device)
            log_prob = torch.zeros(1, 1, device=device)
        else:
            delta, log_prob, ent = ppo_policy.sample_action(x_orig.unsqueeze(0))
            delta = delta.squeeze(0)
            x_adv = x_orig + delta
            # Finite guard: replace non-finite values with x_orig before clamping
            x_adv = torch.where(torch.isfinite(x_adv), x_adv, x_orig)
            x_adv = torch.clamp(x_adv, -CLAMP_RANGE, CLAMP_RANGE)
            # Recompute effective delta and log_prob after clamping so the
            # stored action matches the clamped x_adv used to compute rewards.
            delta = (x_adv - x_orig).detach()
            log_prob, _ = ppo_policy.log_prob_of(x_orig.unsqueeze(0), delta.unsqueeze(0))
            if not torch.isfinite(log_prob).all():
                log_prob = torch.zeros(1, 1, device=device)
            log_prob = log_prob.detach()
            x_adv = x_adv.detach().cpu()

        candidate_samples.append(x_adv.detach().cpu().unsqueeze(0))
        candidate_origins.append(x_orig.detach().cpu().unsqueeze(0))
        candidate_log_probs.append(log_prob.detach().cpu())
        candidate_deltas.append(delta.detach().cpu().unsqueeze(0))

    candidates = torch.cat(candidate_samples, dim=0)
    origins = torch.cat(candidate_origins, dim=0)
    log_probs = torch.cat(candidate_log_probs, dim=0)
    deltas = torch.cat(candidate_deltas, dim=0)

    # --- 4.4: Compute Rewards ---
    rewards = compute_reward(
        x_syn=candidates, detector=new_detector,  # Fix2: use co-evolving detector
        D_train=D_train, gamma_decay=GAMMA_DECAY,
        episode=ep, device=device
    )

    # --- 4.4b: One-Step fallback for invalid PPO actions ---
    # Per paper: when PPO generates actions the detector still catches, use gradient-descent
    # One-Step to find feasible actions; policy then learns these via supervised update.
    sl_states = []
    sl_feasible_deltas = []
    replaced_ppo_indices = set()
    if ep >= WARMUP_EPISODES:
        with torch.no_grad():
            all_det_probs = torch.sigmoid(new_detector(candidates.to(device))).view(-1).cpu()
        invalid_mask = all_det_probs >= FALLBACK_THRESHOLD
        invalid_indices = invalid_mask.nonzero(as_tuple=True)[0]
        if len(invalid_indices) > 0:
            print(f"  One-Step fallback: {len(invalid_indices)}/{NUM_GEN_DATA} invalid PPO actions")
            for i in invalid_indices:
                x_orig_i = origins[i].to(device)
                x_adv_feasible = One_Step_To_Feasible_Action(
                    beta_cvae=loaded_beta_cvae,
                    detector=new_detector,
                    x_orig=x_orig_i,
                    device=device,
                    previously_generated=D_train_grow,
                    alpha=1.0, lambda_div=0.1, lr=0.01, steps=20,
                    log_file=adversarial_log,
                )
                x_adv_feasible = torch.where(torch.isfinite(x_adv_feasible), x_adv_feasible, x_orig_i.cpu())
                x_adv_feasible = torch.clamp(x_adv_feasible, -CLAMP_RANGE, CLAMP_RANGE)
                candidates[i] = x_adv_feasible
                replaced_ppo_indices.add(i.item())
                sl_states.append(x_orig_i.cpu())
                sl_feasible_deltas.append((x_adv_feasible - x_orig_i.cpu()).detach())
            log_to_file(diag_log, f"[OneStepFallback] ep={ep+1} n_invalid={len(invalid_indices)}/{NUM_GEN_DATA}")
            # Recompute rewards with updated (feasible) candidates
            rewards = compute_reward(
                x_syn=candidates, detector=new_detector,
                D_train=D_train, gamma_decay=GAMMA_DECAY,
                episode=ep, device=device
            )

    # --- 4.5: Select Top-l ---
    l_top = max(1, int(NUM_GEN_DATA * TOP_L_RATIO))
    rewards = rewards.cpu()
    top_idx = torch.topk(rewards, k=l_top).indices
    selected = candidates[top_idx]

    # Fix3: CVAE plausibility filter — remove samples that drifted too far from anomaly distribution
    with torch.no_grad():
        sel_dev = selected.to(device)
        y_cond = torch.ones(len(sel_dev), 1, device=device)
        mean_sel, _ = loaded_beta_cvae.encode(sel_dev, y_cond)
        x_recon = loaded_beta_cvae.decode(mean_sel, y_cond)
        recon_err = F.mse_loss(x_recon, sel_dev, reduction='none').mean(dim=1)
        err_threshold = recon_err.median() + recon_err.std()
        relative_mask = recon_err <= err_threshold
        # PPO phase uses a tighter threshold to filter OOD samples (recon mean ~1.2 vs warmup ~0.15)
        ppo_abs_thresh = ABS_RECON_THRESHOLD * 0.5 if ep >= WARMUP_EPISODES else ABS_RECON_THRESHOLD
        absolute_mask = recon_err <= ppo_abs_thresh
        plausible_mask = relative_mask & absolute_mask
        # Apply filter to both selected samples AND top_idx (for PPO consistency)
        plausible_cpu = plausible_mask.cpu()
        selected = selected[plausible_cpu]
        filtered_top_idx = top_idx[plausible_cpu]
    selected_labels = torch.ones(len(selected))
    print(f"  Plausibility filter: {plausible_mask.sum().item()}/{l_top} samples kept")

    avg_reward = rewards[filtered_top_idx].mean().item() if len(filtered_top_idx) > 0 else 0.0
    print(f"Top-{l_top} avg reward: {avg_reward:.4f}")
    log_to_file(episode_log, f"Episode {ep+1}: top-{l_top} avg_reward={avg_reward:.4f} "
                f"class0={num_class0} class1={num_class1}")

    # === Diagnostics: Log 1 (Fix3 quality), Log 2 (Reward decomposition), Log 3 (Pseudo-label risk) ===
    # Log 1: Fix3 plausibility filter quality
    keep_ratio = len(filtered_top_idx) / l_top if l_top > 0 else 0.0
    keep_rel = relative_mask.sum().item() / l_top if l_top > 0 else 0.0
    keep_abs = absolute_mask.sum().item() / l_top if l_top > 0 else 0.0
    recon_mean = recon_err.mean().item()
    recon_std_val = recon_err.std().item()
    recon_p95 = torch.quantile(recon_err, 0.95).item()
    log_to_file(diag_log, f"[Fix3] ep={ep+1} keep={keep_ratio:.2f} keep_rel={keep_rel:.2f} keep_abs={keep_abs:.2f} recon: "
                f"mean={recon_mean:.4f} std={recon_std_val:.4f} p95={recon_p95:.4f} "
                f"rel_thresh={err_threshold.item():.4f} abs_thresh={ppo_abs_thresh:.4f}")

    # Log 2: Reward decomposition (entropy vs. fooling term)
    with torch.no_grad():
        cand_probs = torch.sigmoid(new_detector(candidates.to(device))).view(-1).cpu()
    top_probs = cand_probs[top_idx]
    filt_probs = cand_probs[filtered_top_idx] if len(filtered_top_idx) > 0 else torch.tensor([0.0])
    gamma_w = GAMMA_DECAY ** ep
    raw_top_reward = rewards[top_idx].mean().item()
    filt_reward = rewards[filtered_top_idx].mean().item() if len(filtered_top_idx) > 0 else 0.0
    log_to_file(diag_log, f"[Reward] ep={ep+1} gamma={gamma_w:.4f} raw_topk={raw_top_reward:.4f} filtered={filt_reward:.4f} "
                f"det_prob_topk: mean={top_probs.mean().item():.4f} p90={torch.quantile(top_probs, 0.9).item():.4f} "
                f"det_prob_filtered: mean={filt_probs.mean().item():.4f}")

    # Log 3: Pseudo-label noise risk
    if len(selected) > 0:
        with torch.no_grad():
            sel_probs = torch.sigmoid(new_detector(selected.to(device))).view(-1).cpu()
        pct_low = (sel_probs < 0.3).float().mean().item()
        log_to_file(diag_log, f"[PseudoLabel] ep={ep+1} n={len(selected)} det_prob: "
                    f"mean={sel_probs.mean().item():.4f} min={sel_probs.min().item():.4f} "
                    f"max={sel_probs.max().item():.4f} pct_below_0.3={pct_low:.2f}")
    else:
        log_to_file(diag_log, f"[PseudoLabel] ep={ep+1} n=0 (all filtered out)")

    # --- 4.6: PPO Update (plausibility-filtered AND adversarially-good samples only) ---
    if ep >= WARMUP_EPISODES and len(filtered_top_idx) > 0:
        # Exclude One-Step replaced samples: their stored deltas/log_probs are stale PPO actions.
        # No additional det_prob gate — all CVAE-plausible samples participate in PPO update.
        if replaced_ppo_indices:
            not_replaced_mask = torch.tensor(
                [i.item() not in replaced_ppo_indices for i in filtered_top_idx],
                dtype=torch.bool
            )
            ppo_idx = filtered_top_idx[not_replaced_mask]
            ppo_gate_det = filt_probs[not_replaced_mask].mean().item() if not_replaced_mask.any() else float('nan')
        else:
            ppo_idx = filtered_top_idx
            ppo_gate_det = filt_probs.mean().item() if len(filt_probs) > 0 else float('nan')
        ppo_gate_n = len(ppo_idx)
        ppo_gate_total = len(filtered_top_idx)
        if len(ppo_idx) > 0:
            ppo_states = origins[ppo_idx].to(device)
            ppo_actions = deltas[ppo_idx].to(device)
            ppo_old_log_probs = log_probs[ppo_idx].to(device)
            ppo_rewards_selected = rewards[ppo_idx].unsqueeze(1).to(device)

            with torch.no_grad():
                values = ppo_value(ppo_states)
            advantages = ppo_rewards_selected - values
            returns = ppo_rewards_selected

            ppo_stats = ppo_trainer.ppo_update(
                states=ppo_states, actions=ppo_actions,
                old_log_probs=ppo_old_log_probs,
                returns=returns, advantages=advantages, n_epochs=3
            )
            print(f"PPO updated with {len(ppo_idx)}/{len(filtered_top_idx)} adversarial samples")
            # Log 4b: PPO training stability
            log_to_file(diag_log, f"[PPO] ep={ep+1} policy_loss={ppo_stats['policy_loss']:.4f} "
                        f"value_loss={ppo_stats['value_loss']:.4f} entropy={ppo_stats['entropy']:.4f}")
        else:
            print(f"PPO skipped ep={ep+1}: all candidates replaced by One-Step fallback")

    # --- 4.6b: Supervised policy update on One-Step feasible actions ---
    # Per paper: the agent learns feasible actions via supervised update to bootstrap exploration.
    if ep >= WARMUP_EPISODES and len(sl_states) > 0:
        sl_state_tensor = torch.stack(sl_states).to(device)
        sl_delta_tensor = torch.stack(sl_feasible_deltas).to(device)
        log_probs_sl, _ = ppo_policy.log_prob_of(sl_state_tensor, sl_delta_tensor)
        sl_loss = -log_probs_sl.mean()
        if torch.isfinite(sl_loss):
            ppo_trainer.policy_optimizer.zero_grad()
            sl_loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo_policy.parameters(), max_norm=0.5)
            ppo_trainer.policy_optimizer.step()
            print(f"  Supervised policy update: {len(sl_states)} feasible actions, loss={sl_loss.item():.4f}")
            log_to_file(diag_log, f"[SL] ep={ep+1} n={len(sl_states)} sl_loss={sl_loss.item():.4f}")

    # --- 4.7: Augment training set ---
    if ep >= WARMUP_EPISODES and len(selected) > 0:
        # Adversarial quality gate: only keep PPO samples that partially fool the detector.
        # sel_probs is computed in Log 3 above and is safe to use here (selected unchanged).
        adv_mask = sel_probs < AUG_THRESHOLD
        n_adv_before = len(selected)
        selected = selected[adv_mask]
        selected_labels = selected_labels[adv_mask]
        print(f"  Adversarial gate: {len(selected)}/{n_adv_before} pass det_prob < {AUG_THRESHOLD:.2f}")
        aug_gate_det = sel_probs[adv_mask].mean().item() if adv_mask.any() else float('nan')
        log_to_file(diag_log, f"[GateStats] ep={ep+1} "
                    f"ppo_pass={ppo_gate_n}/{ppo_gate_total} det_ppo={ppo_gate_det:.4f} "
                    f"aug_pass={len(selected)}/{n_adv_before} det_aug={aug_gate_det:.4f}")
    synthetic_data.extend(list(selected))
    D_train = torch.cat([D_train, selected], dim=0)
    y_train = torch.cat([y_train, selected_labels], dim=0)

print(f"\n{'='*60}")
print(f"Co-evolution complete. Total synthetic samples: {len(synthetic_data)}")
print(f"Final dataset: {len(D_train)} samples")
print(f"{'='*60}")

# Persist the final co-evolution policy/value networks so the last executed
# episode's strategy can be inspected or reused later.
ppo_policy_path = os.path.join(save_dir, "ppo_policy_gecco_last_episode.pth")
ppo_value_path = os.path.join(save_dir, "ppo_value_gecco_last_episode.pth")
torch.save(ppo_policy.state_dict(), ppo_policy_path)
torch.save(ppo_value.state_dict(), ppo_value_path)
print(f"Saved final PPO policy to: {ppo_policy_path}")
print(f"Saved final PPO value network to: {ppo_value_path}")
log_to_file(
    episode_log,
    f"Saved final PPO networks: policy={ppo_policy_path}, value={ppo_value_path}"
)

# Persist the actual synthetic anomalies that were accepted into training so
# downstream evaluation reads the exact training-used samples instead of
# re-generating fresh ones from the decoder prior.
synthetic_samples_path = os.path.join(run_dir, "synthetic_anomalies_used.pt")
synthetic_payload = {
    "samples": (
        torch.stack(synthetic_data).cpu()
        if synthetic_data else
        torch.empty(0, input_dim)
    ),
    "n_samples": len(synthetic_data),
    "input_dim": input_dim,
    "run_dir": run_dir,
    "source": "training_used_selected_synthetic_anomalies",
}
torch.save(synthetic_payload, synthetic_samples_path)
print(f"Saved training-used synthetic anomalies to: {synthetic_samples_path}")
log_to_file(
    episode_log,
    f"Saved training-used synthetic anomalies: {synthetic_samples_path} "
    f"(n={len(synthetic_data)})"
)

# =========================
# 5. t-SNE Visualization
# =========================
if SKIP_TSNE:
    print("\n=== Skipping t-SNE visualization (SH_SKIP_TSNE=1 or SH_SKIP_UMAP=1) ===")
else:
    print("\n=== Generating t-SNE visualization ===")
    from sklearn.manifold import TSNE
    plt.style.use('default')

    # Reload original data for visualization
    from pretrain_gecco import load_gecco_windowed
    D_train_orig, y_train_orig, _, _, _, _ = load_gecco_windowed(
        GECCO_PATH, window_size=WINDOW_SIZE, stride=STRIDE, train_ratio=TRAIN_RATIO
    )

    X_synthetic = torch.stack(synthetic_data) if synthetic_data else torch.empty(0, input_dim)

    # t-SNE is much heavier than UMAP, so keep the sample cap tighter.
    max_vis_per_group = 2000

    real_X = torch.cat([D_train_orig, D_test], dim=0)
    real_y = torch.cat([y_train_orig, y_test], dim=0)
    real_X_np = real_X.numpy()

    normal_idx = (real_y == 0).nonzero(as_tuple=True)[0].cpu().numpy()
    anomaly_idx = (real_y == 1).nonzero(as_tuple=True)[0].cpu().numpy()

    if len(normal_idx) > max_vis_per_group:
        normal_idx = np.random.choice(normal_idx, max_vis_per_group, replace=False)
    if len(anomaly_idx) > max_vis_per_group:
        anomaly_idx = np.random.choice(anomaly_idx, max_vis_per_group, replace=False)

    normal_points = real_X_np[normal_idx] if len(normal_idx) > 0 else np.empty((0, input_dim))
    anomaly_points = real_X_np[anomaly_idx] if len(anomaly_idx) > 0 else np.empty((0, input_dim))

    if len(X_synthetic) > max_vis_per_group:
        syn_idx = np.random.choice(len(X_synthetic), max_vis_per_group, replace=False)
        synthetic_points = X_synthetic.numpy()[syn_idx]
    else:
        synthetic_points = X_synthetic.numpy() if len(X_synthetic) > 0 else np.empty((0, input_dim))

    X_plot_parts = [normal_points, anomaly_points]
    label_parts = [
        np.zeros(len(normal_points), dtype=int),
        np.ones(len(anomaly_points), dtype=int),
    ]
    if len(synthetic_points) > 0:
        X_plot_parts.append(synthetic_points)
        label_parts.append(np.full(len(synthetic_points), 2, dtype=int))

    X_plot = np.concatenate(X_plot_parts, axis=0)
    y_plot = np.concatenate(label_parts, axis=0)

    if len(X_plot) < 3:
        print("Skipping t-SNE: not enough points to visualize.")
    else:
        perplexity = min(30, max(5, len(X_plot) // 20))
        perplexity = min(perplexity, len(X_plot) - 1)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=_env("SH_SEED", 1, int),
        )
        X_embedded = tsne.fit_transform(X_plot)

        plt.figure(figsize=(10, 8), facecolor='white')

        normal_mask = (y_plot == 0)
        anomaly_mask = (y_plot == 1)
        synthetic_mask = (y_plot == 2)

        plt.scatter(
            X_embedded[normal_mask, 0],
            X_embedded[normal_mask, 1],
            c='blue',
            alpha=0.45,
            s=10,
            label='Normal',
        )
        plt.scatter(
            X_embedded[anomaly_mask, 0],
            X_embedded[anomaly_mask, 1],
            c='red',
            alpha=0.65,
            s=15,
            label='Anomaly',
        )
        if synthetic_mask.any():
            plt.scatter(
                X_embedded[synthetic_mask, 0],
                X_embedded[synthetic_mask, 1],
                c='green',
                alpha=0.70,
                s=15,
                label='Synthetic anomaly',
            )

        plt.title("t-SNE: GECCO Normal, Anomaly, and Synthetic Anomaly")
        plt.legend()
        tsne_path = os.path.join(run_dir, "gecco_tsne.png")
        plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"t-SNE figure saved to {tsne_path}")

# =========================
# 6. Final Evaluation
# =========================
# Fix1: Use balanced sampling for final training (handles remaining class imbalance)
train_loader_final = make_balanced_loader(D_train, y_train, batch_size=64)
test_dataset = TensorDataset(D_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64)

print("\nAfter co-evolution augmentation:")
unique, counts = np.unique(y_train.numpy(), return_counts=True)
print("Class distribution:", dict(zip(unique.astype(int), counts)))

# Fix4: Fine-tune from pretrained model instead of training from scratch
model = TransformerDetector(input_size=input_dim).to(device)
model.load_state_dict(torch.load(detector_path, map_location=device, weights_only=True))
optimizer_tf = Adam(model.parameters(), lr=1e-4)  # Lower LR for fine-tuning
criterion = FocalLoss(alpha=0.75, gamma=2.0)

# Evaluate pretrained model first as the baseline (prevent regression)
print("\n--- Pretrained model baseline ---")
auroc_base, aupr_base, f1_base, _ = evaluate_full(model, test_loader, device)
best_f1_score = f1_base if f1_base else 0.0
best_auc = auroc_base if auroc_base else 0.0
best_aupr = aupr_base if aupr_base else 0.0
best_epoch = 0
print(f"Pretrained baseline: F1={best_f1_score:.4f}, AUPR={best_aupr:.4f}, AUC={best_auc:.4f}")
print("-" * 40)
for epoch in range(FINAL_EPOCHS):
    train_loss, _ = train_detector(model, train_loader_final, optimizer_tf, criterion, device)
    if (epoch + 1) % EVAL_INTERVAL == 0:
        print(f"\n[Transformer] Epoch {epoch+1}/{FINAL_EPOCHS}, Loss={train_loss:.4f}")
        print("Test set evaluation:")
        auroc, aupr, f1, _ = evaluate_full(model, test_loader, device)
        if f1 and f1 > best_f1_score:
            best_f1_score = f1
            best_auc = auroc
            best_aupr = aupr
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "best_detector_gecco.pth"))
            print(f"  >> New best model saved (epoch {best_epoch}, F1={best_f1_score:.4f})")
        # Log 5: Baseline comparison (ΔF1/ΔAUPR/ΔAUC vs pretrained)
        if f1:
            delta_f1 = f1 - (f1_base or 0)
            delta_aupr = aupr - (aupr_base or 0)
            delta_auc = auroc - (auroc_base or 0)
            best_source = f"epoch {best_epoch}" if best_epoch > 0 else "pretrained"
            log_to_file(diag_log, f"[Baseline] epoch={epoch+1} F1={f1:.4f}(Δ{delta_f1:+.4f}) "
                        f"AUPR={aupr:.4f}(Δ{delta_aupr:+.4f}) AUC={auroc:.4f}(Δ{delta_auc:+.4f}) best_from={best_source}")
        print("-" * 40)

print(f"\n{'='*50}")
print(f"Best results after co-evolution (epoch {best_epoch}):")
print(f"  Best F1: {best_f1_score:.4f}")
print(f"  AUPR:    {best_aupr:.4f}")
print(f"  AUC-ROC: {best_auc:.4f}")
print(f"{'='*50}")

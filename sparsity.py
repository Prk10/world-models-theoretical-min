# Experiment 2: Activation Sparsity During World Model Planning


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RSSM(nn.Module):
    def __init__(self, latent_dim=512, stoch_dim=32, stoch_classes=32, action_dim=6):
        super().__init__()
        self.latent_dim    = latent_dim
        self.stoch_dim     = stoch_dim
        self.stoch_classes = stoch_classes
        self.stoch_size    = stoch_dim * stoch_classes

        self.gru   = nn.GRUCell(self.stoch_size + action_dim, latent_dim)
        self.prior = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ELU(),
            nn.Linear(latent_dim, self.stoch_size),
        )
        self.proj = nn.Linear(latent_dim + self.stoch_size, latent_dim)

    def imagine_step(self, det, stoch, action):
        x      = torch.cat([stoch, action], dim=-1)
        det    = self.gru(x, det)
        logits = self.prior(det).view(-1, self.stoch_dim, self.stoch_classes)
        stoch  = torch.softmax(logits, dim=-1).view(-1, self.stoch_size)
        feat   = self.proj(torch.cat([det, stoch], dim=-1))
        return det, stoch, feat

def measure_sparsity(tensor, threshold=0.01):
    """Fraction of elements with absolute value below threshold."""
    return (tensor.abs() < threshold).float().mean().item()

def run_sparsity_experiment(
    model,
    horizon=20,
    batch_size=64,
    action_dim=6,
    threshold=0.01,
    n_trials=10
):
    """
    Run imagined rollouts and record activation sparsity at each step.
    Returns: dicts mapping step -> list of sparsity values across trials
    """
    model.eval()
    det_sparsity   = {h: [] for h in range(horizon)}
    stoch_sparsity = {h: [] for h in range(horizon)}
    feat_sparsity  = {h: [] for h in range(horizon)}

    with torch.no_grad():
        for trial in range(n_trials):
            # Random initial state
            det   = torch.randn(batch_size, model.latent_dim,  device=device) * 0.5
            stoch = torch.softmax(
                torch.randn(batch_size, model.stoch_dim, model.stoch_classes, device=device),
                dim=-1
            ).view(batch_size, -1)

            for h in range(horizon):
                action = torch.randn(batch_size, action_dim, device=device)
                det, stoch, feat = model.imagine_step(det, stoch, action)

                det_sparsity[h].append(measure_sparsity(det,   threshold))
                stoch_sparsity[h].append(measure_sparsity(stoch, threshold))
                feat_sparsity[h].append(measure_sparsity(feat,  threshold))

    # Average across trials
    det_mean   = [np.mean(det_sparsity[h])   for h in range(horizon)]
    stoch_mean = [np.mean(stoch_sparsity[h]) for h in range(horizon)]
    feat_mean  = [np.mean(feat_sparsity[h])  for h in range(horizon)]

    return det_mean, stoch_mean, feat_mean


def plot_sparsity(det_mean, stoch_mean, feat_mean, save_path="figure_sparsity.png"):
    horizon = len(det_mean)
    steps   = range(horizon)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, [v * 100 for v in det_mean],   'o-', color='#3498DB', label='Deterministic State', linewidth=2)
    ax.plot(steps, [v * 100 for v in stoch_mean], 's-', color='#E74C3C', label='Stochastic Latent',   linewidth=2)
    ax.plot(steps, [v * 100 for v in feat_mean],  '^-', color='#2ECC71', label='Feature Projection',  linewidth=2)

    ax.axhline(np.mean(det_mean)   * 100, color='#3498DB', linestyle='--', alpha=0.4)
    ax.axhline(np.mean(stoch_mean) * 100, color='#E74C3C', linestyle='--', alpha=0.4)

    ax.set_xlabel("Planning Rollout Step", fontsize=12)
    ax.set_ylabel("Activation Sparsity (%)", fontsize=12)
    ax.set_title("Activation Sparsity Across Planning Horizon", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {save_path}")
    plt.close()



if __name__ == "__main__":
    model = RSSM().to(device)

    HORIZON    = 20
    BATCH_SIZE = 64
    THRESHOLD  = 0.01

    print(f"Running sparsity experiment: horizon={HORIZON}, batch={BATCH_SIZE}, threshold={THRESHOLD}")
    det_mean, stoch_mean, feat_mean = run_sparsity_experiment(
        model, horizon=HORIZON, batch_size=BATCH_SIZE, threshold=THRESHOLD
    )

    print("\nSparsity Results:")
    print(f"  Deterministic state:  {np.mean(det_mean)*100:.1f}% avg (range {min(det_mean)*100:.1f}–{max(det_mean)*100:.1f}%)")
    print(f"  Stochastic latent:    {np.mean(stoch_mean)*100:.1f}% avg (range {min(stoch_mean)*100:.1f}–{max(stoch_mean)*100:.1f}%)")
    print(f"  Feature projection:   {np.mean(feat_mean)*100:.1f}% avg (range {min(feat_mean)*100:.1f}–{max(feat_mean)*100:.1f}%)")

    print(f"\n-> Section 4.3: sparsity rates of "
          f"{np.mean(det_mean)*100:.0f}-{np.mean(stoch_mean)*100:.0f}% "
          f"in the deterministic state and stochastic latent.")

    plot_sparsity(det_mean, stoch_mean, feat_mean, save_path="figure_sparsity.png")

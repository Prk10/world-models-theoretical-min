
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("Run this on a GPU. Kaggle: Settings → Accelerator → GPU.")
print(f"Running on: {torch.cuda.get_device_name(0)}")


GPU_NAME = torch.cuda.get_device_name(0).upper()
if   "T4"   in GPU_NAME: PEAK_FLOPS, PEAK_BW = 8.1e12,  300e9
elif "P100" in GPU_NAME: PEAK_FLOPS, PEAK_BW = 9.3e12,  732e9
elif "V100" in GPU_NAME: PEAK_FLOPS, PEAK_BW = 14e12,   900e9
elif "A100" in GPU_NAME: PEAK_FLOPS, PEAK_BW = 19.5e12, 2000e9
else:
    print(f"[WARNING] Unknown GPU '{GPU_NAME}'. Using conservative defaults.")
    PEAK_FLOPS, PEAK_BW = 10e12, 300e9

RIDGE_POINT = PEAK_FLOPS / PEAK_BW
print(f"Hardware: {PEAK_FLOPS/1e12:.1f} TFLOPS | {PEAK_BW/1e9:.0f} GB/s | "
      f"Ridge = {RIDGE_POINT:.1f} FLOPs/byte\n")

BYTES_PER_FLOAT = 4  # FP32


class RSSM(nn.Module):
    def __init__(self, latent_dim=512, stoch_dim=32, stoch_classes=32, action_dim=6):
        super().__init__()
        self.latent_dim    = latent_dim
        self.stoch_dim     = stoch_dim
        self.stoch_classes = stoch_classes
        self.stoch_size    = stoch_dim * stoch_classes
        self.action_dim    = action_dim

        self.gru   = nn.GRUCell(self.stoch_size + action_dim, latent_dim)
        self.prior = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ELU(),
            nn.Linear(latent_dim, self.stoch_size),
        )
        self.proj  = nn.Linear(latent_dim + self.stoch_size, latent_dim)


def analytical_flops(m, B):
    S, D, A = m.stoch_size, m.latent_dim, m.action_dim
    return {
        "GRU Transition":
            3 * 2 * (S + A + D) * D * B,          
        "Prior Network":
            (2 * D * D + 2 * D * S) * B,           
        "Projection":
            2 * (D + S) * D * B,                   
        "Stochastic Sampling":
            m.stoch_dim * m.stoch_classes * 3 * B, 
    }


def analytical_bytes(m, B):
    
    S, D, A = m.stoch_size, m.latent_dim, m.action_dim
    bp = BYTES_PER_FLOAT

    
    gru_weights = (3*D*(S+A) + 3*D*D + 2*3*D) * bp
    gru_input   = B * (S + A) * bp    
    gru_hidden  = B * D * bp         
    gru_output  = B * D * bp          
    gru_bytes   = gru_weights + gru_input + gru_hidden + gru_output

    
    prior_w1  = D * D * bp            
    prior_b1  = D * bp                # bias layer 1
    prior_w2  = D * S * bp            # weight matrix layer 2
    prior_b2  = S * bp                # bias layer 2
    prior_in  = B * D * bp            # input (deterministic state)
    prior_mid = B * D * bp            # intermediate activation
    prior_out = B * S * bp            # output logits
    prior_bytes = prior_w1 + prior_b1 + prior_w2 + prior_b2 + \
                  prior_in + prior_mid + prior_out

    
    proj_w    = (D + S) * D * bp      # weight matrix
    proj_b    = D * bp                # bias
    proj_in   = B * (D + S) * bp     # input (concat det + stoch)
    proj_out  = B * D * bp            # output feature
    proj_bytes = proj_w + proj_b + proj_in + proj_out

    
    samp_in   = B * S * bp            
    samp_out  = B * S * bp            
    samp_bytes = samp_in + samp_out

    return {
        "GRU Transition":      gru_bytes,
        "Prior Network":       prior_bytes,
        "Projection":          proj_bytes,
        "Stochastic Sampling": samp_bytes,
    }


def compute_roofline(model, B=32):
    flops = analytical_flops(model, B)
    bytes_ = analytical_bytes(model, B)

    results = {}
    for name in flops:
        f = flops[name]
        b = bytes_[name]
        results[name] = {
            "intensity":   f / b,
            "flops":       f,
            "bytes":       b,
            "byte_source": "analytical (weights+IO)",
        }
        print(f"  {name:<25}  {f/b:>7.2f} FLOPs/byte  "
              f"({f:,} FLOPs / {b:,} bytes)")
    return results


def plot_roofline(results, save_path="figure1_roofline.png"):
    x    = np.logspace(-2, 4, 500)
    roof = np.minimum(x * PEAK_BW, PEAK_FLOPS)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(x, roof, 'k-', linewidth=2.5, label='Roofline (hardware limit)')
    ax.axvline(RIDGE_POINT, color='gray', linestyle='--', alpha=0.6,
               label=f'Ridge point ({RIDGE_POINT:.0f} FLOPs/byte)')
    ax.axvspan(x[0], RIDGE_POINT, alpha=0.06, color='red')
    ax.text(x[0] * 1.4, PEAK_FLOPS * 0.04, 'Memory-\nBound',
            color='red', alpha=0.7, fontsize=10, fontweight='bold')

    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']

    
    x_jitter = {
        "GRU Transition":      1.25,   # nudge right
        "Prior Network":       1.00,   # centre
        "Projection":          0.80,   # nudge left
        "Stochastic Sampling": 1.00,   # far left, no overlap
    }

    
    label_offsets = {
        "GRU Transition":      ( 10,  6),
        "Prior Network":       ( 10,  6),
        "Projection":          ( 10, -16),
        "Stochastic Sampling": ( 10,  6),
    }

    for (name, data), color in zip(results.items(), colors):
        I_true  = data["intensity"]               
        I_plot  = I_true * x_jitter[name]         # jittered x for visual separation
        perf    = min(I_plot * PEAK_BW, PEAK_FLOPS)  # must stay on roofline
        ox, oy  = label_offsets[name]

        ax.scatter(I_plot, perf, s=160, color=color, zorder=5,
                   label=f'{name}  ({I_true:.1f} FLOPs/byte)',
                   edgecolors='white', linewidths=0.8)
        ax.annotate(
            name, (I_plot, perf),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=8.5,
        )

    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte)", fontsize=12)
    ax.set_ylabel("Attainable Performance (FLOPs / sec)", fontsize=12)
    ax.set_title(
        f"Roofline Analysis — World Model Planning\n"
        f"({torch.cuda.get_device_name(0)})", fontsize=13)
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([1e-2, 1e4])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    BATCH_SIZE = 32
    model = RSSM().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    print("Computing arithmetic intensity (analytical FLOPs / analytical bytes)...\n")

    results = compute_roofline(model, B=BATCH_SIZE)

    print(f"\n{'Operation':<25} {'Intensity':>10} {'FLOPs':>15} {'Bytes':>15}")
    print("-" * 70)
    for name, d in results.items():
        print(f"{name:<25} {d['intensity']:>10.2f} {d['flops']:>15,} {d['bytes']:>15,}")

    n_mem = sum(1 for d in results.values() if d["intensity"] < RIDGE_POINT)
    print(f"\nKey finding: {n_mem}/{len(results)} operations are memory-bandwidth-bound")
    print(f"  (arithmetic intensity < ridge point of {RIDGE_POINT:.1f} FLOPs/byte)")

    plot_roofline(results)

    print("\nMethodological note: bytes = weight parameters + input tensors + output")
    print("tensors, all in FP32. This is the minimum DRAM traffic model; actual")
    print("traffic is higher due to cache misses and memory bus alignment.")
    print("Reported intensities are therefore upper bounds — memory-bound")
    print("conclusion is conservative.")
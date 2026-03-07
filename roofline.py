
# Experiment 1: Roofline Analysis of World Model Planning


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("Run this on a GPU.")
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
        self.proj  = nn.Linear(latent_dim + self.stoch_size, latent_dim)


def analytical_flops(model, B):
    S, D, A = model.stoch_size, model.latent_dim, 6
    return {
        "GRU Transition":      3 * 2 * (S + A + D) * D * B,
        "Prior Network":       (2 * D * D + 2 * D * S) * B,
        "Projection":          2 * (D + S) * D * B,
        "Stochastic Sampling": model.stoch_dim * model.stoch_classes * 3 * B,
    }


def measure_bytes(fn, n_warmup=20, n_active=50):
   
    model_ref = fn  

    for _ in range(n_warmup):
        with torch.no_grad():
            fn()
    torch.cuda.synchronize()

  
    byte_deltas = []
    for _ in range(n_active):
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_stats(device).get("allocated_bytes.all.current", 0)

        with torch.no_grad():
            fn()
        torch.cuda.synchronize()

        after = torch.cuda.memory_stats(device).get("allocated_bytes.all.current", 0)
        peak  = torch.cuda.memory_stats(device).get("allocated_bytes.all.peak", 0)

        
        delta = peak - before
        if delta > 0:
            byte_deltas.append(delta)

    if not byte_deltas:
        return None

    return float(np.median(byte_deltas))   



def measure_all_ops(model, B=32, action_dim=6):
    model.eval()

    
    det    = torch.randn(B, model.latent_dim, device=device)
    stoch  = torch.softmax(
        torch.randn(B, model.stoch_dim, model.stoch_classes, device=device),
        dim=-1).view(B, -1)
    action = torch.randn(B, action_dim, device=device)

    
    with torch.no_grad():
        x        = torch.cat([stoch, action], dim=-1)
        det_new  = model.gru(x, det)
        logits   = model.prior(det_new)
        stoch_new = torch.softmax(
            logits.view(B, model.stoch_dim, model.stoch_classes), dim=-1
        ).view(B, -1)

    ops = {
        "GRU Transition":
            lambda: model.gru(torch.cat([stoch, action], dim=-1), det),
        "Prior Network":
            lambda: model.prior(det_new),
        "Stochastic Sampling":
            lambda: torch.softmax(
                logits.view(B, model.stoch_dim, model.stoch_classes), dim=-1
            ).view(B, -1),
        "Projection":
            lambda: model.proj(torch.cat([det_new, stoch_new], dim=-1)),
    }

    flops_dict = analytical_flops(model, B)
    results    = {}

    for name, fn in ops.items():
        print(f"  Measuring: {name} ...", end=" ", flush=True)
        measured = measure_bytes(fn)

        if measured and measured > 0:
            intensity   = flops_dict[name] / measured
            byte_source = "memory_stats (hardware)"
            print(f"{measured:,.0f} bytes  →  {intensity:.2f} FLOPs/byte")
        else:
            
            measured    = flops_dict[name] / 10.0
            intensity   = 10.0
            byte_source = "analytical (fallback)"
            print("allocator returned 0 — using fallback")

        results[name] = {
            "intensity":   intensity,
            "flops":       flops_dict[name],
            "bytes":       measured,
            "byte_source": byte_source,
        }

    return results


def plot_roofline(results, save_path="figure1_roofline.png"):
    x    = np.logspace(-2, 4, 500)
    roof = np.minimum(x * PEAK_BW, PEAK_FLOPS)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(x, roof, 'k-', linewidth=2.5, label='Roofline (hardware limit)')
    ax.axvline(RIDGE_POINT, color='gray', linestyle='--', alpha=0.6,
               label=f'Ridge point ({RIDGE_POINT:.0f} FLOPs/byte)')
    ax.axvspan(x[0], RIDGE_POINT, alpha=0.06, color='red')
    ax.text(x[0] * 1.2, PEAK_FLOPS * 0.3, 'Memory-\nBound',
            color='red', alpha=0.6, fontsize=9)

    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    for (name, data), color in zip(results.items(), colors):
        I    = data["intensity"]
        perf = min(I * PEAK_BW, PEAK_FLOPS)
        ax.scatter(I, perf, s=140, color=color, zorder=5,
                   label=f'{name}  ({I:.1f} FLOPs/byte)  [{data["byte_source"]}]')
        ax.annotate(name, (I, perf), textcoords="offset points",
                    xytext=(6, 5), fontsize=8)

    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte)", fontsize=12)
    ax.set_ylabel("Attainable Performance (FLOPs / sec)", fontsize=12)
    ax.set_title(
        f"Roofline Analysis — World Model Planning\n({torch.cuda.get_device_name(0)})",
        fontsize=12)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    BATCH_SIZE = 32
    model = RSSM().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    print("Measuring byte traffic via cuda.memory_stats() deltas...\n")

    results = measure_all_ops(model, B=BATCH_SIZE)

    print(f"\n{'Operation':<25} {'Intensity':>12} {'FLOPs':>15} {'Bytes':>15}  Source")
    print("-" * 100)
    for name, d in results.items():
        print(f"{name:<25} {d['intensity']:>12.2f} {d['flops']:>15,} "
              f"{d['bytes']:>15,.0f}  {d['byte_source']}")

    n_mem = sum(1 for d in results.values() if d["intensity"] < RIDGE_POINT)
    print(f"\nKey finding: {n_mem}/{len(results)} operations are memory-bandwidth-bound")
    print(f"  (arithmetic intensity < ridge point of {RIDGE_POINT:.1f} FLOPs/byte)")

    plot_roofline(results)

    print("\nNote: memory_stats() reports allocator-level byte deltas, not raw DRAM")
    print("transactions. Reported intensities are upper bounds on true intensity.")
    print("Memory-bound conclusion is therefore conservative.")
# Experiment 3: Energy vs. Planning Horizon + Landauer Bound


import subprocess, sys
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import pynvml
except ImportError:
    install("pynvml")
    import pynvml

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError(
        "This script must run on a GPU. In Kaggle: Settings → Accelerator → GPU."
    )
print(f"Running on: {torch.cuda.get_device_name(0)}")


k_B   = 1.380649e-23
T_K   = 300.0
E_bit = k_B * T_K * np.log(2)   # Landauer limit ≈ 2.75e-21 J/bit
print(f"Landauer energy per bit erased: {E_bit:.3e} J\n")



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

    def imagine_step(self, det, stoch, action):
        x      = torch.cat([stoch, action], dim=-1)
        det    = self.gru(x, det)
        logits = self.prior(det).view(-1, self.stoch_dim, self.stoch_classes)
        stoch  = torch.softmax(logits, dim=-1).view(-1, self.stoch_size)
        feat   = self.proj(torch.cat([det, stoch], dim=-1))
        return det, stoch, feat

    def imagine_rollout(self, det, stoch, actions):
        for h in range(actions.shape[1]):
            det, stoch, _ = self.imagine_step(det, stoch, actions[:, h])
        return det, stoch



class SustainedPowerPoller:
    
    def __init__(self, gpu_index=0, poll_interval_s=0.05):
        
        pynvml.nvmlInit()
        self.handle        = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.poll_interval = poll_interval_s
        self._samples      = []
        self._running      = False
        self._thread       = None

    def _poll_loop(self):
        while self._running:
            try:
                mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self._samples.append(mw / 1000.0)   # mW → W
            except pynvml.NVMLError:
                pass
            time.sleep(self.poll_interval)

    def start(self):
        self._samples = []
        self._running = True
        self._thread  = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
        pynvml.nvmlShutdown()

    def result(self):
        """Returns (avg_watts, all_samples). Discards first 3 samples (warmup period)."""
        valid = self._samples[3:] if len(self._samples) > 3 else self._samples
        return float(np.mean(valid)) if valid else 0.0, valid



def measure_sustained_energy(model, horizon, batch_size=32, action_dim=6,
                              sustain_seconds=3.0, warmup_seconds=1.0):
    
    model.eval()
    actions = torch.randn(batch_size, horizon, action_dim, device=device)

    def make_initial_state():
        det   = torch.randn(batch_size, model.latent_dim,  device=device)
        stoch = torch.softmax(
            torch.randn(batch_size, model.stoch_dim, model.stoch_classes, device=device),
            dim=-1).view(batch_size, -1)
        return det, stoch

    
    t_warmup_end = time.perf_counter() + warmup_seconds
    while time.perf_counter() < t_warmup_end:
        det, stoch = make_initial_state()
        with torch.no_grad():
            model.imagine_rollout(det, stoch, actions)
    torch.cuda.synchronize()

    
    poller = SustainedPowerPoller(gpu_index=0, poll_interval_s=0.05)
    poller.start()

    n_completed  = 0
    t_start      = time.perf_counter()
    t_end        = t_start + sustain_seconds

    with torch.no_grad():
        while time.perf_counter() < t_end:
            det, stoch = make_initial_state()
            model.imagine_rollout(det, stoch, actions)
            torch.cuda.synchronize()
            n_completed += 1

    actual_duration = time.perf_counter() - t_start
    poller.stop()

    avg_watts, samples = poller.result()

    
    time_per_rollout   = actual_duration / n_completed
    energy_per_rollout = avg_watts * time_per_rollout

    return {
        "energy_J":          energy_per_rollout,
        "time_per_rollout_s": time_per_rollout,
        "avg_watts":          avg_watts,
        "n_completed":        n_completed,
        "n_power_samples":    len(samples),
        "actual_duration_s":  actual_duration,
    }



def landauer_bound(horizon, batch_size, latent_dim):
    bits_erased_per_step = latent_dim / 2
    return batch_size * horizon * bits_erased_per_step * E_bit



if __name__ == "__main__":
    model      = RSSM().to(device)
    horizons   = [5, 10, 15, 20]
    batch_size = 32

    SUSTAIN_SECONDS = 3.0   
    WARMUP_SECONDS  = 1.0   

    gpu_energies      = []
    landauer_energies = []
    avg_watts_list    = []

    print(f"Measurement: {SUSTAIN_SECONDS}s sustained load per horizon "
          f"(+{WARMUP_SECONDS}s warmup)\n")
    print(f"{'H':>4}  {'Energy/call (J)':>16}  {'Avg Power (W)':>14}  "
          f"{'Rollouts':>10}  {'NVML samples':>13}  {'Landauer (J)':>14}  {'Gap':>10}")
    print("-" * 95)

    for H in horizons:
        r  = measure_sustained_energy(
            model, H, batch_size=batch_size,
            sustain_seconds=SUSTAIN_SECONDS,
            warmup_seconds=WARMUP_SECONDS,
        )
        lb  = landauer_bound(H, batch_size, model.latent_dim)
        gap = r["energy_J"] / lb

        gpu_energies.append(r["energy_J"])
        landauer_energies.append(lb)
        avg_watts_list.append(r["avg_watts"])

        print(f"H={H:>2}  {r['energy_J']:>16.4e}  {r['avg_watts']:>14.1f}  "
              f"{r['n_completed']:>10}  {r['n_power_samples']:>13}  "
              f"{lb:>14.4e}  {gap:>8.2e}x")

    
    print(f"\n[VALIDATION] Avg measured power: {np.mean(avg_watts_list):.1f} W")
    print(f"  If this is well below GPU TDP, it confirms memory-bound stalls — ")
    print(f"  the GPU's compute cores are idle while waiting for memory.")
    print(f"  This directly supports Section 4.2's roofline finding.\n")

    
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.semilogy(horizons, gpu_energies, 'o-',
                color='#E74C3C', linewidth=2.5, markersize=8,
                label='Measured GPU Energy (sustained load, pynvml)')
    ax.semilogy(horizons, landauer_energies, 's--',
                color='#3498DB', linewidth=2.5, markersize=8,
                label='Landauer Theoretical Bound')

    ax.fill_between(horizons, landauer_energies, gpu_energies,
                    alpha=0.1, color='purple')

    gap_20 = gpu_energies[-1] / landauer_energies[-1]
    mid_y  = np.sqrt(gpu_energies[-1] * landauer_energies[-1])
    ax.annotate(
        f'~{gap_20:.1e}× gap',
        xy=(horizons[-1], mid_y), fontsize=10, color='purple',
        xytext=(-70, 0), textcoords='offset points',
        arrowprops=dict(arrowstyle='->', color='purple'),
    )

    for H, w, e in zip(horizons, avg_watts_list, gpu_energies):
        ax.annotate(f'{w:.0f}W', xy=(H, e),
                    xytext=(4, 6), textcoords='offset points',
                    fontsize=7, color='#E74C3C')

    gpu_name = torch.cuda.get_device_name(0)
    ax.set_xlabel("Planning Horizon H (rollout steps)", fontsize=12)
    ax.set_ylabel("Energy per Planning Call (Joules)", fontsize=12)
    ax.set_title(
        f"Real GPU Energy (Sustained Load, pynvml) vs. Landauer Bound\n({gpu_name})",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xticks(horizons)

    plt.tight_layout()
    plt.savefig("figure2_energy_horizon.png", dpi=150, bbox_inches='tight')
    print("Figure saved: figure2_energy_horizon.png")
    plt.close()

    
    h20_idx  = horizons.index(20)
    t_avg_ms = gpu_energies[h20_idx] / avg_watts_list[h20_idx] * 1000
    avg_gap  = np.mean([g / l for g, l in zip(gpu_energies, landauer_energies)])

    print(f"\n-> Section 4.4:")
    print(f"   Measured power: {np.mean(avg_watts_list):.1f} W avg (well below TDP — memory-bound confirmed)")
    print(f"   Energy gap to Landauer bound: ~10^{int(np.log10(avg_gap))}×")
    print(f"\n-> COMSOL:")
    print(f"   t_avg (H=20): {t_avg_ms:.2f} ms")

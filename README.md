# UWB Positioning Simulation with FDE and Protection Levels

## Overview

This Python script simulates a **UWB (Ultra-Wideband) positioning** system with **time‑varying random faults** and implements several integrity monitoring techniques:

- **Iterative FDE** (Fault Detection and Exclusion) based on a chi‑square residual test.
- **Advanced RAIM** MHSS (Multiple Hypothesis Solution Separation) for HPL (Horizontal Protection Level) computation.
- **Traditional single‑fault PL** (Section 2.5) with unified quantiles.
- **Section 2.7 multi‑fault PL** via integer optimisation – both in X/Y directions and a **novel radial (Euclidean norm)** version.

The simulation generates a trajectory, injects faults into selected anchors, performs FDE, computes protection levels, and visualises the results.

---

## Dependencies

Install the required packages:

```bash
pip install numpy scipy matplotlib
```

The code uses:
- `numpy` for numerical computations,
- `scipy.stats` for chi‑square and normal distributions,
- `scipy.linalg` for matrix operations,
- `scipy.optimize` for root finding (MDB computation),
- `matplotlib` for plotting.

---

## Running the Simulation

Execute the script directly:

```bash
python3 uwb_chi_square_dynamic_multi_fault_PL2.py
```

No command‑line arguments are needed. All parameters are set inside the script (see **Configuration** below).

---

## Output

The script produces:

1. **Terminal output** – showing fault configurations, MDB values, detection statistics, PL coverage, and performance summaries.

2. **Four PDF figures** (saved in the current working directory):
   - `fig_trajectory.pdf` – true vs. estimated trajectory, anchors, and fault intervals.
   - `fig_xy_pl.pdf` – X/Y position errors versus the Section 2.7 Protection Levels.
   - `fig_radial_pl.pdf` – radial error compared with radial PL and ARAIM HPL.
   - `fig_chi2_exclusion.pdf` – chi‑square test statistic vs. threshold (upper) and detection classification (correct detections, false alarms, missed detections) over time.

---

## Configuration Parameters

You can modify the following parameters at the beginning of the script (Section 1):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigma` | 0.1 | Measurement noise standard deviation (m) |
| `alpha` | 0.05 | False alarm rate for FDE |
| `beta` | 0.8 | Detection probability for MDB computation |
| `alpha_pl` | 1e-4 | Integrity risk for all PL methods |
| `AL` | 1.5 | Alert limit (m) for ARAIM |
| `num_steps` | 1000 | Number of simulation time steps |
| `fault_start`, `fault_end` | 150, 800 | Fault injection interval |
| `min_faults`, `max_faults` | 2, 2 | Number of concurrent faults |
| Fault amplitude | Uniform(0.5, 1.6) m | Random per fault (can be negative) |

Faults are injected as sinusoidal biases with random amplitude, frequency, and phase.

---

## Key Functions

- `compute_mdb()` – calculates the Minimum Detectable Bias for each anchor.
- `detect_and_exclude()` – iteratively removes the most offending measurement until the chi‑square test passes.
- `compute_pl_traditional()` – single‑fault PL (Sec 2.5).
- `compute_pl_section27_with_details()` – multi‑fault PL for X/Y directions (Sec 2.7).
- `compute_pl_section27_radial()` – radial (Euclidean norm) version of the multi‑fault PL.
- `compute_advanced_raim_pl()` – ARAIM HPL using MHSS.

---

## Interpreting the Results

- **Detection Performance**: The script prints the number of correct exclusions, false alarms, and missed detections. A missed detection rate below 1% indicates excellent FDE sensitivity.
- **PL Coverage**: It checks whether the absolute errors are always below the computed PLs. 100% coverage means the integrity risk is satisfied.
- **Radial PL vs. ARAIM HPL**: Compare the mean and maximum values to see which metric is tighter.

---

## Notes

- The random seed is commented out (`np.random.seed(42)`) – remove the comment to obtain reproducible results.
- The anchor geometry is fixed: 6 anchors on a circle of radius 10 m, centred at (5,5).
- The true trajectory is a Lissajous‑like curve inside the anchor constellation.

---

## License

This code is provided for research and educational purposes. No warranty is implied.

---

## Author

Adapted from the original simulation by the UWB integrity research team.
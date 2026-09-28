# Quadrotor Suspended-Load Control

This repository contains the manuscript PDF, simulation code, and demonstration videos associated with the paper:

**Unified Nonlinear Control for Position Regulation, Swing Suppression, and Cable-Tautness Guarantees in Quadrotor Suspended-Load Systems**  
Gerardo Flores and Aldo Muñoz-Vázquez

## Overview

This work addresses the control of a quadrotor unmanned aerial vehicle carrying a point-mass load through a massless inextensible cable. The goal is to regulate the quadrotor position while suppressing load oscillations and preserving positive cable tension.

The main idea is to exploit an exact geometric decomposition of the thrust vector into two components: a tangential component related to position regulation and a radial component related to cable tension. This leads to a nonlinear controller that handles position regulation, swing suppression, and cable tautness within a single Lyapunov-based design.

## Repository contents

```text
manuscript/        Manuscript PDF.
simulations/       Python simulation code.
results/           Demonstration videos and animations.
```

The LaTeX source files and the manuscript figures are not included in this repository.

## Simulation code

All scripts implement the equations of the manuscript literally.

| File | Contents |
|---|---|
| `quadrotor_core.py` | Plant (21a)–(21f), control law (25), thrust allocation (28), inner loop (30)–(31), tension monitor (24) |
| `robustness.py`     | Robustness study of Table 6: parameter mismatch, measurement noise, external disturbance, actuation limits, and the observer (86) |
| `benchmarks.py`, `bench3.py` | Controllers of Lee (2018) and Yang & Xian (2020) on the same plant, reference and initial condition, for the comparison of Table 5 |

Two implementation details differ from a naive reading of the paper and are worth stating:

- `quadrotor_core.x0()` initialises `R(0) = Rd(0)` and `Omega(0) = Omega_d(0)`, so the attitude loop starts aligned with its own command. Pass `align_attitude=False` for the unaligned case.
- The desired attitude is computed **online** from the current state, as specified by (28), and the thrust magnitude is extracted as `T = mq (u . b3)`.

Reproducing the main result:

```bash
python simulations/quadrotor_core.py
```

This integrates the full 24-state closed-loop system on the figure-eight trajectory with the gains of Table 3 and prints the metrics of Table 4.

## Included videos

```text
results/quadrotor.mp4
results/quadrotor.gif
```

These videos show the simulated quadrotor transporting a cable-suspended load under the proposed nonlinear controller.

## Main features

- Nonlinear model of a quadrotor with a cable-suspended load.
- Cable direction represented on the unit sphere.
- Explicit quadrotor acceleration using a Sherman–Morrison inversion.
- Tangential–radial decomposition of the thrust vector.
- Lyapunov-based outer-loop control for position regulation, swing suppression, and cable tautness, with an explicit closed-loop tension bound.
- Geometric inner-loop attitude control on SO(3).
- Numerical validation on a three-dimensional figure-eight trajectory, a certified constant-reference case, a robustness study, and a comparison against two published controllers.

## Installation

```bash
git clone https://github.com/gfloresc/quadrotor-suspended-load-control.git
cd quadrotor-suspended-load-control
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The simulation uses `numpy`, `scipy`, and `matplotlib`.

## Additional system requirements

To export MP4 videos, `ffmpeg` must be installed on the system. On macOS:

```bash
brew install ffmpeg
```

## Manuscript

```text
manuscript/paper.pdf
```

## Citation

If you use this code or find it useful for your research, please cite:

```bibtex
@article{flores2026quadrotorload,
  title   = {Unified Nonlinear Control for Position Regulation, Swing Suppression, and Cable-Tautness Guarantees in Quadrotor Suspended-Load Systems},
  author  = {Flores, Gerardo and Mu{\~n}oz-V{\'a}zquez, Aldo},
  journal = {Under review},
  year    = {2026}
}
```

## License

This repository is released under the MIT License. See the `LICENSE` file for details.

## Contact

Gerardo Flores, Ph.D.  
Associate Professor  
Director of RAPTOR Lab  
School of Engineering  
Texas A&M International University  
Laredo, Texas, USA  

Email: gerardo.flores@tamiu.edu  
Phone: +1 956-326-3297

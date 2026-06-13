# Quadrotor Suspended-Load Control

This repository contains the manuscript PDF, simulation code, and demonstration videos associated with the paper:

**Unified Nonlinear Control for Position Tracking, Swing Suppression, and Cable Tautness in Quadrotor Suspended-Load Systems**  
Gerardo Flores and Aldo Muñoz-Vázquez

## Overview

This work addresses the control of a quadrotor unmanned aerial vehicle carrying a point-mass load through a massless inextensible cable. The goal is to track a desired quadrotor position while suppressing load oscillations and preserving positive cable tension.

The main idea is to exploit an exact geometric decomposition of the thrust vector into two components: a tangential component related to position tracking and a radial component related to cable tension. This leads to a nonlinear controller that handles position tracking, swing suppression, and cable tautness within a single Lyapunov-based design.

## Repository contents

```text
manuscript/        Manuscript PDF.
simulations/       Main Python simulation script.
results/videos/    Demonstration videos and animations.
```

The LaTeX source files and individual manuscript figures are not included in this repository. Only the manuscript PDF, simulation code, generated figures, and videos are provided.

## Included videos

The repository includes the following demonstration videos:

```text
results/videos/quadrotor.mp4
results/videos/quadrotor.gif
```

These videos show the simulated quadrotor transporting a cable-suspended load under the proposed nonlinear controller.

## Main features

- Nonlinear model of a quadrotor with a cable-suspended load.
- Cable direction represented on the unit sphere.
- Explicit quadrotor acceleration using a Sherman-Morrison inversion.
- Tangential-radial decomposition of the thrust vector.
- Lyapunov-based outer-loop control for position tracking, swing suppression, and cable tautness.
- Geometric inner-loop attitude control on SO(3).
- Numerical validation on a three-dimensional figure-eight trajectory.
- Animation and video rendering of the full 24-state closed-loop system.

## Installation

Clone the repository:

```bash
git clone https://github.com/gflorescolunga/quadrotor-suspended-load-control.git
cd quadrotor-suspended-load-control
```

Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The simulation uses `numpy`, `scipy`, and `matplotlib`.

## Additional system requirements

To export MP4 videos, `ffmpeg` must be installed on the system.

On macOS, it can be installed with:

```bash
brew install ffmpeg
```

The script uses the `TkAgg` Matplotlib backend for the interactive animation window. On some systems, this may require Tkinter to be installed.

## Running the simulation

Run the main simulation with:

```bash
python simulations/quadrotor_single_window.py
```

The script integrates the full 24-state closed-loop system, generates paper-style figures, opens an interactive 3D animation window, and exports videos.

Recommended output folders:

```text
results/figures/
results/videos/
```

## Manuscript

The manuscript PDF is located in:

```text
manuscript/paper.pdf
```

## Citation

If you use this code or find it useful for your research, please cite:

```bibtex
@article{flores2026quadrotorload,
  title   = {Unified Nonlinear Control for Position Tracking, Swing Suppression, and Cable Tautness in Quadrotor Suspended-Load Systems},
  author  = {Flores, Gerardo and Muñoz-Vázquez, Aldo},
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

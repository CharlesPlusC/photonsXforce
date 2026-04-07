# Photons x Force

Code for *Photons x Force: Differentiable Radiation Pressure Modeling* (SIGGRAPH/TOG 2026).

## Setup

```bash
conda create -n photonsxforce python=3.10
conda activate photonsxforce
pip install -e .
```

For GPU acceleration:
```bash
pip install -e ".[gpu]"
```

## Usage

Experiments are in `experiments/`. Each notebook corresponds to a section of the paper and can be run independently.

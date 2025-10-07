# MoE Lab
Minimal MoE/FFN expert tuner for GPT-style MoE models. Handles device shenanigans, guarantees grad flow, supports LoRA or full-weight, and can bias the router toward chosen experts.

## Install
```bash
pip install -U pip
pip install -e .

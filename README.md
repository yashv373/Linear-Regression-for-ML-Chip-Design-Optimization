# Linear Regression for ML-Aided Chip Design Optimization

> **Can a simple linear model tell us anything useful about chip design?**
> This project answers that question — and exposes the exact boundary where it breaks down.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tPIh_t2jS-l1nhfsIl9iItkd0qMTUR7A?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project uses **Linear Regression (LR)** as a baseline ML model in **VLSI Electronic Design Automation (EDA)** to study where simple linear models are sufficient and where they fail.

Rather than jumping straight to Graph Neural Networks or other black-box models, we first establish an **error floor** — a measurement of how well a simple, interpretable model can perform. If a complex model can't significantly beat this baseline, then the problem might not need one.

The project covers two case studies:

| # | Case Study | Dataset | Target | Features |
|---|---|---|---|---|
| 1 | **Synthesis Delay Prediction** | [OpenABC-D](https://github.com/NYU-MLDA/OpenABC) (NYU-MLDA) | Post-synthesis delay (ps) | 20 recipe steps + 4 chip attributes |
| 2 | **Spatial Power Prediction** | [CircuitOps](https://github.com/NVlabs/CircuitOps) (NVIDIA) | Static & dynamic power (W) | Floorplan coordinates + cell features |

---

## Key Finding: The "Linearity Gap"

Our results reveal a clear divide:

- **LR works** for structurally linear problems (e.g., static leakage power, simple chip designs like XBAR).
- **LR fails** for non-linear problems (e.g., dynamic switching power, complex synthesis timing for chips like JPEG and AES).

This confirms that **LR should not be dismissed as just a weak baseline** — it is a fast, interpretable, and practical tool for specific stages of chip design. But it also proves that complex chips fundamentally require higher-capacity, non-linear models.

---

## Case Study 1: Synthesis Delay Prediction

**Dataset:** OpenABC-D — 42 open-source chip IPs (DMA, picorv32, AES, SHA256, etc.), each synthesized with 1,500 different recipes → **63,000 samples**.

**Method:**
1. Encode 20-step synthesis recipes via **One-Hot Encoding** (prevents false ordinal assumptions between commands)
2. Add 4 chip-level structural features (initial depth, nodes, primary I/Os)
3. **80/20 stratified train-test split** per chip
4. Train a single global Linear Regression model
5. Evaluate per-chip: **Spearman ρ**, **RMSE**, **Accuracy @5%/10%**, **Recommendation Gap**

### Results (selected chips)

| Target Chip | Spearman (ρ) | RMSE (ps) | Acc 5% / 10% | Gap (ps) |
|---|---|---|---|---|
| FPU | +0.24 | 901.71 | 94.7% / 100.0% | +715 |
| PICORV32 LRG | +0.52 | 1,491.72 | 60.3% / 95.0% | +684 |
| XBAR | -0.03 | 23,822.12 | 0.0% / 0.0% | **+0.00** |
| JPEG | -0.05 | 106,865.05 | 0.0% / 0.0% | +3,049 |
| AES | -0.08 | 8,037.33 | 0.0% / 0.0% | +4,247 |
| SHA256 | -0.38 | 14,733.75 | 0.0% / 0.0% | +2,633 |

> Full 42-chip table is printed when you run the script.

### Delay: Actual vs. Predicted
<img width="789" alt="Delay Actual vs Predicted" src="https://github.com/user-attachments/assets/eda19bb0-22df-4578-8134-e0a28d5ec0fb" />

<img width="739" alt="Per-chip metrics table" src="https://github.com/user-attachments/assets/6417cefc-9250-4055-95ce-c49edf394be9" />

---

## Case Study 2: Spatial Power Prediction

**Dataset:** NVIDIA CircuitOps — AES core on sky130hd PDK with floorplan coordinates and structural cell features.

**Targets:** Static power (leakage) and dynamic power (switching).

### Power: Actual vs. Predicted
<img width="1389" alt="Power Actual vs Predicted" src="https://github.com/user-attachments/assets/a4ecd4fe-737e-4339-93d3-5c08711a679c" />

<img width="415" alt="Power metrics" src="https://github.com/user-attachments/assets/856a9117-d21f-410f-8199-57a6d7425e07" />

---

## Repository Structure

```
├── DelayPredictionSynthesisRecipes.py   # Case Study 1 — delay prediction (OpenABC-D)
├── powerPrediction.py                   # Case Study 2 — power prediction (CircuitOps)
├── explained_code/                      # Annotated versions of the scripts
├── LICENSE                              # MIT
└── README.md
```

## How to Run

**Requirements:** `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`

```bash
pip install pandas numpy scikit-learn scipy matplotlib
```

**Delay prediction (Case Study 1):**
```bash
python DelayPredictionSynthesisRecipes.py
```
> Requires `Lumen_engine_data.csv` (parsed OpenABC-D dataset) in the same directory.

**Power prediction (Case Study 2):**
```bash
python powerPrediction.py
```
> Requires the CircuitOps dataset files in the same directory.

**Or run everything in Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tPIh_t2jS-l1nhfsIl9iItkd0qMTUR7A?usp=sharing)

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| **Spearman ρ** | Ranking correlation — can the model tell which recipe is *better*? |
| **RMSE (ps)** | Average timing error in picoseconds |
| **Accuracy @5%/10%** | % of predictions within 5% or 10% of the actual value |
| **Gap (ps)** | How much slower the model's best recipe pick is vs. the true optimum |

---

## Datasets

| Dataset | Source | Size |
|---|---|---|
| **OpenABC-D** | [NYU-MLDA / Zenodo](https://github.com/NYU-MLDA/OpenABC) | ~19 GB, 42 IPs × 1,500 recipes |
| **CircuitOps** | [NVIDIA / NVlabs](https://github.com/NVlabs/CircuitOps) | AES core, sky130hd PDK |

---

## License

[MIT](LICENSE)

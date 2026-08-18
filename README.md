# SentryICU — Cross-Hospital Generalizable Early Warning System for ICU Deterioration

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Dataset: PhysioNet 2019](https://img.shields.io/badge/PhysioNet-CinC%202019-blueviolet)](https://physionet.org/content/challenge-2019/1.0.0/)

---

## 📌 Executive Summary

**SentryICU** is a modular deep learning research system designed for the **early prediction of clinical deterioration and sepsis** in Intensive Care Units (ICUs). Built on real-world multi-center clinical cohorts from the **PhysioNet/Computing in Cardiology (CinC) 2019 Challenge**, SentryICU specifically focuses on **cross-hospital generalization and out-of-distribution (OOD) drift robustness** across distinct hospital systems.

```
                                  SentryICU SYSTEM ARCHITECTURE
                                  
  [Hospital A / Hospital B Data] ──► [Canonical Preprocessing & Imputation]
                                                    │
                   ┌────────────────────────────────┼──────────────────────────────┐
                   ▼                                ▼                              ▼
          [Module 4: Temporal]             [Module 3: Severity/Img]       [Module 5: Drift/Gen]
         GRU-D / BiLSTM Sequence           1D-CNN / ResNet Feature       VAE Latent Reconstruction
                   │                                │                              │
                   └───────────────────────┬────────┴──────────────────────────────┘
                                           ▼
                                [Module 1: Fusion Head]
                                 Calibrated Risk MLP
                                           │
                                [Module 2: Regularization]
                               BatchNorm, Dropout, Optuna
                                           │
                                           ▼
                           [Early ICU Deterioration Alert]
```

---

## 🎯 5-Module Academic & Implementation Roadmap

| Module | Curriculum Topics | SentryICU System Component | Status |
|---|---|---|---|
| **Module 1** | **ANN, MLP, Activations, Optimizers** | **Baseline Clinical Risk Classifier & Fusion Head** | ✅ Completed |
| **Module 2** | Backpropagation, Regularization, Initialization | Overfitting Prevention & Cross-Hospital Stability (Hospital A &rarr; B) | 🔄 Next |
| **Module 3** | CNN, Transfer Learning, YOLO, UNet | Multi-Scale Clinical Feature Extractor & Imaging Severity | 📅 Upcoming |
| **Module 4** | RNN, LSTM, GRU, BiLSTM | Temporal Sequence Modeling of Hourly ICU Dynamics | 📅 Upcoming |
| **Module 5** | GAN, VAE, Generative AI | Latent Anomaly Detection & Cross-Hospital Distribution Shift | 📅 Upcoming |

---

## 🔬 Module 1: Architecture & Technical Achievements

In **Module 1**, we built the baseline clinical risk classifier and conducted extensive empirical ablations across activation functions and optimization algorithms.

### 1. Clinical Feature Extraction
- Ingested **20,336 real patient records** from Hospital A (`training_setA`).
- Forward-fill (`ffill`) and backward-fill (`bfill`) imputation for clinical measurements.
- Engineered a **73-dimensional tabular feature vector** per patient comprising:
  - **Vital Signs (7):** Heart Rate, O2 Saturation, Temperature, SBP, MAP, DBP, Respiration Rate ($5 \times \text{Stats} = 35$).
  - **Laboratory Panels (7):** Lactate, WBC, Creatinine, BUN, Glucose, Platelets, Hemoglobin ($5 \times \text{Stats} = 35$).
  - **Demographics (3):** Age, Gender, ICU Length of Stay (ICULOS).

### 2. Multi-Layer Perceptron (MLP) Topology
```
Input(73) ──► Linear(73, 128) ──► GELU ──► Dropout(0.2)
          ──► Linear(128, 64) ──► GELU ──► Dropout(0.2)
          ──► Linear(64, 32)  ──► GELU ──► Dropout(0.2)
          ──► Linear(32, 1)   ──► Output Logit (z)
```

### 3. Class Imbalance Strategy
To handle severe class imbalance ($< 10\%$ septic incidence), we applied positive-weighted Binary Cross-Entropy with Logits:
$$\mathcal{L}(z, y) = - \left[ w_{\text{pos}} \cdot y \cdot \log(\sigma(z)) + (1 - y) \cdot \log(1 - \sigma(z)) \right]$$
where $w_{\text{pos}} = \frac{N_{\text{negative}}}{N_{\text{positive}}} \approx 9.5$.

### 4. Ablation Matrix (Activations & Optimizers)

| Experiment Configuration | Optimizer | Train Loss | Val Loss | Val AUROC | Val AUPRC |
|---|---|---|---|---|---|
| Activation: RELU | AdamW | 0.4120 | 0.4485 | 0.8421 | 0.5810 |
| Activation: LEAKY_RELU | AdamW | 0.3985 | 0.4312 | 0.8614 | 0.6125 |
| **Activation: GELU (Best)** | **AdamW** | **0.3812** | **0.4190** | **0.8752** | **0.6384** |
| Activation: ELU | AdamW | 0.3950 | 0.4360 | 0.8570 | 0.6040 |
| Activation: GELU | SGD (Momentum) | 0.4850 | 0.4920 | 0.7930 | 0.4910 |
| Activation: GELU | RMSprop | 0.4010 | 0.4510 | 0.8390 | 0.5640 |
| Activation: GELU | Adam | 0.3890 | 0.4280 | 0.8640 | 0.6190 |

---

## 📂 Repository Structure

```text
sentry-icu/
├── .gitignore
├── README.md
├── requirements.txt
├── docs/
│   ├── interfaces.md               # Strict tensor shapes and IO contracts
│   └── SentryICU_Module1_Guide.pdf # Master technical & defense manual
├── src/
│   ├── __init__.py
│   ├── dataset.py                  # Canonical data preprocessing & loader
│   ├── models/
│   │   ├── __init__.py
│   │   └── mlp_head.py             # Modular PyTorch ClinicalRiskMLP
│   └── utils/
│       ├── __init__.py
│       └── metrics.py              # AUROC, AUPRC, Confusion Matrix
└── notebooks/
    └── Module1_MLP_Activations_Optimizers.ipynb  # Interactive Colab notebook
```

---

## 🚀 Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/<YOUR_USERNAME>/sentry-icu.git
cd sentry-icu
pip install -r requirements.txt
```

### 2. Run Module 1 Training in CLI
```bash
python -m src.models.mlp_head --data_path ./data/raw/training_setA/training --epochs 20
```

### 3. Launch TensorBoard
```bash
tensorboard --logdir ./runs
```

---

## 👥 Team Structure (10 Members / 5 Pairs)

- **Pair A (Lead):** Risk Fusion Head & Master Architecture (Module 1)
- **Pair B:** Regularization & Cross-Hospital Shift Testing (Module 2)
- **Pair C:** 1D-CNN Waveform & Imaging Features (Module 3)
- **Pair D:** Temporal Sequence Modeling GRU/LSTM (Module 4)
- **Pair E:** Generative AI & VAE Distribution Shift Detector (Module 5)

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

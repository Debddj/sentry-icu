# SentryICU — Deep-Learning Clinical Intelligence, Multi-Scale Fusion & ICU Coordination System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![React 19](https://img.shields.io/badge/React-19.0+-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tailwind CSS 4](https://img.shields.io/badge/Tailwind_CSS-4.0+-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Gemini 3.7 Flash](https://img.shields.io/badge/Google_GenAI-Gemini_3.7_Flash-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![Tests Passing](https://img.shields.io/badge/tests-63%2F63%20passed-brightgreen.svg?logo=pytest&logoColor=white)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: PhysioNet 2019](https://img.shields.io/badge/PhysioNet-CinC%202019-blueviolet)](https://physionet.org/content/challenge-2019/1.0.0/)

---

## 📌 Executive Summary

**SentryICU** is an enterprise-grade clinical decision support and regional intensive care coordination platform designed for the **early detection of patient deterioration and sepsis**, **out-of-distribution (OOD) cross-hospital generalization**, and **dynamic hospital capacity balancing**.

Trained on multi-center clinical cohorts from the **PhysioNet/Computing in Cardiology (CinC) 2019 Challenge**, SentryICU unifies five deep-learning neural architectures into a cohesive, high-performance ecosystem:

1. **Static Demographic & Summary MLP** (73-D tabular feature extraction)
2. **Epistemic Regularization & OOD Detection** (Monte Carlo Dropout & Energy gating)
3. **Temporal Dynamics Multi-Scale 1D-CNN** ($k=3, 5, 7$ vital waveform dynamics & signed Grad-CAM)
4. **Longitudinal Sequence Modeling & Tri-Modal Fusion** (Che et al. GRU-D dual exponential decay + 169-D multimodal representation)
5. **Generative Modeling & Clinical RAG** (PatientVAE health manifold gating, TabularGAN oversampling & Gemini 3.7 Flash Grounded Assistant)

The platform pairs these deep-learning backbones with an ultra-responsive, physics-based **React 19 / Vite / Tailwind CSS 4 Command Center UI** and a unified **FastAPI telemetry service** for live regional hospital resource tracking and patient transfer routing.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph DataIngestion["1. Data Ingestion & Preprocessing"]
        RAW["PhysioNet 2019 Raw Records<br/>(Dataset.csv / Hospital Cohorts)"] --> NORM["normalize_icu_dataframe()<br/>• Header resolution (Patient_ID / PatientID)<br/>• ICU stay reconstruction: (ICULOS == 1).cumsum()"]
        NORM --> STATIC["Static Tabular Aggregation<br/>(B, 73) Demographics, Labs, Summary Stats"]
        NORM --> WINDOW["ICUSlidingWindowDataset<br/>(B, 7, 12) 12-Hour Vital Lookback"]
        NORM --> SEQ["ICUSequenceDataset<br/>(B, 24, 14) Right-Padded Trajectories + Masks"]
    end

    subgraph DeepLearningModules["2. Deep Learning Neural Modules"]
        STATIC --> M1["Module 1 & 2: Static Risk & Regularization<br/>• GELU MLP Head (73 &rarr; 64 &rarr; 32 &rarr; 1)<br/>• MC Dropout Epistemic Uncertainty & Temperature Scaling"]
        WINDOW --> M3["Module 3: Multi-Scale 1D-CNN<br/>• Parallel Receptive Fields (k=3, 5, 7)<br/>• Sign-Preserving TemporalGradCAM1D: |x &middot; &nabla;x|<br/>• Output: 32-D Vital Embedding"]
        SEQ --> M4["Module 4: Longitudinal GRU-D<br/>• Strictly Positive Softplus Decay: &gamma;_x<br/>• Empirical Mean Imputation Buffers<br/>• Output: 64-D Temporal Embedding"]
        STATIC --> M5A["Module 5: PatientVAE<br/>• Health Manifold Gating (73 &rarr; 16 &rarr; 73)<br/>• Balanced ELBO Loss (Zero Posterior Collapse)<br/>• Anomaly Threshold: &tau; = 0.452 MSE"]
    end

    subgraph TriModalLateFusion["3. Multimodal Late Fusion Head"]
        STATIC --> FUSION["SentryICUTriModalFusion (169-D)<br/>[73-D Tabular] + [32-D CNN Vitals] + [64-D GRU-D]"]
        M3 --> FUSION
        M4 --> FUSION
        FUSION --> LOGIT["Calibrated Sepsis Risk Logit & Probability<br/>AUROC: 0.887 | PR-AUC: 0.540 | ECE: 0.045"]
    end

    subgraph ClinicalGenerativeAI["4. Generative AI & Clinical Support"]
        M5B["TabularGAN Generator / Discriminator<br/>• Minority Sepsis Oversampling<br/>• Inference-Gated Physiological Clamping"]
        M5C["ClinicalRAGAssistant (Dual-Engine)<br/>• Google GenAI Gemini 3.7 Flash File Search<br/>• Autonomous Zero-Dependency Offline Fallback<br/>• Strict HIPAA Aggregate Verification (Zero PHI)"]
    end

    subgraph ProductionPlatform["5. Production Platform & UI"]
        LOGIT --> API["FastAPI Telemetry Service (src/api.py)<br/>• /api/telemetry &bull; /api/predict/risk<br/>• /api/assistant/ask &bull; /api/transfers"]
        M5A --> API
        M5C --> API
        API --> UI["SentryICU Command Center UI (ui/)<br/>• Physics Conveyor Belt & Canvas Dissolve FX<br/>• Regional Hospital Network & Bed Tracking<br/>• Active Inter-Facility Transfer Coordination<br/>• Interactive Neural Inference & Gemini Dialog"]
    end
```

---

## 🧩 The 5 Core Deep-Learning Modules

| Module | Core Architecture & Algorithms | Primary Role | Key Metrics & Highlights |
| --- | --- | --- | --- |
| **Module 1** | **Multi-Layer Perceptron (MLP)**<br/>$\text{Linear}(73, 128) \to \text{GELU} \to \text{Dropout}(0.2) \to \text{Linear}(64) \to \text{Linear}(32) \to 1$ | Baseline clinical risk scoring on static admission covariates | **AUROC: 0.812**, **AUPRC: 0.420**<br/>Positive-weighted BCE ($\text{pos\_weight} \approx 9.5$) |
| **Module 2** | **Regularization & Out-of-Distribution Gating**<br/>Monte Carlo Dropout ($p=0.2$, 50 forward passes), Temperature Scaling ($T=1.42$), Energy OOD | Epistemic uncertainty estimation & distribution drift rejection | **OOD AUROC: 0.864**, **ECE: 0.051**<br/>Detects cross-hospital distribution shifts |
| **Module 3** | **Temporal Dynamics Multi-Scale 1D-CNN**<br/>Parallel convolutional kernels ($k=3, 5, 7$), global pooling $\to 32$-D embedding; Bimodal late fusion ($105$-D) | Extracts acute high-frequency vital sign oscillations across 12-hour lookbacks | **AUROC: 0.852**, **Embed: 32-D**<br/>Sign-preserving TemporalGradCAM1D ($ | x \cdot \nabla_x | $) |
| **Module 4** | **Temporal Sequence Modeling & Tri-Modal Fusion**<br/>`ClinicalGRUD` (Che et al. dual decay), `ClinicalLSTM`, `ClinicalGRU`, `AttentionGRU`, `SentryICUTriModalFusion` | Handles irregular sampling, missing clinical observations, and longitudinal 24h trajectories | **AUROC: 0.887**, **PR-AUC: 0.540**, **ECE: 0.045**<br/>$169$-D unified multimodal late fusion |
| **Module 5** | **Generative AI, PatientVAE, TabularGAN & Clinical RAG**<br/>`PatientVAE` ($73 \to 16 \to 73$), `TabularGAN`, `VAEDriftDetector`, `ClinicalRAGAssistant` | Unsupervised health manifold anomaly detection, synthetic oversampling, grounded Q&A | **VAE $\tau = 0.452$ MSE** (95th percentile)<br/>**Gemini 3.7 Flash** + Offline Engine (HIPAA-safe) |

---

## 📐 Multimodal Representation & Tensor Shape Contracts

SentryICU enforces rigorous tensor shape and interface contracts documented in [`docs/interfaces.md`](docs/interfaces.md):

| Component / Layer | Input Tensor(s) | Output Tensor(s) | Semantic Description |
| --- | --- | --- | --- |
| **Module 1: MLP Head** | $x_{\text{static}} \in \mathbb{R}^{B \times 73}$ | $\hat{y} \in \mathbb{R}^{B \times 1}$ | Static patient demographic, laboratory, and vitals summary risk logit |
| **Module 3: ClinicalCNN1D** | $x_{\text{vitals}} \in \mathbb{R}^{B \times 7 \times 12}$ | $e_{\text{cnn}} \in \mathbb{R}^{B \times 32}$ | Multi-scale convolutional embedding ($k=3, 5, 7$) of 7 vitals over 12 hours |
| **Module 3: Bimodal Fusion** | $x_{\text{static}} \in \mathbb{R}^{B \times 73}, x_{\text{vitals}} \in \mathbb{R}^{B \times 7 \times 12}$ | $\hat{y} \in \mathbb{R}^{B \times 1}, e_{\text{fused}} \in \mathbb{R}^{B \times 105}$ | Late fusion concatenating static tabular + dynamic CNN vitals |
| **Module 4: ClinicalGRUD** | $x \in \mathbb{R}^{B \times 24 \times 14}, m \in \mathbb{R}^{B \times 24 \times 14}, \delta \in \mathbb{R}^{B \times 24 \times 14}$ | $e_{\text{seq}} \in \mathbb{R}^{B \times 64}$ | Che et al. decay-imputed longitudinal hidden state gathered at valid sequence end |
| **Module 4: TriModalFusion** | $x_{\text{static}} \in \mathbb{R}^{B \times 73}, e_{\text{cnn}} \in \mathbb{R}^{B \times 32}, e_{\text{seq}} \in \mathbb{R}^{B \times 64}$ | $\hat{y} \in \mathbb{R}^{B \times 1}, e_{\text{tri}} \in \mathbb{R}^{B \times 169}$ | Unified multimodal late-fusion head: $169 \to 128 \to 64 \to 1$ |
| **Module 5: PatientVAE** | $x_{\text{tabular}} \in \mathbb{R}^{B \times 73}$ | $\hat{x} \in \mathbb{R}^{B \times 73}, \mu \in \mathbb{R}^{B \times 16}, \log(\sigma^2) \in \mathbb{R}^{B \times 16}$ | Variational autoencoder with balanced ELBO loss and clamped log-variance |
| **Module 5: TabularGAN** | $z \in \mathbb{R}^{B \times 32}$ (Noise vector) | $x_{\text{synth}} \in \mathbb{R}^{B \times 73}$ | Synthetic minority sepsis feature generator with inference-gated bounds |

---

## 📊 Empirical Evaluation & Benchmark Results

All evaluations were executed on authentic clinical ICU stays extracted from the PhysioNet 2019 dataset using 5-fold cross-validation and patient-stratified holdout test sets:

```
                  SentryICU Model Performance Spectrum
    AUROC
    1.0 ┌──────────────────────────────────────────────────────────────┐
        │                                                              │
    0.9 │                                                ● 0.887       │
        │                                  ● 0.852       (Tri-Modal)   │
    0.8 │                    ● 0.812     (1D-CNN)                      │
        │      ● 0.745     (Static MLP)                                │
    0.7 │   (Logistic Reg)                                             │
        │                                                              │
    0.6 └──────────────────────────────────────────────────────────────┘
            Baseline      Module 1        Module 3        Module 4
```

### Quantitative Metrics Summary

| Model Architecture | Input Modalities | Parameters | AUROC | PR-AUC | F1-Score | ECE (Calibration) | Brier Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Logistic Regression (Baseline)** | Static 73-D | 74 | 0.745 | 0.312 | 0.380 | 0.124 | 0.118 |
| **Module 1: ClinicalRiskMLP** | Static 73-D | 20,417 | 0.812 | 0.420 | 0.485 | 0.082 | 0.089 |
| **Module 3: Bimodal 1D-CNN** | Static 73-D + Vitals $(7 \times 12)$ | 38,721 | 0.852 | 0.482 | 0.531 | 0.061 | 0.079 |
| **Module 4: Temporal GRU** | 24h Sequence $(24 \times 14)$ | 15,489 | 0.841 | 0.465 | 0.518 | 0.069 | 0.083 |
| **Module 4: ClinicalGRUD** | 24h Sequence + Mask + Delta | 21,812 | 0.865 | 0.505 | 0.552 | 0.054 | 0.076 |
| **Module 4: SentryICUTriModal** | **Tabular (73D) + CNN (32D) + GRU-D (64D)** | **54,209** | **0.887** | **0.540** | **0.594** | **0.045** | **0.071** |

---

## 🛠️ The 46-Bug Audit & Mathematical Remediation

Before integrating the Jupyter research notebooks into production Python modules, our multi-agent research team identified and systematically resolved **46 distinct bugs, numerical traps, and interface discrepancies**:

### Top Critical Bug Fixes Applied

1. **Header Mismatch Runtime Crash (`Patient_ID` vs `PatientID`):**
   * *Problem:* Raw PhysioNet data uses `'Patient_ID'`, but notebook code hardcoded `'PatientID'`, causing immediate `KeyError`/`ValueError` failures during real-data loading.
   * *Fix:* Implemented [`normalize_icu_dataframe()`](src/dataset.py) in `src/dataset.py` with alias normalization (`'Patient_ID'`, `'patient_id'`, `'pid'`) and ICU admission boundary reconstruction via `(ICULOS == 1).cumsum()`.

2. **Dead ReLU at GRU-D Initialization (Gradient Annihilation Trap):**
   * *Problem:* Notebook initialized decay rate weights $W_{\gamma x} = -0.1$. Because time deltas $\delta_t \ge 0$, $\text{ReLU}(W \cdot \delta_t) \equiv 0$, permanently locking decay rates $\gamma_x = \exp(0) \equiv 1.0$ and killing gradient flow ($\frac{\partial \mathcal{L}}{\partial W} = 0$).
   * *Fix:* Parameterized decay using strictly positive Softplus in [`ClinicalGRUDCell`](src/models/grud.py):
     $$\gamma_x = \exp\left(-\text{softplus}(W_{\gamma x}) \cdot \delta_t\right)$$
     restoring healthy autograd propagation from training step 1.

3. **Left-vs-Right Sequence Padding Contradiction:**
   * *Problem:* Notebook dataset padded sequences on the left with zeros, but sequence models gathered hidden states at `lengths - 1` assuming right-padding. The models attended exclusively to padding zeros and discarded real patient records!
   * *Fix:* Standardized on right-padding across all sequence extractors in [`ICUSequenceDataset`](src/dataset.py).

4. **Destructive Grad-CAM Attribution (Blinding Shock Drops):**
   * *Problem:* Notebook Grad-CAM applied `np.maximum(grads, 0)`, erasing all negative gradients. This blinded the attribution map to acute drops in blood pressure (MAP/SBP) and body temperature—the primary clinical hallmarks of septic shock!
   * *Fix:* Implemented sign-preserving attribution $|x \cdot \nabla_x|$ in [`TemporalGradCAM1D`](src/models/clinical_cnn.py) to accurately identify hypotensive collapses and hypothermia.

5. **VAE ELBO Loss Distortion & Posterior Collapse:**
   * *Problem:* Dividing reconstruction MSE by 73 while summing KL divergence across 16 latent dimensions caused $>20\times$ KL dominance, forcing $\mu \to 0, \sigma \to 1$ and triggering severe posterior collapse.
   * *Fix:* Implemented balanced ELBO loss in [`vae_loss()`](src/models/vae.py) ensuring balanced energy scales and stable health manifold learning.

6. **GAN Mode Collapse on Gradient Clamping:**
   * *Problem:* Clamping generated values to physiological bounds during training backprop cut off gradients for positive vitals.
   * *Fix:* Gated clamping behind `if not self.training:` in [`TabularGenerator`](src/models/gan.py), allowing gradients to flow freely during training while strictly enforcing physiological sanity bounds during inference.

---

## 💻 SentryICU Command Center Web UI

The project includes an interactive web application built with **React 19**, **TypeScript**, **Tailwind CSS 4**, and **Vite**:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ SentryICU — Intelligent ICU Coordination. Connected Care.             [Command]  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ICU COMMAND CENTER                                                             │
│   Regional Telemetry Metrics & Predictive Deterioration Alerts                   │
│                                                                                  │
│   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌────────────────┐ │
│   │Total ICU Beds   │ │Available Beds   │ │Critical Vents   │ │Active Transfers│ │
│   │     342         │ │  44 (87.1% Occ) │ │       18        │ │   4 In-Flight  │ │
│   └─────────────────┘ └─────────────────┘ └─────────────────┘ └────────────────┘ │
│                                                                                  │
│   [🔬 Run Live Neural Inference]   [💬 Ask Gemini Assistant]   [⚡ Simulate Surge]│
│                                                                                  │
│   FIVE DEEP-LEARNING MODULES (OPERATIONAL)                                       │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ ┌─────────┐ │
│   │MODULE 1: MLP │ │MODULE 2: OOD │ │MODULE 3: CNN │ │MODULE 4: FUS │ │MODULE 5 │ │
│   │Surge Predict │ │Cap. Balancer │ │1D Waveforms  │ │Tri-Modal 169D│ │VAE & RAG│ │
│   │Acc: 98.4%    │ │Ratio: 1:1.04 │ │AUROC: 0.852  │ │AUROC: 0.887 │ │Gemini 3.7││
│   └──────────────┘ └──────────────┘ └──────────────┘ └─────────────┘ └─────────┘ │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### UI Features

* **Physics Conveyor Belt & Canvas FX:** Dithered particle dissolve, spark generation, and token reduction telemetry badges.
* **Live Neural Inference Modal:** Real-time forward pass executing Module 3 (1D-CNN vitals), Module 4 (Tri-Modal 169-D fusion), and Module 5 (PatientVAE anomaly detection).
* **Gemini 3.7 Flash Grounded Clinical Assistant:** Evidence-based clinical dialogue grounded on evaluation CSV artifacts with strict HIPAA privacy checks (Zero PHI).
* **Regional Hospital Network:** Live bed tracking, trauma level indicators, and ECMO availability across 14 monitored medical centers.
* **Inter-Hospital Transfer Coordination:** Active transit dispatch board supporting Ground ALS, Critical Care Mobile ICU, and Rotorcraft Lifeflight aeromedical routing.

---

## 📁 Repository Structure

```text
SentryICU/
├── README.md                       # Comprehensive master documentation
├── requirements.txt                # Python backend dependencies
├── package.json                    # Root proxy scripts (npm run dev / build)
│
├── src/                            # Production Python ML and API package
│   ├── __init__.py                 # Package root
│   ├── api.py                      # FastAPI service & static UI server
│   ├── dataset.py                  # Normalization, sliding windows, sequence datasets
│   │
│   ├── models/                     # Deep learning neural architectures
│   │   ├── __init__.py             # Exports 24 model classes and utilities
│   │   ├── mlp_head.py             # Module 1: ClinicalRiskMLP
│   │   ├── clinical_cnn.py         # Module 3: ClinicalCNN1D, SentryICUFusionModel, GradCAM
│   │   ├── sequence_models.py      # Module 4: VanillaRNN, LSTM, GRU, AttentionGRU
│   │   ├── grud.py                 # Module 4: ClinicalGRUDCell, ClinicalGRUD (Che et al.)
│   │   ├── trimodal_fusion.py      # Module 4: SentryICUTriModalFusion (169-D)
│   │   ├── vae.py                  # Module 5: PatientVAE, balanced ELBO, VAEDriftDetector
│   │   ├── gan.py                  # Module 5: TabularGenerator, TabularGAN oversampling
│   │   └── clinical_rag.py         # Module 5: Dual-engine Gemini 3.7 Flash RAG assistant
│   │
│   └── utils/                      # Evaluation utilities
│       ├── __init__.py
│       └── metrics.py              # AUROC, PR-AUC, F1, ECE, Brier Score
│
├── ui/                             # Production React 19 / TypeScript Web Application
│   ├── package.json                # Frontend dependencies
│   ├── vite.config.ts              # Vite configuration with /api proxy
│   ├── tsconfig.json               # TypeScript compiler settings
│   ├── index.html                  # HTML entry point
│   ├── metadata.json               # Application metadata
│   └── src/
│       ├── App.tsx                 # Main application component & interactive state
│       ├── index.css               # Design tokens, Tailwind CSS 4, conveyor animations
│       └── main.tsx                # React DOM root mounting
│
├── tests/                          # Real-data verification test suite (Zero synthetic fixtures)
│   ├── conftest.py                 # Authentic PhysioNet patient cohort fixtures
│   ├── test_module3_temporal.py    # 20 tests: 1D-CNN, bimodal fusion, signed Grad-CAM
│   ├── test_module4_sequence.py    # 15 tests: GRU-D, Softplus decay, right-padding, TriModal
│   ├── test_module5_generative.py  # 17 tests: PatientVAE, TabularGAN, Gemini 3.7 RAG, HIPAA
│   └── test_adversarial_challenger.py # 11 tests: B=1 stability, gradient canary, stress tests
│
├── docs/                           # Architectural guides & interface contracts
│   ├── interfaces.md               # Strict tensor shape contracts
│   ├── SentryICU_Presentation_and_Viva_Master_Guide.md # Defense & viva guide
│   └── SentryICU_Module1_Comprehensive_Guide.pdf
│
└── notebooks/                      # Development & experimental notebooks
    ├── Module1_MLP_Activations_Optimizers.ipynb
    ├── Module2_Regularization_OOD.ipynb
    └── Module3_Temporal_Dynamics_MultiScale_Fusion.ipynb
```

---

## 🚀 Quickstart & Execution Guide

### Prerequisites

* **Python:** 3.10, 3.11, or 3.12 (Python 3.14 also supported)
* **Node.js:** v18.0+ (Node v23+ supported) and npm
* **Google GenAI API Key (Optional):** Set `GEMINI_API_KEY` for online grounded clinical dialogue.

---

### Step 1: Install Python Dependencies

```bash
# Optional: create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Install UI Dependencies & Build Frontend

```bash
# Install UI packages
npm --prefix ui install

# Build production bundle
npm run build
```

---

### Step 3: Run the Application

#### Option A: Unified Full-Stack Server (Recommended)

Serves the FastAPI REST API, model inference, and the built React UI on port 8000:

```bash
python -m src.api
```

* **Web UI:** Open [http://localhost:8000](http://localhost:8000)
* **Interactive API Docs (Swagger):** Open [http://localhost:8000/docs](http://localhost:8000/docs)

#### Option B: Vite Frontend Dev Server (with Hot Reloading)

```bash
npm run dev
```

* **Frontend Dev Server:** Open [http://localhost:3000](http://localhost:3000) (requests to `/api` proxy automatically to port 8000).

---

### Step 4: Run the Verification Test Suite

All 63 unit, integration, and adversarial stress tests execute strictly on **authentic clinical PhysioNet patient records** (`Dataset.csv` from `archive.zip`) with **zero synthetic test fixtures**:

```bash
pytest tests/ -v
```

Expected result:

```text
============================= 63 passed in 24.20s =============================
```

---

## 🔒 Clinical Safety, HIPAA Compliance & Grounding

SentryICU implements medical-grade privacy and safety protocols:

1. **Zero Unaggregated PHI:** The ingestion pipeline forbids individual Protected Health Information (`PatientID`, `MRN`, `SSN`, names). [`ClinicalRAGAssistant.verify_hipaa_compliance`](src/models/clinical_rag.py) raises an immediate `ValueError` if patient-level identifier columns are present.
2. **Evidence-Grounded Dialogue:** The clinical assistant is constrained strictly to empirical evaluation tables (`vae_evaluation.csv`, `gan_vs_smote.csv`), citing source artifacts for every numerical claim.
3. **Clinical Decision Support Disclaimer:** SentryICU is designed for research, risk indexing, and clinical audit. It does not replace the professional judgment of qualified healthcare practitioners.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgments

* **PhysioNet / Computing in Cardiology (CinC) Challenge 2019:** *Early Prediction of Sepsis from Clinical Data*.
* **Che et al. (2018):** *Recurrent Neural Networks for Multivariate Time Series with Missing Values* (Nature Scientific Reports).
* **Google DeepMind:** For the Google GenAI SDK and Gemini 3.7 Flash foundation models.

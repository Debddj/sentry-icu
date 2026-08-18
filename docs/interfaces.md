# SentryICU — System Interface & Tensor Specification (Contract)

To prevent breaking changes and ensure smooth integration among the 5 sub-teams (10 members), all models and pipelines MUST strictly adhere to the tensor dimensions and data formats defined below.

---

## 1. Data Schema & Patient Features (Canonical Pipeline)

- **Source:** PhysioNet/CinC 2019 Challenge (`Set A` for training, `Set B` for OOD evaluation).
- **Tabular Features Tensor Shape:** `(Batch_Size, 73)` — float32.
- **Normalization:** Zero-mean, unit-variance standardized (`StandardScaler`), fitted ONLY on Hospital A train split.

---

## 2. Module Interfaces

### Module 1 — Risk Fusion Head (Pair A)
- **Class:** `ClinicalRiskMLP(nn.Module)`
- **Input:** `(Batch_Size, 73)` or concatenated latent embeddings `(Batch_Size, 73 + D_cnn + D_gru)`
- **Output:** Raw logit `(Batch_Size, 1)` (applied with `BCEWithLogitsLoss`)
- **Inference Output:** Calibrated Probability $\in [0.0, 1.0]$ via `torch.sigmoid(logit)`

### Module 2 — Regularization & Initialization (Pair B)
- **Input / Output:** Identical to Module 1.
- **Contract:** Adds `nn.BatchNorm1d`, dropout rates $\in [0.1, 0.4]$, Kaiming/He weight initialization, and Optuna hyperparameter optimization.
- **Deliverable:** Zero-shot evaluation metrics table comparing Set A vs. Set B.

### Module 3 — Multi-Scale / CNN Feature Extractor (Pair C)
- **Class:** `ClinicalCNN1D(nn.Module)`
- **Input:** Sliding vital sign windows `(Batch_Size, Num_Vitals=7, Window_Length=12)`
- **Output Embedding:** `(Batch_Size, 32)` float32 representation for concatenation into Fusion Head.

### Module 4 — Temporal Sequence Model (Pair D)
- **Class:** `TemporalGRU(nn.Module)` or `TemporalGRUD(nn.Module)`
- **Input:** Multivariate hourly time-series `(Batch_Size, Seq_Len=24, Num_Features=14)` with binary mask `(Batch_Size, Seq_Len=24, Num_Features=14)`.
- **Output Embedding:** `(Batch_Size, 64)` float32 hidden representation of temporal trajectory.

### Module 5 — Generative VAE & Drift Detector (Pair E)
- **Class:** `PatientVAE(nn.Module)`
- **Input:** `(Batch_Size, 73)`
- **Outputs:**
  - Reconstructed input `(Batch_Size, 73)`
  - Latent distribution parameters: `mu (Batch_Size, 16)`, `logvar (Batch_Size, 16)`
  - Anomaly / Drift Score: Mean Squared Error (MSE) reconstruction error.

---

## 3. Loss & Metric Standard
- **Loss Function:** `nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.5]))`
- **Primary Metrics:**
  - `AUROC` (Area Under ROC Curve)
  - `AUPRC` (Area Under Precision-Recall Curve)
  - `ECE` (Expected Calibration Error)
  - `Brier Score` (Mean Squared Probability Error)

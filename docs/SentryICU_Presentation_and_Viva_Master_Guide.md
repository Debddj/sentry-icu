# SentryICU — 10-Minute Presentation Master Guide & Viva Defense
**Group 8 | CSE(AI) | Presentation & Viva Master Blueprint**  
**Exhaustive Slide-by-Slide Script, Module Mappings, Theoretical Foundations & Professor Question-Defense Bank**

- **Team:** Group 8 (10 Members)
- **Presentation Duration:** 10 Minutes Flat
- **Selection Rule:** Any Member Can Be Chosen
- **Coverage:** Slides 1–15 | Modules 1–5 (Modules 1, 2, and 3 Completed; Modules 4 and 5 Planned)

---

## 0. Master 10-Minute Presentation Pacing & Strategy Guide

> [!IMPORTANT]
> **The Golden Rule of 10-Minute Group Presentations:** Panels evaluate your team on clarity, clinical relevance, deep technical mastery, and seamless transitions. Because anyone out of all 10 members can be randomly picked, every single member must know this exact flow by heart. Deliver with confidence, stick strictly to the second-by-second timeline below, and leave exactly 30 seconds of buffer at the end.

| Slide | Title / Focus | Time Allocation | Running Clock | Core Talking Point to Nail |
|---|---|---|---|---|
| **Slide 1** | Title Slide | 15 sec | `0:00 - 0:15` | Introduce project name, Group 8, and core vision in one punchy sentence. |
| **Slide 2** | Team Members | 15 sec | `0:15 - 0:30` | Acknowledge CSE(AI) Section A, Group 8, 10 members working across 5 DL modules. |
| **Slide 3** | The Sepsis Crisis & Failure | 45 sec | `0:30 - 1:15` | 1 in 5 deaths, 7.6% mortality jump/hr; why NEWS/qSOFA fail (static, uncalibrated). |
| **Slide 4** | Objectives & Research Gap | 45 sec | `1:15 - 2:00` | 5 deep learning paradigms unified; zero-shot cross-hospital test (the core research gap). |
| **Slide 5** | The Solution & Architecture | 50 sec | `2:00 - 2:50` | Walk through Multi-Modal Fusion: Tabular MLP + CNN + GRU + VAE Drift Gate. |
| **Slide 6** | Syllabus Mapping & Progress | 40 sec | `2:50 - 3:30` | **Clearly state: Modules 1, 2 & 3 Completed & Validated; Modules 4 & 5 Planned & architected.** |
| **Slide 7** | Dataset & Preprocessing | 50 sec | `3:30 - 4:20` | PhysioNet Set A (20k) & Set B (20k); class imbalance (<10%), 73-dim engineered schema. |
| **Slide 8** | Module 1: MLP Foundation | 60 sec | `4:20 - 5:20` | ClinicalRiskMLP (73->128->64->32->1); 4x4 ablation; GELU+AdamW won (AUROC 0.875). |
| **Slide 9** | Module 2: Regularization & OOD | 65 sec | `5:20 - 6:25` | RegularizedMLP, Optuna; BatchNorm vs LayerNorm domain shift finding; zero-shot B test. |
| **Slide 10** | **Module 3: CNN & Waveforms** | **50 sec** | `6:25 - 7:15` | **12h sliding windows; multi-scale 1D-CNN ($k=3,5,7$) -> 32-D spatial embedding + 105-D Multimodal Fusion; Leak-free target indexing & stratified split; 1D Grad-CAM (hypotension/hypothermia sensitive); checkpoint export (`module3_best_cnn_fusion.pt`).** |
| **Slide 11** | Module 4: Temporal GRU | 50 sec | `7:15 - 8:05` | 24h sequences -> 2-layer GRU -> 64-D embedding; GRU-D temporal decay; 169-D Tri-Fusion (73-D MLP + 32-D CNN + 64-D GRU). |
| **Slide 12** | Module 5: VAE & Drift | 45 sec | `8:05 - 8:50` | PatientVAE (73->16->73); Reconstruction MSE as anomaly & cross-hospital drift gate. |
| **Slide 13** | Impact & Scalability | 30 sec | `8:50 - 9:20` | Quantifies deployment risk, reduces alert fatigue via ECE, scales to Federated Learning. |
| **Slide 14-15** | Thank You & Q&A | 10 sec | `9:20 - 9:30` | Conclude strong, welcome panel questions. Total time: 9:30 (30s buffer). |

---

## 1. Slide-by-Slide Walkthrough, Speaker Scripts & Panel Defense

### Slide 1 & 2: Title & Team Members (Duration: 30s | Clock: 0:00 - 0:30)

#### Verbatim Speaker Pitch:
> "Respected professors and panel members, good morning. We are Group 8 from CSE (Artificial Intelligence), Section A. Today, we present **SentryICU** — a modular, cross-hospital generalizable deep learning system for early ICU deterioration and sepsis detection. Our team consists of 10 members collaborating across 5 foundational deep learning paradigms to solve one of the deadliest clinical challenges in modern healthcare."

#### Panel Trap Questions & High-Scoring Defense:
- **Q: Why the name 'SentryICU'? What is the central hypothesis of your project?**  
  *Defense Answer:* A 'sentry' is a continuous, vigilant guard. Current ICU systems are retrospective or static; SentryICU's central hypothesis is that an early warning system must combine static statistical extremes (MLP), local waveform morphology (CNN), long-range deterioration velocity (GRU), and uncalibrated distribution shift detection (VAE) while proving it can survive zero-shot deployment at an unseen hospital.

---

### Slide 3: The Problem — The Sepsis Crisis & Why Current Detection Fails (Duration: 45s | Clock: 0:30 - 1:15)

#### Verbatim Speaker Pitch:
> "Sepsis accounts for 1 in every 5 deaths worldwide — approximately 11 million deaths annually. In the ICU, every single hour of delayed antibiotic treatment increases patient mortality by roughly 7.6%. Yet, the clinical scoring tools hospitals currently use — like NEWS, SOFA, and qSOFA — are rule-based, static, and completely blind to individual patient trajectories. Even worse, modern ML models built for single hospitals suffer catastrophic performance collapse when deployed to new hospital environments with different demographics, lab protocols, and equipment. Uncalibrated models produce severe false alarms, leading to clinical alert fatigue where nurses literally mute the monitors. SentryICU is designed to solve this exact trust and transfer crisis."

#### Deep Technical Foundation:
- **qSOFA / SOFA Failure Mode:** qSOFA uses static thresholds (RR $\ge 22$, altered mentation, SBP $\le 100$). It misses early compensatory septic shock where blood pressure is maintained via tachycardia.
- **Alert Fatigue:** High false positive rates cause clinicians to ignore alarms. Expected Calibration Error (ECE) measures whether a 70% predicted probability truly means 7 out of 10 patients are septic.

#### Panel Trap Questions & High-Scoring Defense:
- **Q: What is sepsis clinically, and why is an hourly time-series approach necessary instead of a single admission snapshot?**  
  *Defense Answer:* Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection (Sepsis-3 definition). It evolves dynamically over hours. An admission snapshot misses the rapid progression from Systemic Inflammatory Response Syndrome (SIRS) to septic shock. Tracking hourly physiological velocity (e.g. rising heart rate with falling mean arterial pressure) is critical for early therapeutic intervention.
- **Q: Why do machine learning models fail when transferred between hospitals? What is 'Dataset Shift'?**  
  *Defense Answer:* Machine learning models assume in-distribution independent and identically distributed (i.i.d.) data: $P_{\text{train}}(X, y) = P_{\text{test}}(X, y)$. Across hospitals, three types of distribution shift occur: Covariate shift $P(X)$ changes (different age/demographics), Concept shift $P(y|X)$ changes (different ICU clinical intervention protocols), and Prior probability shift $P(y)$ changes (differing sepsis baseline prevalence).

---

### Slide 4: Project Objectives & Research Gap (Duration: 45s | Clock: 1:15 - 2:00)

#### Verbatim Speaker Pitch:
> "To address this crisis, SentryICU establishes five concrete engineering and research objectives: First, build a modular, end-to-end pipeline on real-world ICU records. Second, rigorously evaluate cross-hospital generalizability by training strictly on Hospital A and testing zero-shot on Hospital B. Third, unify five core deep learning paradigms — Multi-Layer Perceptrons, Regularization, CNNs, Recurrent Models, and Generative AI — into one cohesive framework. Fourth, execute systematic empirical ablations across activations, optimizers, and normalization techniques. And fifth, deliver a fully reproducible codebase with transparent experiment tracking."

#### Panel Trap Questions & High-Scoring Defense:
- **Q: What is 'Zero-Shot Cross-Hospital Evaluation'? Why is it a research gap?**  
  *Defense Answer:* Most academic papers report 5-fold cross-validation on a single hospital dataset (e.g. MIMIC-III or PhysioNet Set A alone). Zero-shot cross-hospital evaluation means the model is trained exclusively on Hospital A, and evaluated on Hospital B (an entirely distinct clinical center) without fine-tuning weights and without re-fitting standard scalers. This directly measures true real-world clinical deployment survivability.

---

### Slide 5 & 6: Proposed Architecture & Syllabus Mapping (Duration: 90s | Clock: 2:00 - 3:30)

#### Verbatim Speaker Pitch:
> "Our proposed architecture bridges the deep learning syllabus with clinical needs through a multi-modal hierarchical pipeline: Raw ICU data enters our feature engineering pipeline. Modules 1 and 2 process global summary statistics through a regularized MLP. **Module 3 extracts short-term waveform spikes and acute physiological drift through our multi-scale 1D-CNN.** Module 4 captures long-term deterioration trajectories through a GRU. Module 5 uses a Variational Autoencoder to act as a pre-deployment anomaly and distribution drift gate. These latent embeddings fuse into a unified risk prediction head. As mapped in Slide 6: **Modules 1, 2, and 3 are fully completed, empirically ablated, and validated.** Modules 4 and 5 have finalized architectural contracts and are scheduled as our next milestones."

| Module | Deep Learning Syllabus Topic | SentryICU Implementation | Latent Output | Status |
|---|---|---|---|---|
| **Module 1** | ANN, MLP, Activations, Optimizers | `ClinicalRiskMLP` baseline ($73 \rightarrow 128 \rightarrow 64 \rightarrow 32 \rightarrow 1$) | 73-dim tabular | **COMPLETED** |
| **Module 2** | Backprop, Regularization, Initialization | `RegularizedMLP`, BatchNorm vs LayerNorm, Optuna, OOD Test | Calibrated logit | **COMPLETED** |
| **Module 3** | **CNN, Waveforms, Transfer Learning, Grad-CAM** | **`ClinicalCNN1D` ($k=3,5,7$) + 105-D `SentryICUFusionModel` on 12h sliding vital windows, leak-free windowing, stratified split, AUROC/AUPRC/Brier metrics, 1D Grad-CAM, checkpoint exported** | **32-dim spatial + 105-dim fused** | **COMPLETED** |
| **Module 4** | RNN, LSTM, GRU, GRU-D Temporal Decay | 2-layer GRU on 24h sequence + decay for missing labs | 64-dim temporal | Planned / Spec |
| **Module 5** | GAN, VAE, Generative AI, Drift Detection | `PatientVAE` ($73 \rightarrow 16 \rightarrow 73$) reconstruction error drift gate | Anomaly score | Planned / Spec |

#### Panel Trap Questions & High-Scoring Defense:
- **Q: Why not use a single model (like XGBoost or a Transformer) instead of 5 separate deep learning modules?**  
  *Defense Answer:* Tree-based models like XGBoost cannot natively process continuous sliding time-series waveforms, cannot perform end-to-end gradient backpropagation through convolutional kernels, and cannot generate latent generative distributions for unsupervised drift detection. Our multi-modal modular design separates orthogonal clinical dimensions: extremes (MLP), acute waveform morphology (CNN), long-term velocity (GRU), and out-of-distribution trust (VAE).

---

### Slide 7: Data Analysis — Dataset & Preprocessing Pipeline (Duration: 50s | Clock: 3:30 - 4:20)

#### Verbatim Speaker Pitch:
> "Our data pipeline utilizes the PhysioNet/Computing in Cardiology 2019 Challenge dataset. Hospital A represents Beth Israel Deaconess Medical Center with 20,336 ICU stays, while Hospital B represents Emory University Hospital with 20,000 stays. We tackle four major clinical data hurdles: First, extreme class imbalance: less than 10% of ICU patients develop sepsis. We resolve this using positive-weighted Binary Cross-Entropy loss with pos_weight = 9.5 to 10.4. Second, massive missingness: 70 to 90% of lab analytes are unobserved at any given hour. We apply clinical forward-fill and backward-fill. Third, irregular sampling: hourly vitals versus labs drawn every 6 to 24 hours. And fourth, feature engineering: 14 key clinical variables — 7 vitals and 7 labs — aggregated across 5 statistical moments (mean, min, max, std, last) plus 3 demographics (Age, Gender, Max ICULOS), constructing a standardized 73-dimensional feature vector."

#### Exact Mathematical Preprocessing Details:
- **73 Features:** (7 vitals + 7 labs) $\times$ 5 stats = 70 features + Age + Gender + Max_ICULOS = 73 Features.
- **StandardScaler Rule:** Scaler is fitted strictly on Hospital A train split: $z = (x - \mu_A) / \sigma_A$. Hospital B is transformed using $\mu_A, \sigma_A$ with zero refitting.

#### Panel Trap Questions & High-Scoring Defense:
- **Q: How did you calculate pos_weight in BCEWithLogitsLoss? What does it do mathematically?**  
  *Defense Answer:* $\text{pos\_weight} = N_{\text{neg}} / N_{\text{pos}}$ (e.g. $18,000 / 2,000 = 9.0$). In BCE loss:
  $$\mathcal{L} = -\left[ \text{pos\_weight} \cdot y \log(\sigma(x)) + (1 - y) \log(1 - \sigma(x)) \right]$$
  It scales the gradient of the minority positive class by 9x, forcing the backpropagation optimizer to penalize false negatives 9 times more heavily than false positives, which is vital for life-critical sepsis alarms.
- **Q: Why forward-fill missing labs instead of mean imputation or dropping missing rows?**  
  *Defense Answer:* Dropping rows would destroy 90% of the dataset. Mean imputation destroys physiological variance. Clinically, a normal creatinine of 0.9 mg/dL drawn at hour 4 remains the physician's assumed value until the next lab at hour 12. Forward-fill (`ffill`) reflects clinical reality by carrying forward the most recent valid measurement.

---

### Slide 8: Module 1 — MLP Foundation & Baseline Risk Model (Duration: 60s | Clock: 4:20 - 5:20)

#### Verbatim Speaker Pitch:
> "Module 1 establishes our neural baseline through `ClinicalRiskMLP` — a fully differentiable architecture: 73 inputs $\rightarrow$ 128 $\rightarrow$ 64 $\rightarrow$ 32 $\rightarrow$ 1 output logit with Dropout 0.2. We executed a comprehensive 4x4 ablation matrix comparing 4 activation functions (ReLU, LeakyReLU, GELU, ELU) across 4 optimizers (Adam, AdamW, SGD with Momentum 0.9, and RMSprop). The winning combination was GELU with AdamW, achieving an impressive AUROC of 0.875 and AUPRC of 0.638. GELU won decisively because clinical z-score standardized features contain substantial negative values. Standard ReLU suffers from 'Dying ReLU' syndrome by zeroing all negative inputs, whereas GELU softly weights negative inputs via Gaussian cumulative distribution, preserving subtle sub-zero physiological signals."

#### Deep Technical Formulas:
- **GELU Definition:** $\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x)$, where $X \sim \mathcal{N}(0, 1)$. Approximation: $0.5x \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 x^3)))$.
- **AdamW vs Adam:** Adam couples L2 penalty with gradients: $\theta_{t+1} = \theta_t - \text{lr} \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) - \text{lr} \cdot \lambda \cdot \theta_t$. AdamW decouples weight decay directly from adaptive moments, preventing weights with large past gradients from escaping regularization.

#### Panel Trap Questions & High-Scoring Defense:
- **Q: Why does the model output raw logits instead of applying Sigmoid inside the forward function?**  
  *Defense Answer:* PyTorch's `nn.BCEWithLogitsLoss` combines Sigmoid and Binary Cross-Entropy into a single layer using the log-sum-exp trick: $\log(1 + \exp(-x))$. This avoids numerical instability and gradient saturation (underflow/overflow) that occurs when computing standalone `torch.sigmoid(x)` followed by `torch.log()`.
- **Q: Why did SGD perform worse than Adam/AdamW?**  
  *Defense Answer:* Clinical features have wildly different scales and gradient magnitudes (e.g. Heart Rate variations vs. White Blood Cell counts). SGD uses a uniform learning rate across all parameters, struggling with ill-conditioned loss surfaces, whereas AdamW maintains per-parameter adaptive learning rates via second-moment estimation.

---

### Slide 9: Module 2 — Regularization & Cross-Hospital Stress Test (Duration: 65s | Clock: 5:20 - 6:25)

#### Verbatim Speaker Pitch:
> "Module 2 answers the central research question: Does our model survive deployment at an unseen hospital? We built `RegularizedMLP` and conducted 6 rigorous experimental tracks: weight initialization (Default vs Xavier vs Kaiming), normalization ablation (BatchNorm vs LayerNorm vs None), dropout sweeps (0.1 to 0.5), weight decay sweeps, learning rate schedulers (Cosine vs ReduceLROnPlateau), and a 50-trial Optuna Bayesian search. Our core scientific finding: While BatchNorm achieves strong in-distribution convergence on Hospital A, it degrades significantly under zero-shot transfer to Hospital B. This is because BatchNorm locks in Hospital A's batch running mean and variance. When Hospital B's patient demographics differ, those frozen running stats cause covariate mismatch. LayerNorm, which normalizes across features per patient independently, demonstrated superior OOD robustness."

#### Deep Technical Formulas:
- **BatchNorm:** $y = \frac{x - \mathbb{E}_{\text{batch}}[x]}{\sqrt{\text{Var}_{\text{batch}}[x] + \epsilon}} \cdot \gamma + \beta$. Stores running mean $\mu_{\text{running}}$ and $\sigma^2_{\text{running}}$.
- **LayerNorm:** $y = \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} \cdot \gamma + \beta$, where $\mu_L = \frac{1}{D}\sum_{i=1}^D x_i$. Independent of batch statistics.
- **Optuna TPE:** Tree-structured Parzen Estimator models $P(x|y)$ using two density functions $\ell(x)$ for top trials and $g(x)$ for remaining trials, maximizing Expected Improvement (EI).

#### Panel Trap Questions & High-Scoring Defense:
- **Q: What is the Expected Calibration Error (ECE) and why did it increase 2-3x on Hospital B?**  
  *Defense Answer:* ECE groups predictions into $M$ bins and computes:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
  On Hospital A, predicted probabilities match observed sepsis rates (low ECE). Under distribution shift on Hospital B, the model becomes overconfident in its erroneous predictions, causing ECE to rise from ~0.10 to ~0.27. This proves why calibration assessment is mandatory before hospital deployment.

---

### Slide 10: Module 3 — CNN Feature Extraction & Interpretability (Duration: 50s | Clock: 6:25 - 7:15)

#### Verbatim Speaker Pitch:
> "Module 3 tackles the limitation of static summary statistics by capturing localized waveform dynamics using 1D Convolutional Neural Networks. We re-pipelined raw vital signs into 12-hour sliding windows with a 6-hour stride, producing $(7, 12)$ time-series tensors across 7 key vitals. Our `ClinicalCNN1D` architecture deploys multi-scale parallel kernels of size 3, 5, and 7—inspired by Inception modules. A kernel of size 3 captures acute spikes like sudden blood pressure crashes; size 5 captures sub-acute trends like rising fever; size 7 captures sustained organ decline. These branches concatenate into a 32-dimensional spatial embedding, which fuses with our 73-D static features into a 105-D multimodal representation in `SentryICUFusionModel`. We solved temporal target leakage by strictly indexing labels to window termination, enforced stratified patient splitting, and achieved superior AUROC and AUPRC over the tabular-only baseline. Finally, we implemented sign-preserving 1D Temporal Grad-CAM to highlight bedside clinical drivers without blinding acute hypotension or hypothermia, and exported `module3_best_cnn_fusion.pt` for downstream Module 4 and 5 integration."

#### Deep Technical Formulas:
- **1D Temporal Convolution:**
  $$y_t = \sum_{c=1}^{C_{\text{in}}} \sum_{i=0}^{k-1} w_{c,i} \cdot x_{c, t+i} + b$$
- **Multi-Scale Inception Branch Concatenation:**
  $$F = \left[ F_{k=3}; F_{k=5}; F_{k=7} \right] \in \mathbb{R}^{B \times 48 \times 12} \xrightarrow{\text{AdaptivePool1d}} \mathbb{R}^{B \times 48} \xrightarrow{\text{Linear}} \mathbb{R}^{B \times 32}$$
- **105-D Multimodal Fusion Head:**
  $$h_{\text{fused}} = \left[ x_{\text{static}}(73); h_{\text{cnn}}(32) \right] \in \mathbb{R}^{B \times 105} \xrightarrow{\text{Regularized MLP Head}} \text{Logit} \in \mathbb{R}^{B \times 1}$$
- **Sign-Preserving 1D Temporal Grad-CAM:**
  $$\text{Attribution} = \left| x_{\text{norm}} \odot \nabla_{x} \right|$$
  Standard Grad-CAM applies $\text{ReLU}(x \cdot \alpha)$. Because vitals are standardized $z$-scores, life-threatening hypotension ($\text{MAP} < 65\text{ mmHg}, z < 0$) and hypothermia ($\text{Temp} < 36.0^\circ\text{C}, z < 0$) are negative numbers. Naive ReLU truncates them to zero. Absolute magnitude attribution $|x \odot \nabla_x|$ guarantees bidirectional clinical deviations remain fully visible to clinicians.

#### Panel Trap Questions & High-Scoring Defense:
- **Q: Why use 1D-CNN instead of 2D-CNN for clinical vitals? What does kernel sliding along the time axis mean?**  
  *Defense Answer:* 2D convolutions assume spatial adjacency in both height and width (like neighboring pixels in an image). In clinical tables, the channel ordering of vitals (e.g. Heart Rate next to O2Sat) is arbitrary, not continuous. 1D convolutions treat vitals as input channels across which full dot-products are taken, sliding kernels strictly along the 1D temporal axis. This guarantees translation invariance in time.
- **Q: In time-series sliding windows, how did your team prevent temporal target leakage, and why does taking the maximum label across the window fail clinically?**  
  *Defense Answer:* In PhysioNet 2019, sepsis onset marks the point of clinical deterioration. If a window spanning hours 0 to 11 is labeled positive using `max()` when sepsis occurred at hour 3, the model observes post-sepsis vitals from hours 4 to 11 to 'predict' an event that already occurred. SentryICU eliminates this future lookahead by indexing the prediction target strictly to window termination (`window['SepsisLabel'].iloc[-1]`), ensuring the network only observes historical data prior to the prediction horizon.
- **Q: Standard Grad-CAM applies a ReLU to the weighted feature maps. Why did you modify this for ICU vital sign waveforms?**  
  *Defense Answer:* ICU vitals are standardized ($z$-scores with mean 0). Pathological indicators of septic shock include both positive excursions (tachycardia, tachypnea, fever) and critical negative depressions—specifically severe hypotension (MAP < 65 mmHg, $z < 0$) and severe hypothermia (Temp < 36.0°C, $z < 0$). In naive Grad-CAM, multiplying a negative $z$-score by a positive importance weight yields a negative number that standard ReLU truncates to zero. SentryICU uses absolute magnitude-aware gradient attribution $|x \odot \nabla_x|$ to guarantee that life-threatening drops in blood pressure and core temperature remain vividly interpretable on bedside heatmaps.
- **Q: How does Module 3 connect upstream with Modules 1 & 2, and what artifact is passed forward to Module 4 and Module 5?**  
  *Defense Answer:* Module 3 strictly complies with `docs/interfaces.md`: it extracts a 32-D embedding from $(B, 7, 12)$ vital windows and fuses it with the canonical 73-D tabular vector to create a 105-D fused representation. It saves `models/module3_best_cnn_fusion.pt` containing the model state dict, the standalone CNN encoder weights, the 73-D scaler parameters, and vital sign `ts_mean` and `ts_std`. Module 4 directly ingests this 32-D CNN embedding alongside its 64-D GRU embedding to form the 169-D Tri-Modal fusion vector.

---

### Slide 11: Module 4 — Temporal Sequence Modeling (RNN, LSTM & GRU) (Duration: 50s | Clock: 7:15 - 8:05)

#### Verbatim Speaker Pitch:
> "While CNNs capture localized 12-hour waveform shapes, ICU stays span days. Module 4 captures long-range physiological trajectories using Recurrent Neural Networks. We feed 24-hour sequences across 14 clinical features into a 2-layer GRU with hidden dimension 64, outputting a 64-dimensional temporal embedding. GRU was selected over LSTM because its 2-gate mechanism achieves equivalent clinical accuracy with 25% fewer parameters, speeding up training and reducing overfitting. Furthermore, we incorporate GRU-D — which uses trainable exponential decay parameters ($\gamma_x$ and $\gamma_h$) to explicitly model measurement staleness as elapsed time $\Delta t$ increases without new lab draws. Fusing Module 1 Tabular (73-D) + Module 3 CNN (32-D) + Module 4 GRU (64-D) yields a comprehensive 169-dimensional Tri-Modal representation."

#### Deep Technical Formulas:
- **GRU-D Decay:** $\gamma_t = \exp(-\max(0, W_\gamma \cdot \Delta t + b_\gamma))$. Decays unobserved values toward empirical mean $x_{\text{mean}}$ and hidden states toward 0.
- **Tri-Modal Fusion:**
  $$\text{Input} = \left[ h_{\text{tabular}}(73); h_{\text{cnn}}(32); h_{\text{gru}}(64) \right] \in \mathbb{R}^{169} \rightarrow \text{Linear}(169, 128) \rightarrow \text{BatchNorm} \rightarrow \text{GELU} \rightarrow \text{Linear}(128, 1)$$

#### Panel Trap Questions & High-Scoring Defense:
- **Q: Why not use Bidirectional GRU (BiGRU) for real-time bedside alerting?**  
  *Defense Answer:* In BiGRU, the backward pass $h_{\text{backward}, t}$ aggregates future time steps $[t+1, T]$. While acceptable in retrospective research audits, future data does not exist at hour $t$ during real-time bedside monitoring. Deploying BiGRU online causes future data leakage during training, yielding artificially high metrics that crash in clinical production. SentryICU strictly enforces causal unidirectional recurrence for bedside alerting.

---

### Slide 12: Module 5 — Generative AI & Anomaly Detection (VAE & GAN) (Duration: 45s | Clock: 8:05 - 8:50)

#### Verbatim Speaker Pitch:
> "Module 5 introduces Generative AI for unsupervised anomaly detection and distribution drift monitoring. We build `PatientVAE`: an Encoder compresses the 73 clinical features into a 16-dimensional latent space (mean $\mu$ and log-variance $\sigma$), and a symmetric Decoder reconstructs the patient vector. We train the VAE exclusively on normal, non-septic Hospital A patients using Reconstruction MSE plus KL-Divergence loss. When deployed to Hospital B, the VAE acts as an automated safety gate: patients exhibiting high reconstruction error represent severe physiological anomalies or significant cross-hospital data drift. We quantify this drift using Kolmogorov-Smirnov statistical tests and MMD distance to alert clinicians before deploying the model in a new facility."

#### Deep Technical Formulas:
- **ELBO Loss:** $\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \,||\, p(z))$, where $p(z) \sim \mathcal{N}(0, I)$.
- **Reparameterization Trick:** $z = \mu(x) + \sigma(x) \odot \epsilon$, where $\epsilon \sim \mathcal{N}(0, I)$. Enables backpropagation through stochastic nodes.

#### Panel Trap Questions & High-Scoring Defense:
- **Q: Why is the Reparameterization Trick necessary in Variational Autoencoders?**  
  *Defense Answer:* If we sampled $z$ directly from $\mathcal{N}(\mu, \sigma^2)$, the sampling operation is stochastic and non-differentiable; gradients cannot backpropagate from decoder to encoder. By expressing $z = \mu + \sigma \odot \epsilon$ where $\epsilon$ is drawn from an independent standard normal $\mathcal{N}(0, I)$, the stochasticity is isolated into an external parameter-free input, allowing backpropagation of $\partial \mathcal{L} / \partial \mu$ and $\partial \mathcal{L} / \partial \sigma$ via standard autograd.

---

### Slide 13, 14 & 15: Impact, Scalability & Conclusion (Duration: 40s | Clock: 8:50 - 9:30)

#### Verbatim Speaker Pitch:
> "To conclude, SentryICU delivers three immediate clinical breakthroughs: First, it quantifies rather than assumes cross-hospital deployment risk. Second, it reduces clinical alert fatigue through calibrated ECE probabilities and Grad-CAM interpretability. Third, its modular architecture enables independent upgrades and scales seamlessly to Federated Learning across hospital networks without centralizing sensitive patient telemetry. By uniting Multi-Layer Perceptrons, Regularization, CNNs, Recurrent Models, and Generative AI, SentryICU provides a reliable, clinically actionable early warning paradigm. Thank you. We are now open for questions from the panel."

#### Panel Trap Questions & High-Scoring Defense:
- **Q: How would Federated Learning work with SentryICU?**  
  *Defense Answer:* Under Federated Averaging (FedAvg), Hospital A and Hospital B train local SentryICU model weights on their private patient records. Only weight updates (gradients or $\Delta W$) are transmitted to a central coordinator for aggregation ($W_{\text{global}} = \sum \frac{|D_k|}{|D|} W_k$), completely preserving patient privacy and HIPAA compliance while overcoming single-center bias.

---

## 2. Top 22 Rapid-Fire Master Viva Questions & Panel Traps

1. **Why do deep networks suffer from vanishing gradients, and how do ResNet, LSTM, and GELU prevent it?**  
   *Defense Answer:* Vanishing gradients occur during backpropagation when the chain rule multiplies many Jacobian matrices with eigenvalues $< 1$ (or sigmoid/tanh derivatives with max values of $0.25$ and $1.0$). In deep MLPs, gradients decay exponentially toward zero at early layers. ResNet solves this via additive identity shortcuts ($x + F(x)$, gradient flow $= 1 + dF/dx$); LSTM solves it via the linear cell state constant error carousel ($dC_t/dC_{t-1} = f_t$); GELU solves it by providing non-zero gradients for negative inputs unlike hard-clipping ReLU.

2. **What is the mathematical difference between Kaiming (He) and Xavier (Glorot) initialization?**  
   *Defense Answer:* Xavier assumes linear activations and sets $\text{Var}(W) = 2 / (n_{\text{in}} + n_{\text{out}})$. Under ReLU/GELU, half the neurons are inactive, halving variance at each layer. Kaiming Normal compensates for this by setting $\text{Var}(W) = 2 / n_{\text{in}}$ (or $2 / ((1 + a^2) \cdot n_{\text{in}})$ for LeakyReLU), maintaining constant signal variance across arbitrarily deep non-linear networks.

3. **Why is AUPRC more informative than AUROC for ICU sepsis detection?**  
   *Defense Answer:* Because sepsis has severe class imbalance ($<10\%$ positive). AUROC plots True Positive Rate vs. False Positive Rate ($\text{FPR} = \text{FP} / (\text{FP} + \text{TN})$). Because True Negatives (TN) are massive ($\sim 90\%$), FPR stays deceptively tiny even with hundreds of false alarms. AUPRC plots Precision ($\text{TP} / (\text{TP} + \text{FP})$) vs. Recall, directly penalizing false alarms without being diluted by the large negative class.

4. **What is the difference between Batch Normalization and Layer Normalization in PyTorch?**  
   *Defense Answer:* BatchNorm computes mean and variance across the batch dimension for each individual feature channel (requires batch_size $> 1$, stores running statistics, sensitive to batch composition). LayerNorm computes mean and variance across all feature channels for a single sample independently (batch-size independent, stores no running stats, ideal for sequence models and OOD transfer).

5. **What is the Brier Score and how does it relate to calibration?**  
   *Defense Answer:* Brier Score is the mean squared error of probabilistic predictions: $\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$. It is a strictly proper scoring rule that simultaneously measures both discrimination (refinement) and calibration (reliability). A lower Brier score indicates better calibrated clinical risk.

6. **What is Dropout and why does it act as an ensemble method?**  
   *Defense Answer:* Dropout randomly zeroes neurons with probability $p$ during training, scaling remaining activations by $1/(1-p)$ (Inverted Dropout). It prevents co-adaptation of features. Because each mini-batch trains a different sub-network of the $2^N$ possible architectures with shared weights, inference with all neurons active approximates the geometric mean of an exponential ensemble of models.

7. **Explain Backpropagation Through Time (BPTT) and why truncated BPTT is used.**  
   *Defense Answer:* BPTT unrolls an RNN over $T$ time steps and applies standard backpropagation. For long sequences (e.g. 100 hours), computing gradients across the entire unrolled graph requires massive VRAM and causes vanishing/exploding gradients. Truncated BPTT splits the sequence into chunks of length $k$ (e.g. 24h), executing forward and backward passes only within each window.

8. **What is the difference between L1 regularization (Lasso) and L2 regularization (Ridge)?**  
   *Defense Answer:* L1 adds $\lambda \sum |w_i|$ to the loss, creating diamond-shaped constraint boundaries whose corners intersect axes, driving non-informative weights strictly to zero (sparse feature selection). L2 adds $\lambda \sum w_i^2$, creating circular constraint contours that shrink weights smoothly toward zero without exact sparsity.

9. **In CNNs, what is receptive field and how does stacking 3x1 kernels compare to a single 7x1 kernel?**  
   *Defense Answer:* Receptive field is the span of input time steps influencing an output neuron. Three stacked Conv1D layers of kernel size 3 have the same effective receptive field ($3 + 2 + 2 = 7$) as a single kernel of size 7, but use fewer parameters ($3 \cdot (3 \cdot C^2) = 9C^2$ vs $7C^2$ is comparable) and incorporate 3 non-linear activations instead of 1, allowing the network to learn vastly more expressive, non-linear feature hierarchies. In Module 3, we combine $k=3, 5, 7$ in parallel Inception branches to simultaneously capture multi-scale temporal frequencies.

10. **How does the learning rate scheduler ReduceLROnPlateau differ from CosineAnnealingLR?**  
    *Defense Answer:* CosineAnnealingLR follows a deterministic cosine schedule: $\text{lr}_t = \text{lr}_{\text{min}} + 0.5(\text{lr}_{\text{max}} - \text{lr}_{\text{min}})(1 + \cos(\pi t / T_{\text{max}}))$. ReduceLROnPlateau is dynamic and metric-driven: it monitors validation loss and reduces the learning rate by a factor (e.g. 0.5) only when the validation metric fails to improve for a predefined patience (e.g. 3 epochs).

11. **What is the Kullback-Leibler (KL) Divergence in VAE loss?**  
    *Defense Answer:* $D_{\text{KL}}(q_\phi(z|x) \,||\, p(z))$ measures the statistical divergence between the encoder's approximate posterior $q_\phi(z|x)$ and the standard normal prior $p(z) \sim \mathcal{N}(0, I)$. Mathematically: $-0.5 \sum [1 + \log(\sigma^2) - \mu^2 - \sigma^2]$. It acts as a regularizer preventing the encoder from clustering patients into discrete isolated points, forcing a continuous, smooth latent manifold.

12. **What is Mode Collapse in GANs and why did you select VAE as your primary generative model?**  
    *Defense Answer:* Mode collapse occurs when the GAN Generator discovers a small subset of outputs that consistently fool the Discriminator and repeatedly produces only those samples, losing dataset diversity. VAEs optimize the principled Evidence Lower Bound (ELBO) with stable maximum-likelihood training, avoiding adversarial minimax instability and guaranteeing coverage across the entire patient manifold.

13. **How does Optuna's Median Pruner work?**  
    *Defense Answer:* The Median Pruner compares the intermediate validation score (e.g. validation AUROC at epoch 5) of the current trial against the median score of previous completed trials at the same step. If the trial falls below the median, Optuna raises `TrialPruned`, terminating unpromising hyperparameter configurations early to conserve GPU hours.

14. **What is the role of the 169-dimensional Tri-Modal Fusion vector?**  
    *Defense Answer:* It is the concatenation of Module 1 Tabular MLP features (73-D, capturing stay-wide statistical extremes), Module 3 1D-CNN features (32-D, capturing localized multi-scale vital sign waveforms), and Module 4 GRU features (64-D, capturing long-term longitudinal trajectory and deterioration velocity): $73 + 32 + 64 = 169$ dimensions.

15. **What is the difference between Sequence-to-One and Sequence-to-Sequence in ICU modeling?**  
    *Defense Answer:* Sequence-to-One processes a time series of length $T$ and outputs a single prediction at the end of the window (e.g. patient sepsis diagnosis at discharge). Sequence-to-Sequence computes a risk prediction at every single time step $t$ using causal masking, providing continuous real-time bedside risk monitoring.

16. **Why is Platt Scaling / Temperature Scaling used post-hoc?**  
    *Defense Answer:* Deep neural networks with modern regularizers are often overconfident. Temperature Scaling divides model logits by a scalar parameter $T > 0$: $p_i = \sigma(z_i / T)$. When $T > 1$, it softens the probability distribution, lowering ECE and restoring true empirical calibration without changing AUROC or ranking accuracy.

17. **How do you handle patients with varying ICU stay lengths (e.g. 8 hours vs 200 hours)?**  
    *Defense Answer:* In tabular modules (1 & 2), statistical aggregation across the stay produces a fixed 73-D vector regardless of duration. In sequence modules (3 & 4), we apply sliding lookback windows (e.g. fixed 12h or 24h horizons) and use zero-padding with `pack_padded_sequence` and boolean masks for variable sequence lengths.

18. **What is the clinical significance of qSOFA vs SOFA?**  
    *Defense Answer:* SOFA (Sequential Organ Failure Assessment) requires extensive lab values (PaO2, platelets, bilirubin, creatinine) and is comprehensive but slow. qSOFA (quick SOFA) uses only 3 bedside vitals (Respiratory Rate $\ge 22$, Glasgow Coma Scale $< 15$, Systolic BP $\le 100$) for rapid triage outside the ICU.

19. **What is Maximum Mean Discrepancy (MMD) used in Module 5?**  
    *Defense Answer:* MMD is a non-parametric kernel-based statistical test that measures the distance between two probability distributions $P$ and $Q$ in a Reproducing Kernel Hilbert Space (RKHS): $\text{MMD}^2(P, Q) = \mathbb{E}[k(x, x')] - 2\mathbb{E}[k(x, y)] + \mathbb{E}[k(y, y')]$. It quantifies cross-hospital data drift without making restrictive parametric assumptions.

20. **What is your team's key scientific contribution in SentryICU?**  
    *Defense Answer:* Our key contribution is proving that standard deep learning models built on a single hospital fail under cross-center transfer, and demonstrating a modular, calibration-aware, multi-modal system combining **tabular extremes, convolutional waveform dynamics, recurrent trajectory velocity, and generative VAE drift gates** that rigorously quantifies and withstands zero-shot cross-hospital deployment.

21. **How did your team prevent architectural confounding in the Module 3 ablation experiments?**  
    *Defense Answer:* In our ablation suite, when comparing single-scale ($k=5$) against multi-scale ($k=3, 5, 7$), kernel sizes (3 vs 5 vs 7), pooling methods (max vs avg), and network depth, we wrapped every encoder into the exact same 2-layer regularized MLP fusion head with BatchNorm and Dropout. This isolated the convolutional inductive bias as the single variable under test, preventing head capacity differences from corrupting experimental conclusions.

22. **Why was stratified patient-level splitting critical in Module 3's sliding window pipeline?**  
    *Defense Answer:* Sepsis prevalence in ICU cohorts is low ($\sim 2-8\%$). If you perform random splitting on patient IDs without stratification, the small cohort of positive patients can be heavily skewed into the training set, leaving validation with near-zero positive cases. Stratifying patient IDs by their overall sepsis status ensures that both train and validation splits contain identical positive-to-negative patient ratios before temporal sliding window expansion.

---

## 3. Team Member Roster & Confidence Matrix

Every member should know the overarching narrative and have specialized mastery in their primary domain:

| Member Name | Roll No. | Primary Module | Specialized Viva Domain |
|---|---|---|---|
| **Debnil Dey** | 19 | Module 1 & 2 Lead | MLP Architecture, BCE loss pos_weight, Cross-Hospital OOD Protocol |
| **Arnav Sharma** | 12 | Module 1 Associate | Activation functions (GELU vs ReLU), Optimizers (AdamW vs Adam) |
| **Koustav Dey** | 29 | Module 2 Lead | BatchNorm vs LayerNorm shift, Optuna Bayesian TPE, Calibration (ECE) |
| **Oaishi Saha** | 33 | Module 2 Associate | Dropout sweep, Weight decay regularization, Learning rate schedulers |
| **Sayuri Ghosh** | 56 | **Module 3 Lead** | **1D-CNN multi-scale kernels ($k=3,5,7$), Inception branch design, 105-D `SentryICUFusionModel`, Leak-free sliding windowing, Stratified patient splitting, Checkpoint export** |
| **Soumita Mondal** | 71 | **Module 3 Associate** | **1D Temporal Grad-CAM with sign preservation (hypotension/hypothermia), Channel importance weights $\alpha_k$, Clinical metric evaluation (AUPRC, Brier Score, ECE), Ablation de-confounding** |
| **Apurbo Singha** | 10 | Module 4 Lead | GRU vs LSTM gating, BPTT vanishing gradients, Sequence-to-one |
| **Ayana Roy** | 15 | Module 4 Associate | GRU-D temporal decay for missing labs, BiGRU prospective data leakage |
| **Shrijeeta Das** | 61 | Module 5 Lead | PatientVAE architecture, ELBO loss, Anomaly score reconstruction |
| **Tithi Roy** | 89 | Module 5 Associate | Reparameterization trick, Cross-hospital drift detection, Tabular GAN |

---

*CONFIDENTIAL — SentryICU Complete Team Viva & Slide-by-Slide Defense Manual*

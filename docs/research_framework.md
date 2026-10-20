# Research framework: SEM microstructure → suture retention strength

Methods and plan only. Results, lot and sample identifiers and internal data
locations are deliberately kept out of this repository.

**Why lot-aware evaluation:** samples from the same production lot share
microstructure and strength. A random sample-level split puts the same lots
in both training and test, so a model can score well just by recognising the
lot. Every model here is judged first on **leave-one-lot-out** CV: does it
predict strength for a lot it has never seen?

## 3. Research questions

| ID | Question | Primary evidence |
|---|---|---|
| **RQ1** Transfer | Do SEM features predict strength for **unseen lots** better than a training-mean baseline? | Nested leave-one-lot-out (LOLO) MAE / R², lot-level error |
| **RQ2** Within-lot | Do SEM features explain **within-lot** variation? | Within-lot Spearman, mixed-model fixed effects with a lot random intercept |
| **RQ3** Feature selection | Is there a **small, stable, lot-robust** feature set that does as well as all 48? | Selection stability across lot-folds, LOLO performance against set size |
| **RQ4** Model class | Do NNs (tabular or image-based) add value over linear and tree models **under LOLO**? | Paired fold-wise differences against the best classical model |
| **RQ5** Tail / heterogeneity | What separates the high-variance lots, and can uncertainty be flagged? | Conformal interval coverage, error on the upper tail of strength |
| **RQ6** Multi-outcome | Do other mechanical outcomes (45°/heel/toe suture, tensile, burst) share an image signal? | Multi-task LOLO, after adjudicating the evidence |

---

## 4. Evaluation protocol (shared by all tracks)

1. **Outer loop:** leave-one-lot-out (16 folds) as the primary scheme. Repeated GroupKFold(4–5) as a
   secondary check. Shuffled CV is reported only as a "lot-recognition" reference.
2. **Inner loop:** grouped CV inside each training fold for every data-adaptive step: imputation,
   variance/missingness filters, monotonic sign learning, feature selection, hyperparameters and early stopping.
3. **Baselines in every table:**
   - Training-fold mean and median (deployable floor).
   - **Oracle lot mean** (not deployable): the ceiling on what lot recognition can achieve.
   - Report's monotonic RF settings, rerun inside the same folds.
4. **Metrics:**
   - Sample level: MAE, MedAE, RMSE, R², Spearman.
   - Lot level: error of predicted versus measured lot median.
   - Within-lot Spearman.
   - Tail MAE for samples above the 90th percentile.
   - Interval coverage and width from split-conformal or CQR, calibrated per lot fold.
5. **Statistics:** compare models on fold-paired differences with a lot-level bootstrap. Pick one primary
   metric in advance (suggested: LOLO MAE). Report every configuration tried, not just the best.
6. **Noise ceiling:** use repeated suture-retention observations in `collected_measurements.csv`
   (after adjudication) to estimate test–retest noise and the maximum achievable R².
7. **Sensitivity analyses:** exclude flagged images or weak `linkage_grade`, use log versus raw target,
   drop the two high-variance lots, and run inner-only and outer-only versions.
8. **Final confirmation:** fresh lots only. Lots already used for model selection don't count as
   independent confirmation.

---

## 5. Track A: Feature selection

**Goal:** find compact, interpretable, *lot-robust* feature sets. Every step runs inside the training folds.

| Step | Method | Notes |
|---|---|---|
| A1 Structure | Group the 48 features by **view** (inner/outer), **family** (IND, tortuosity, fibrilLength, fibrilIndRatio, opening ×4, scalars) and **moment** (mean/std versus skew/kurtosis) | Ablations: view-only, family-drop, means and SDs only. Higher moments are unstable when few structures are detected |
| A2 Redundancy | Hierarchical clustering on \|Spearman\|. Keep one representative per cluster, or the cluster's first PC | Handles highly correlated feature pairs. Report importance at cluster level |
| A3 Filter | Fold-wise Spearman / mutual information / **mRMR** | Fast baseline ranking |
| A4 Embedded | Elastic net, **group lasso** (groups = family or view), sparse PLS | Group lasso matches the measurement structure |
| A5 Wrapper | RFE / sequential selection scored by **inner grouped CV** | Never score by shuffled CV |
| A6 Stability | **Stability selection**: lasso or RFE on lot-bootstrap resamples, keep features chosen in ≥60–70% | Plot stability against LOLO error |
| A7 Lot-invariance screen | Within each training fold, compute η² by lot and within-lot ρ. Penalise or remove features that are mostly lot identity | Proposed new criterion: favour features with within-lot signal (RQ2) |
| A8 Attribution | Held-out **grouped permutation importance** (permute whole families) and SHAP on out-of-fold predictions | Replaces training impurity importance |
| A9 Extended features | `image_statistics.csv`: per-image n/min/max per metric. Counts (n) and ranges may capture sampling reliability | Add only through the same nested protocol |

**Deliverable:** for each selection method, a curve of LOLO MAE against number of features, plus a
selection-frequency heatmap across lot folds.

---

## 6. Track B: Model families

Ordered by how well each suits a few hundred samples from a few dozen lots.

### B1. Interpretable and statistical
- **Mixed-effects regression** (lot random intercept, with random slopes as an option). Separates the
  within-lot effect (RQ2) from the between-lot effect, so it is the standard way to test whether tortuosity
  matters within a lot. **MERF / GPBoost** add a tree fixed effect on top of a random lot effect.
- **Monotone GAMs / Explainable Boosting Machines.** Shape functions per feature with optional monotone
  constraints. These replace hand-drawn response curves with fitted ones.
- **Two-stage lot model.** Predict the lot median from lot-aggregated image features (n = 16, needs strong
  regularisation), then predict within-lot deviation separately. This makes the lot-QC framing explicit.

### B2. Classical ML
- Ridge / elastic net / PLS on log target (currently the strongest under LOLO).
- RF, ExtraTrees, LightGBM / XGBoost with and without monotonic constraints, comparing fixed signs
  against fold-learned signs.
- Gaussian process regression (ARD kernel). Gives native uncertainty and works well at small n.
- **Quantile regression forests / CQR** to model the heavy upper tail and flag high-uncertainty samples.

### B3. Neural networks on tabular features
Expect little gain at this sample size. Include them to answer RQ4 honestly.
- **TabPFN** (prior-fitted transformer for small tabular data). The strongest NN candidate at n < 1000
  and needs no tuning.
- Small MLP: 1–2 layers, strong weight decay and dropout, early stopping on an inner grouped fold, a
  5–10-seed ensemble, log target.
- **Monotone NNs**: lattice / Deep Lattice Networks (TF-Lattice) or min-max networks. These keep the
  report's monotonic interpretability with a smoother response.
- **Two-tower view fusion**: separate inner and outer encoders combined late, since the two views correlate only weakly.
- Optional **domain-adversarial** or **GroupDRO** training with lot as the domain, to suppress lot-identity
  features (links to A7).

### B4. Deep learning on the SEM images (main NN track)
Uses the full-resolution TIFFs (original files only, hash-verified; previews must not be used).
- **Frozen embeddings → linear head**: tile each image (e.g. 224–518 px tiles at fixed nm/px) and embed with
  ImageNet CNNs, **DINOv2** and a microscopy-pretrained backbone (e.g. MicroNet). Mean-pool or
  attention-pool per view, then ridge or GP. Cheap, low variance, best first step.
- **Multiple-instance learning**: attention MIL over tiles from both views, with the sample as the bag.
  Attention maps show which regions matter.
- **Self-supervised pretraining** (SimCLR / DINO / MAE) on all available unlabelled SEM tiles, followed by
  linear probing.
- **Fine-tuning** only with heavy augmentation (rotation and flips if fibril orientation doesn't matter
  physically, contrast/brightness jitter) and LOLO-based early stopping.
- **Hybrid**: embeddings concatenated with the handcrafted 48 features to test whether the handcrafted set misses information.
- **Main risk:** CNNs pick up lot or session signatures (coating, contrast, charging) more easily than
  handcrafted features. Mitigations: per-image intensity normalisation, lot-adversarial heads, LOLO
  evaluation only, and a **"predict lot from image" probe** to measure how much lot identity is encoded.
  Saliency checks should confirm the model attends to microstructure, not to masked labels or calipers.

### B5. Multi-task and auxiliary targets (RQ6)
- Jointly predict straight-across, 45°, heel and toe suture retention plus tensile and burst outcomes,
  using shared encoders and auxiliary losses.
- **Requirement:** adjudicate source, unit, statistic role and review status first. Many rows are
  `machine_extracted_not_reviewed` or OCR. Don't pool reported means with observations or treat
  duplicate source rows as replicates. Keep `linkage_scope` and `association` as covariates.

---

## 7. Phased plan and decision gates

| Phase | Work | Exit criterion / decision |
|---|---|---|
| **0. Harness** (~1–2 wk) | Nested LOLO pipeline, baselines (including oracle lot mean), reproduce the report's RF under the old split, noise-ceiling estimate, sensitivity flags | Reproduces the original report's holdout result on its fixed split |
| **1. Diagnose** | Mixed-effects models (RQ2), lot-level analysis (n = number of lots), "predict lot from features" probe, close look at the high-variance lots (images, flags, test records) | **Gate 1:** any within-lot signal? If not, reframe the goal as lot-level QC and focus on getting more lots |
| **2. Feature selection** (Track A) | Stability, group lasso, lot-invariance screen, family ablations | A set of ≤10 features with LOLO MAE no worse than all 48, chosen in ≥70% of folds |
| **3. Tabular models** (B1–B3) | GP, EBM, monotone GBM, TabPFN, MLP ensemble, CQR | **Gate 2:** a model beats the training mean on LOLO MAE (paired bootstrap CI excludes 0) |
| **4. Image deep learning** (B4) | Frozen embeddings → MIL → SSL. Lot-probe and saliency audits | Adds over the best tabular model on LOLO *and* doesn't increase lot-probe accuracy |
| **5. Multi-task + uncertainty** (B5, RQ5) | After evidence adjudication | Calibrated intervals with ~90% coverage on held-out lots |
| **6. Confirmation** | Pre-registered model on **fresh lots** | Single pre-specified primary metric |

## 8. Data recommendations
The biggest constraint is the number of lots, not the number of samples or the model class.
- **More lots** matter more than more samples per lot. Power for RQ1 grows with lot count; aim for 30–40+.
- **Several SEM fields per side** per sample. One field per side gives a noisy estimate of that sample's
  microstructure and weakens the within-lot signal.
- **Image the actual test coupon**, or record coupon ↔ SEM-field location. Current linkage is by sample
  number only.
- Record process and acquisition metadata (session, operator, coating, detector settings) so lot effects
  can be broken into explainable parts.


# Physics-informed digital twins coupling tumor biomechanics with multiphasic imaging predict personalized hepatocellular carcinoma trajectories across cohorts

PCNO-DT maps baseline multiphasic liver imaging to continuous tumor trajectories. The model joins a 3D ResNet-26 hemodynamic branch, a continuous-time DeepONet trunk, Fisher–KPP reaction–diffusion dynamics, linear poroelastic mechanics, stress-modulated proliferation, and split-conformal uncertainty calibration. Training has a synthetic FEM stage followed by clinical fine-tuning on HCC-TACE-Seg. ATLAS is held out for cross-site assessment.

## Installation

The reported environment is Python 3.11, PyTorch 2.3.1, CUDA 12.4, and cuDNN 9.

```bash
conda env create -f environment.yml
conda activate pcno-dt
python -m pip install -e .
```

Docker requires an NVIDIA Container Toolkit installation and a CUDA-capable host.

```bash
docker build -t pcno-dt:1.0 .
docker run --gpus all --shm-size=32g -v "$PWD/data:/workspace/data" pcno-dt:1.0 --stage synthetic
```

## Data

Verified source pages, licenses, and access terms are listed in `datasets.txt`. HCC-TACE-Seg is the primary cohort and requires pre-contrast, arterial, portal venous, and delayed CT volumes plus liver and tumor masks. ATLAS is an external CE-MRI cohort and must not enter hyperparameter selection.

The expected patient layout is:

```text
data/hcc_tace_seg/
  patient_identifier/
    pre.nii.gz
    arterial.nii.gz
    portal.nii.gz
    delayed.nii.gz
    liver.nii.gz
    tumor.nii.gz
    trajectory.npz
```

Create the manifest after entering time-to-progression and event values from the approved clinical table:

```bash
pcno-prepare --root data/hcc_tace_seg --output data/hcc_tace_seg/manifest.csv
```

Volumes are rigidly registered before use, resampled to 1 mm isotropic spacing, cropped to the liver, and converted into three enhancement-ratio channels plus the raw four-phase stack. Patient identifiers remain local and are not written to logs.

## Synthetic trajectories

Create the 10,000-case FEBio parameter registry:

```bash
pcno-synthesize --output data/febio_registry --count 10000 --seed 0
```

Each case spans 180 days at one-day intervals on a tetrahedral liver mesh of approximately 50,000 elements. The sampled priors are diffusion 0.01–1.0 mm²/day, proliferation 0.001–0.1 day⁻¹, tumor Young's modulus 3–50 kPa, liver Young's modulus 0.4–6 kPa, Poisson ratio 0.45–0.49, and arterial fraction 0.1–0.9. FEBio outputs are rasterized to `images`, `liver_mask`, `tumor_mask`, `coordinates`, and `density` arrays in compressed NumPy archives.

## Training

Synthetic pre-training uses a global batch of 128 across four A100 40 GB GPUs, Adam at 1e-3, weight decay 1e-5, 5,000 warm-up steps, cosine decay, gradient clipping at 1.0, 50,000 collocation points per step, and 200 epochs.

```bash
torchrun --standalone --nproc-per-node=4 -m pcno_dt.cli.train --config configs/main.yaml --stage synthetic --synthetic-root data/synthetic --seed 0 --precision bf16
```

Clinical fine-tuning freezes the stem and first two residual stages, uses batch size 8 on one A100, lowers the learning rate to 1e-4, and stops on validation volume MAE with patience 20. Run every fold with seeds 0, 17, 42, 1729, and 65535. A complete synthetic stage takes about 72 hours on four A100 40 GB GPUs. Each clinical fold takes about eight hours on one A100.

```bash
pcno-train --config configs/main.yaml --stage clinical --seed 0 --precision bf16
```

The composite objective contains density error, a segmentation term weighted by 0.1, and reaction–diffusion, mechanics, and coupling residuals initialized at unit weight. Physics weights are rebalanced every 500 optimization steps. Model state, optimizer state, scheduler state, seed, and random-generator state are saved atomically.

## Evaluation

HCC-TACE-Seg uses five-fold cross-validation stratified by BCLC stage and tumor-size tertile. Each fold reserves 20% of its training partition for early stopping and conformal calibration. Report means over 25 runs. Primary outcomes are trajectory C-index, volume MAE, Dice, time-to-progression MAE, and normalized PDE residual. Pairwise comparisons use paired t-tests with Bonferroni correction across 13 baselines and Wilcoxon signed-rank confirmation. C-index confidence intervals use 10,000 bootstrap samples.

```bash
pcno-evaluate --proposed artifacts/pcno_cindex.csv --baseline artifacts/xgb_cindex.csv --metric c_index --bootstrap-samples 10000 --output outputs/main/c_index.json
```

Expected primary-cohort values are C-index 0.783 ± 0.024, volume MAE 2.91 ± 0.38 mL, Dice 0.901 ± 0.031, time-to-progression MAE 21.3 ± 3.8 days, and normalized PDE residual 6.8e-4 ± 1.2e-4. Cross-site evaluation applies the HCC-TACE-Seg model to all 90 ATLAS cases without retraining or test-time adaptation. Cases above a normalized PDE residual of 1.2e-3 are deferred for review.

## Compute budget

The main run uses four NVIDIA A100 GPUs with 40 GB VRAM each for about 72 hours, then one A100 for about eight hours per clinical fold. Five folds and five seeds require roughly 1,000 A100-hours for clinical fine-tuning in addition to synthetic training. FEBio generation uses a 32-core AMD EPYC Milan node with 64 GB RAM. A classical solve takes a median 4.2 hours per patient. Model evaluation takes about 0.83 seconds per patient on one A100, including imaging I/O. Storage depends on mesh and raster settings; reserve at least 2 TB for 10,000 FEM trajectories and intermediate volumes.

## Configuration

`configs/main.yaml` records all reported architectural, physics, optimizer, scheduling, validation, and hardware values. Command-line paths may be changed, but the numerical defaults should remain fixed for reported experiments. The branch has stages `(2, 2, 2, 2)`, initial width 128, coefficient rank 256, and six bounded biomechanical outputs. The trunk has six layers of width 512 and 16 harmonics for each of the 30-, 90-, and 180-day base periods.

## Failure handling

The evaluation protocol inspects multifocal disease, metal artifacts from TACE coils, portal-vein thrombosis, infiltrative tumor margins, atypical post-treatment enhancement, respiratory motion, and extrahepatic involvement. These conditions can invalidate an imaging boundary condition or the closed liver-domain assumption. The residual threshold is an abstention criterion, not a replacement for clinical review.

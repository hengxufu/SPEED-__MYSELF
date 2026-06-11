# Research Method and Reproducibility

## 1. Problem Definition

The model estimates the six-degree-of-freedom pose of a known non-cooperative
spacecraft from a monocular image. Synthetic SPEED+ images provide source
supervision, while Lightbox and Sunlamp images represent target domains with
different illumination and sensor characteristics.

The method focuses on reducing the synthetic-to-HIL domain gap without using
target pose or keypoint labels in the adaptation loss. The evaluation and
adaptation pipelines use the benchmark target bounding boxes for deterministic
cropping. This assumption must be retained when comparing the reported values.

## 2. Synthetic-Domain Estimator

The input ROI is resized to 224 x 224. Swin-Tiny extracts hierarchical
features, and an FPN fuses stages 1, 2, and 3. The decoder predicts 11 heatmaps
at 56 x 56 resolution. A differentiable soft-argmax with beta 100 converts
each heatmap into a normalized 2D keypoint.

The training objective combines heatmap cross-entropy/KL supervision and a
coordinate auxiliary loss with weight 0.2. Keypoints 3, 4, 5, 6, and 10
receive a 1.5 hard-keypoint weight. The model is optimized with AdamW, a
five-epoch warm-up, and cosine decay.

Training set: 47,966 synthetic images. Validation set: 11,994 synthetic
images. The best thresholded SPEED score occurs at epoch 72.

## 3. Pose Recovery and Geometry Audit

Predicted 2D keypoints and the known 3D Tango keypoint model are passed to
PnP/RANSAC. A prediction is accepted as a pseudo-label candidate only when:

| Filter | Setting |
| --- | ---: |
| Minimum RANSAC inliers | 6 |
| Maximum median inlier reprojection error | 8 px |
| Valid translation norm | 0.5 to 20.0 m |
| Minimum valid reprojected keypoints | 6 |

An accepted pose is used to reproject the 3D model. These reprojected
keypoints, rather than the original noisy network peaks, supervise the target
update. This makes the pseudo-labels mutually consistent with a single rigid
pose.

## 4. Lightweight Iterative Pre-Adaptation

The synthetic checkpoint initializes both teacher and student. The teacher
generates target keypoints and remains frozen during an epoch. The student is
updated only on geometry-accepted samples. At the end of an epoch, the student
can replace the teacher for the next iteration.

Trainable parameters are restricted to the FPN, neck/decoder, heatmap head,
and backbone normalization parameters. In the final configuration this is
approximately 1.76M of 29.26M parameters, or 6.02%.

The final schedule uses two epochs at learning rate 1e-5 followed by one
fine-tuning epoch at 5e-6.

## 5. Results

### Final HIL Results

| Domain | Method | SPEED | Mean KP RMSE | Median eT | Median eR | RANSAC fail |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Lightbox | Synthetic baseline | 0.9076 | 39.61 px | 0.5807 m | 17.9567 deg | 51.57% |
| Lightbox | Final adaptation | **0.5696** | **24.74 px** | **0.3191 m** | **6.8619 deg** | **32.69%** |
| Sunlamp | Synthetic baseline | 1.0180 | 46.72 px | 0.6268 m | 26.2371 deg | 68.36% |
| Sunlamp | Final adaptation | **0.8794** | **38.67 px** | **0.5671 m** | **17.4310 deg** | **57.69%** |

### Ablation Interpretation

Iterative teacher refresh and normalization-layer adaptation consistently
improve Lightbox. Sunlamp remains more difficult because severe illumination
effects reduce the number and reliability of geometry-accepted samples.

Relaxing Sunlamp acceptance thresholds increases pseudo-label coverage from
24.72% to 45.15%, but worsens the SPEED score from 0.8794 to 0.9495. This
shows that pseudo-label quantity is not a sufficient objective: pose
consistency and scale accuracy matter more than raw acceptance rate.

Detailed values are stored in
[`../research/results/domain_preadapt_summary.csv`](../research/results/domain_preadapt_summary.csv).

## 6. Result Provenance

- Synthetic curves and summary statistics were compiled from the 75-epoch
  validation log.
- HIL rows were compiled from full-domain evaluations: 6,740 Lightbox images
  and 2,791 Sunlamp images.
- Published SPEED+ rows were transcribed from Tables 3 and 4 of the SPEED+
  paper for context.
- The committed CSV/JSON files are the curated, paper-facing records. Raw
  logs and checkpoints are excluded because of size.

## 7. Reproducibility Commands

```powershell
# Prepare bbox-conditioned target-domain files.
python research\scripts\prepare_target_dataset.py `
  --dataroot <SPEED_PLUS_ROOT> --outroot work\target_dataset

# Two-stage final adaptation.
.\research\scripts\run_target_preadapt.ps1 `
  -DataRoot work\target_dataset `
  -BaseCheckpoint <SYNTHETIC_CHECKPOINT> `
  -Domain lightbox

# Full-domain evaluation.
.\research\scripts\evaluate_heatmap_domain.ps1 `
  -DataRoot work\target_dataset `
  -Checkpoint outputs\preadapt\lightbox\stage2\model_adapted.pth.tar `
  -Domain lightbox
```

The result compiler expects raw metric JSON files under `outputs/domain_eval`
and adaptation histories under `outputs/preadapt`:

```powershell
python research\scripts\compile_final_results.py
```

## 8. Limitations and Submission Claims

1. Target bounding boxes are used for cropping. The method is not an
   end-to-end detector-plus-pose system.
2. Target pose/keypoint labels are excluded from the adaptation loss but are
   used for evaluation and for preparing benchmark-compatible CSV records.
3. Published SPEED+ values provide context, not a strict ranking, because
   model implementations and evaluation details differ.
4. Sunlamp performance remains limited by specular highlights, shadows, and
   low pseudo-label acceptance.
5. A multi-seed study and runtime/memory analysis should be added before a
   strong archival submission claim.

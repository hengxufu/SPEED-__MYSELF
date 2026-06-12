# Experiment Result Records

This directory contains the curated tables used by the repository README and
paper figures.

| File | Contents |
| --- | --- |
| `synthetic_key_epochs.csv` | Selected epochs from the 75-epoch synthetic validation run |
| `synthetic_experiment_summary.json` | Best-epoch and final-epoch synthetic metrics |
| `domain_preadapt_summary.csv/json` | Baseline, ablation, and final HIL metrics |
| `speedplus_horizontal_comparison.csv/json` | Contextual comparison with published SPEED+ tables |
| `spacecraft_uda_preadapt_summary.csv/json` | Multi-round Spacecraft-UDA-inspired pre-adaptation results |

Raw logs, generated target CSVs, datasets, and checkpoints are intentionally
not committed. `NaN` pseudo-label fields in baseline rows mean that no target
pre-adaptation was performed.

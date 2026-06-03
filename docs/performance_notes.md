# Performance Notes

The local machine has an NVIDIA GeForce RTX 3080 Laptop GPU with 16 GB VRAM. `nvidia-smi` reports driver CUDA 13.0, and the project environment installs `torch==2.12.0+cu130`.

## Current Smoke Profile

Command:

```powershell
conda run -n cpgdetector python -m cpgdetector.profile --config configs/smoke.yaml --batches 10
```

Observed on the smoke model:

- Peak allocated GPU memory: about 71 MB
- Peak reserved GPU memory: about 92 MB
- Dataset build time: about 0.7 s
- Data loading fraction: about 4%
- Throughput: about 277 windows/s

Interpretation: smoke training is compute/model-size limited rather than DataLoader-limited, and it underuses the GPU by design. The full config uses a wider model and larger batch size. Prediction uses a separate large batch size because it is inference-only and can safely use more VRAM.

## Practical Tuning

- Increase `training.batch_size` until GPU memory reaches a useful range without OOM; the default full config is intended for server GPU training and uses batch size 2048.
- On Linux/A100, use `training.num_workers: 8`, `persistent_workers: true`, and `prefetch_factor: 4`. On Windows or memory-constrained machines, reduce `training.num_workers` to 0.
- The training DataLoader now returns encoded integer windows and creates one-hot tensors in a batch-level collate function. This avoids per-sample one-hot work in `Dataset.__getitem__`.
- Validation threshold metrics now stay on GPU through a histogram/threshold-grid accumulator. ROC-AUC and PR-AUC during training are approximate histogram estimates; final ROC/PR curve plots still use exact sklearn curves on collected scores.
- `torch.compile` is enabled for the full config and disabled for smoke runs. Compile has a large first-iteration cost but can pay off in long A100 training runs.
- The full config uses warmup + cosine learning-rate decay. Warmup reduces early instability with large A100 batches and mixed precision; cosine decay lowers the step size late in training without requiring manual milestones.
- Increase `prediction.batch_size` aggressively for full-chromosome prediction; 2048 is the default.
- If GPU utilization remains low, increase model channels, add more dilated blocks, or increase window length to 1024.
- If CPU becomes the bottleneck, the next optimization target is vectorized batch one-hot generation and cached mask arrays for chromosomes used in training.

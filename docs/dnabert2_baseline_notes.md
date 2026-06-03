# DNABERT2 Baseline Notes

This project now includes a DNABERT2 fine-tuning baseline adapted from the training pattern in `WeitangSun/CpG_transformer`, but it does not reuse that repository's data preparation implementation.

References:

- `WeitangSun/CpG_transformer`: https://github.com/WeitangSun/CpG_transformer
- DNABERT2 model used there: `zhihan1996/DNABERT-2-117M`
- Their training script: `train_cpg.py`
- Their data script: `prepare_cpg_data.py`

## What Was Adapted

The adapted baseline follows the DNABERT2 fine-tuning structure:

- load `zhihan1996/DNABERT-2-117M` with `trust_remote_code=True`
- attach a 2-label sequence classification head
- tokenize raw DNA sequences with the DNABERT2 tokenizer
- train through HuggingFace `Trainer`
- select the best checkpoint by validation F1
- report accuracy, precision, recall, F1, MCC, ROC-AUC, and PR-AUC

The entry point is:

```bash
python -m cpgdetector.dnabert2_baseline --config configs/default.yaml
```

It consumes our existing `CpGWindowDataset` rather than generating `train.csv`, `dev.csv`, and `test.csv`.

## Dataset Difference

`CpG_transformer` defines CpG prediction as a sequence-level binary classification task. Its documented data flow extracts positive windows centered on CpG island midpoints and samples random non-overlapping genomic regions as negatives. It then writes plain CSV files with `sequence,label`.

Our primary model defines the task as base-level segmentation with an auxiliary window head. A sample is a fixed-size genomic window with:

- per-base CpG island mask
- CpG fraction over the window
- binary `has_cpg` target derived from whether the window overlaps any CpG island

The DNABERT2 baseline in this project therefore uses our fixed windows and trains only against the window binary target. It does not predict base-level masks.

## Expected Impact

The DNABERT2 baseline is not directly comparable to the CNN base head. It is comparable to the CNN window head and to the traditional/logistic window baselines.

Centering positives on island midpoints can make the classification task easier because the discriminative region is usually near the middle of the sequence. Our sliding windows include partial islands, boundaries, and windows with small CpG fractions; this is closer to the segmentation setting but usually harder for a pure sequence classifier.

Random negative sampling can inflate AUC if negatives are GC-poor and far from island boundaries. Our training sampler includes hard negatives near island boundaries and GC-rich negatives, so DNABERT2 performance under this adapter may be lower than the reported numbers in `CpG_transformer`, but the comparison is more relevant to our use case.

Chromosome-level splitting remains important. Both approaches avoid random window-level leakage across neighboring loci, but our split is controlled by `configs/*.yaml`, while `CpG_transformer` documents a fixed train/dev/test chromosome split.

Because DNABERT2 produces one score per input window, interval prediction still requires sliding-window inference and post-processing. It cannot replace the base segmentation output without adding a token/base alignment head, which is a different model design.

## Practical Notes

For A100 training, start with:

- `batch_size: 32` to `64`
- `eval_batch_size: 64` to `128`
- `fp16: true`
- `num_workers: 4` or higher
- `model_max_length: 128` for 512 bp windows

If the baseline overfits quickly, reduce `epochs`, increase `weight_decay`, or select by PR-AUC instead of F1 when positives are sparse.

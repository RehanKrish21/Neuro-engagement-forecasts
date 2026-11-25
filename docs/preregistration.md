# Preregistration: run 2 and later (SnapUGC, ECR)

**Written 2026-11-24, before any run-2 data were collected.** The git commit that adds this file is the timestamp. Any change after run 2 starts will be listed at the bottom with a date and a reason.

## Background
Run 1 (2026-11-23) analysed 135 SnapUGC training videos (see the README's *Results so far*). It was exploratory. Brain features alone predicted ECR on held-out videos (Spearman 0.29, 95% CI 0.13–0.44) about as well as metadata (0.35), and a single combined ridge model did not improve on the baseline. *After* seeing those results, a fixed 50/50 rank average of the baseline and brain models looked better (0.38 vs 0.35; gain CI −0.07 to +0.14). Because that comparison was chosen after looking at the data, it has to be tested on new videos.

## Question
Do TRIBE v2-predicted brain responses add predictive value for short-video hook retention **beyond simple metadata**?

## Data
- SnapUGC `train_data.csv`; outcome **ECR** (share of viewers who watch ≥ 5 s). NAWP is absent from this file.
- Videos: stratified sample over ECR (10 quantile bins, seed 0), from links that still download.
- Videos are accumulated across Kaggle runs until **at least 700 videos in total**, or **three more runs**, whichever comes first. The primary analysis is run **once**, at that point, and reported whatever it shows. Intermediate runs are reported as progress, without claims.
- TRIBE v2 checkpoint `facebook/tribev2`; precision fp32 unless the in-notebook check (all networks r ≥ 0.98 vs fp32, ≥ 1.3× faster) selects fp16. The precision used is reported.

## Models (fixed)
- `baseline`: ridge on `duration_s`, `title_n_words`, `has_description`.
- `brain`: ridge on all 98 network-summary features (10 networks + global + pattern shift + 2 composites; mean, std, hook (first 5 s), hook vs rest, slope, late vs early, peak time).
- `avg(baseline,brain)`: 50/50 average of the two models' ranks; nothing fitted.
- Secondary: `brain_compact` (mean + hook statistics only), `baseline+brain`, `baseline+brain_compact`, `avg(baseline,brain_compact)`.
- 5-fold shuffled cross-validation (seed 0); ridge alpha chosen internally by RidgeCV.

## Primary test
- **Comparison:** `avg(baseline,brain)` vs `baseline`.
- **Metric:** pooled out-of-fold Spearman with ECR. The gain is computed with a paired bootstrap over videos (2,000 resamples, seed 0).
- **Evaluated on confirmatory videos only:** videos **not** in run 1 (IDs in `src/neuroengage/data/snapugc_run1_ids.txt`). Models still train on all videos.
- **Decision rule:**
  - **Supported** if the 95% CI of the gain is entirely above 0.
  - **Not supported** if the CI is entirely ≤ 0.
  - **Inconclusive** otherwise. The estimate and CI are reported either way.

## Secondary (reported, not used for the main claim)
- Pairwise accuracy (share of video pairs ordered correctly; chance 0.5) with 95% CIs, for every model.
- `brain` vs `baseline` (does brain alone match or beat metadata?).
- The same metrics on all videos, including run 1.
- `brain_compact` and combined-model variants.

## Not changing after seeing data
The outcome, primary comparison, feature definitions, baseline columns, CV scheme, bootstrap settings and decision rule above.

## Deviations and additions
- **2026-11-24 (before run 2 started), exploratory addition, not a change to the primary test:** a *raw-feature baseline*. TRIBE's own input features (V-JEPA 2 video, Wav2Vec-BERT audio, Llama 3.2 text) are recorded during extraction and summarised as whole-video and first-5-s averages. They are fed to a ridge model with a wide regularisation grid (alpha 1 to 10^8), using the same 5-fold CV and bootstrap. Reported: `raw` vs `brain`, per-modality raw sets, and a 50/50 rank average of raw and brain. It covers only videos processed with v0.8.0+, so not run 1. It is labelled exploratory in every report.
- **2026-11-24 (before run 2 started), data source, not an analysis change:** about two-thirds of SnapUGC's official video links are dead, so video files for dead links are fetched from an unofficial Hugging Face copy of the SnapUGC videos (`tcm03/SnapUGC_0..3`, `tcm03/SnapUGC`), matched by SnapUGC video ID. Spot checks: 121,506 IDs, full-length videos (5 to 55 s), all 135 run-1 IDs present. ECR labels still come only from the official `train_data.csv`. This widens which videos can be sampled, which should *reduce* any bias from which links happened to survive. Each video's source (`ok` official, `ok_mirror` copy) is recorded in `_download_log.csv`.

## Progress log
- **2026-11-25, run 2 (interim, 531 videos; 396 confirmatory):** primary comparison `avg(baseline,brain)` vs `baseline` on confirmatory videos: Spearman 0.219 vs 0.091, gain +0.128 [0.065, 0.189], P(gain > 0) = 1.00. The interval is entirely above 0, but the sample is below the pre-registered 700-video analysis point, so this is **interim**. The primary test will be run once more at ≥ 700 videos and reported whatever it shows. Files: `docs/results/run2_*`.

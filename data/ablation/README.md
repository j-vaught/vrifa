# VRIFA ablation settings

This folder keeps the per-video settings that were selected from the ablation sweeps. The input `1-11` settings are restored from the latest historical ablation result before cleanup commit `87b0fd8`, which removed the larger ablation outputs from `main`. The input `12-14` settings come from the ROI95 sweep against the newly labeled frames.

The file `input_1_11_per_video_best.json` contains the historical selected settings for inputs `1-11`. The file `input_12_14_per_video_best.json` contains the selected settings for inputs `12-14`, with `input_13` intentionally using the ellipse morphology variant. Input `15` is excluded.

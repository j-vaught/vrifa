# Tuning, ablation methodology, and segmentation metrics

VRIFA's 91-trial ablation sweeps a mixed parameter space (colorspace, threshold offset, blur and morphology kernels, minimum component area, lock-frames) under a multi-metric objective. This section surveys how to evaluate segmentations on multiple metrics, tune classical pipelines, and report ablations.

## Multi-metric segmentation evaluation

Single-metric reporting is increasingly viewed as inadequate. Maier-Hein et al.'s "Metrics Reloaded" [1] recommends matching metrics to the task and reporting region-overlap, boundary, and counting metrics jointly because each is blind to a different failure mode. Reinke et al.'s companion piece [2] catalogs how Dice and Jaccard/IoU can be inflated by tiny or large objects while boundary errors remain invisible. Cityscapes [3] popularized iIoU for unbalanced class scales, and COCO panoptic [4] standardized PQ as a region+counting hybrid. Hausdorff distance [5] and average symmetric surface distance from MICCAI benchmarks [6] complement overlap with worst-case and mean boundary error; VRIFA's mean-boundary-distance term sits in this family.

## Boundary-aware metrics and why they matter for front localization

Region overlap saturates when an object is mostly segmented but its boundary is shifted by a few pixels. Csurka et al.'s boundary F-measure [7] formalizes a boundary-precision/recall match within a small tolerance and is the metric VRIFA reports as "boundary F1." Cheng et al.'s Boundary IoU [8] showed that mask-only metrics under-penalize errors on large objects' boundaries. DAVIS [9] adopted contour-accuracy F, and trimap evaluation [10] restricts scoring to a band around the ground-truth boundary. For a *resin-front* localization task the quantity of physical interest is the wet/dry interface position; an IoU of 0.95 with a uniformly inflated mask hides exactly the error the user cares about. Boundary F1 directly measures that interface error, which is why VRIFA's jump from 0.206 to 0.559 is the most physically meaningful entry in its results table.

## Hyperparameter tuning frameworks and their use on classical CV pipelines

Modern HPO frameworks make tuning small pipelines almost free. Optuna [11] popularized define-by-run search spaces with TPE [12] and pruning, and Hyperopt [13] is its closest predecessor. The same machinery applies to classical CV. Earlier work tuned image pipelines with evolutionary search: Olague's evolutionary CV [14] and Ebner's GP approaches [15] auto-designed feature detectors and color-constancy operators, and Treptow and Zell [16] evolved object-detection cascades. Brochu et al. [17] established Bayesian optimization for expensive black-box searches, and BO is now the standard backbone for image-pipeline parameter sweeps. VRIFA's Optuna study is conventional in tooling, with an unconventional mix of classical-CV knobs.

## Ablation reporting standards

Lipton and Steinhardt [18] argued that many ML papers conflate genuine ablation with confound-laden comparisons and called for parameter-by-parameter sensitivity analysis. Sturm [19] and Pineau et al.'s reproducibility checklist [20] codify expectations to report the search budget, seed policy, and per-parameter effects rather than only the best configuration. Bouthillier et al. [21] showed how variance across seeds and splits routinely exceeds reported gains, motivating multi-trial summaries.

## Tuning's effect on downstream learned detectors

Upstream-pipeline quality propagates into detector mAP. Frenay and Verleysen [22] surveyed label-noise effects, Northcutt et al. [23] showed label errors in COCO/ImageNet alter benchmark rankings, and Rolnick et al. [24] documented how moderate noise compresses achievable mAP. A tuned mask source feeding YOLO or Mask R-CNN training is therefore materially different from an untuned one.

## Positioning of VRIFA

VRIFA's 91-trial ablation reports mask IoU, Dice, boundary F1, box IoU, and mean boundary distance jointly, exactly the multi-metric posture advocated by Maier-Hein et al. [1] and Reinke et al. [2]. Reporting a single IoU would have hidden the boundary-F1 jump from 0.206 to 0.559, the metric most diagnostic for front-position error.

Tuning a classical pipeline with explicit, named knobs is a recognized methodology going back to Olague's evolutionary CV [14] and Ebner's GP-for-CV [15]. VRIFA modernizes it with Optuna-style [11] reproducible run summaries, an objective including a boundary-aware term [7, 8], and a direct line of sight to downstream-detector training [23, 24]. The combination, classical knobs plus modern HPO plus multi-metric boundary-aware evaluation, distinguishes the study from a simple grid search.

## References

[1] L. Maier-Hein, A. Reinke, P. Godau, et al., "Metrics reloaded: recommendations for image analysis validation," *Nature Methods*, vol. 21, pp. 195-212, 2024.

[2] A. Reinke, M. D. Tizabi, M. Eisenmann, et al., "Common limitations of image processing metrics: a picture story," *arXiv:2104.05642*, 2024.

[3] M. Cordts, M. Omran, S. Ramos, et al., "The Cityscapes dataset for semantic urban scene understanding," CVPR, 2016.

[4] A. Kirillov, K. He, R. Girshick, C. Rother, P. Dollar, "Panoptic segmentation," CVPR, 2019.

[5] D. P. Huttenlocher, G. A. Klanderman, W. J. Rucklidge, "Comparing images using the Hausdorff distance," IEEE TPAMI, vol. 15, no. 9, pp. 850-863, 1993.

[6] B. H. Menze, A. Jakab, S. Bauer, et al., "The multimodal brain tumor image segmentation benchmark (BRATS)," IEEE TMI, vol. 34, no. 10, pp. 1993-2024, 2015.

[7] G. Csurka, D. Larlus, F. Perronnin, "What is a good evaluation measure for semantic segmentation?," BMVC, 2013.

[8] B. Cheng, R. Girshick, P. Dollar, A. C. Berg, A. Kirillov, "Boundary IoU: improving object-centric image segmentation evaluation," CVPR, 2021.

[9] F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, A. Sorkine-Hornung, "A benchmark dataset and evaluation methodology for video object segmentation," CVPR, 2016.

[10] P. Kohli, L. Ladicky, P. H. S. Torr, "Robust higher order potentials for enforcing label consistency," CVPR, 2008.

[11] T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama, "Optuna: a next-generation hyperparameter optimization framework," KDD, 2019.

[12] J. Bergstra, R. Bardenet, Y. Bengio, B. Kegl, "Algorithms for hyper-parameter optimization," NeurIPS, 2011.

[13] J. Bergstra, D. Yamins, D. D. Cox, "Hyperopt: a Python library for model selection and hyperparameter optimization," SciPy, 2013.

[14] G. Olague, *Evolutionary Computer Vision: The First Footprints*. Springer, 2016.

[15] M. Ebner, "Evolving color constancy," *Pattern Recognition Letters*, vol. 27, no. 11, pp. 1220-1229, 2006.

[16] A. Treptow and A. Zell, "Combining adaboost learning and evolutionary search to select features for real-time object detection," CEC, 2004.

[17] E. Brochu, V. M. Cora, N. de Freitas, "A tutorial on Bayesian optimization of expensive cost functions," *arXiv:1012.2599*, 2010.

[18] Z. C. Lipton and J. Steinhardt, "Troubling trends in machine learning scholarship," ICML Debates, 2018.

[19] B. L. Sturm, "A simple method to determine if a music information retrieval system is a 'horse'," IEEE Trans. Multimedia, vol. 16, no. 6, pp. 1636-1644, 2014.

[20] J. Pineau, P. Vincent-Lamarre, K. Sinha, et al., "Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program)," JMLR, vol. 22, pp. 1-20, 2021.

[21] X. Bouthillier, P. Delaunay, M. Bronzi, et al., "Accounting for variance in machine learning benchmarks," MLSys, 2021.

[22] B. Frenay and M. Verleysen, "Classification in the presence of label noise: a survey," IEEE TNNLS, vol. 25, no. 5, pp. 845-869, 2014.

[23] C. G. Northcutt, A. Athalye, J. Mueller, "Pervasive label errors in test sets destabilize machine learning benchmarks," NeurIPS Datasets and Benchmarks, 2021.

[24] D. Rolnick, A. Veit, S. Belongie, N. Shavit, "Deep learning is robust to massive label noise," *arXiv:1705.10694*, 2017.

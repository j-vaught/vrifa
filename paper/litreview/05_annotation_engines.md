# Annotation engines and weak supervision for vision in manufacturing

## Programmatic labeling and weak supervision (Snorkel and successors)

The modern theory of programmatic supervision was crystallized by Ratner et al. in the *data programming* paper and the Snorkel system [1, 2, 3]. Rather than hand-labeling examples, domain experts write *labeling functions* (LFs) — heuristics, pattern matchers, or third-party models — whose noisy votes are denoised by a generative label model that learns each LF's accuracy without ground truth. Snorkel-DryBell extended this to industrial scale at Google [4], and Bach et al. generalized the approach to multi-task settings [5]. The core insight, that *programs can replace annotators when their errors are diverse and modelable*, is exactly the regime VRIFA inhabits. A VRIFA pipeline run is a single, highly-correlated labeling function over pixels; its tunable knobs (gate quantile, morphology kernel, peak-reference offset) play the role Ratner et al. assign to LF design. Where Snorkel denoises *across* LFs, VRIFA's quality control comes *within* a single LF via tuning against a 20-frame human subset, the 0.583 to 0.807 jump being the single-LF analog of a Snorkel accuracy estimate.

## Pseudo-labels and self-training in modern semi-supervised learning

The pseudo-label paradigm dates to Lee's 2013 note [6], where high-confidence model predictions on unlabeled data are recycled as targets. It was rehabilitated at scale by Noisy Student [7], which trained ImageNet teachers and students iteratively, and by FixMatch [8] and MixMatch [9], which combined consistency regularization with confidence-thresholded pseudo-labels. Self-training with augmentation has since become the default semi-supervised baseline [10]. VRIFA differs in that the "teacher" is not a neural network but a deterministic classical-CV pipeline; the pseudo-labels are produced *before* any training rather than during it. This makes VRIFA a *pre-training-time* analog of FixMatch — the unlabeled video is converted to supervision once, then any detector can be trained on the resulting COCO/YOLO files using ordinary supervised loops.

## Classical-CV-to-CNN bootstrapping in scientific imaging

Bootstrapping deep models on classical-CV-derived masks is widespread in microscopy and biomedicine. CellProfiler [11] and ilastik [12] established threshold-and-morphology pipelines as the default before deep learning; CellPose [13] and StarDist [14] then trained CNNs on masks that were largely originally produced by such pipelines. In materials science, MIPAR [15] and the work of DeCost and Holm on microstructure segmentation [16] follow the same pattern, and Stan et al. [17] explicitly trained U-Nets on weakly-labeled tomography. In medical imaging, Rajchl et al. (DeepCut) used bounding-box-only weak labels with a CRF refinement loop to train a CNN [18]. Agricultural plant-phenotyping work by Minervini et al. likewise begins with color-threshold masks before fine-tuning [19]. VRIFA fits squarely into this lineage but addresses VARTM resin infusion, a process whose optics (specularity, dye gradient, draping wrinkles) make naive thresholding fail without the *darken-only + peak-reference* prior that VRIFA bakes in.

## Manual and model-assisted annotation tools

The complementary paradigm is human-in-the-loop tooling. CVAT [20], Label Studio [21], COCO Annotator, V7, and FiftyOne [22] structure manual labeling and review. Active-learning frameworks select which frames a human should label next [23]. Model-assisted annotation has accelerated dramatically since Segment Anything (SAM) [24] and SAM-2 [25] enabled prompt-driven mask generation, with Roboflow and Encord wrapping such models in commercial labeling stacks. SAM-style universal segmenters are, however, weakest precisely on translucent low-contrast process imagery like resin flow, which is why a domain-specific physical-prior pipeline can still outperform a foundation-model click-prompt.

## Synthetic data as an alternative

A third route is to render labels rather than infer them. Domain randomization [26], the Synthia and GTA-to-Cityscapes work [27], and procedural-rendering pipelines for industrial inspection produce pixel-perfect masks at the cost of a sim-to-real gap. For VARTM, no validated physics renderer of resin-front optics exists, so synthetic data is not currently a substitute for video-derived pseudo-labels.

## Process-imaging-specific bootstrapping

Within manufacturing process monitoring, several groups have trained CNNs on classical-CV labels. Scime and Beuth's L-PBF melt-pool work [28] and Gobert et al. [29] used thresholded high-speed images to label anomalies, then trained classifiers. Tow-placement and AFP defect detection by Sacco et al. [30] and Schmidt et al. follow the same recipe. Thermography-based monitoring of composite cure has been auto-labeled by gradient detectors before CNN training [31]. These works share VRIFA's structure — a hand-tuned CV gate produces noisy masks, a CNN is trained — but typically do not export standard annotation schemas, do not publish the tuning ablation, and do not separate the labeling engine from the detector.

## Studies of label noise and how it propagates to detector quality

Frenay and Verleysen [32] survey label-noise taxonomies; Karimi et al. [33] specifically catalog noisy-label effects in medical segmentation; Zlateski et al. [34] show that segmentation CNNs tolerate substantial boundary noise but degrade sharply under systematic bias. For detection, Chadwick and Newman [35] quantify how bounding-box jitter affects mAP. The collective message is that *random* label noise is forgiven by deep models but *systematic* label bias is not — a direct motivation for VRIFA's tunable knobs and for measuring detector mAP as a function of pipeline configuration.

## Positioning of VRIFA

VRIFA is a domain-specific instance of "classical-CV-bootstrapped supervision". Its differentiators are (i) tunable explicit knobs whose effect on label quality is measured by the 0.583 to 0.807 ablation; (ii) a physical prior baked in, since *darken-only + peak-reference* is justified by the optics of resin wetting; (iii) export-format coverage (COCO, YOLOv5-segmentation, Darknet), where most weak-label tools target one schema; and (iv) reproducibility via `run_summary.yaml`. The principal limitation is that VRIFA does not yet quantify how downstream detector mAP responds to label-tuning, which is the natural next experiment per the label-noise literature summarized above.

## References

[1] Ratner, A., De Sa, C., Wu, S., Selsam, D., and Re, C. *Data Programming: Creating Large Training Sets, Quickly*. NeurIPS 2016.
[2] Ratner, A., Bach, S. H., Ehrenberg, H., Fries, J., Wu, S., and Re, C. *Snorkel: Rapid Training Data Creation with Weak Supervision*. VLDB 2017.
[3] Ratner, A. et al. *Snorkel: rapid training data creation with weak supervision*. VLDB Journal 2020.
[4] Bach, S. H. et al. *Snorkel DryBell: A Case Study in Deploying Weak Supervision at Industrial Scale*. SIGMOD 2019.
[5] Bach, S. H., He, B., Ratner, A., and Re, C. *Learning the Structure of Generative Models without Labeled Data*. ICML 2017.
[6] Lee, D.-H. *Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks*. ICML Workshop 2013.
[7] Xie, Q., Luong, M.-T., Hovy, E., and Le, Q. V. *Self-training with Noisy Student improves ImageNet classification*. CVPR 2020.
[8] Sohn, K. et al. *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence*. NeurIPS 2020.
[9] Berthelot, D. et al. *MixMatch: A Holistic Approach to Semi-Supervised Learning*. NeurIPS 2019.
[10] Zoph, B. et al. *Rethinking Pre-training and Self-training*. NeurIPS 2020.
[11] Carpenter, A. E. et al. *CellProfiler: image analysis software for identifying and quantifying cell phenotypes*. Genome Biology 2006.
[12] Berg, S. et al. *ilastik: interactive machine learning for (bio)image analysis*. Nature Methods 2019.
[13] Stringer, C., Wang, T., Michaelos, M., and Pachitariu, M. *Cellpose: a generalist algorithm for cellular segmentation*. Nature Methods 2021.
[14] Schmidt, U., Weigert, M., Broaddus, C., and Myers, G. *Cell Detection with Star-Convex Polygons*. MICCAI 2018.
[15] Sosa, J. M. et al. *Development and Application of MIPAR: a Novel Software Package for Two- and Three-Dimensional Microstructural Characterization*. Integrating Materials and Manufacturing Innovation 2014.
[16] DeCost, B. L. and Holm, E. A. *A computer vision approach for automated analysis and classification of microstructural image data*. Computational Materials Science 2015.
[17] Stan, T., Thompson, Z. T., and Voorhees, P. W. *Optimizing convolutional neural networks to perform semantic segmentation on large materials imaging datasets: X-ray tomography and serial sectioning*. Materials Characterization 2020.
[18] Rajchl, M. et al. *DeepCut: Object Segmentation from Bounding Box Annotations using Convolutional Neural Networks*. IEEE TMI 2017.
[19] Minervini, M., Fischbach, A., Scharr, H., and Tsaftaris, S. A. *Finely-grained annotated datasets for image-based plant phenotyping*. Pattern Recognition Letters 2016.
[20] Sekachev, B. et al. *Computer Vision Annotation Tool (CVAT)*. https://github.com/opencv/cvat, 2019.
[21] Tkachenko, M. et al. *Label Studio: Data labeling software*. https://labelstud.io, 2020.
[22] Moore, B. E. and Corso, J. J. *FiftyOne*. Voxel51, 2020.
[23] Settles, B. *Active Learning Literature Survey*. Univ. Wisconsin-Madison TR 1648, 2009.
[24] Kirillov, A. et al. *Segment Anything*. ICCV 2023.
[25] Ravi, N. et al. *SAM 2: Segment Anything in Images and Videos*. arXiv 2024.
[26] Tobin, J. et al. *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*. IROS 2017.
[27] Richter, S. R., Vineet, V., Roth, S., and Koltun, V. *Playing for Data: Ground Truth from Computer Games*. ECCV 2016.
[28] Scime, L. and Beuth, J. *Anomaly detection and classification in a laser powder bed additive manufacturing process using a trained computer vision algorithm*. Additive Manufacturing 2018.
[29] Gobert, C., Reutzel, E. W., Petrich, J., Nassar, A. R., and Phoha, S. *Application of supervised machine learning for defect detection during metallic powder bed fusion additive manufacturing using high resolution imaging*. Additive Manufacturing 2018.
[30] Sacco, C., Radwan, A. B., Anderson, A., Harik, R., and Gregory, E. *Machine learning in composites manufacturing: A case study of Automated Fiber Placement inspection*. Composite Structures 2020.
[31] Schmidt, C., Hocke, T., and Denkena, B. *Deep learning-based classification of production defects in automated-fiber-placement processes*. Production Engineering 2019.
[32] Frenay, B. and Verleysen, M. *Classification in the Presence of Label Noise: A Survey*. IEEE TNNLS 2014.
[33] Karimi, D., Dou, H., Warfield, S. K., and Gholipour, A. *Deep learning with noisy labels: Exploring techniques and remedies in medical image analysis*. Medical Image Analysis 2020.
[34] Zlateski, A., Jaroensri, R., Sharma, P., and Durand, F. *On the Importance of Label Quality for Semantic Segmentation*. CVPR 2018.
[35] Chadwick, S. and Newman, P. *Training Object Detectors with Noisy Data*. IEEE Intelligent Vehicles Symposium 2019.

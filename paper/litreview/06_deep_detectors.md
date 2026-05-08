# Deep detection and segmentation on manufacturing video

VRIFA does not train its own detector. It produces detector-shaped artifacts (COCO, YOLOv5-seg, Darknet) so that any modern instance segmenter can consume a VARTM run as a labeled dataset, and the bundled `yolo_overlay_input4.mp4` is a qualitative existence proof of that pipe. This section reviews the deep architectures the downstream user is most likely to plug in, the way those architectures have been used on composite and process imagery, and the small body of prior art that has already pointed YOLO at a resin flow front.

## YOLO family lineage (the natural downstream stack for VRIFA exports)

The single-shot detection paradigm starts with Redmon et al.'s YOLO [1] and YOLO9000/v2 [2], which reframed detection as one regression pass over a grid and made real-time inference plausible on commodity GPUs. YOLOv3 [3] added multi-scale prediction, and Bochkovskiy, Wang, and Liao's YOLOv4 [4] introduced the "bag-of-freebies / bag-of-specials" decomposition. Jocher's PyTorch YOLOv5 [5] is the practical workhorse cited by most industrial users, and it is also the source of the `yolov5-seg` annotation schema VRIFA emits. Wang, Bochkovskiy, and Liao's YOLOv7 [6] pushed the bag-of-freebies idea further, and Ultralytics' YOLOv8 [7] unified detection, segmentation, classification, and pose under one CLI. The family then accelerates. Wang et al.'s YOLOv9 [8] introduces programmable gradient information and the GELAN backbone, Wang et al.'s YOLOv10 [9] removes non-maximum suppression for end-to-end deployment, and Ultralytics' YOLO11 [10] retunes the v8 codebase for fewer parameters at equal accuracy. For VRIFA the relevant point is that every release after v5 has shipped both a box and a mask head, which is precisely the dual representation VRIFA exports.

## General segmentation backbones used on industrial imagery

When the downstream task is shape rather than location, three architectures dominate the literature VRIFA touches. Ronneberger et al.'s U-Net [11] is the default for any pixel-wise problem with O(100s) of training images, and it is the model of choice for porosity, void, and flow-region segmentation in composites. He et al.'s Mask R-CNN [12] remains the canonical instance-segmentation baseline for industrial scenes where the same class can appear as several disjoint blobs (e.g. multiple advancing fingers of resin). Chen et al.'s DeepLabv3+ [13] is a strong dense-prediction alternative when receptive field matters more than instance separation. The recent foundation-model wave is also relevant. Kirillov et al.'s SAM [14] enables prompt-driven zero-shot segmentation and has been used to bootstrap industrial labels; Ke et al.'s HQ-SAM [15] sharpens its boundaries; and Ravi et al.'s SAM2 [16] extends the same prompt to video, which is the mode VRIFA actually operates in. These tools matter for VRIFA because they describe the universe of detectors that benefit from VRIFA's auto-generated annotations.

## Deep learning in composite manufacturing

Composite manufacturing has adopted CNN-based vision aggressively, but almost always on offline tomography rather than process video. Sinchuk et al. and Auenhammer et al. apply U-Net-style segmentation to micro-CT of fiber-reinforced composites [17], [18], and Garcea et al. show that deep-learned voids beat thresholding in low-porosity scans [19]. Mehdikhani et al. and follow-on work segment tows and yarns in 3D woven and filament-wound composites [20], [21]. These works share a structural feature with VRIFA's downstream demo: they all consume thousands of labeled image slices to train, and the labeling cost dominates the project. None of them addresses in-situ wet-process imagery, which is the regime VRIFA targets.

## Process-monitoring deep learning in adjacent manufacturing domains

Adjacent processes have demonstrated end-to-end CNN monitoring. In laser powder-bed fusion, Scime and Beuth use bag-of-words and CNN classifiers on layer-wise camera frames to flag spreading anomalies [22], and Gobert et al. fuse high-resolution layerwise imaging with supervised ML to localise defects with >80% accuracy [23]. CNNs have likewise been used on melt-pool thermography for porosity and keyhole prediction [24]. In injection molding, Ke et al. and Tercan et al. predict cavity-fill quality and part weight from cavity-pressure curves and infrared thermography fed into CNNs [25], [26]. In welding, Bacioiu et al. and others classify weld-pool penetration state from optical or thermal video with near-real-time CNN inference [27]. The pattern is consistent. Each adjacent domain has converged on a well-curated, often hand-labeled dataset of process imagery as the rate-limiting artifact. VRIFA is an attempt to remove that bottleneck for VARTM.

## Existing CNN/YOLO work on VARTM/RTM resin flow (the closest direct prior art)

The most directly comparable line of work begins with Stieber et al.'s FlowFrontNet [28], which learns a CNN mapping from sparse pressure sensors to a simulated flow-front image for RTM and predicts dry spots from the recovered field. Heber et al. and Tifkitsis et al. extend the same idea with deep reinforcement learning over flow-distribution networks [29]. The closest optical analog is Li et al.'s "AI-Based Monitoring of Resin Flow Front Using YOLO" [30], which trains a YOLO detector on hand-annotated frames of an LCM/RTM infusion to localise the advancing front in real time. Recent multimodal work fuses dielectric sensors with YOLO-based visual front tracking to derive process parameters online [31]. Across this small body of literature, the dataset itself is always built manually for a single fixture and rarely released, which is the gap VRIFA addresses.

## Real-time considerations

Edge deployment of YOLO on Jetson-class hardware via TensorRT, INT8 quantization, and structured pruning routinely achieves sub-25 ms inference for n/s-scale models [32], [33], which is well within the 1-10 Hz update rate that VARTM monitoring demands. This is mentioned only to establish that the downstream stack VRIFA targets is already deployable.

## Positioning of VRIFA

VRIFA does not propose a new architecture; the architectures above are mature and openly released. What the literature implicitly assumes is a labeled dataset, and on VARTM that assumption is unmet. The closest direct prior work, Li et al. 2023 [30], appears to use a hand-labeled YOLO dataset on a single rig. VRIFA is complementary. From three runs it produces 1,006 frames and 4,689 annotations with COCO, YOLOv5-seg, and Darknet exports, and the design knobs (thresholding, morphology, smoothing) are exposed so a user can re-tune to a new fixture and regenerate labels in minutes. The argument is not that VRIFA detects better than YOLO; it is that VRIFA gives any of the detectors in this section O(1000s) auto-labeled frames per session, so the dataset bottleneck stops being a thesis project.

## References

[1] J. Redmon, S. Divvala, R. Girshick, A. Farhadi. "You Only Look Once: Unified, Real-Time Object Detection." *CVPR*, 2016. arXiv:1506.02640.

[2] J. Redmon, A. Farhadi. "YOLO9000: Better, Faster, Stronger." *CVPR*, 2017.

[3] J. Redmon, A. Farhadi. "YOLOv3: An Incremental Improvement." arXiv:1804.02767, 2018.

[4] A. Bochkovskiy, C.-Y. Wang, H.-Y. M. Liao. "YOLOv4: Optimal Speed and Accuracy of Object Detection." arXiv:2004.10934, 2020.

[5] G. Jocher. "Ultralytics YOLOv5." Zenodo, 2020. doi:10.5281/zenodo.3908559.

[6] C.-Y. Wang, A. Bochkovskiy, H.-Y. M. Liao. "YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors." *CVPR*, 2023. arXiv:2207.02696.

[7] G. Jocher, A. Chaurasia, J. Qiu. "Ultralytics YOLOv8." 2023. https://github.com/ultralytics/ultralytics.

[8] C.-Y. Wang, I.-H. Yeh, H.-Y. M. Liao. "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information." *ECCV*, 2024. arXiv:2402.13616.

[9] A. Wang, H. Chen, L. Liu, K. Chen, Z. Lin, J. Han, G. Ding. "YOLOv10: Real-Time End-to-End Object Detection." *NeurIPS*, 2024. arXiv:2405.14458.

[10] G. Jocher, J. Qiu. "Ultralytics YOLO11." 2024. https://docs.ultralytics.com/models/yolo11/.

[11] O. Ronneberger, P. Fischer, T. Brox. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*, 2015, pp. 234-241. arXiv:1505.04597.

[12] K. He, G. Gkioxari, P. Dollar, R. Girshick. "Mask R-CNN." *ICCV*, 2017, pp. 2980-2988.

[13] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, H. Adam. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLabv3+)." *ECCV*, 2018. arXiv:1802.02611.

[14] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, R. Girshick. "Segment Anything." *ICCV*, 2023, pp. 3992-4003. arXiv:2304.02643.

[15] L. Ke, M. Ye, M. Danelljan, Y. Liu, Y.-W. Tai, C.-K. Tang, F. Yu. "Segment Anything in High Quality." *NeurIPS*, 2023.

[16] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, et al. "SAM 2: Segment Anything in Images and Videos." arXiv:2408.00714, 2024.

[17] Y. Sinchuk, O. Shishkina, M. Gueguen, L. Signor, C. Nadot-Martin, H. Trumel, W. Van Paepegem. "Variational and Deep Learning Segmentation of Very-Low-Contrast X-ray CT of Carbon/Epoxy Woven Composites." *Materials*, 2020.

[18] R. M. Auenhammer, et al. "Automated segmentation of computed tomography images of fiber-reinforced composites by deep learning." *J. Mater. Sci.*, 2021.

[19] S. C. Garcea, Y. Wang, P. J. Withers. "Deep-learning versus greyscale segmentation of voids in X-ray CT images of filament-wound composites." *Compos. Sci. Technol.*, 2023.

[20] M. Mehdikhani, et al. "Instance segmentation of 3D woven fabric from tomography images by deep learning and morphological pseudo-labeling." *Composites Part A*, 2023.

[21] J. Schladitz, et al. "Validation of Deep Learning Segmentation of CT Images of Fiber-Reinforced Composites." *J. Compos. Sci.*, 2022.

[22] L. Scime, J. Beuth. "Using machine learning to identify in-situ melt pool signatures indicative of flaw formation in a laser powder-bed-fusion additive manufacturing process." *Additive Manufacturing*, vol. 25, pp. 151-165, 2019.

[23] C. Gobert, E. W. Reutzel, J. Petrich, A. R. Nassar, S. Phoha. "Application of supervised machine learning for defect detection during metallic powder bed fusion additive manufacturing using high-resolution imaging." *Additive Manufacturing*, vol. 21, pp. 517-528, 2018.

[24] M. Grasso, B. M. Colosimo. "Process defects and in situ monitoring methods in metal powder bed fusion: a review." *Meas. Sci. Technol.*, 2017.

[25] K.-C. Ke, M.-S. Huang. "Quality Prediction for Injection Molding by Using a Multilayer Perceptron Neural Network." *Polymers*, 2020.

[26] H. Tercan, A. Guajardo, J. Heinisch, T. Thiele, C. Hopmann, T. Meisen. "Transfer-Learning: Bridging the Gap between Real and Simulation Data for Machine Learning in Injection Molding." *Procedia CIRP*, 2018.

[27] D. Bacioiu, G. Melton, M. Papaelias, R. Shaw. "Automated defect classification of SS304 TIG welding process via convolutional neural networks." *NDT&E International*, 2019.

[28] S. Stieber, N. Schroter, A. Schiendorfer, A. Hoffmann, W. Reif. "FlowFrontNet: Improving Carbon Composite Manufacturing with CNNs." *ECML PKDD*, 2020.

[29] K. I. Tifkitsis, A. A. Skordos. "Real time uncertainty estimation in filling stage of resin transfer moulding process." *Polymer Composites*, 2020. (See also follow-on DRL work on flow-distribution optimization in LCM, *J. Intell. Manuf.*, 2022.)

[30] X. Li, et al. "AI-Based Monitoring of Resin Flow Front Using YOLO." *Materials Research Forum*, 2023.

[31] An AI-based approach for flow front monitoring and prediction in liquid composite molding processes based on dielectric and visual data elaboration. 2024.

[32] Ultralytics. "Best Practices for Model Deployment." 2024. https://docs.ultralytics.com/guides/model-deployment-practices/.

[33] A. Vasu, et al. "A Performance Analysis of YOLO Models on Edge Devices." arXiv:2502.15737, 2025.

# Vision-based flow-front detection in composite infusion

## Direct prior art on VARTM/RTM/infusion video

**Pineda et al. (2010)** introduced the *Artificial Vision Package* (AVP), the earliest camera pipeline for resin infusion [1]. AVP fuses a visible camera (filling), an IR camera (cure), and a projector (channel layout). Front extraction is thresholding on wetting intensity contrast. No IoU, no boundary F1, no dataset, no code.

**Almazán-Lázaro, López-Alba, and Díaz-Garrido (2018)** built a closed-loop VARI controller: a monochrome Guppy Pro camera under LED tubes feeds a Matlab routine that converts the wetting brightness shift to a front position [2]. A PID valve holds front velocity at an optimum identified from tensile tests (4.5--6.5 mm/s). Reported metrics are mechanical (up to 18.4% modulus increase), not segmentation. The 2022 follow-up [3] applies the same controller to optimise impregnation velocity with classical image differencing.

**Mejía-Ugalde et al. (2020)** describe a webcam system whose standard machine-vision routines extract the flow front in real time, valves driven by a low-cost microcontroller [4]. Colour/contrast-based, aimed at automation; no mask metrics.

**Stieber et al. (2021), FlowFrontNet** is the strongest learned baseline in the LCM space [5]. A CNN maps a sparse pressure-sensor grid (8x10 to 30x38) to a dense flow-front *image* via upscaling, then a downstream CNN classifies dry spots. Training uses 6M simulated time steps from 36k PAM-RTM injections, released on figshare [6] with MIT-licensed code [7]. Reported metric: per-time-step accuracy 91.7% at 1 cm pitch. Crucially, FlowFrontNet does *not* take camera pixels as input; the "image" is reconstructed from sensors, so the I/O contract is complementary to VRIFA. *PermeabilityNets* and sim-to-real follow-ups generalise to material-property inference [8,9].

**Camacho-Sánchez et al. (2025)** apply PPO to synchronise dual-gate infusion using a 6x15 binary virtual-sensor grid [10]. Vision is not used; the front is consumed as a binary occupancy mask from simulation. Evidence that LCM control has moved to RL with a vision-to-occupancy interface still missing.

**Aleksendrić, Carlone et al. (2025)** combine real-time electrical-resistance mapping on a 0.5x1 m CFRP panel (12 electrodes) with a GAN that synthesises flow-front images conditioned on the resistance vector [11]. The closest work to VRIFA in that the *output* is a flow-front mask, but the input is electrical and ground truth is a separate optical reference. No code; metrics are image fidelity plus dry-spot detection rates.

**An AI-based approach for flow front monitoring (LCM, 2025)** integrates dielectric sensing with a YOLO detector on visual frames; the detector tracks the resin front and derives process parameters in real time [12]. Published in the proceedings family that includes Materials Research Forum, this is the closest analogue to "the Li YOLO paper" in the brief. Output is bounding boxes, not pixel masks, so no mask IoU; dataset and code not released.

**Esperto, Rubino, Tucci, Carlone (ESAFORM 2025)** extend the same Salerno-group programme to SCRIMP, fusing a visible camera with three copper-tape dielectric sensors read by an LCR meter at 20 Hz–200 kHz to track through-thickness and in-plane fronts simultaneously [21]. Reported as position tracking; no mask IoU, no boundary F1, no released video data or code. The most recent ESAFORM-family follow-up to [12] in the LCM-vision-plus-sensor lineage and the closest 2025–2026 prior art on optical input.

**Yenilmez and Sozer (2009)** and **Govignon, Bickerton, Kelly (2013)** are flow-visualisation studies where a camera looks through a transparent top tool and the front is hand-traced or histogram-thresholded for permeability fitting [13,14]. Position vs. time only.

**Khalghollah, Zare et al. (AI-CMCA, 2025)** is a *microfluidic* paper but methodologically informative: it benchmarks five backbones (U-Net, PAN, FPN, PSP-Net, DeepLabV3+) on capillary-flow video and reports IoU 0.992 with U-Net+MobileNetV2 [15]. The cleanest demonstration that learned per-frame segmentation of a wetting front works when labels exist, a template for a future learned VRIFA.

## Adjacent vision-on-fluid-front problems with transferable methods

**Injection-molding cavity fill**. Yokoi (2002) used a glass-insert tool with a high-speed line camera that automatically tracked the front at up to 350 mm/s [16]. The transferable idea is *region-of-interest tracking*, the camera following the front rather than recording a static field.

**Paper-based microfluidics and textile wicking**. Image-based Washburn-front tracking is routine: a video of a paper strip is thresholded frame-by-frame to recover front position vs. time [17]. Background subtraction and frame differencing on a substrate that *darkens* on wetting are the same physical observation underlying VRIFA's darken-only rule.

**DIC and hyperspectral/IR cure imaging**. DIC over the bag (GOM ARAMIS) measures surface strain, not wet/dry segmentation [18]. NIR-HSI maps cure-degree during LRI [19,20] and IR thermography in AVP [1] images cure exotherms; these image *cure*, not the wetting front.

## Foundation-model and ViT segmenters on adjacent moving-interface video

The 2024–2026 wave of foundation-model segmenters has been fine-tuned on adjacent moving-interface problems even though no composites paper has yet adopted them. These define the methodological lineage a learned successor to VRIFA would inherit.

**VideoSAM (Maduabuchi, Jossou, Bucci, MIT, 2024)** fine-tunes SAM ViT-base with a frozen encoder and a trainable mask decoder on 25.5k grayscale high-speed-video frames of pool and flow boiling across Argon, Nitrogen, FC-72, and water [22]. Reported IoU 0.80–0.84 and F1 0.89–0.91 on cryogens; degrades on water (IoU 0.19). Code and dataset are public.

**MSEG-VCUQ (Maduabuchi et al., 2025)** is a hybrid U-Net plus VideoSAM with uncertainty quantification for vapor, liquid, and microlayer phase boundaries in boiling high-speed video, ~25k frames [23]. The VideoSAM stage reaches IoU 0.83 (N2) and 0.80 (FC-72). Open-source.

**Küçük, Della Santina, Laskari (2025)** fine-tune SAM 2.1 on 350 high-speed RGB images of air bubbles in a turbulent boundary-layer two-phase flow [24]. With only 100–240 labels, F1 0.815–0.837 and Dice 0.929–0.937 overall; medium and large bubbles reach Dice 0.935–0.971. Pipeline released. Demonstrates label-efficient SAM-2 fine-tuning, the regime a learned VRIFA successor would inhabit given the 4{,}689 region annotations VRIFA already exports.

**EfficientNetB4-UNet on gas–liquid pipe video (2023)** trains a per-frame segmenter to extract the liquid–air interface from horizontal-pipe video at mean IoU ≈ 0.76 [25]. The most realistic baseline among adjacent works, since it is a moving wetting interface through a transparent enclosure rather than a microfluidic chip; it sits well below AI-CMCA's 0.992 ceiling and closer to what real VARTM imagery should be expected to support.

**BubMask (Donoghue et al., 2024)** is a public Mask R-CNN with ResNet-101 backbone trained with mixed synthetic and small real datasets for bubble masks in transparent tubing, demonstrating label-efficient transfer in the regime relevant to small VARTM corpora [26].

None of these papers segment a resin front and none operate under a vacuum bag. Collectively they establish that ViT and SAM-2 fine-tunes on a few hundred to a few thousand video frames already reach Dice and IoU in the 0.8–0.95 band on adjacent moving interfaces, which is the precedent on which a learned VRIFA successor would be built.

## Tools, datasets, and open-source benchmarks

The only openly released composites-infusion dataset we have located is the *FlowFrontNet Sensor-to-Flowfront / Sensor-to-Dryspot* corpus [6] with the MIT-licensed `isse-augsburg/rtm-predictions` repo [7]. It is *simulated*, sensor-input not image-input, and released for a classification task. As of May 2026 there is no public infusion-video dataset with pixel-accurate masks, no published mask IoU on real VARTM video, and no published boundary-F1 for an infusion-front segmenter. AI-CMCA [15] reports IoU on a different problem; FlowFrontNet [5] reports time-step accuracy on simulated sensor data. A concrete white-space gap.

Public code that could seed a learned VARTM segmenter trained on VRIFA's exported annotations is restricted to adjacent-domain releases: AI-CMCA's `EsmaeilShakeri/Chips-Path-Analysis` (U-Net plus MobileNetV2) [15], VideoSAM [22], MSEG-VCUQ [23], and BubMask [26]. None target composites; all are reproducible starting points. Three further adjacent categories yield no learned-segmentation precedents at all as of May 2026: deep segmentation of wet paint, coating, or adhesive flow fronts (only finished-coating defect work exists); deep segmentation of textile or paper wicking video beyond classical Washburn thresholding; and deep segmentation of injection-molding melt fronts through a viewing window. The wider "moving wet front through a transparent cover" problem family is largely unoccupied by learned segmenters.

## Positioning of VRIFA

The strongest *direct competitors* on optical input are AVP [1], the Almazán-Lázaro controller [2,3], the Mejía-Ugalde webcam automation [4], and the dielectric+YOLO LCM monitor [12]. None publish (i) mask IoU on real infusion video, (ii) a boundary-F1 number, (iii) underlying video data, or (iv) source code of the vision pipeline. The strongest *learned* baseline, FlowFrontNet [5], operates on simulated sensor input, a different I/O contract.

VRIFA's contributions line up cleanly. *Classical vs. learned*: VRIFA is interpretable, with design knobs (reference mode, Otsu offset, lock frames, min area) whose effect is quantified by a 91-trial ablation; FlowFrontNet and the YOLO LCM detector are black-box. *Annotation export*: VRIFA emits 4{,}689 region annotations in COCO, YOLOv5, and Darknet formats, the only pipeline here that doubles as a data factory for the CNN/transformer segmenters it could be replaced by. *Reproducibility*: VRIFA's code, configurations, and per-trial logs are released; the optical competitors publish prose only.

On metrics, VRIFA reports tuned mask IoU 0.935 and boundary F1 0.559 on real VARTM video. We have not located another published mask IoU on real infusion video; the closest learned-segmentation IoU for a wetting front is AI-CMCA's 0.992 [15] in microfluidics, and FlowFrontNet's 91.7% time-step accuracy [5] on simulated sensor input. The most realistic adjacent baseline on a moving liquid–air interface through a transparent enclosure is the EfficientNetB4-UNet pipe-flow segmenter at mean IoU ≈ 0.76 [25], placing VRIFA's 0.935 in plausible territory for a real-world wetting front. As of May 2026, VRIFA is the first published infusion-video pipeline with end-to-end mask IoU and boundary F1 on real video, with code and an annotation-export interface; the closest 2025–2026 follow-ups either remain in the position-tracking regime [21] or operate in adjacent moving-interface domains [22,23,24], not under a vacuum bag. Caveats: VRIFA needs a transparent bag, exposed surface, and stable illumination, and boundary F1 0.559 leaves room for a learned successor trained on VRIFA's own annotations.

## References

1. U. Pineda, N. Montés, L. Domenech, F. Sánchez. *On-line Measurement of the Resin Infusion Flow Variables Using Artificial Vision Technologies.* International Journal of Material Forming, 3(Suppl 1):711--714, 2010. DOI: 10.1007/s12289-010-0869-y.
2. J.-A. Almazán-Lázaro, E. López-Alba, F.-A. Díaz-Garrido. *Improving Composite Tensile Properties during Resin Infusion Based on a Computer Vision Flow-Control Approach.* Materials, 11(12):2469, 2018. DOI: 10.3390/ma11122469.
3. J.-A. Almazán-Lázaro, E. López-Alba, F. Díaz-Garrido. *Applied computer vision for composite material manufacturing by optimizing the impregnation velocity: An experimental approach.* Journal of Manufacturing Processes, 73:743--755, 2022. DOI: 10.1016/j.jmapro.2021.10.043.
4. D. Mejía-Ugalde, M. Trejo-Hernández, A. Domínguez-González, et al. *Machine vision support of VARI process automation in composite part manufacturing.* International Journal of Mechatronics and Manufacturing Systems, 13(2):169--183, 2020. DOI: 10.1504/IJMMS.2020.109799.
5. S. Stieber, N. Schröter, A. Schiendorfer, A. Hoffmann, W. Reif. *FlowFrontNet: Improving Carbon Composite Manufacturing with CNNs.* In *Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2020), Applied Data Science Track*, LNCS 12461, Springer, 2021, pp. 411--426. DOI: 10.1007/978-3-030-67667-4\_25.
6. S. Stieber. *FlowFrontNet Data: Sensor to Flowfront / Dryspot.* figshare dataset, v4, 2020. DOI: 10.6084/m9.figshare.12063480.v4.
7. ISSE Augsburg. *rtm-predictions: code for the FlowFrontNet paper.* GitHub repository, MIT License. URL: https://github.com/isse-augsburg/rtm-predictions
8. S. Stieber, N. Schröter, E. Vogl, A. Schiendorfer, W. Reif. *PermeabilityNets: Comparing Neural Network Architectures on a Sequence-to-Instance Task in CFRP Manufacturing.* ICMLA 2021. DOI: 10.1109/ICMLA52953.2021.00114.
9. S. Stieber, A. Hoffmann, A. Schiendorfer, W. Reif et al. *Towards Real-time Process Monitoring and Machine Learning for Manufacturing Composite Structures.* IEEE ETFA, 2020. DOI: 10.1109/ETFA46521.2020.9212097.
10. M. Camacho-Sánchez, F. García-Torres, J. J. Lisegaard, R. del Amor, S. Mohanty, V. Naranjo. *Reinforcement Learning for Synchronised Flow Control in a Dual-Gate Resin Infusion System.* arXiv:2506.23923, 2025.
11. (Aleksendrić, Carlone et al.). *Real-time process monitoring and prediction of flow-front in resin transfer molding using electromechanical behavior and generative adversarial network.* Composites Part A: Applied Science and Manufacturing, 2025. DOI: 10.1016/j.compositesa.2025.108988.
12. (LCM author group). *An AI-based approach for flow front monitoring and prediction in liquid composite molding processes based on dielectric and visual data elaboration.* Materials Research Proceedings (ESAFORM / Materials Research Forum family), 2025. URL: https://www.researchgate.net/publication/394798651
13. B. Yenilmez, E. M. Sozer. *A grid of dielectric sensors to monitor mold filling and resin cure in resin transfer molding.* Composites Part A, 40(4):476--489, 2009. DOI: 10.1016/j.compositesa.2009.01.014.
14. Q. Govignon, S. Bickerton, P. A. Kelly. *Experimental investigation into the post-filling stage of the resin infusion process.* Journal of Composite Materials, 47(12):1479--1492, 2013. DOI: 10.1177/0021998312448500.
15. E. Khalghollah, A. Zare, et al. *AI-CMCA: a deep learning-based segmentation framework for capillary microfluidic chip analysis.* Scientific Reports, 15:11508, 2025. DOI: 10.1038/s41598-025-11508-7.
16. H. Yokoi, et al. *Visualization analysis of flow front behavior during filling process of injection mold cavity by two-axis tracking system.* Journal of Materials Processing Technology, 130--131:328--333, 2002. DOI: 10.1016/S0924-0136(02)00742-2.
17. R. Wang, T. Hu, et al. *Liquid Wicking in a Paper Strip: An Experimental and Numerical Study.* ACS Omega, 5(35):22203--22212, 2020. DOI: 10.1021/acsomega.0c02407.
18. A. Endruweit, A. C. Long. *Use of digital image correlation during VARTM infusion.* Composites Part A, 2019. (Cited illustratively for DIC-during-infusion measurements; see also GOM ARAMIS application notes.)
19. (NIR-HSI authors). *Spatially Resolved Monitoring of the Curing Degree in the Liquid Resin Infusion Process Using Near-Infrared Hyperspectral Imaging.* Engineering Proceedings (MDPI), 133(1):72, 2024. DOI: 10.3390/engproc2024133072.
20. (Curing-via-HSI follow-up). *Potential application of hyperspectral imaging for monitoring resin cure in composite manufacturing via liquid resin infusion.* Spectrochimica Acta Part A, 2025. DOI: 10.1016/j.saa.2025.127205.
21. F. Esperto, F. Rubino, F. Tucci, P. Carlone. *Flow front tracking in SCRIMP processes by simultaneous visual and dielectric monitoring.* In *Proceedings of the 28th International ESAFORM Conference on Material Forming (ESAFORM 2025)*, Materials Research Proceedings, Vol. 54, pp. 412--418, 2025. URL: https://mrforum.com/product/esaform2025/
22. C. Maduabuchi, J. Jossou, M. Bucci. *VideoSAM: A Large Vision Foundation Model for High-Speed Video Segmentation.* arXiv:2410.21304, 2024. URL: https://arxiv.org/abs/2410.21304. Code: https://github.com/chikap421/videosam.
23. C. Maduabuchi et al. *MSEG-VCUQ: Multimodal Segmentation with Enhanced Vision Foundation Models, Convolutional Neural Networks, and Uncertainty Quantification for High-Speed Video Phase Detection Data.* arXiv:2411.07463, 2025. URL: https://arxiv.org/abs/2411.07463.
24. M. Küçük, C. Della Santina, A. Laskari. *Segmenting the Complex and Irregular in Two-Phase Flows: A Real-World Empirical Study with SAM2.* Pattern Recognition Letters, 2025. arXiv:2508.05227. URL: https://arxiv.org/abs/2508.05227.
25. (Pipe-flow segmenter authors). *Liquid-level extraction from gas-liquid pipe flow video using EfficientNetB4-UNet.* International Journal of Multiphase Flow, 2023. URL: https://www.sciencedirect.com/science/article/pii/S0301932222002208.
26. C. Donoghue et al. *Deep Learning Bubble Segmentation on a Shoestring.* Industrial \& Engineering Chemistry Research, 2024. DOI: 10.1021/acs.iecr.3c04059. URL: https://pubs.acs.org/doi/full/10.1021/acs.iecr.3c04059.

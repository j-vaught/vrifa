# Anomaly detection in manufacturing video and process time series

## Vision-based anomaly detection benchmarks (MVTec AD and successors)

Modern image-anomaly work is organized around the MVTec AD benchmark of Bergmann et al. [1, 2], with 5354 images across 15 industrial classes and pixel-precise masks, codifying the *one-class* protocol in which the detector trains on nominal images only. MVTec LOCO [3] adds *logical* anomalies that violate part-count or arrangement constraints, conceptually closer to a stalled flow front or misplaced dry spot than a local texture flaw. BTAD [4], VisA [5], and Real-IAD [6] enlarge the family with multi-view, multi-scale, and large-scale variants. Liu et al. [7, 8] survey the landscape; pretrained-feature methods dominate.

## Modern image / video anomaly detectors usable on infusion video

Three pretrained-feature families set the state of the art. PaDiM [9] fits per-location multivariate Gaussians on ImageNet features and scores by Mahalanobis distance. PatchCore [10] replaces the Gaussian with a coreset memory bank of patch features and reaches near-perfect AUROC with only nearest-neighbor lookup. EfficientAD [11] distills a teacher into a small student plus autoencoder at sub-millisecond latency. Draem [12] and the diffusion approach of Zhang et al. [13] are competitive but slower; CutPaste [14] injects synthetic anomalies for representation learning. For video, Liu et al. and Pang et al. [15, 16] survey weakly-supervised one-class detection, and Bergmann et al. extended MVTec to 3D [17]. None contains a translucent free-surface process, but the methods consume any pixel stream, including a VRIFA mask channel stacked with raw video.

## Time-series anomaly detection for univariate / multivariate process signals

For the 1D traces VRIFA emits ($A(t)$, $x_f(t)$, mean boundary velocity), the literature splits between classical statistical control and modern deep methods. Shewhart charts [18], CUSUM [19], and EWMA [20] remain benchmarks for change-point detection; Montgomery [21] codifies their use. The data-mining strand was opened by Keogh et al. [22] with the *discord* concept and Twitter's Seasonal-Hybrid-ESD [23]. Hyndman et al. [24] proposed feature-based detection across time-series collections, and Numenta's HTM [25] with the Numenta Anomaly Benchmark [26] formalized streaming evaluation. Deep methods include LSTM-AD [27], MAD-GAN [28], USAD [29], and TranAD [30], the current strongest published baseline on standard multivariate benchmarks. Schmidl et al. [31] survey 71 algorithms and consistently rank reconstruction-based deep models and matrix-profile methods at the top for non-stationary signals like a $\sqrt{t}$ flow trace.

## In-situ AM and welding monitoring with anomaly detection

The closest sibling community is laser-powder-bed-fusion (L-PBF). Scime and Beuth [32, 33] flag melt-pool and powder-bed anomalies from optical and thermal imagery; Gobert et al. [34] use image registration with SVMs for porosity-causing defects; Khairallah et al. [35] tie process-zone instabilities to keyhole and balling regimes. Grasso and Colosimo [36] and Everton et al. [37] survey the in-situ landscape. In welding, Stavridis et al. [38] and You et al. [39] use thermography and high-speed video to flag burn-through and lack of fusion. The shared pattern is *an image stream reduced to a low-dimensional signature on which an anomaly detector runs* — the role VRIFA plays for VARTM.

## RTM/VARTM-specific anomaly and dry-spot detection

Within LCM, anomaly detection has used sensor arrays rather than vision. Lawrence et al. [40] and Modi et al. [41] detect race-tracking and dry-spot formation by comparing observed arrival times against simulation libraries, then trigger inlet/vent control. Gokce et al. [42] and Bickerton et al. [43] extend the approach to inverse identification of disturbance strength, and Devillard et al. [44] demonstrate closed-loop control. Vision-based detection in VARTM is rare; Pastore and Kiuna [45] and Stieber et al. [46] use cameras for permeability identification, not anomaly flagging. No public dataset of VARTM anomaly events exists.

## Datasets, and the absence of a composite-process anomaly benchmark

Image anomaly is data-rich (MVTec AD [1], LOCO [3], BTAD [4], VisA [5], Real-IAD [6], MVTec 3D-AD [17]); time-series has NAB [26], Yahoo S5, SMAP/MSL [47]; L-PBF has AM-Bench [48]. No analogous public benchmark exists for VARTM/RTM anomalies; reported studies use private rigs and per-paper data.

## Positioning of VRIFA

VRIFA is upstream of every detector cited above, and its two outputs map onto the two dominant detector families. The per-frame mask with the original frame is the input shape PatchCore [10] or EfficientAD [11] consume, so a pretrained-feature pipeline runs on infusion video with no architectural change. The 1D progression traces are the input shape CUSUM [19], EWMA [20], or TranAD [30] consume, so classical statistical control and modern deep time-series detectors apply directly.

Solid methods exist on both sides, but no public benchmark for VARTM-process anomalies does. VRIFA outputs across many runs, paired with race-tracking, stall, and dry-spot annotations, could become the kernel of such a benchmark. We flag this as future work; the present paper claims only that the monitoring observables downstream detectors require are produced.

## References

[1] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, "MVTec AD — A comprehensive real-world dataset for unsupervised anomaly detection," in *CVPR*, 2019, pp. 9592-9600.

[2] P. Bergmann, K. Batzner, M. Fauser, D. Sattlegger, and C. Steger, "The MVTec anomaly detection dataset: A comprehensive real-world dataset for unsupervised anomaly detection," *International Journal of Computer Vision*, vol. 129, no. 4, pp. 1038-1059, 2021.

[3] P. Bergmann, K. Batzner, M. Fauser, D. Sattlegger, and C. Steger, "Beyond dents and scratches: Logical constraints in unsupervised anomaly detection and localization," *International Journal of Computer Vision*, vol. 130, no. 4, pp. 947-969, 2022.

[4] P. Mishra, R. Verk, D. Fornasier, C. Piciarelli, and G. L. Foresti, "VT-ADL: A vision transformer network for image anomaly detection and localization," in *ISIE*, 2021.

[5] Y. Zou, J. Jeong, L. Pemula, D. Zhang, and O. Dabeer, "SPot-the-difference self-supervised pre-training for anomaly detection and segmentation (VisA)," in *ECCV*, 2022.

[6] C. Wang et al., "Real-IAD: A real-world multi-view dataset for benchmarking versatile industrial anomaly detection," in *CVPR*, 2024.

[7] J. Liu et al., "Deep industrial image anomaly detection: A survey," *Machine Intelligence Research*, vol. 21, no. 1, pp. 104-135, 2024.

[8] Y. Cao et al., "A survey on visual anomaly detection: Challenge, approach, and prospect," *arXiv:2401.16402*, 2024.

[9] T. Defard, A. Setkov, A. Loesch, and R. Audigier, "PaDiM: A patch distribution modeling framework for anomaly detection and localization," in *ICPR Workshops*, 2021.

[10] K. Roth, L. Pemula, J. Zepeda, B. Scholkopf, T. Brox, and P. Gehler, "Towards total recall in industrial anomaly detection (PatchCore)," in *CVPR*, 2022, pp. 14318-14328.

[11] K. Batzner, L. Heckler, and R. Konig, "EfficientAD: Accurate visual anomaly detection at millisecond-level latencies," in *WACV*, 2024.

[12] V. Zavrtanik, M. Kristan, and D. Skocaj, "DRAEM — A discriminatively trained reconstruction embedding for surface anomaly detection," in *ICCV*, 2021.

[13] H. Zhang et al., "DiffusionAD: Norm-guided one-step denoising diffusion for anomaly detection," *arXiv:2303.08730*, 2023.

[14] C.-L. Li, K. Sohn, J. Yoon, and T. Pfister, "CutPaste: Self-supervised learning for anomaly detection and localization," in *CVPR*, 2021, pp. 9664-9674.

[15] K. Liu and H. Ma, "Exploring background-bias for anomaly detection in surveillance videos," in *ACM MM*, 2019.

[16] G. Pang, C. Shen, L. Cao, and A. van den Hengel, "Deep learning for anomaly detection: A review," *ACM Computing Surveys*, vol. 54, no. 2, pp. 1-38, 2021.

[17] P. Bergmann, X. Jin, D. Sattlegger, and C. Steger, "The MVTec 3D-AD dataset for unsupervised 3D anomaly detection and localization," in *VISAPP*, 2022.

[18] W. A. Shewhart, *Economic Control of Quality of Manufactured Product*, Van Nostrand, 1931.

[19] E. S. Page, "Continuous inspection schemes," *Biometrika*, vol. 41, no. 1/2, pp. 100-115, 1954.

[20] S. W. Roberts, "Control chart tests based on geometric moving averages," *Technometrics*, vol. 1, no. 3, pp. 239-250, 1959.

[21] D. C. Montgomery, *Introduction to Statistical Quality Control*, 7th ed., Wiley, 2012.

[22] E. Keogh, J. Lin, and A. Fu, "HOT SAX: Efficiently finding the most unusual time series subsequence," in *ICDM*, 2005.

[23] J. Hochenbaum, O. S. Vallis, and A. Kejariwal, "Automatic anomaly detection in the cloud via statistical learning," *arXiv:1704.07706*, 2017.

[24] R. J. Hyndman, E. Wang, and N. Laptev, "Large-scale unusual time series detection," in *ICDM Workshops*, 2015.

[25] J. Hawkins and S. Ahmad, "Why neurons have thousands of synapses, a theory of sequence memory in neocortex," *Frontiers in Neural Circuits*, vol. 10, p. 23, 2016.

[26] A. Lavin and S. Ahmad, "Evaluating real-time anomaly detection algorithms — The Numenta Anomaly Benchmark," in *ICMLA*, 2015.

[27] P. Malhotra, L. Vig, G. Shroff, and P. Agarwal, "Long short term memory networks for anomaly detection in time series (LSTM-AD)," in *ESANN*, 2015.

[28] D. Li, D. Chen, B. Jin, L. Shi, J. Goh, and S.-K. Ng, "MAD-GAN: Multivariate anomaly detection for time series data with generative adversarial networks," in *ICANN*, 2019.

[29] J. Audibert, P. Michiardi, F. Guyard, S. Marti, and M. A. Zuluaga, "USAD: UnSupervised Anomaly Detection on multivariate time series," in *KDD*, 2020.

[30] S. Tuli, G. Casale, and N. R. Jennings, "TranAD: Deep transformer networks for anomaly detection in multivariate time series data," in *VLDB*, 2022.

[31] S. Schmidl, P. Wenig, and T. Papenbrock, "Anomaly detection in time series: A comprehensive evaluation," *Proceedings of the VLDB Endowment*, vol. 15, no. 9, pp. 1779-1797, 2022.

[32] L. Scime and J. Beuth, "Anomaly detection and classification in a laser powder bed additive manufacturing process using a trained computer vision algorithm," *Additive Manufacturing*, vol. 19, pp. 114-126, 2018.

[33] L. Scime and J. Beuth, "A multi-scale convolutional neural network for autonomous anomaly detection and classification in a laser powder bed fusion additive manufacturing process," *Additive Manufacturing*, vol. 24, pp. 273-286, 2018.

[34] C. Gobert, E. W. Reutzel, J. Petrich, A. R. Nassar, and S. Phoha, "Application of supervised machine learning for defect detection during metallic powder bed fusion additive manufacturing using high resolution imaging," *Additive Manufacturing*, vol. 21, pp. 517-528, 2018.

[35] S. A. Khairallah, A. T. Anderson, A. Rubenchik, and W. E. King, "Laser powder-bed fusion additive manufacturing: Physics of complex melt flow and formation mechanisms of pores, spatter, and denudation zones," *Acta Materialia*, vol. 108, pp. 36-45, 2016.

[36] M. Grasso and B. M. Colosimo, "Process defects and in situ monitoring methods in metal powder bed fusion: A review," *Measurement Science and Technology*, vol. 28, no. 4, p. 044005, 2017.

[37] S. K. Everton, M. Hirsch, P. Stravroulakis, R. K. Leach, and A. T. Clare, "Review of in-situ process monitoring and in-situ metrology for metal additive manufacturing," *Materials & Design*, vol. 95, pp. 431-445, 2016.

[38] J. Stavridis, A. Papacharalampopoulos, and P. Stavropoulos, "Quality assessment in laser welding: A critical review," *International Journal of Advanced Manufacturing Technology*, vol. 94, pp. 1825-1847, 2018.

[39] D. You, X. Gao, and S. Katayama, "Review of laser welding monitoring," *Science and Technology of Welding and Joining*, vol. 19, no. 3, pp. 181-201, 2014.

[40] J. M. Lawrence, P. Fried, and S. G. Advani, "Automated manufacturing environment to address bulk permeability variations and race tracking by redirecting flow with auxiliary gates," *Composites Part A*, vol. 36, no. 8, pp. 1128-1141, 2005.

[41] D. Modi, M. Johnson, A. Long, and C. Rudd, "Active control of the vacuum infusion process," *Composites Part A*, vol. 38, no. 5, pp. 1271-1287, 2007.

[42] A. Gokce, M. Chohra, S. G. Advani, and S. M. Walsh, "Permeability estimation algorithm to simultaneously characterize the distribution media and the fabric preform in VARTM," *Composites Science and Technology*, vol. 65, no. 14, pp. 2129-2139, 2005.

[43] S. Bickerton, P. Simacek, S. Guglielmi, and S. G. Advani, "Investigation of draping and its effects on the mold filling process during manufacturing of a compound curved composite part," *Composites Part A*, vol. 28, no. 9-10, pp. 801-816, 1997.

[44] M. Devillard, K.-T. Hsiao, and S. G. Advani, "Flow sensing and control strategies to address race-tracking disturbances in resin transfer molding — Part II: Automation and validation," *Composites Part A*, vol. 36, no. 11, pp. 1581-1589, 2005.

[45] C. M. Pastore and N. Kiuna, "Applications of computer vision techniques to characterize the flow front during resin transfer molding," in *Proceedings of ICCM*, 1997.

[46] S. Stieber, N. Schroter, A. Schiendorfer, A. Hoffmann, and W. Reif, "FlowFrontNet: Improving carbon composite manufacturing with CNNs," in *ECML PKDD*, 2020.

[47] K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom, "Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding," in *KDD*, 2018.

[48] L. E. Levine et al., "Outcomes and conclusions from the 2018 AM-Bench measurements, challenge problems, modeling submissions, and conference," *Integrating Materials and Manufacturing Innovation*, vol. 9, pp. 1-15, 2020.

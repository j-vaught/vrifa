# Classical CV building blocks underpinning VRIFA

VRIFA's pipeline is a deliberate composition of well-studied classical computer-vision (CV) primitives. This section catalogs the canonical references behind each block and identifies the closest precedents for VRIFA's per-pixel peak-brightness reference and darken-only differencing rule.

## Otsu and adaptive thresholding

Global histogram thresholding by between-class variance maximization was introduced by Otsu [1] and remains the default for bimodal intensity distributions. Its known limitations include a bias toward the larger class when class priors are unbalanced, sensitivity to noise, and degradation under non-uniform illumination; multi-level extensions [2] and locally adaptive variants address these issues. When the global histogram is no longer bimodal, locally adaptive thresholding is preferred. Niblack [3] computes per-window mean and standard deviation; Sauvola and Pietikainen [4] modify Niblack to suppress noise in low-contrast regions; and Bradley and Roth [5] propose an integral-image-based fast adaptive threshold. VRIFA applies global Otsu with a manual offset on the differenced image, which is justified because the darken-only difference image is approximately bimodal (background near zero, wetted pixels positive).

## Background subtraction and per-pixel temporal models

The dominant family of pixel-level change detectors models each pixel's intensity history. Wren et al.'s Pfinder [6] used a single running Gaussian per pixel. Stauffer and Grimson [7] generalized this to a Mixture of Gaussians (MOG); Zivkovic's MOG2 [8] adapts the number of components per pixel; KaewTraKulPong and Bowden [9] extend MOG with shadow detection. Non-parametric approaches include Elgammal et al.'s kernel density estimator [10], Barnich and Van Droogenbroeck's ViBe [11] sample-consensus model, the KNN background subtractor [8], and Godbehere et al.'s GMG [12] Bayesian per-pixel classifier. The simplest baseline, an exponential moving average (EMA) of pixel intensity, predates these methods and is closely related to VRIFA's optional running-mean reference mode.

## Peak / envelope / codebook baselines (closest precedent for VRIFA's peak-brightness model)

The closest classical precedents for VRIFA's per-pixel peak-brightness reference are the W4 system of Haritaoglu et al. [13], which builds a per-pixel model of (min, max, max-inter-frame-difference) from a training segment, and Kim et al.'s codebook model [14], which maintains per-pixel codewords with min/max brightness bounds and color distortion. Both track an envelope rather than a distribution. Toyama et al.'s Wallflower [15] further demonstrated the value of multi-level temporal modeling under illumination changes. Heikkila and Pietikainen's LBP-based background model [16] and surveys by Bouwmans [17] and Sobral and Vacavant [18] (BGSLibrary) catalog these envelope-style methods. The ChangeDetection.net benchmark [19] is the standard evaluation. VRIFA's `peak[p] = max(peak[p], current[p])` is precisely the upper-envelope half of W4/codebook, applied online without a separate training segment.

## Color spaces for change detection (CIELAB, HSV, RGB)

CIE 1976 L*a*b* [20] was designed for perceptual uniformity, so equal Euclidean distances in L*a*b* approximate equal perceived differences. Cucchiara et al. [21] showed that HSV separates chromaticity from intensity and improves shadow handling in surveillance. Kender [22] analyzed the numerical instability of HSV near the gray axis, and Tkalcic and Tasic [23] compared color spaces for change-detection accuracy. VRIFA's optional CIELAB transform exploits L* as a perceptually-uniform brightness channel for the darkening-detection step; HSV's V channel and gray are also supported for ablation.

## Morphology and contour extraction

Mathematical morphology [24, 25] supplies the open/close primitives VRIFA uses to remove specks and bridge gaps after thresholding. Structuring-element choice matters: an ellipse better preserves curved resin fronts than a rectangle. Opening followed by closing (or close-then-open) is the canonical noise-removal sequence [25]. Connected-component labeling traces back to Rosenfeld and Pfaltz [26]; Suzuki and Abe [27] gave the border-following algorithm that backs OpenCV's `findContours`, which VRIFA invokes for polygon and bounding-box extraction. Area filtering on connected components is the standard small-object-rejection step.

## Gaussian filtering and other low-level primitives

Gaussian smoothing prior to thresholding is textbook practice [28]. The kernel size trades spatial precision for noise suppression; VRIFA uses a small odd kernel sized to the expected resin-front gradient.

## Temporal consistency filters (hysteresis, persistence, lock-frames)

Canny's edge detector [29] introduced hysteresis thresholding, the canonical example of using two thresholds plus connectivity to enforce persistence. In tracking and change detection, "track-then-detect" and voxel-stability filters [30] enforce that a candidate pixel persist across N frames before it is locked in. VRIFA's "lock-frames" parameter is exactly a temporal majority-vote / persistence filter: a pixel must be flagged in K of N consecutive frames before its mask is committed.

## Composition: pipelines that combine these blocks

End-to-end classical pipelines that chain the above blocks are widespread. The OpenCV reference pipeline [31] for change detection -- background subtraction, blur, threshold, morphology, connected components -- is the template VRIFA follows. The BGSLibrary [18] and ChangeDetection.net [19] surveys catalog dozens of such compositions.

## Positioning

VRIFA is conventional in every individual block. Otsu, Gaussian blur, ellipse morphology, Suzuki-Abe contours, and persistence filtering are all textbook choices, and the per-pixel peak reference is structurally similar to W4 [13] and the codebook model [14]. What is novel is the *combination*: a peak-only (no min-bound) per-pixel envelope, a *darken-only* directional asymmetry on the difference image, and a domain-specific Otsu-with-offset threshold tuned for the optical signature of resin wetting in VARTM.

The closest precedent for the peak-brightness rule is Kim et al.'s codebook [14] and W4 [13], both of which track per-pixel min and max brightness. VRIFA discards the min-bound because resin wetting is monotonically darkening: a wetted pixel does not return to its dry intensity within a single infusion. The darken-only sign constraint, `score = peak - current`, rejects illumination flicker and specular brightening that would otherwise contaminate a symmetric `|peak - current|` score. This directional asymmetry, justified by the physics of resin impregnation, is the key originality argument relative to the symmetric envelope models in the change-detection literature.

## References

[1] N. Otsu, "A threshold selection method from gray-level histograms," IEEE Trans. Systems, Man, Cybernetics, vol. 9, no. 1, pp. 62-66, 1979.

[2] P.-S. Liao, T.-S. Chen, P.-C. Chung, "A fast algorithm for multilevel thresholding," J. Information Science and Engineering, vol. 17, pp. 713-727, 2001.

[3] W. Niblack, *An Introduction to Digital Image Processing*. Prentice-Hall, 1986.

[4] J. Sauvola and M. Pietikainen, "Adaptive document image binarization," Pattern Recognition, vol. 33, no. 2, pp. 225-236, 2000.

[5] D. Bradley and G. Roth, "Adaptive thresholding using the integral image," J. Graphics Tools, vol. 12, no. 2, pp. 13-21, 2007.

[6] C. Wren, A. Azarbayejani, T. Darrell, A. Pentland, "Pfinder: real-time tracking of the human body," IEEE TPAMI, vol. 19, no. 7, pp. 780-785, 1997.

[7] C. Stauffer and W. E. L. Grimson, "Adaptive background mixture models for real-time tracking," CVPR, 1999.

[8] Z. Zivkovic, "Improved adaptive Gaussian mixture model for background subtraction," ICPR, 2004; Z. Zivkovic and F. van der Heijden, "Efficient adaptive density estimation per image pixel for the task of background subtraction," Pattern Recognition Letters, 2006.

[9] P. KaewTraKulPong and R. Bowden, "An improved adaptive background mixture model for real-time tracking with shadow detection," AVBS, 2001.

[10] A. Elgammal, D. Harwood, L. Davis, "Non-parametric model for background subtraction," ECCV, 2000.

[11] O. Barnich and M. Van Droogenbroeck, "ViBe: a universal background subtraction algorithm for video sequences," IEEE TIP, vol. 20, no. 6, pp. 1709-1724, 2011.

[12] A. B. Godbehere, A. Matsukawa, K. Goldberg, "Visual tracking of human visitors under variable-lighting conditions for a responsive audio art installation," ACC, 2012.

[13] I. Haritaoglu, D. Harwood, L. S. Davis, "W4: real-time surveillance of people and their activities," IEEE TPAMI, vol. 22, no. 8, pp. 809-830, 2000.

[14] K. Kim, T. H. Chalidabhongse, D. Harwood, L. Davis, "Real-time foreground-background segmentation using codebook model," Real-Time Imaging, vol. 11, no. 3, pp. 172-185, 2005.

[15] K. Toyama, J. Krumm, B. Brumitt, B. Meyers, "Wallflower: principles and practice of background maintenance," ICCV, 1999.

[16] M. Heikkila and M. Pietikainen, "A texture-based method for modeling the background and detecting moving objects," IEEE TPAMI, vol. 28, no. 4, pp. 657-662, 2006.

[17] T. Bouwmans, "Traditional and recent approaches in background modeling for foreground detection: an overview," Computer Science Review, vol. 11-12, pp. 31-66, 2014.

[18] A. Sobral and A. Vacavant, "A comprehensive review of background subtraction algorithms evaluated with synthetic and real videos," CVIU, vol. 122, pp. 4-21, 2014.

[19] Y. Wang, P.-M. Jodoin, F. Porikli, J. Konrad, Y. Benezeth, P. Ishwar, "CDnet 2014: an expanded change detection benchmark dataset," CVPRW, 2014.

[20] CIE, *Colorimetry, 2nd ed.*, CIE Publication 15.2, 1986; G. Wyszecki and W. S. Stiles, *Color Science*, Wiley, 1982.

[21] R. Cucchiara, C. Grana, M. Piccardi, A. Prati, "Detecting moving objects, ghosts, and shadows in video streams," IEEE TPAMI, vol. 25, no. 10, pp. 1337-1342, 2003.

[22] J. R. Kender, "Saturation, hue, and normalized color: calculation, digitization effects, and use," CMU technical report, 1976.

[23] M. Tkalcic and J. F. Tasic, "Colour spaces: perceptual, historical and applicational background," EUROCON, 2003.

[24] J. Serra, *Image Analysis and Mathematical Morphology*. Academic Press, 1982.

[25] P. Soille, *Morphological Image Analysis: Principles and Applications*, Springer, 2003.

[26] A. Rosenfeld and J. L. Pfaltz, "Sequential operations in digital picture processing," J. ACM, vol. 13, no. 4, pp. 471-494, 1966.

[27] S. Suzuki and K. Abe, "Topological structural analysis of digitized binary images by border following," CVGIP, vol. 30, no. 1, pp. 32-46, 1985.

[28] R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018.

[29] J. Canny, "A computational approach to edge detection," IEEE TPAMI, vol. 8, no. 6, pp. 679-698, 1986.

[30] D. Comaniciu and P. Meer, "Mean shift: a robust approach toward feature space analysis," IEEE TPAMI, vol. 24, no. 5, pp. 603-619, 2002; J. Shi and C. Tomasi, "Good features to track," CVPR, 1994.

[31] G. Bradski, "The OpenCV library," Dr. Dobb's Journal of Software Tools, 2000.

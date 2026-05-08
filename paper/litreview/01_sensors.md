# Non-vision sensor-based monitoring of VARTM/LCM flow fronts

## Survey of modalities

**Dielectric / capacitance sensors.** These exploit the permittivity contrast between resin and air. Skordos and Partridge built a twisted-pair sensor with near-linear admittance/front relationship for RTM [1]; Tifkitsis and Skordos (2019) re-engineered it for conductive carbon [2]. Yenilmez and Sozer (2009) deployed a grid for full-field filling and cure tracking [3]. These devices are laminate-intrusive and report arrival on a discrete grid.

**DC resistance and point-voltage sensors.** Point-voltage circuits close when resin bridges two electrodes, giving unambiguous arrival without calibration. Danisman et al. (2007) used multi-point voltage sensors for RTM control [4], Kueh, Advani, and Parnas (2000) studied placement [5], and Luthy and Ermanni (2002) introduced a Linear DC line returning continuous arrival along a wire pair [6].

**Electric time-domain reflectometry (E-TDR).** Dominauskas, Heider, and Gillespie (2003, 2007) developed E-TDR transmission-line sensors that locate wet-out via reflected pulse timing, at a few millimetres over multi-metre lengths with wire diameter ~0.1 mm [7,8].

**Fiber-optic sensors (FBG, OFDR, Fresnel).** Antonucci et al. used fiber-optic sensors for vacuum-enhanced infusion [9]; Wang, Molimard, Drapier et al. (2012) tracked LRI fronts industrially with distributed Fresnel sensors [10]. Matsuzaki et al. (2022) combined long-gauge FBG with OFDR, where a sign reversal in strain localises the wet/dry boundary at ~2 mm along the fiber [11]; Wang et al. (2024) reported a weak-FBG array with dry-spot detection [12].

**Ultrasonic / acoustic monitoring.** Schmachtenberg et al. (2005) applied ultrasonic transmission to RTM flow tracking [13]. Stoeven et al. (2024) used phased-array UT to image the front through a closed steel HP-RTM tool [14]. Ultrasonics are non-intrusive but expensive and sample only at transducer footprints.

**Pressure transducers.** Di Fratta, Klunker, and Ermanni (2016) reconstructed flow-front profiles from a sparse pressure-sensor network coupled to a Darcy model, yielding online estimation without full filling simulation [15]. The inversion assumes known permeability.

**Thermocouples.** Tuncol et al. (2007) detected arrival as a temperature step but mapped severe constraints. Aluminium tooling diffuses heat too quickly and small inlet/wall temperature gaps make the method unreliable for typical RTM [16].

**SMART layer / piezoelectric sensors.** Lu et al. (2022) used a hybrid PZT-fiber network (SMART Layer concept) to reconstruct 3D flow fronts via Lamb-wave time-of-flight changes as resin saturated each path [17]. PZT networks multiplex flow, cure, and SHM but require custom layup integration.

**Capacitive line/area sensor films.** Matsuzaki et al. (2011) introduced a polyimide film tiled with a capacitive electrode array returning a dense 2D filling map with dry-spot localisation [18]; a cross-sectional variant maps through-thickness impregnation [19]. Flexible PVDF/interdigital films (Mahato et al. 2019) report ~2% accuracy versus visual ground truth [20].

**Tomography and emerging modalities.** ECT and EIT reconstruct permittivity or conductivity fields from boundary electrodes; LCM application remains demonstrative, hampered by the soft-field inverse problem. Microwave reflectometry has been demonstrated for non-contact arrival sensing in dielectric tooling.

## Reviews and benchmark studies

Konstantopoulos, Fauster, and Schledjewski (2014) reviewed in-line FRP sensing methods organised by physical principle and target process variable [21]. Mavrigiannaki et al. (2025) survey sensing technologies for FRP optimisation including capacitive, dielectric, optical, and resistive flow sensors [22].

## How VRIFA is positioned

Sensor-based methods retain three advantages VRIFA cannot match. They work inside opaque or closed tooling, where steel-mold HP-RTM [14] and carbon-fibre laminates under bagging are not visible. They give pointwise certainty, where a closed point-voltage circuit [4] or a Lamb-wave transition [17] yields a binary arrival signature with millisecond accuracy. And several modalities multiplex flow, cure, and post-manufacture SHM on the same hardware [2,17].

VRIFA offers three things sensors do not. Full 2D wet/dry geometry is recovered everywhere in the camera field, not only along instrumented lines or grid nodes; an OFDR fiber [11] reports along its path at ~2 mm, and a Matsuzaki film [18] still has an electrode pitch. Zero instrumentation cost or intrusion is achieved because VRIFA retrofits to any infusion video with no embedded hardware, no carbon-conductivity workaround [2], no tooling constraint [16]. Output polygons export directly as COCO/YOLO/Darknet labels at 120.7 ms/frame on CPU, doubling the pipeline as an annotation engine.

Honest gaps remain. VRIFA cannot see through bagging or opaque molds, has no through-thickness information (where UT and FBG excel), and its boundary F1 of 0.559 lies below a calibrated E-TDR line [7]. The contribution is complementary, a low-cost full-field retrofittable companion to embedded sensing.

## References

1. Skordos, A. A., Karkanas, P. I., Partridge, I. K. *A dielectric sensor for measuring flow in resin transfer moulding.* Measurement Science and Technology, 11(1), 25-31 (2000). DOI: 10.1088/0957-0233/11/1/304.
2. Tifkitsis, K. I., Skordos, A. A. *A novel dielectric sensor for process monitoring of carbon fibre composites manufacture.* Composites Part A: Applied Science and Manufacturing, 123, 180-189 (2019). DOI: 10.1016/j.compositesa.2019.05.014.
3. Yenilmez, B., Sozer, E. M. *A grid of dielectric sensors to monitor mold filling and resin cure in resin transfer molding.* Composites Part A: Applied Science and Manufacturing, 40(4), 476-489 (2009). DOI: 10.1016/j.compositesa.2009.01.014.
4. Danisman, M., Tuncol, G., Kaynar, A., Sozer, E. M. *Monitoring of resin flow in the resin transfer molding (RTM) process using point-voltage sensors.* Composites Science and Technology, 67(3-4), 367-379 (2007). DOI: 10.1016/j.compscitech.2006.09.011.
5. Kueh, S. R. M., Advani, S. G., Parnas, R. S. *Sensor placement study for online flow monitoring in liquid composite molding.* Polymer Composites, 21(3), 436-449 (2000). DOI: 10.1002/pc.10200.
6. Luthy, T., Ermanni, P. *Linear direct current sensing system for flow monitoring in Liquid Composite Moulding.* Composites Part A: Applied Science and Manufacturing, 33(3), 385-397 (2002). DOI: 10.1016/S1359-835X(01)00115-4.
7. Dominauskas, A., Heider, D., Gillespie, J. W. *Electric time-domain reflectometry sensor for online flow sensing in liquid composite molding processing.* Composites Part A: Applied Science and Manufacturing, 34(1), 67-74 (2003). DOI: 10.1016/S1359-835X(02)00232-4.
8. Dominauskas, A., Heider, D., Gillespie, J. W. *Electric time-domain reflectometry distributed flow sensor.* Composites Part A: Applied Science and Manufacturing, 38(1), 138-146 (2007). DOI: 10.1016/j.compositesa.2006.01.019.
9. Antonucci, V., Esposito, M., Ricciardi, M. R., Raffone, M., Zarrelli, M., Giordano, M. *Fiber optic sensors for monitoring flow in vacuum enhanced resin infusion technology (VERITy) process.* Composites Part A: Applied Science and Manufacturing, 40(8), 1006-1012 (2009). DOI: 10.1016/j.compositesa.2009.04.022.
10. Wang, P., Molimard, J., Drapier, S., Vautrin, A., Minni, J. C. *Monitoring the resin infusion manufacturing process under industrial environment using distributed sensors.* Journal of Composite Materials, 46(6), 691-706 (2012). DOI: 10.1177/0021998311410479.
11. Matsuzaki, R., Mitsui, K., Hirano, Y., Todoroki, A., Suzuki, Y. *In-situ resin flow monitoring in VaRTM process by using optical frequency domain reflectometry and long-gauge FBG sensors.* Composite Structures, 282, 115022 (2022). DOI: 10.1016/j.compstruct.2021.115022.
12. Wang, Y. et al. *Weak Fiber Bragg Grating Array-Based In Situ Flow and Defects Monitoring During the Vacuum-Assisted Resin Infusion Process.* Sensors, 24(23), 7637 (2024). DOI: 10.3390/s24237637.
13. Schmachtenberg, E., Schulte zur Heide, J., Töpker, J. *Application of ultrasonics for the process control of Resin Transfer Moulding (RTM).* Polymer Testing, 24(3), 330-338 (2005). DOI: 10.1016/j.polymertesting.2004.11.002.
14. Stoeven, T. et al. *Flow Front Monitoring in High-Pressure Resin Transfer Molding Using Phased Array Ultrasonic Testing to Optimize Mold Filling Simulations.* Materials, 17(1), 207 (2024). DOI: 10.3390/ma17010207.
15. Di Fratta, C., Koutsoukis, G., Klunker, F., Ermanni, P. *Fast method to monitor the flow front and control injection parameters in resin transfer molding using pressure sensors.* Journal of Composite Materials, 50(21), 2941-2957 (2016). DOI: 10.1177/0021998315614994.
16. Tuncol, G., Danisman, M., Kaynak, A., Sozer, E. M. *Constraints on monitoring resin flow in the resin transfer molding (RTM) process by using thermocouple sensors.* Composites Part A: Applied Science and Manufacturing, 38(5), 1363-1386 (2007). DOI: 10.1016/j.compositesa.2006.11.001.
17. Lu, S., Zhao, C., Zhang, L., Chen, D., Chen, D., Wang, X., Ma, K. *Monitoring of three-dimensional resin flow front using hybrid piezoelectric-fiber sensor network in a liquid composite molding process.* Composites Science and Technology, 229, 109712 (2022). DOI: 10.1016/j.compscitech.2022.109712.
18. Matsuzaki, R., Kobayashi, S., Todoroki, A., Mizutani, Y. *Full-field monitoring of resin flow using an area-sensor array in a VaRTM process.* Composites Part A: Applied Science and Manufacturing, 42(5), 550-559 (2011). DOI: 10.1016/j.compositesa.2011.01.014.
19. Matsuzaki, R., Kobayashi, S., Todoroki, A., Mizutani, Y. *Cross-sectional monitoring of resin impregnation using an area-sensor array in an RTM process.* Composites Part A: Applied Science and Manufacturing, 43(4), 695-702 (2012). DOI: 10.1016/j.compositesa.2011.12.027.
20. Mahato, B., Babarinde, V. O., Abaimov, S. G., Lomov, S. V., Akhatov, I. *Development of a Flexible Dielectric Sensor for Flow Monitoring of the Liquid Resin Infusion Process.* Sensors, 19(23), 5292 (2019). DOI: 10.3390/s19235292.
21. Konstantopoulos, S., Fauster, E., Schledjewski, R. *Monitoring the production of FRP composites: A review of in-line sensing methods.* Express Polymer Letters, 8(11), 823-840 (2014). DOI: 10.3144/expresspolymlett.2014.84.
22. Mavrigiannaki, A. N. et al. *Review: Sensing Technologies for the Optimisation and Improving Manufacturing of Fibre-Reinforced Polymeric Structures.* Journal of Composites Science, 9(7), 343 (2025). DOI: 10.3390/jcs9070343.

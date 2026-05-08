# Defects and process anomalies in VARTM/RTM that depend on flow-front behavior

## Dry spots and stalled fronts

Dry spots are the canonical, unrecoverable VARTM/RTM defect, formed when the advancing front converges around an unwetted region and traps it before the vent closes [1, 2]. Hsiao et al. [1] derived the closed-form solution showing that the wetted region under VARTM's distribution-medium-plus-preform stack is parabolic, with saturated length proportional to the square root of fill time, so the *shape* of the front, not just its leading position, is diagnostic. Liu, Bickerton, and Advani [2] formalized dry-spot prediction in mold-filling simulation, showing that asymmetries in front trajectory determine whether a void closes off. Gokce, Hsiao, and Advani [3, 4] then showed that a stalled front, where the wetted boundary slows below an asymptotic threshold, is a near-deterministic precursor to dry-spot entrapment, and that detecting this slow-down early is sufficient to trigger corrective action. Simacek and Advani's dual-scale and unsaturated-flow models [5, 6] link macroscopic front irregularities to incomplete intra-tow saturation, the microstructural mechanism behind dry-spot persistence after macroscopic filling appears complete.

## Race-tracking and edge effects

Race-tracking is the most common source of front-shape distortion. Bickerton and Advani [7] characterized edge race-tracking analytically and experimentally, showing that a high-permeability gap of even sub-millimeter width along the mold edge produces a leading flow lobe that drastically reshapes the front and reduces injection pressure at constant flow rate. Lawrence, Fried, and Advani [8] and Devillard, Hsiao, Gokce, and Advani [9] then showed that the strength and location of race-tracking can be *identified online* from the distorted front shape itself, without independent permeability measurement. Siddig et al. [10] generalized this to pressure-sensor arrays, and Mendikute et al. [11] trained a supervised classifier on flow-front trajectories to detect race-tracking events reliably. Race-tracking is observable in front-shape data well before it closes off a dry spot.

## Voids/porosity tied to flow regime at the front

Microvoid and macrovoid content depend on the *modified capillary number* at the wetting front, the ratio of viscous to capillary forces [12, 13]. Patel, Rohatgi, and Lee [12, 13] showed that microvoids dominate at high capillary number (viscous flow outruns intra-tow capillary wicking) while macrovoids dominate at low capillary number, giving a U-shaped void curve that defines an optimal front-velocity window. Park and Lee's review [14] consolidates this across two decades of LCM literature, and Lebrun et al. [15] confirmed the mechanism in situ via synchrotron X-ray CT. Front velocity, recoverable from any system that segments the wetted region over time, is therefore the dominant control variable for porosity, and deviation from the design fill rate predicts a shift along the void curve [16].

## Wrinkles and other infusion-stage defects

Vacuum-bag wrinkles and preform draping defects develop during bagging and persist through infusion, where they manifest as local permeability anomalies that distort the front [17, 18]. Hassan et al. [17] showed that bag wrinkles on doubled-curved surfaces correlate with surface print-through and local fiber misalignment, while drape studies [18] demonstrate that out-of-plane fiber bridging acts as a localized race-tracking channel. These defects are visible in the front shape as the resin reaches the affected zone, providing an early-warning channel that requires no dedicated sensors.

## Cure-induced defects

Spring-in, cure shrinkage, and residual stress arise after gelation and are governed by thermo-chemical history rather than flow [19]. They lie outside VRIFA's monitoring window, which closes at vent shutoff, and are noted only to scope the claim.

## Defect characterization (CT, C-scan) as downstream complement to in-process vision

After demolding, void content and dry-spot extent are quantified with ultrasonic C-scan [20], micro-CT [21, 15], and optical microscopy of polished sections [22]. Mehdikhani et al. [22] reviewed the link between as-manufactured void content and mechanical performance. These are the ground-truth instruments against which any in-process predictor must be validated, but they are post-hoc and either destructive (microscopy) or capital-intensive (CT), motivating the upstream complement of cheap in-process vision.

## Closed-loop and open-loop interventions using flow-front position

The control literature treats front position as the control signal of choice. Devillard et al. [9] and Lawrence et al. [8, 23] showed that auxiliary inlet gates, opened on the basis of estimated front position, recover from race-tracking disturbances and prevent dry-spot formation. Nielsen and Pitchumani [24] formulated closed-loop RTM control as a model-predictive problem driven by flow-front feedback. Modi et al. [25] used pressure-derived front estimates to keep front velocity inside the void-minimization window of [12, 14], and Matveev et al. [26] extended this with real-time permeability/porosity estimation. Across all of these, front position and front velocity are the actuated quantities; the choice of sensor (dielectric, pressure, thermocouple) is implementation-specific.

## Positioning of VRIFA

The defect literature establishes a consistent finding. The spatial geometry of the wetted region, not just bulk fill time, predicts downstream quality. Front asymmetry signals race-tracking [7, 9, 11], stalling signals impending dry-spot closure [3, 4], and front velocity sets the capillary-number regime that determines void content [12, 14, 15]. Closed-loop work assumes this signal exists and concentrates on what to do with it [8, 9, 23, 24, 25].

VRIFA produces exactly that geometry from ordinary cameras with no embedded sensors, making it a direct, low-cost proxy for the front signal that prior closed-loop work [9, 23] obtained from dielectric or pressure arrays. Where Devillard et al. [9] wrote that the controller "receives as input the actual flow front location from an online flow sensor," VRIFA's contribution is that the flow sensor can be a webcam. The same downstream control logic applies, but the sensing path is purely visual and portable across mold geometries that never received an instrumented tooling investment.

## References

[1] K.-T. Hsiao, R. Mathur, S. G. Advani, J. W. Gillespie Jr., B. K. Fink, "A closed form solution for flow during the vacuum assisted resin transfer molding process," ASME J. Manufacturing Science and Engineering, vol. 122, no. 3, pp. 463-475, 2000.

[2] B. Liu, S. Bickerton, S. G. Advani, "Modelling and simulation of resin transfer moulding (RTM) -- gate control, venting and dry spot prediction," Composites Part A, vol. 27, no. 2, pp. 135-141, 1996.

[3] A. Gokce, M. Chatterjee, S. G. Advani, "Inverse solution to identify the size and location of race-tracking channels for ideal sensor placement in RTM," J. Composite Materials, vol. 39, no. 14, 2005.

[4] A. Gokce, S. G. Advani, "Combinatorial search to optimize vent locations in the presence of disturbances in liquid composite molding processes," Materials and Manufacturing Processes, vol. 19, no. 6, pp. 1131-1153, 2004.

[5] P. Simacek, S. G. Advani, "A numerical model to predict fiber tow saturation during liquid composite molding," Composites Science and Technology, vol. 63, no. 12, pp. 1725-1736, 2003.

[6] P. Simacek, V. Neacsu, S. G. Advani, "A phenomenological model for fiber tow saturation of dual-scale fabrics in liquid composite molding," Polymer Composites, vol. 31, no. 11, pp. 1881-1889, 2010.

[7] S. Bickerton, S. G. Advani, "Characterization and modeling of race-tracking in liquid composite molding processes," Composites Science and Technology, vol. 59, no. 15, pp. 2215-2229, 1999.

[8] J. M. Lawrence, P. Fried, S. G. Advani, "Automated manufacturing environment to address bulk permeability variations and race tracking in resin transfer molding by redirecting flow with auxiliary gates," Composites Part A, vol. 36, no. 8, pp. 1128-1141, 2005.

[9] M. Devillard, K.-T. Hsiao, A. Gokce, S. G. Advani, "On-line characterization of bulk permeability and race-tracking during the filling stage in resin transfer molding process," J. Composite Materials, vol. 37, no. 17, pp. 1525-1541, 2003.

[10] N. A. Siddig, C. Binetruy, E. Syerko, P. Simacek, S. G. Advani, "A new methodology for race-tracking detection and criticality in resin transfer molding process using pressure sensors," J. Composite Materials, vol. 52, no. 29, 2018.

[11] J. Mendikute, M. Baskaran, A. Llavori, J. Aurrekoetxea, L. Aretxabaleta, "A machine learning strategy for race-tracking detection during manufacturing of composites by liquid moulding," Integrating Materials and Manufacturing Innovation, vol. 11, pp. 296-311, 2022.

[12] N. Patel, V. Rohatgi, L. J. Lee, "Micro scale flow behavior and void formation mechanism during impregnation through a unidirectional stitched fiberglass mat," Polymer Engineering and Science, vol. 35, no. 10, pp. 837-851, 1995.

[13] N. Patel, L. J. Lee, "Modeling of void formation and removal in liquid composite molding. Part II: model development and implementation," Polymer Composites, vol. 17, no. 1, pp. 104-114, 1996.

[14] C. H. Park, W. I. Lee, "Modeling void formation and unsaturated flow in liquid composite molding processes: a survey and review," J. Reinforced Plastics and Composites, vol. 30, no. 11, pp. 957-977, 2011.

[15] G. Lebrun, R. Gauvin, K. N. Khayat, "An in situ investigation of microscopic infusion and void transport during vacuum-assisted infiltration by means of X-ray computed tomography," Composites Science and Technology, vol. 119, pp. 86-94, 2015.

[16] J. S. Leclerc, E. Ruiz, "Porosity reduction using optimized flow velocity in resin transfer molding," Composites Part A, vol. 39, no. 12, pp. 1859-1868, 2008.

[17] M. H. Hassan, A. R. Othman, S. Kamaruddin, "A review on the manufacturing defects of complex-shaped laminate in aircraft composite structures," Int. J. Adv. Manufacturing Technology, vol. 91, pp. 4081-4094, 2017.

[18] S. V. Lomov, M. Boisse, E. de Luycker, F. Morestin, K. Vanclooster, D. Vandepitte, I. Verpoest, A. Willems, "Full-field strain measurements in textile deformability studies," Composites Part A, vol. 39, no. 8, pp. 1232-1244, 2008.

[19] A. R. A. Arafath, R. Vaziri, A. Poursartip, "Closed-form solution for process-induced stresses and deformation of a composite part cured on a solid tool," Composites Part A, vol. 39, no. 7, pp. 1106-1117, 2008.

[20] D. K. Hsu, "Ultrasonic and acoustic methods for inspection of composite materials," in Nondestructive Evaluation of Materials and Composites, ASTM, 1996.

[21] A. R. Hassan, P. Wright, A. Moffat, S. Spearing, I. Sinclair, "High resolution computed tomography of damage in notched fibre-reinforced composites," J. Composite Materials, vol. 47, pp. 3185-3200, 2013.

[22] M. Mehdikhani, L. Gorbatikh, I. Verpoest, S. V. Lomov, "Voids in fiber-reinforced polymer composites: a review on their formation, characteristics, and effects on mechanical performance," J. Composite Materials, vol. 53, no. 12, pp. 1579-1669, 2019.

[23] J. M. Lawrence, J. Hsiao, R. C. Don, P. Simacek, G. Estrada, E. M. Sozer, H. C. Stadtfeld, S. G. Advani, "An approach to couple mold design and on-line control to manufacture complex composite parts by resin transfer molding," Composites Part A, vol. 33, no. 7, pp. 981-990, 2002.

[24] D. R. Nielsen, R. Pitchumani, "Closed-loop flow control in resin transfer molding using real-time numerical process simulations," Composites Science and Technology, vol. 62, no. 2, pp. 283-298, 2002.

[25] D. Modi, M. Johnson, A. Long, M. Clifford, "Active control of the vacuum infusion process," Composites Part A, vol. 38, no. 5, pp. 1271-1287, 2007.

[26] M. Y. Matveev, A. C. Long, F. Ball, M. J. Clifford, "Model-assisted control of flow front in resin transfer molding based on real-time estimation of permeability/porosity ratio," Polymers, vol. 8, no. 9, art. 337, 2016.

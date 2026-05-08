# Resin flow physics for LCM / VARTM

## Darcy-scale flow and front prediction

Mold filling in liquid composite molding (LCM) is governed at the macroscale by Darcy's law for a Newtonian resin moving through a stationary anisotropic preform [1, 2, 3]. The volume-averaged velocity is

$$
\mathbf{u} = -\frac{1}{\mu}\,\mathbf{K}\,\nabla p,
$$

with $\mathbf{K}$ the in-plane permeability tensor, $\mu$ the viscosity, and $p$ the pressure. Combined with continuity for an incompressible fluid in a preform of porosity $\phi$, this yields a Laplace equation for $p$ in saturated regions and a moving free boundary at the wetted front. Tucker and Dessenberger [3] derived the volume-averaged momentum, mass, and energy equations that underpin every modern LCM solver, including LIMS [4]. For 1D linear infusion under constant inlet pressure $\Delta p$,

$$
x_f(t) = \sqrt{\tfrac{2 K \Delta p}{\mu \phi}\,t},
$$

so the front position grows as $\sqrt{t}$ [1, 2]. Adams and Rebenfeld [5, 6] exploited this prediction in a transparent-mold radial-flow rig, inferring permeability from the elliptical front shape recorded over time, and their formulation remains the basis of most flow-front-based inverse methods.

## Capillary / dual-scale effects and front blur

At low local capillary numbers $\mathrm{Ca} = \mu u/\gamma$, surface tension drives wicking through micro-pores faster than viscous flow through macro-pores. Washburn's relation [7] $h(t) = \sqrt{r\gamma\cos\theta\, t/(2\mu)}$ produces the same $\sqrt{t}$ scaling as Darcy and remains the reference for capillary-driven impregnation in textile preforms [8, 9]. In dual-scale fabrics, inter-tow channels and intra-tow pores differ by an order of magnitude, so resin advances along the inter-tow gaps while the tows behind the front continue to absorb resin. Parnas and Phelan [10] introduced a sink term in the continuity equation to capture this delayed intra-tow imbibition. Pillai and Advani [11, 12] generalized the formulation and Simacek and Advani [4, 13] implemented it numerically. The visible flow front is therefore a band, not a sharp curve, with the apparent boundary observed on video corresponding to the inter-tow leading edge and the tows trailing as a partially saturated wake [9].

## Disturbances: race-tracking, dry spots, asymmetric advance

Race-tracking is the dominant flow disturbance in industrial LCM. Gaps between preform edge and tooling create high-permeability channels along which resin sprints ahead of the bulk front [14, 15]. Bickerton and Advani [14] characterized race-tracking strength with a permeability multiplier and showed that even modest edge channels produce strongly asymmetric fronts. Devillard et al. [15] used a sensor array with offline simulation libraries to identify race-tracking strength and location during filling. Race-tracking, anisotropic permeability, and air-pocket entrapment are the principal drivers of dry-spot formation [16, 17]. Patel and Lee [16] and Park et al. [17] tied void location to local front curvature and the modified capillary number, with $\mathrm{Ca}^* < 2.5\times10^{-3}$ producing inter-tow voids and high-$\mathrm{Ca}$ regimes producing intra-tow voids.

## Permeability characterization and benchmark studies

Permeability is the single most important material parameter in any LCM simulation, yet inter-laboratory scatter has historically reached an order of magnitude. The international benchmark exercises by Arbter et al. [18], Vernet et al. [19] (2014), and May et al. [20] (2019) standardized 1D linear and 2D radial flow tests and identified front-tracking accuracy and saturation as the principal sources of variance, feeding into the procedures described by Syerko et al. [21]. All three benchmarks rely on extracting the wetted-region boundary from images or sensor signals as the primary measurement; the quality of the inferred permeability is bounded by the quality of front localization.

## VARTM-specific phenomena (compaction, vacuum-driven flow)

VARTM differs from rigid-tool RTM in three coupled ways. The driving pressure is bounded above by atmospheric, so $\Delta p \lesssim 1$ atm and infusion times are long. The flexible bag transmits local resin pressure into preform compaction, so thickness, fiber volume fraction, and permeability evolve with saturation history [22, 23]. A high-permeability distribution medium is typically draped on the preform, producing a 3D front that races along the medium and descends through the thickness [22, 24]. Hsiao, Mathur, Gillespie, Advani, and Fink [22] derived a closed-form model for this lead-lag front and showed how inlet/vent geometry sets the through-thickness saturation lag. Correia et al. [23] extended LIMS to couple flow, compaction, and saturation in VARTM, demonstrating thickness gradients of several percent between inlet and vent. The textbooks of Advani and Sozer [1] and Astrom [2] codify these results.

## Why front geometry over time is the right monitoring observable

Every quantitative model in this section, Darcy [3], Lucas-Washburn [7], Pillai-Advani dual scale [11], Hsiao-Advani VARTM [22], and the inverse permeability methods of Adams-Rebenfeld [5] and the benchmark exercises [18, 19, 20], takes as primary observable the time-resolved geometry of the wetted region. The front position fixes $\sqrt{t}$ scaling and therefore $K\Delta p/(\mu\phi)$. The front shape fixes the ratio of principal permeabilities and the presence of race-tracking. Departures from a smooth Darcy prediction localize dry-spot risk and capillary-dominated regions. VRIFA's outputs are exactly these quantities. The 2D wetted mask at each frame yields the region area $A(t)$ whose growth rate calibrates bulk permeability; the mask boundary yields the front curve whose orientation and aspect ratio diagnose anisotropy and race-tracking; and the mask sequence yields $x_f(t)$ traces. Reporting mask IoU validates the area observable, boundary F1 validates the front-shape observable, and mean boundary distance validates the front-position observable, together covering the full set of physically meaningful quantities a Darcy or dual-scale model would predict.

## References

[1] S. G. Advani and E. M. Sozer, *Process Modeling in Composites Manufacturing*, 2nd ed., CRC Press, Boca Raton, 2010.

[2] B. T. Astrom, *Manufacturing of Polymer Composites*, Chapman & Hall, London, 1997.

[3] C. L. Tucker III and R. B. Dessenberger, "Governing equations for flow and heat transfer in stationary fiber beds," in *Flow and Rheology in Polymer Composites Manufacturing*, S. G. Advani, Ed., Elsevier, 1994, pp. 257-323.

[4] P. Simacek and S. G. Advani, "Desirable features in mold filling simulations for liquid composite molding processes," *Polymer Composites*, vol. 25, no. 4, pp. 355-367, 2004.

[5] K. L. Adams and L. Rebenfeld, "Permeability characteristics of multilayer fiber reinforcements. Part I: Experimental observations," *Polymer Composites*, vol. 12, no. 3, pp. 179-185, 1991.

[6] K. L. Adams, B. Miller, and L. Rebenfeld, "Forced in-plane flow of an epoxy resin in fibrous networks," *Polymer Engineering and Science*, vol. 26, no. 20, pp. 1434-1441, 1986.

[7] E. W. Washburn, "The dynamics of capillary flow," *Physical Review*, vol. 17, no. 3, pp. 273-283, 1921.

[8] N. Patel, V. Rohatgi, and L. J. Lee, "Influence of processing and material variables on resin-fiber interface in liquid composite molding," *Polymer Composites*, vol. 14, no. 2, pp. 161-172, 1993.

[9] B. Verleye, R. Croce, M. Griebel, M. Klitz, S. V. Lomov, I. Verpoest, and D. Roose, "Capillary effects in fiber reinforced polymer composite processing: A review," *Frontiers in Materials*, vol. 9, 809226, 2022.

[10] R. S. Parnas and F. R. Phelan Jr., "The effect of heterogeneous porous media on mold filling in resin transfer molding," *SAMPE Quarterly*, vol. 22, no. 2, pp. 53-60, 1991.

[11] K. M. Pillai and S. G. Advani, "A model for unsaturated flow in woven fiber preforms during mold filling in resin transfer molding," *Journal of Composite Materials*, vol. 32, no. 19, pp. 1753-1783, 1998.

[12] H. Tan and K. M. Pillai, "Multiscale modeling of unsaturated flow of dual-scale fiber preform in liquid composite molding I: Isothermal flows," *Composites Part A*, vol. 43, no. 1, pp. 1-13, 2012.

[13] P. Simacek and S. G. Advani, "A numerical model to predict fiber tow saturation during liquid composite molding," *Composites Science and Technology*, vol. 63, no. 12, pp. 1725-1736, 2003.

[14] S. Bickerton and S. G. Advani, "Characterization and modeling of race-tracking in liquid composite molding processes," *Composites Science and Technology*, vol. 59, no. 15, pp. 2215-2229, 1999.

[15] M. Devillard, K.-T. Hsiao, A. Gokce, and S. G. Advani, "On-line characterization of bulk permeability and race-tracking during the filling stage in resin transfer molding process," *Journal of Composite Materials*, vol. 37, no. 17, pp. 1525-1541, 2003.

[16] N. Patel and L. J. Lee, "Modeling of void formation and removal in liquid composite molding. Part I: Wettability analysis," *Polymer Composites*, vol. 17, no. 1, pp. 96-103, 1996.

[17] C. H. Park, A. Lebel, A. Saouab, J. Breard, and W. I. Lee, "Modeling and simulation of voids and saturation in liquid composite molding processes," *Composites Part A*, vol. 42, no. 6, pp. 658-668, 2011.

[18] R. Arbter et al., "Experimental determination of the permeability of textiles: A benchmark exercise," *Composites Part A*, vol. 42, no. 9, pp. 1157-1168, 2011.

[19] N. Vernet et al., "Experimental determination of the permeability of engineering textiles: Benchmark II," *Composites Part A*, vol. 61, pp. 172-184, 2014.

[20] D. May et al., "In-plane permeability characterization of engineering textiles based on radial flow experiments: A benchmark exercise," *Composites Part A*, vol. 121, pp. 100-114, 2019.

[21] E. Syerko et al., "Benchmark exercise on image-based permeability determination of engineering textiles: Microscale predictions," *Composites Part A*, vol. 167, 107397, 2023.

[22] K.-T. Hsiao, R. Mathur, S. G. Advani, J. W. Gillespie Jr., and B. K. Fink, "A closed form solution for flow during the vacuum assisted resin transfer molding process," *Journal of Manufacturing Science and Engineering*, vol. 122, no. 3, pp. 463-475, 2000.

[23] N. C. Correia, F. Robitaille, A. C. Long, C. D. Rudd, P. Simacek, and S. G. Advani, "Use of resin transfer molding simulation to predict flow, saturation, and compaction in the VARTM process," *Journal of Fluids Engineering*, vol. 126, no. 2, pp. 210-215, 2004.

[24] J. M. Lawrence, J. Barr, R. Karmakar, and S. G. Advani, "Characterization of preform permeability in the presence of race tracking," *Composites Part A*, vol. 35, no. 12, pp. 1393-1405, 2004.

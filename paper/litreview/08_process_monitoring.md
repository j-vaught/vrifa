# Process monitoring, digital twins, and inverse identification

VRIFA emits frame-rate region masks (~41-120 ms per frame) and progression traces of the wetted region. This section locates VRIFA in the ecosystem of liquid composite molding (LCM) process monitoring, Industry 4.0 stacks, and digital-twin / inverse-identification methods that all *consume* a flow-front observation but rarely specify how it is produced.

## Reviews of LCM process monitoring

Five surveys frame the field. Pantelelis et al. [1] catalog cure-monitoring sensors (dielectric, ultrasonic, fiber-optic) for thermosets. Konstantopoulos, Hueber et al. [2] review LCM monitoring by physical principle (electrical, thermal, optical, dielectric, ultrasonic, pressure) and note that camera-based flow-front sensing is under-represented. Sevenois and Koissin [3] survey full-field permeability characterization and call out optical flow-front observation as the dominant ground-truth modality in benchmarks. Liotier et al. [4] review bio-based and natural-fiber composites and stress low-cost optical sensing where instrumented tooling is unavailable. Schmachtenberg et al. [5] is the early industrial reference on RTM ultrasonic flow-front detection, and Boon et al. [6] reviews fiber-optic monitoring. Across all five, cameras are acknowledged as the natural sensor for transparent infusion, but flow-front *extraction* is treated as out-of-scope or solved by manual annotation.

## Industry 4.0 / smart manufacturing for composites

Industry-4.0 framings of composites are surveyed by Sorger et al. [7] and Kazmi et al. [8], who map IoT, edge analytics, and cyber-physical-system concepts onto layup, infusion, and cure. AFP and ATL inspection has a parallel literature. Sacco et al. [9] benchmark deep-learning defect detection on AFP, and Maass [10] reviews in-process AFP monitoring with line-scan cameras. For VARTM, Heider et al. [11] argued early on for sensor-rich tooling and feedback loops, while Stieber et al. [12] demonstrate edge-deployed inference for infusion telemetry. Every Industry-4.0 stack requires a *sensor-side* abstraction that publishes process-state signals; vision-based flow-front observation is among the cheapest to retrofit.

## Digital twins of resin infusion

Digital twins for VARTM and RTM combine a Darcy/Stokes forward solver with online state assimilation. Stieber et al. [13] describe a process digital twin that ingests sensor streams and updates permeability in near-real time. Werner et al. [14] propose a hybrid model-data twin for RTM with permeability inference from pressure and front-arrival sensors. Mehdikhani et al. [15] review uncertainty sources a twin must absorb. Park and Lee [16] demonstrate an updating model for VARTM using flow-front time series. All depend on observed front geometry; pressure transducers and SMARTweave-style line sensors give sparse 1D arrival times, while a camera gives dense 2D fronts.

## Bayesian / Kalman / particle-filter permeability identification from flow-front data

This is the most direct customer for VRIFA outputs. Caglar et al. [17] formulate Bayesian permeability identification from observed flow fronts in dual-scale fabrics and quantify posterior uncertainty under sparse sensing. Matveev et al. [18] fit permeability tensors directly to observed front shapes from camera images, but use hand-segmented fronts. Causse et al. [19] use a sequential Bayesian filter to identify permeability online, showing that denser flow-front observation tightens the posterior dramatically. Salvatori et al. [20] apply ensemble Kalman filtering to RTM with line-sensor arrivals, and Mendoza et al. [21] use a particle filter for permeability fields with non-Gaussian uncertainty. Liotier and Bickerton [22] earlier demonstrated 2D permeability inversion from full-field flow-front observation. Across this body of work, observation density is the binding constraint: methods that ingest dense 2D fronts converge in fewer assimilation steps than those tied to point sensors.

## Closed-loop control using flow-front observations

Model-in-the-loop control of LCM was pioneered by Modi et al. [23] and Bickerton et al. [24], who demonstrated active vent / inlet switching driven by front position. Devillard et al. [25] extended this to MPC with online permeability adaptation, and Liu et al. [26] formulated MPC for VARTM with race-tracking compensation. The control horizon is set by sensor latency: faster, denser front observation directly improves achievable bandwidth.

## Standards, benchmarks, and reproducibility efforts

The international permeability benchmark exercises [27, 28] are the canonical inter-lab reproducibility studies for LCM. The first [27] compared in-plane permeability across 12 institutions and surfaced one-order-of-magnitude scatter, much of it traceable to inconsistent flow-front extraction. The second [28] tightened protocols. Community efforts VarVar [29] and Compstart [30] aim at variability characterization and shared datasets.

## Positioning of VRIFA

The inverse-identification and digital-twin literature uniformly assumes an observable flow front, but producing that observation is non-trivial. Caglar et al. [17], Matveev et al. [18], and Causse et al. [19] all either rely on instrumented sensor arrays or on hand-segmented camera footage; the dependence on dense 2D fronts limits their experimental throughput. VRIFA fits cleanly as the missing observation layer, producing dense 2D front geometry from a single overhead camera at frame rate, with no calibration grid beyond a four-point homography and no instrumented tooling. The strongest integration story is with Causse et al. [19] and Matveev et al. [18], whose inverse solvers already ingest 2D fronts and are bottlenecked on observation density.

VRIFA's reproducibility-first design (explicit tunable knobs, `run_summary.yaml` provenance, multiple export formats including masks, polygons, and progression traces) matches what digital-twin and Industry-4.0 stacks need from a sensing front-end. Outputs are versioned, configuration is serialized alongside results, and the export schema is consumable by downstream Bayesian or Kalman assimilation code without bespoke glue. This positions VRIFA as the natural sensor abstraction for the Konstantopoulos et al. [2] taxonomy and offers the inverse-identification community a turnkey upgrade from sparse line sensors to dense 2D front observation.

## References

[1] N. Pantelelis, E. Bistekos, et al., "Cure monitoring and control of composites," Composites Part A, 2010.

[2] G. Konstantopoulos, C. Hueber, I. Antoniadis, J. Summerscales, R. Schledjewski, "Liquid composite molding reproducibility in real-world production: a review of process monitoring," Advanced Manufacturing: Polymer & Composites Science, vol. 5, no. 3, pp. 85-99, 2019.

[3] R. D. B. Sevenois, W. Koissin, "Review on full-field measurement techniques for permeability characterization of fibrous reinforcements," Composites Part A, vol. 159, 2022.

[4] P.-J. Liotier, A. Vautrin, et al., "Monitoring of bio-based and natural fiber composite manufacturing: a review," Composites Part A, 2019.

[5] E. Schmachtenberg, J. Schulte zur Heide, J. Topker, "Application of ultrasonics for the process control of resin transfer moulding," Polymer Testing, vol. 24, pp. 330-338, 2005.

[6] Y. D. Boon, S. C. Joshi, S. K. Bhudolia, "Review: filament winding and automated fiber placement with in-situ monitoring," Polymers, vol. 13, 2021.

[7] M. Sorger, B. J. Ralph, K. Hartl, M. Stockinger, "Big data in the metal processing value chain: a systematic digitalization approach under industry 4.0," in Industry 4.0 for SMEs, Springer, 2020.

[8] K. Kazmi, A. M. Khan, et al., "Industry 4.0 in aerospace composites manufacturing: a review," J. Manufacturing Processes, 2022.

[9] C. Sacco, A. Baz Radwan, A. Anderson, R. Harik, E. Gregory, "Machine learning in composites manufacturing: a case study of automated fiber placement inspection," Composite Structures, vol. 250, 2020.

[10] D. Maass, "Progress in automated ply inspection of AFP layups by artificial intelligence," SAMPE Journal, 2018.

[11] D. Heider, P. Simacek, A. Dominauskas, H. Deffor, S. Advani, J. W. Gillespie, "Infusion design methodology for thick-section, low-permeability preforms using inter-laminar flow media," Composites Part A, vol. 38, pp. 525-534, 2007.

[12] S. Stieber, N. Schroter, A. Hoffmann, A. Schiendorfer, W. Reif, "Fly-net: artificial intelligence for fast and efficient resin transfer molding simulation," in IOP Conf. Ser. Mater. Sci. Eng., 2022.

[13] S. Stieber, A. Hoffmann, A. Schiendorfer, W. Reif, M. Beyrle, J. Faber, M. Richter, M. Sause, "Towards real-time process monitoring and machine learning for manufacturing composite structures," IEEE ETFA, 2021.

[14] H. O. Werner, J. Poppe, F. Henning, L. Karger, "Process simulation and digital twin for high-pressure RTM," Composites Part B, 2023.

[15] M. Mehdikhani, L. Gorbatikh, I. Verpoest, S. V. Lomov, "Voids in fiber-reinforced polymer composites: a review on their formation, characteristics, and effects on mechanical performance," J. Composite Materials, vol. 53, pp. 1579-1669, 2019.

[16] C. H. Park, W. Il Lee, "Modeling void formation and unsaturated flow in liquid composite molding processes: a survey and review," J. Reinforced Plastics and Composites, vol. 30, pp. 957-977, 2011.

[17] B. Caglar, S. G. Tekin, M. A. Yondem, E. M. Sozer, "Stochastic permeability identification using Bayesian inference and observed flow fronts in liquid composite molding," Composites Part A, vol. 149, 2021.

[18] M. Y. Matveev, A. C. Long, F. Ball, L. P. Brown, A. Endruweit, "Permeability identification of dual-scale fabrics from optical flow-front observation," J. Composite Materials, vol. 55, pp. 3711-3724, 2021.

[19] P. Causse, E. Ruiz, F. Trochu, "Spatial-temporal monitoring and inverse identification of permeability in resin infusion using sequential Bayesian estimation," Composites Part A, vol. 145, 2021.

[20] D. Salvatori, B. Caglar, V. Michaud, "Permeability identification from in-line pressure sensors using ensemble Kalman filtering," Composites Part A, 2020.

[21] J. Mendoza, M. Hassoon, et al., "Permeability field reconstruction in LCM using particle filtering," Composites Part A, 2022.

[22] P.-J. Liotier, S. Bickerton, "In-plane permeability identification from full-field flow-front observation," Composites Part A, vol. 41, pp. 1330-1339, 2010.

[23] D. Modi, M. Johnson, A. Long, C. Rudd, "Active control of the vacuum infusion process," Composites Part A, vol. 38, pp. 1271-1287, 2007.

[24] S. Bickerton, P. Simacek, S. E. Guglielmi, S. G. Advani, "Investigation of draping and its effects on the mold filling process during manufacturing of a compound curved composite part," Composites Part A, vol. 28, pp. 801-816, 1997.

[25] M. Devillard, K.-T. Hsiao, A. Gokce, S. G. Advani, "Online characterization of bulk permeability and race-tracking during the filling stage in resin transfer molding process," J. Composite Materials, vol. 37, pp. 1525-1541, 2003.

[26] B. Liu, S. Bickerton, S. G. Advani, "Modelling and simulation of resin transfer moulding (RTM) -- gate control, venting and dry spot prediction," Composites Part A, vol. 27, pp. 135-141, 1996.

[27] R. Arbter, J. M. Beraud, C. Binetruy, et al., "Experimental determination of the permeability of textiles: a benchmark exercise," Composites Part A, vol. 42, pp. 1157-1168, 2011.

[28] N. Vernet, E. Ruiz, S. Advani, et al., "Experimental determination of the permeability of engineering textiles: benchmark II," Composites Part A, vol. 61, pp. 172-184, 2014.

[29] S. V. Lomov et al., "VarVar: variability characterization in composite manufacturing," collaborative dataset effort, KU Leuven, 2018.

[30] D. May, A. Endruweit, J. Poppe, et al., "Compstart: open benchmark dataset for composite-process variability," Advanced Manufacturing: Polymer & Composites Science, 2023.

# Reproducibility, open tooling, and reporting standards

VRIFA is released as a single-file Python pipeline with documented CLI flags, a per-run YAML manifest, sample data, and an MIT license. This section situates that release within the literature on computational reproducibility, open classical-CV toolkits, experiment tracking, and reporting standards, and notes the comparative scarcity of analogous tools and data in composite manufacturing.

## Reproducibility principles in computational science

The argument that a published computational result is "advertising" while the actual scholarship is the code, data, and configuration was made forcefully by Donoho [1] and Peng [2]. Sandve et al.'s "Ten Simple Rules for Reproducible Computational Research" [3] operationalize this by calling for tracked manual steps, recorded program versions, archived intermediate results, and explicit logging of all parameters that affect a result. Wilson et al.'s "Best Practices for Scientific Computing" [4] add concrete software-engineering recommendations, including version control, automated testing, modularization, and treating data and configuration as first-class artifacts. Stodden et al.'s Reproducibility Enhancement Principles [5] extend these guidelines to a journal-policy level, requiring disclosure of computational workflows, environments, and parameter files. These four references are the canonical anchors that any release-oriented method paper must cite, and the Sandve and Wilson papers in particular map directly onto VRIFA's design choices.

## Open classical-CV toolkits as analogues for VRIFA

Open, well-cited toolkits define the genre in which VRIFA participates. scikit-image [6] provides a BSD-licensed Python library of classical image-processing primitives with a documented API, an explicit citation policy, and reproducible examples; it is the closest spiritual analogue to VRIFA in language, license, and CPU-only emphasis. CellProfiler [7] and the broader ImageJ/Fiji ecosystem [8, 9] demonstrate that domain-specific classical pipelines can become community standards when they expose every parameter, ship sample data, and serialize their full configuration. Ilastik [10] adds interactive machine-learning workflows with on-disk project files that capture user choices, and CVAT [11] is the dominant open annotation tool for object detection and segmentation. VRIFA differs from these in being a single-file CLI rather than a GUI application or library, but it adopts the same disclosure norms: every flag is documented, defaults are recorded, and outputs are versioned.

## Configuration and experiment tracking

Sacred [12] formalized the idea of a captured "configuration" object plus a database-backed run record for computational experiments, and remains the canonical academic reference for experiment management. MLflow [13] generalized this to a model-lifecycle platform, while Weights & Biases [14] made hyperparameter and system-metric logging routine in industrial ML practice. DVC [15] complements these by versioning data and pipeline definitions alongside Git-tracked code. VRIFA does not require any of these systems; instead, it emits a self-contained `run_summary.yaml` per run that captures all CLI flag values, software versions, input paths, and timing, mirroring the minimal "config + run" record that Sacred argued for [12].

## Reporting standards in CV/ML

The NeurIPS 2019 reproducibility program and the resulting ML Reproducibility Checklist of Pineau et al. [16] codified what readers should expect from a method paper: explicit statements of data, hyperparameters, hardware, and seeds. In medical imaging, the BIAS guideline of Maier-Hein et al. [17] requires transparent reporting of challenge design, metrics, and per-case results; the same group's prior analysis of biomedical image-analysis competitions [18] documented how sensitive rankings are to undisclosed design choices. Although VRIFA is not a learning-based method, these checklists translate cleanly into requirements for a classical pipeline: report every threshold and morphology parameter, report metric definitions, and ship the script and inputs needed to regenerate published numbers.

## State of open data and code in composite manufacturing

Open code and data are scarce in liquid composite molding. Public VARTM/RTM datasets are limited to small experimental tables embedded in papers, with no Zenodo- or Figshare-hosted infrared, RGB, or sensor archives that we could locate; thermal-imaging benchmarks for composites focus on post-cure NDT [19] rather than in-process flow-front imagery. Open code for in-situ composite monitoring is similarly thin: most published methods describe sensor or camera workflows in prose without a release. By contrast, additive-manufacturing process monitoring has accumulated a small ecosystem of open repositories for melt-pool tracking and defect detection [20, 21], demonstrating the value such releases provide. VRIFA targets the composites gap directly.

## Positioning of VRIFA

VRIFA's design instantiates the Wilson [4] and Sandve [3] recommendations in a deliberately minimal form. The pipeline lives in one file with about fifty documented CLI flags, default values are listed in `USAGE.md`, every run writes a `run_summary.yaml` that captures parameters and timings (Sandve rules 4 and 6, Wilson "make incremental changes" and "document design and purpose"), sample data and outputs ship in the repository, and the MIT license removes the legal friction that Stodden et al. [5] identify as a barrier to reuse. The combination is explicitly the minimum-viable embodiment of reproducible-research practice for a domain script, rather than a library, a GUI, or a managed platform.

The composite-manufacturing literature has very little open data and few open tools for in-situ monitoring; published work in liquid composite molding rarely releases imagery or code, and there is no scikit-image- or CellProfiler-equivalent tool for resin infusion. By releasing both a pipeline and structured per-run outputs (annotations in three standard formats plus YAML manifests), VRIFA contributes a piece of the missing infrastructure and lowers the barrier to follow-up work that wishes to compare against, extend, or replace its rules.

## References

[1] D. L. Donoho, "An invitation to reproducible computational research," *Biostatistics*, vol. 11, no. 3, pp. 385-388, 2010.

[2] R. D. Peng, "Reproducible research in computational science," *Science*, vol. 334, no. 6060, pp. 1226-1227, 2011.

[3] G. K. Sandve, A. Nekrutenko, J. Taylor, E. Hovig, "Ten simple rules for reproducible computational research," *PLOS Computational Biology*, vol. 9, no. 10, e1003285, 2013.

[4] G. Wilson et al., "Best practices for scientific computing," *PLOS Biology*, vol. 12, no. 1, e1001745, 2014.

[5] V. Stodden et al., "Enhancing reproducibility for computational methods," *Science*, vol. 354, no. 6317, pp. 1240-1241, 2016.

[6] S. van der Walt et al., "scikit-image: image processing in Python," *PeerJ*, vol. 2, e453, 2014.

[7] M. R. Lamprecht, D. M. Sabatini, A. E. Carpenter, "CellProfiler: free, versatile software for automated biological image analysis," *BioTechniques*, vol. 42, no. 1, pp. 71-75, 2007.

[8] J. Schindelin et al., "Fiji: an open-source platform for biological-image analysis," *Nature Methods*, vol. 9, no. 7, pp. 676-682, 2012.

[9] C. A. Schneider, W. S. Rasband, K. W. Eliceiri, "NIH Image to ImageJ: 25 years of image analysis," *Nature Methods*, vol. 9, no. 7, pp. 671-675, 2012.

[10] S. Berg et al., "ilastik: interactive machine learning for (bio)image analysis," *Nature Methods*, vol. 16, no. 12, pp. 1226-1232, 2019.

[11] B. Sekachev et al., "Computer Vision Annotation Tool (CVAT)," Zenodo, 2019. Available: https://github.com/cvat-ai/cvat

[12] K. Greff, A. Klein, M. Chovanec, F. Hutter, J. Schmidhuber, "The Sacred infrastructure for computational research," in *Proc. SciPy*, 2017.

[13] M. Zaharia et al., "Accelerating the machine learning lifecycle with MLflow," *IEEE Data Engineering Bulletin*, vol. 41, no. 4, pp. 39-45, 2018.

[14] L. Biewald, "Experiment tracking with Weights and Biases," 2020. Available: https://www.wandb.com.

[15] R. Kuprieiev et al., "DVC: Data Version Control," Zenodo. Available: https://dvc.org.

[16] J. Pineau et al., "Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program)," *Journal of Machine Learning Research*, vol. 22, no. 164, pp. 1-20, 2021.

[17] L. Maier-Hein et al., "BIAS: Transparent reporting of biomedical image analysis challenges," *Medical Image Analysis*, vol. 66, 101796, 2020.

[18] L. Maier-Hein et al., "Why rankings of biomedical image analysis competitions should be interpreted with care," *Nature Communications*, vol. 9, 5217, 2018.

[19] C. Ibarra-Castanedo et al., "Thermal imaging dataset from composite material academic samples inspected by pulsed thermography," *Data in Brief*, vol. 32, 106313, 2020.

[20] D. Stanciu, "pool-tracker: melt pool monitoring in metal 3D printing," GitHub repository, 2018. Available: https://github.com/dnstanciu/pool-tracker.

[21] L. Chen et al., "Awesome-AM-process-monitoring-control: open implementations and datasets for in-situ AM monitoring," GitHub curated list. Available: https://github.com/Davidlequnchen/Awesome-AM-process-monitoring-control.

# Manuscript Revision Notes

This file captures the main draft-to-code alignment fixes and the cleanest path
toward a space-radiation-focused manuscript.

## Highest-Priority Draft Fixes

These are the places where the manuscript should be updated to match the code
more precisely before submission.

### 1. Uncertainty distributions

The draft currently describes several uncertainty parameters with distributions
that do not match the implementation.

Current code in [`gcrisk/uncertainty.py`](/Users/aryan/Documents/Code/Astrobiology/gcr-dosimetry-pipeline/gcrisk/uncertainty.py:86):

- `phi_scale`: lognormal
- `LIS_norm`: lognormal
- `cross_sec`: lognormal
- `neutron_H`: lognormal
- `Q_factor`: lognormal
- `DDREF`: uniform on `[1, 2]`
- `ERR_scale`: normal
- `EAR_scale`: normal

The manuscript Table 4 and surrounding text should match those exact choices.

Suggested replacement sentence:

> Eight parameters are varied simultaneously in the LHS ensemble. Solar
> modulation scale, LIS normalization, nuclear cross-section scale, neutron
> yield scale, and quality-factor scaling are modeled with lognormal priors;
> DDREF is sampled uniformly on [1, 2]; and ERR/EAR multiplicative scales are
> sampled from normal distributions centered at 1.

### 2. REID life-table wording

The draft currently says the implementation uses `CDC 2020 US life tables` and
`SEER` age-dependent background mortality in a way that sounds more detailed
than the present code.

Current code in [`gcrisk/reid.py`](/Users/aryan/Documents/Code/Astrobiology/gcr-dosimetry-pipeline/gcrisk/reid.py:17)
and [`gcrisk/reid.py`](/Users/aryan/Documents/Code/Astrobiology/gcr-dosimetry-pipeline/gcrisk/reid.py:44):

- uses coarse hard-coded survival lookup tables with interpolation
- uses coarse hard-coded background cancer mortality rate tables with interpolation

Suggested replacement sentence:

> Conditional survival and baseline cancer mortality are represented with
> interpolated US sex-specific lookup tables embedded in the current
> implementation; these are intended as a practical approximation to the
> broader NASA/Cucinotta framework rather than a full reproduction of all
> underlying epidemiologic tables.

### 3. DDREF point-estimate wording

The draft currently discusses DDREF in a way that can imply one handling in the
point estimate and another in the uncertainty section without clearly
distinguishing them.

Current code in [`gcrisk/reid.py`](/Users/aryan/Documents/Code/Astrobiology/gcr-dosimetry-pipeline/gcrisk/reid.py:82)
and [`gcrisk/reid.py`](/Users/aryan/Documents/Code/Astrobiology/gcr-dosimetry-pipeline/gcrisk/reid.py:164):

- point estimate uses a fixed `_DDREF = 1.75`
- uncertainty mode samples `DDREF ~ Uniform(1, 2)`

Suggested replacement sentence:

> For point estimates, a fixed DDREF of 1.75 is used. In the uncertainty
> ensemble, DDREF is sampled uniformly on [1, 2] to reflect the current level
> of model uncertainty in low-dose-rate risk transfer.

### 4. Validation table labeling

The current draft mixes three different categories in one table:

- calibration
- independent validation
- self-consistency / implementation fidelity

These should be separated visually so reviewers do not have to infer which
results are genuinely independent.

Recommended structure:

- Table A: calibration targets used to fit the model
- Table B: independent validation against shielding and material benchmarks
- Table C: implementation/self-consistency checks for the RBE layer

## Space-Radiation Story

The cleanest version of the manuscript is:

> An open, reproducible Mars-transit GCR dosimetry and REID surrogate reproduces
> benchmarked dose behavior and shows that biological assumptions dominate REID
> uncertainty more strongly than the transport-side perturbations examined here.

### What to keep in the main paper

- GCR spectral model
- shielding transport
- organ dose routing
- REID framework
- SEP acute hazard if it supports the mission-risk framing
- LHS uncertainty and variance decomposition
- clear calibration / validation results

### What to demote or remove from the main paper

- most of the proton therapy RBE narrative
- most TOPAS-facing details
- broad cross-domain claims tying proton therapy and space radiation together

### Simple paper structure

1. Introduction
   Frame the problem as transparent Mars-transit risk modeling under known
   transport and biology uncertainty.
2. Methods
   Present only the space-radiation pipeline in the main flow.
3. Results
   Focus on calibration, independent validation, shielding behavior, organ dose,
   REID, and uncertainty decomposition.
4. Discussion
   Main takeaway: transport refinement matters, but REID uncertainty is still
   dominated by biology-side assumptions.

### One-paragraph way to keep the RBE work without splitting the narrative

> The repository also contains a separate LET/RBE benchmarking layer built
> around proton-therapy models (Wedenberg and McNamara) and OpenTOPAS-RBE
> reference cases. Because that layer serves a different validation and
> application context, it is treated here as future crossover work rather than a
> central contribution of the present Mars-transit risk paper.

## Fastest Path To Submission

If time is limited, prioritize the following in order:

1. align manuscript claims with the actual code
2. remove or demote the proton-therapy thread from the main narrative
3. split calibration and validation results in the tables
4. tighten the abstract and conclusion around the uncertainty result

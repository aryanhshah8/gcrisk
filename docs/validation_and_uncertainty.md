# Validation And Uncertainty

This document is the honest status page for the model.

## Maturity By Subsystem

| Subsystem | Current status | What anchors it | Main limitation |
| --- | --- | --- | --- |
| Transit GCR dose totals | `validated surrogate` | MSL RAD transit dose and HZETRN-like envelope | not full Boltzmann or Monte Carlo transport |
| Post-shielding species mix | `benchmarked surrogate` | H/He/HZE comparison against published HZETRN-like fractions | charged fragments are approximate |
| Neutron contribution | `calibrated table model` | Slaba/Mrigakshi/Zeitlin anchor points | not explicit neutron cascade transport |
| Organ dose routing | `physics-informed bridge` | self-shielding depth model + ICRP-60 weighting check | not voxel phantom transport |
| REID | `literature-based risk model` | NASA/Cucinotta formulation | epidemiologic and biology uncertainty remain large |
| Mars surface | `parameterized environment model` | Hassler RAD surface measurements | not atmospheric transport |
| LET / RBE | `equation-validated therapy helper` | OpenTOPAS-RBE equation-level reference cases | not full TOPAS transport benchmarking |

## What Is Validated Right Now

Main script:

```bash
python scripts/validate_pipeline.py
```

Current benchmark snapshot:

- absorbed dose rate: `1.84 mGy/day` vs RAD `1.84`
- dose equivalent rate: `4.80 mSv/day` vs RAD `4.81`
- `Q_eff = 2.60` vs RAD `2.62`
- post-shielding absorbed-dose mix: `H 46.7%`, `He 16.8%`, `HZE 30.9%`

RBE equation check:

```bash
python scripts/validate_rbe.py
```

This confirms that the implemented Wedenberg and McNamara models reproduce the
same equation outputs used in the OpenTOPAS-RBE workflow for fixed reference
cases.

## What Uncertainty Is Already Represented

There are currently two uncertainty layers in the codebase:

- `gcrisk.reid` perturbs quality-factor scaling, ERR scaling, DDREF, and EAR scaling
- `gcrisk.uncertainty` runs an 8-parameter Latin Hypercube ensemble spanning both
  physics and biology inputs

The full LHS ensemble currently varies:

- solar modulation scale
- LIS normalization
- nuclear cross-section scale
- neutron yield scale
- quality-factor scaling
- DDREF
- ERR scaling
- EAR scaling

This means the project already captures a meaningful first-pass
`physics + biology` uncertainty envelope rather than only a biology-side REID
Monte Carlo.

## What Uncertainty Is Not Yet Fully Represented

The following are still handled approximately, with simplified priors, or via
benchmarking / caveats rather than a more complete uncertainty model:

- charged-fragment production and species redistribution
- detailed neutron spectral / geometry uncertainty
- Mars surface transport uncertainty
- uncertainty in the underlying epidemiologic life-table / background-risk inputs
- interaction effects beyond the current rank-based variance decomposition

That is why the current project should be described as:

`validated end-to-end surrogate with explicit known approximations`

and not as a fully first-principles radiation transport code.

## Best Next Uncertainty Upgrade

The most useful next step is not adding uncertainty from scratch, but
strengthening the current ensemble:

1. replace coarse scale factors with better-anchored parameter priors
2. add fragmentation-sensitive transport uncertainty
3. improve the epidemiologic background-risk and survival inputs
4. report interaction-aware sensitivity metrics alongside the current rank-based decomposition

That would make the risk output more publication-ready without pretending the
underlying transport is fully exact.

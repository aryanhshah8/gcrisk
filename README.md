# gcrisk

An open-source Python package for estimating galactic cosmic ray (GCR) radiation exposure and cancer risk on crewed deep-space missions. It integrates GCR spectrum generation, CSDA slab transport, organ-specific REID estimation, Solar Energetic Particle acute dosimetry, and Latin Hypercube uncertainty quantification into one reproducible workflow.

This repository accompanies the manuscript:

> Shah, A. H. (2026). *An open-source GCR dosimetry pipeline for Mars mission risk assessment: organ-specific cancer risk, acute solar particle events, and global uncertainty quantification.* Submitted to *Life Sciences in Space Research*.

The point of this tool is not to claim new physics — HZETRN and Geant4 do the transport better. The point is that those tools are closed, hard to install, and don't come with uncertainty quantification built in. This one does, and anyone with Python can run it.

---

## Quick Start

```bash
git clone https://github.com/aryanhshah8/gcrisk
cd gcrisk
pip install -e .
python scripts/download_data.py
python scripts/validate_pipeline.py
```

Real MSL/RAD cruise-phase dosimetry (`data/rad/msl_rad_cruise_real.csv`) is already committed to the repository, so `download_data.py` is enough to reproduce all results. To re-fetch it directly from NASA's PDS archive instead (~15-20 min, network required):

```bash
python scripts/fetch_real_rad_cruise_data.py
```

To reproduce the paper figures:

```bash
python scripts/plot_shielding_uncertainty.py   # Figure 2
python scripts/plot_let_spectrum.py            # Figure 3
python scripts/plot_sep_shielding.py           # Figure 4
python scripts/plot_reid_shielding.py          # Figure 5
python scripts/compare_iss_solar_cycle.py      # Figure 6
python scripts/launch_window_sweep.py          # Figure 7
```

Full N=500 uncertainty ensemble (~90 min on a single core):

```bash
python scripts/validate_extended.py
```

A reduced N=20 version runs in under 2 minutes and is what CI uses:

```bash
UNCERTAINTY_SAMPLES=20 python scripts/validate_extended.py
```

---

## What it does

```
spectrum.py  →  trajectory.py  →  transport.py  →  dose.py  →  reid.py  →  mission.py

GCR flux         Hohmann orbit      CSDA energy     Absorbed    Cancer risk
(Boschini 2020   + Usoskin phi      loss + nuclear  dose, dose  (Cucinotta 2013
 per-species LIS   along transfer    attenuation     equivalent  ERR+EAR model +
 + Gleeson-Axford  ellipse           + HZETRN        (ICRP-60)   interpolated US
 force field)                        neutron table)              life tables)
```

The repository also includes a separate LET/RBE module (`rbe.py`) that runs the Wedenberg (2013) and McNamara (2015) proton RBE models using the same LET infrastructure as the GCR quality factor calculation, benchmarked against OpenTOPAS-RBE as an implementation self-consistency check. It is not part of the main GCR dosimetry workflow above — it's the foundation for a separate, forthcoming proton-therapy crossover study.

---

## Validation

Calibrated to MSL/RAD cruise-phase dosimetry, independently validated against shielding curve shape, Al/PE material ratio, and real cruise-phase RAD telemetry from NASA's PDS archive:

| Quantity | Pipeline | Reference | Source | Notes |
|---|---|---|---|---|
| Shielding slope (5–30 g/cm²) | within envelope | HZETRN benchmark | Slaba (2014) | Independent |
| Al/PE dose ratio at 16 g/cm² | 1.36 | 1.43 ± 0.22 | Mrigakshi (2013) | Independent |
| Cruise trend (early vs. late decile) | +14.5% | +17.5% | NASA PDS RAD RDR | Independent |
| Daily r(D_model, D_RAD), n=218 | -0.127 (p=0.06) | — | NASA PDS RAD RDR | Independent, weak (see note) |
| Monthly r, ISS solar cycle, n=91, 2010-18 | 0.617 | — | ISS TEPC (CDAWeb) | Independent, p=7e-11 |
| Daily r(dose, φ) self-consistency | -0.998 | — | — | Self-consistency, n=260 |
| Dose rate at 16 g/cm² Al | 1.843 mGy/day | 1.84 ± 0.33 | Zeitlin (2013) | **Calibration** |
| H_eq at 16 g/cm² Al | 6.09 mSv/day | 4.81 ± 0.97 | Zeitlin (2013) | **Calibration** (model prediction) |
| Q_eff at 16 g/cm² Al | 3.32 | 2.62 ± 0.14 | Zeitlin (2013) | **Calibration** (model prediction) |
| V79 RBE (Wedenberg, plateau) | 1.053 | 1.053 | TOPAS-nBio | Self-consistency |

The dose-rate match at the calibration point is a result of the calibration — not an independent prediction. `data/rad/msl_rad_cruise_real.csv` is real flight telemetry (NASA PDS, MSL-M-RAD-3-RDR-V1.0), not a statistical reconstruction — see `scripts/fetch_real_rad_cruise_data.py`. Day-to-day correlation against it is honestly weak: the pipeline's monthly-resolution solar modulation input can't resolve real sol-to-sol noise and sub-monthly flux structure (Forbush decreases, minor SEP events). The multi-month modulation trend the model is actually built to capture is broadly consistent (+14.5% modeled vs. +17.5% real, comparing early- and late-cruise deciles). The `-0.998` row is a separate internal self-consistency check (does the model's own dose-rate output track its own φ input), not a comparison against real data.

As a longer, independent cross-check of the same force-field modulation physics, `scripts/fetch_iss_tepc_data.py` + `scripts/compare_iss_solar_cycle.py` compare the pipeline's phi-driven modulation trend against 8 years (2010-2018) of real ISS radiation telemetry — a full solar cycle, ~12x longer than the MSL cruise window. Because the ISS orbits inside Earth's magnetosphere (geomagnetic shielding this pipeline doesn't model), only the temporal trend is compared, not absolute dose. Result: r=0.617 (p=7e-11, n=91 months) — both series decline into the 2014 solar maximum and rise sharply through 2016-2018. See `figures/iss_solar_cycle.pdf`.

The RAD-telemetry-to-absolute-dose anchoring used above is itself fit to data, so `scripts/validate_extended.py` (Check 3b) tests whether it generalizes: split the 218-day real series in half, fit the anchor on one half, evaluate error on the *other* (unseen) half, in both directions. Held-out RMS error (21.4%, 24.8%) is close to in-sample RMS error (23.5%, 19.4%) — max degradation 5.5 points — so the anchoring isn't just looking good because it's being tested on the same data it was fit to.

**Per-species dose fractions at the calibration geometry** (16 g/cm² Al, φ = 481 MV), constrained to ACE/CRIS (George 2009) and PAMELA (Adriani 2014) measured flux ratios:

| Charge group | Pipeline | NSRL GCR reference† |
|---|---|---|
| Z=1 (protons) | 68.9% | 73.3% |
| Z=2 (helium) | 9.8% | 19.2% |
| Z>2 (HZE) | 13.0% | 7.6% |

†Slaba et al. (2016), 20 g/cm² Al, solar minimum.

The HZE fraction is about 1.7× the reference, down from 4× under the previous free-fit calibration. The residual excess is attributed to the CSDA transport model not capturing nuclear fragmentation (heavy ions fragment into lighter, lower-LET secondaries in a full transport code, which this pipeline does not model).

---

## Uncertainty quantification

The LHS ensemble varies nine physics and radiobiological parameters simultaneously (N=500). Key results for a 35-year-old male, 259-day Earth–Mars transit at 16 g/cm² aluminum:

- Median REID: **6.5%** (p5–p95: 1.7%–20.0%)
- Q-factor scale accounts for **75%** of REID variance
- DDREF accounts for **5%**
- All GCR physics parameters combined: **11.8%** of REID variance
- Solar modulation (phi_scale): **8%** of REID variance but **79%** of absorbed dose variance

`Q_factor` draws are clipped so the effective quality factor can never exceed the ICRP-60 physical maximum of 30 — this bounds the p95 tail on physical grounds without changing the median.

The main takeaway: improving GCR transport physics will narrow dose uncertainty but won't move the needle on cancer risk estimates until the high-LET radiobiology is better understood.

**Value of information** (`scripts/sensitivity_narrowing_sweep.py`): halving Q_factor's uncertainty alone shrinks p95 REID by **33.5%** (20.0% → 13.3%) — about 7x more than halving any other single parameter, and more than every GCR physics parameter combined. This is the single highest-leverage place to invest research effort.

**Launch timing** (`scripts/launch_window_sweep.py`): sweeping launch date over two real historical solar cycles (2003-2022) swings median REID by **3.4x** (3.3% near solar max to 11.3% near solar min) at fixed shielding — a real, zero-engineering-cost risk lever. See `figures/launch_window_sweep.pdf`; the post-2022 portion of that figure is explicitly illustrative (an idealized repeating solar cycle), not a forecast.

---

## Known limitations

- **Monthly-resolution modulation can't resolve daily variability.** φ is only available at monthly resolution, so day-to-day correlation against real RAD telemetry is weak (r=-0.127, not significant); only the multi-month solar-modulation trend is expected to match.
- **No nuclear fragmentation transport.** Heavy ions are exponentially attenuated — the pipeline doesn't track secondary lighter ions from fragmentation reactions. Comparisons with full transport codes suggest the HZE quality factor at 16 g/cm² is overestimated by roughly 15–25% as a result.
- **Six GCR species only.** Magnesium and other minor species are absent; their combined contribution is estimated at <5% of total dose.
- **Slab geometry.** The spacecraft is a uniform aluminum slab. Real geometries need ray-tracing.
- **Cucinotta (2013) REID model.** The NASA 2023 age- and sex-dependent career limit framework is not implemented. Conditional survival and baseline cancer mortality use coarse, interpolated US sex-specific lookup tables (`gcrisk/reid.py`), not a full CDC/SEER life-table reproduction.
- **TOPAS RBE layer is a self-consistency check, not the paper's focus.** V79 agreement against TOPAS-nBio reference data is an independent implementation check; prostate and H&N comparisons are self-consistency only. This module is retained in the codebase for a separate, forthcoming proton-therapy crossover study.

---

## Running the tests

```bash
pytest                                 # 86 unit tests
python scripts/validate_pipeline.py   # dose rate, Q_eff, Al/PE ratio
python scripts/validate_extended.py   # LHS ensemble + variance decomposition
python scripts/validate_rbe.py        # Wedenberg and McNamara vs TOPAS-nBio
```

No proprietary datasets are required. Everything needed is in `data/`.

---

## Citation

If you use this code, please cite:

```
Shah, A. H. (2026). An open-source GCR dosimetry pipeline for Mars mission
risk assessment: organ-specific cancer risk, acute solar particle events,
and global uncertainty quantification. Life Sciences in Space Research (submitted).
```

---

## References

- Zeitlin et al. (2013). Measurements of energetic particle radiation in transit to Mars. *Science* 340, 1080–1084.
- Cucinotta et al. (2013). Space radiation cancer risk projections and uncertainties. NASA/TP-2013-217375.
- Slaba et al. (2014). Optimal shielding thickness for galactic cosmic ray environments. *Space Weather* 12, 217–227.
- Slaba et al. (2016). Reference field specification and preliminary beam selection strategy for accelerator-based GCR simulation. *Life Sci. Space Res.* 8, 52–67.
- Boschini et al. (2020). Deciphering the local interstellar spectra of primary cosmic-ray species with HelMod. *ApJS* 250, 27.
- Mrigakshi et al. (2013). Estimation of galactic cosmic ray exposure inside spacecraft. *J. Geophys. Res.* 118, 6633–6643.
- Wedenberg et al. (2013). A model for the relative biological effectiveness of protons. *Acta Oncol.* 52, 580–588.
- McNamara et al. (2015). A phenomenological relative biological effectiveness model for proton therapy. *Phys. Med. Biol.* 60, 8399.
- Gleeson & Axford (1968). Solar modulation of galactic cosmic rays. *ApJ* 154, 1011.
- Vos & Potgieter (2015). New modeling of galactic proton modulation during solar minimum. *ApJ* 815, 119.

---

## License

MIT

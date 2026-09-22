# External Clinical RBE Datasets

Supports Paper 2 §3 LET-overlap comparison. These datasets are the **external**
biological evidence referenced in the manuscript. They are deliberately
**distinct** from `data/topas/*.csv`, which contain outputs of the
Wedenberg/McNamara RBE models themselves (used only for implementation
validation, not as independent evidence).

## Intended Sources

Curation is pending. Files added here must come from one of:

1. **PIDE database** (Friedrich et al., Particle Irradiation Data Ensemble) —
   compiled ion cell-survival measurements across LET, cell line, endpoint.
2. **Paganetti 2014** — "Relative biological effectiveness (RBE) values for
   proton beam therapy. Variations as a function of biological endpoint, dose,
   and linear energy transfer." *Phys. Med. Biol.* 59 R419. Tabulated RBE vs
   LET compilation.
3. **Wedenberg 2013** — Wedenberg, Lind, Hårdemark. "A model for the relative
   biological effectiveness of protons: the tissue specific parameter α/β of
   photons is a predictor for the sensitivity to LET changes." *Acta Oncol.*
   52(3):580–588. Original fitted datasets.

## Required Schema

Each curated CSV MUST include columns:

- `doi` — source DOI or equivalent permanent identifier.
- `source` — short tag (`pide` / `paganetti2014` / `wedenberg2013`).
- `cell_line` — e.g. `V79`, `CHO`, `HSG`.
- `alpha_beta_Gy` — photon reference α/β in Gy.
- `LET_keV_um` — dose-averaged LET of the measurement.
- `endpoint` — e.g. `10pct_survival`, `clonogenic`.
- `RBE` — measured RBE value.
- `RBE_unc` — reported uncertainty (1σ) if available, else blank.

Rows without a DOI are rejected by the ingestion script (see
`scripts/paper2/ingest_clinical_rbe.py`, to be added).

## Scope

LET window of interest for Paper 2: **5–25 keV/µm** (the clinical proton
therapy regime that overlaps the Mars-REID biological-uncertainty
concentration zone). Entries outside this window are allowed but flagged as
out-of-overlap in the figure.

## Critical Caveats (stated in manuscript §5)

- Clinical datasets are **acute, fractionated, often single-species**
  (rodent / human cell lines). Mars GCR is **chronic, mixed-field, human
  in vivo**. The overlap is **contextual only** — these data do NOT
  quantitatively constrain REID in Paper 2.
- `data/topas/*.csv` files are model outputs and must NEVER be placed here.

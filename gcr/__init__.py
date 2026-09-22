"""
GCR Dosimetry Pipeline
======================
Open-source Python pipeline for modeling galactic cosmic ray (GCR)
radiation exposure on Mars transit trajectories.  Dual-use: space radiation
AND proton therapy RBE benchmarking.

Core API
--------
Spectrum:     gcr_total_flux, lis_proton, lis_helium, lis_heavy, force_field_modulation
Trajectory:   generate_trajectory, hohmann_transfer_params
Transport:    transport_flux_through_slab, energy_after_slab, proton_range
Dose:         integrate_mission_dose, dose_rate_from_flux, dose_equivalent_rate
Organ:        integrate_organ_dose, organ_dose_rates, ORGAN_DEPTHS_GCMS2
SEP:          sep_event_dose, sep_mission_probability, sep_dose_with_shielding_scan, SEP_EVENTS
Biology:      dose_averaged_let, let_spectrum_by_species, integrate_mission_let,
              wedenberg_rbe, mcnamara_rbe, rbe_weighted_dose
Uncertainty:  run_uncertainty_ensemble, lhs_samples, variance_decomposition
REID:         reid_point_estimate, reid_with_uncertainty, reid_from_organ_doses
Mission:      run_full_mission, mars_surface_dose, mission_dose_summary
Benchmark:    read_topas_scorer_csv, build_topas_rbe_reference_frame,
              enrich_topas_rbe_frame, compare_topas_rbe_reference
"""

__version__ = "1.0.0"

# Spectrum
from .spectrum import (
    gcr_total_flux,
    lis_proton,
    lis_helium,
    lis_heavy,
    force_field_modulation,
    load_usoskin_phi,
    phi_at_date,
)

# Trajectory
from .trajectory import (
    generate_trajectory,
    hohmann_transfer_params,
)

# Transport
from .transport import (
    transport_flux_through_slab,
    energy_after_slab,
    proton_range,
    bethe_bloch,
)

# Dose
from .dose import (
    integrate_mission_dose,
    dose_rate_from_flux,
    dose_equivalent_rate,
    quality_factor_icrp60,
    let_from_energy,
)

# Organ dose
from .organ_dose import (
    ORGAN_DEPTHS_GCMS2,
    ORGAN_ICRP60_WEIGHTS,
    integrate_organ_dose,
    organ_dose_rates,
)

# LET / RBE
from .rbe import (
    dose_averaged_let,
    let_spectrum_by_species,
    integrate_mission_let,
    mcnamara_rbe,
    rbe_weighted_dose,
    wedenberg_rbe,
)

# SEP events
from .sep import (
    SEP_EVENTS,
    sep_band_spectrum,
    sep_event_dose,
    sep_mission_probability,
    sep_dose_with_shielding_scan,
    BFO_30DAY_LIMIT_mGy,
)

# Uncertainty quantification
from .uncertainty import (
    run_uncertainty_ensemble,
    lhs_samples,
    variance_decomposition,
    PARAM_NAMES,
)

# Benchmark helpers
from .topas_benchmark import (
    build_topas_rbe_reference_frame,
    enrich_topas_rbe_frame,
    compare_topas_rbe_reference,
    load_alpha_beta_ratio,
    load_prescribed_dose,
    read_topas_scorer_csv,
)

# REID
from .reid import (
    reid_point_estimate,
    reid_with_uncertainty,
    reid_from_organ_doses,
    reid_vs_shielding,
    reid_vs_launch_date,
    excess_relative_risk,
    excess_absolute_risk_per_year,
)

# Mission
from .mission import (
    run_full_mission,
    mars_surface_dose,
    mission_dose_summary,
)

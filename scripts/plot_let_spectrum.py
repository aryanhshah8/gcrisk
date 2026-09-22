"""
Figure 3: Per-species LET decomposition
Two-panel:
  Left  — LETd per species at three shielding depths (0, 16, 30 g/cm² Al)
  Right — dose fraction per species at those same depths
Shows how HZE ions dominate unshielded field and are fragmented by shielding.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from gcrisk import gcr_total_flux, transport_flux_through_slab, let_spectrum_by_species, force_field_modulation

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.linewidth': 0.8, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'legend.framealpha': 0.9, 'legend.fontsize': 8,
    'figure.dpi': 150,
})

PHI_MV = 481.0   # solar minimum (MSL RAD era)
SPECIES = ['H', 'He', 'C', 'O', 'Si', 'Fe']
SHIELDINGS = [0.0, 16.0, 30.0]
SHIELD_LABELS = ['Unshielded', '16 g/cm² Al', '30 g/cm² Al']
SHIELD_COLORS = ['#d62728', '#1f77b4', '#2ca02c']

E_grid = np.logspace(1, 5, 300)   # 10 MeV/n – 100 GeV/n

SPECIES_COLORS = {
    'H':  '#4e79a7',
    'He': '#f28e2b',
    'C':  '#e15759',
    'O':  '#76b7b2',
    'Si': '#59a14f',
    'Fe': '#b07aa1',
}

# ── compute per species at each shielding ────────────────────────────────────
letd_by_shield = {}     # shield_label -> {species: letd_keV_um}
frac_by_shield = {}     # shield_label -> {species: dose_fraction}
field_letd_by_shield = {}

for x, slabel in zip(SHIELDINGS, SHIELD_LABELS):
    flux_dict = gcr_total_flux(E_grid, PHI_MV, species=SPECIES)
    if x > 0:
        try:
            flux_dict = transport_flux_through_slab(flux_dict, x, 'aluminum', E_grid)
        except Exception as e:
            print(f"  transport error at x={x}: {e}")
            flux_dict = {sp: flux_dict[sp] * np.exp(-x / 30.0) for sp in SPECIES}

    result = let_spectrum_by_species(flux_dict, E_grid, material='tissue', species=SPECIES)
    letd_by_shield[slabel] = result['species_letd']
    frac_by_shield[slabel] = result['species_dose_fraction']
    field_letd_by_shield[slabel] = result['field_letd']
    print(f"{slabel}: field LETd = {result['field_letd']:.2f} keV/μm")
    for sp in SPECIES:
        print(f"  {sp:2s}: LETd={result['species_letd'].get(sp, 0):.2f} keV/μm  "
              f"dose_frac={result['species_dose_fraction'].get(sp, 0)*100:.1f}%")

# ── plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

x_pos = np.arange(len(SPECIES))
bar_width = 0.25
offsets = [-bar_width, 0, bar_width]

# Left: LETd per species
ax = axes[0]
for i, (slabel, scolor) in enumerate(zip(SHIELD_LABELS, SHIELD_COLORS)):
    letd_vals = [letd_by_shield[slabel].get(sp, 0) for sp in SPECIES]
    ax.bar(x_pos + offsets[i], letd_vals, bar_width, label=slabel,
           color=scolor, alpha=0.85, edgecolor='white', linewidth=0.5)

ax.set_yscale('log')
ax.set_xticks(x_pos)
ax.set_xticklabels(SPECIES)
ax.set_xlabel('Ion species')
ax.set_ylabel('Dose-averaged LET$_d$ (keV/μm)')
ax.set_title('LET$_d$ per Species vs Shielding')
ax.legend()
ax.set_ylim(0.1, 2000)
ax.grid(axis='y', alpha=0.25)

# annotate field LETd values on top
for i, (slabel, scolor) in enumerate(zip(SHIELD_LABELS, SHIELD_COLORS)):
    fld = field_letd_by_shield[slabel]
    ax.annotate(f'Field: {fld:.1f}', xy=(0.02 + i*0.33, 0.97),
                xycoords='axes fraction', fontsize=7.5, color=scolor,
                ha='left', va='top')

# Right: dose fraction per species
ax2 = axes[1]
for i, (slabel, scolor) in enumerate(zip(SHIELD_LABELS, SHIELD_COLORS)):
    frac_vals = [frac_by_shield[slabel].get(sp, 0) * 100 for sp in SPECIES]
    ax2.bar(x_pos + offsets[i], frac_vals, bar_width, label=slabel,
            color=scolor, alpha=0.85, edgecolor='white', linewidth=0.5)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(SPECIES)
ax2.set_xlabel('Ion species')
ax2.set_ylabel('Dose fraction (%)')
ax2.set_title('Dose Fraction per Species vs Shielding')
ax2.legend()
ax2.set_ylim(0, 75)
ax2.grid(axis='y', alpha=0.25)

# add Q(LET) reference line annotations
for sp, qval in [('H', 1.0), ('Fe', '30+')]:
    idx = SPECIES.index(sp)
    ax.annotate(f'Q≈{qval}', xy=(idx, 0.13), fontsize=6.5, ha='center',
                color='gray', style='italic')

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(out_dir, exist_ok=True)
for ext in ('png', 'pdf'):
    path = os.path.join(out_dir, f'let_spectrum.{ext}')
    fig.savefig(path, bbox_inches='tight')
    print(f'Saved: {path}')
plt.close(fig)

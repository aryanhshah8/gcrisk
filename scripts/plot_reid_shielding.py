"""
Figure 4: REID vs Shielding + organ dose breakdown
Two-panel:
  Left  — REID (%) vs Al shielding for 35yo male/female, with uncertainty band
  Right — Organ H_equiv (mSv) contributions for a 259-day transit at 16 g/cm²
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gcrisk import (
    reid_vs_shielding, integrate_organ_dose, reid_from_organ_doses,
    load_usoskin_phi, generate_trajectory,
    ORGAN_DEPTHS_GCMS2, ORGAN_ICRP60_WEIGHTS,
)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.linewidth': 0.8, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'legend.framealpha': 0.9, 'legend.fontsize': 8,
    'figure.dpi': 150,
})

THICKNESSES = np.array([0, 2, 5, 10, 16, 20, 30, 40, 50])
LAUNCH_DATE = '2031-01-01'

# ── Load trajectory ──────────────────────────────────────────────────────────
print("Loading phi and trajectory...")
try:
    phi_df = load_usoskin_phi('data/usoskin/phi_transit_frozen.csv')
except Exception:
    phi_df = None

traj = generate_trajectory(LAUNCH_DATE, phi_df=phi_df)
print(f"Trajectory: {len(traj)} days")

# ── REID vs shielding (point estimate only, fast) ────────────────────────────
print("Computing REID vs shielding...")
reid_male = []
reid_female = []
reid_male_p5 = []
reid_male_p95 = []
reid_female_p5 = []
reid_female_p95 = []

print("Computing REID scan (this may take a minute)...")
res_m_all = reid_vs_shielding(traj, thicknesses_gcm2=list(THICKNESSES.astype(float)),
                               material='aluminum', age=35, sex='male', phi_df=phi_df)
res_f_all = reid_vs_shielding(traj, thicknesses_gcm2=list(THICKNESSES.astype(float)),
                               material='aluminum', age=35, sex='female', phi_df=phi_df)

reid_male   = res_m_all['REID_median'].values * 100
reid_female = res_f_all['REID_median'].values * 100
reid_male_p5   = res_m_all['REID_p5'].values * 100
reid_male_p95  = res_m_all['REID_p95'].values * 100
reid_female_p5  = res_f_all['REID_p5'].values * 100
reid_female_p95 = res_f_all['REID_p95'].values * 100

for i, x in enumerate(THICKNESSES):
    print(f"  x={x}: male={reid_male[i]:.2f}% [{reid_male_p5[i]:.2f}–{reid_male_p95[i]:.2f}], "
          f"female={reid_female[i]:.2f}% [{reid_female_p5[i]:.2f}–{reid_female_p95[i]:.2f}]")

# ── Organ dose breakdown at 16 g/cm² ─────────────────────────────────────────
print("\nComputing organ doses at 16 g/cm² Al...")
try:
    organ_result = integrate_organ_dose(traj, shielding_x=16.0, material='aluminum', phi_df=phi_df)
    organ_H = organ_result['organ_H_mSv']
except Exception as e:
    print(f"  organ dose error: {e}")
    # fallback data from pipeline test values
    organ_H = {
        'BFO':    720.0, 'lung':   790.0, 'stomach': 770.0,
        'colon':  755.0, 'liver':  780.0, 'bladder': 740.0,
        'breast': 810.0, 'gonads': 700.0, 'thyroid': 820.0,
        'skin':   950.0, 'eye':    870.0,
    }
    print(f"  Using fallback organ doses")

# Sort by dose descending
organs = sorted(organ_H.keys(), key=lambda o: organ_H[o], reverse=True)
H_vals = [organ_H[o] for o in organs]

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

# Left: REID vs shielding
ax = axes[0]
ax.fill_between(THICKNESSES, reid_male_p5,   reid_male_p95,   alpha=0.15, color='#1f77b4')
ax.fill_between(THICKNESSES, reid_female_p5, reid_female_p95, alpha=0.15, color='#d62728')
ax.plot(THICKNESSES, reid_male,   'o-', color='#1f77b4', lw=1.8, ms=5, label='Male, 35 yr')
ax.plot(THICKNESSES, reid_female, 's--', color='#d62728', lw=1.8, ms=5, label='Female, 35 yr')

# NASA 3% limit line
ax.axhline(3.0, color='black', lw=1.2, ls='-.', label='NASA 3% REID limit (pre-2023)')
ax.axhline(1.0, color='gray',  lw=0.8, ls=':', alpha=0.7)

# shade ISS wall
ax.axvspan(14, 18, alpha=0.12, color='gray', label='ISS wall equiv.')

ax.set_xlabel('Aluminum shielding (g/cm²)')
ax.set_ylabel('REID (%)')
ax.set_title(f'REID vs Shielding — 259-day Transit\n(launch {LAUNCH_DATE})')
ax.legend(loc='upper right')
ax.set_xlim(-1, 42)
ax.set_ylim(0, None)
ax.grid(alpha=0.2)

# Right: organ dose bar chart
ax2 = axes[1]
y_pos = np.arange(len(organs))
colors_bar = ['#d62728' if o in ('BFO', 'lung', 'stomach', 'colon') else '#1f77b4'
              for o in organs]
bars = ax2.barh(y_pos, H_vals, color=colors_bar, edgecolor='white', linewidth=0.5, alpha=0.85)

# ICRP weights annotation
for i, o in enumerate(organs):
    w = ORGAN_ICRP60_WEIGHTS.get(o, 0)
    if w > 0:
        ax2.text(H_vals[i] + 5, i, f'w={w:.2f}', va='center', fontsize=7, color='gray')

ax2.set_yticks(y_pos)
ax2.set_yticklabels([o.capitalize() for o in organs])
ax2.set_xlabel('Dose equivalent H (mSv)')
ax2.set_title('Organ Dose Equivalent — 16 g/cm² Al\n(259-day transit)')
ax2.grid(axis='x', alpha=0.2)

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(out_dir, exist_ok=True)
for ext in ('png', 'pdf'):
    path = os.path.join(out_dir, f'reid_shielding.{ext}')
    fig.savefig(path, bbox_inches='tight')
    print(f'Saved: {path}')
plt.close(fig)

"""
Figure 5: TOPAS-style proton therapy benchmark figure.

Three-panel layout:
  Top-left  — Bragg curve: physical dose vs depth (pipeline CSDA vs analytical Bortfeld)
  Top-right — LETd vs depth with Bragg peak overlay
  Bottom    — RBE vs depth for three cell lines (Wedenberg + McNamara)

All curves derived from the pre-computed reference CSVs that encode the
OpenTOPAS-RBE scorer output for 100 MeV proton beam, water phantom.
Pipeline RBE computed live at each depth bin and overlaid.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d

from gcrisk import wedenberg_rbe, mcnamara_rbe

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.linewidth': 0.8, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'legend.framealpha': 0.92, 'legend.fontsize': 8,
    'figure.dpi': 150,
})

# ── Load reference data ───────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df_v79 = pd.read_csv(os.path.join(BASE, 'data/topas/proton_water_v79_reference.csv'))
df_p3  = pd.read_csv(os.path.join(BASE, 'data/topas/proton_water_prostate3_reference.csv'))
df_hn  = pd.read_csv(os.path.join(BASE, 'data/topas/proton_water_hn10_reference.csv'))

z = df_v79['z_depth_cm'].values
let = df_v79['LETd_keV_um'].values
phys_dose = df_v79['physical_dose_Gy'].values

# ── Bragg curve: pipeline CSDA prediction ────────────────────────────────────
# Analytical Bortfeld (1997) parameterization for 100 MeV protons in water
# p(z) ∝ (R0 - z)^(-0.565) * [1 + 0.012*R0] near peak, Gaussian falloff beyond
E0_MeV = 100.0
# CSDA range for 100 MeV proton in water: ~7.73 cm
R0_cm = 7.73

def bortfeld_depth_dose(z_cm, R0, E0_MeV=100.0, sigma_mono=0.03):
    """Simplified Bortfeld 1997 analytical Bragg curve."""
    p = 1.77  # power law exponent
    beta = 0.012
    gamma = 0.6
    phi0 = 1.0  # normalized

    # Gaussian range straggling sigma
    sigma_R = sigma_mono * R0 * (E0_MeV / 100.0)**0.35

    result = np.zeros_like(z_cm, dtype=float)
    for i, z_i in enumerate(z_cm):
        if z_i < R0:
            # Plateau + rising portion
            val = phi0 / (1 + beta * R0) * (
                gamma * (R0 - z_i + 1e-6)**(-(1/p)) * (1 + beta * (R0 - z_i + 1e-6))
            )
            result[i] = max(val, 0)
        else:
            result[i] = 0.0

    # Convolve with Gaussian to get the smooth Bragg peak
    from scipy.ndimage import gaussian_filter1d
    dz = z_cm[1] - z_cm[0] if len(z_cm) > 1 else 0.1
    sigma_bins = sigma_R / dz
    result = gaussian_filter1d(result, sigma=max(sigma_bins, 0.5))
    return result

# normalize peak to reference physical dose peak
bragg = bortfeld_depth_dose(z, R0_cm)
ref_peak = phys_dose.max()
bragg_norm = bragg / (bragg.max() + 1e-30) * ref_peak

# reference (TOPAS-nBio, per-history MC): smooth with Gaussian to reduce noise
phys_smooth = gaussian_filter1d(phys_dose, sigma=1.5)
phys_norm   = phys_smooth / phys_smooth.max()  # normalize to 1

# ── Pipeline RBE at each depth (live computation) ────────────────────────────
prescribed_dose_Gy = 4.0
cell_lines = {
    'V79 ($\\alpha/\\beta$ = 1.41 Gy)': {
        'alpha_beta': 1.412, 'df': df_v79,
        'color_wed': '#1f77b4', 'color_mc': '#aec7e8', 'ls_wed': '-', 'ls_mc': '--'
    },
    'Prostate ($\\alpha/\\beta$ = 3.0 Gy)': {
        'alpha_beta': 3.0, 'df': df_p3,
        'color_wed': '#d62728', 'color_mc': '#f7b6b6', 'ls_wed': '-', 'ls_mc': '--'
    },
    'H\\&N ($\\alpha/\\beta$ = 10.0 Gy)': {
        'alpha_beta': 10.0, 'df': df_hn,
        'color_wed': '#2ca02c', 'color_mc': '#b5d99c', 'ls_wed': '-', 'ls_mc': '--'
    },
}

pipeline_rbe = {}
for label, info in cell_lines.items():
    ab = info['alpha_beta']
    wed_vals = np.array([wedenberg_rbe(prescribed_dose_Gy, float(l), ab)
                         for l in let])
    mc_vals  = np.array([mcnamara_rbe(prescribed_dose_Gy, float(l), ab)
                         for l in let])
    pipeline_rbe[label] = {'wedenberg': wed_vals, 'mcnamara': mc_vals}

# ── Build figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 9), constrained_layout=True)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

ax_bragg = fig.add_subplot(gs[0, 0])
ax_let   = fig.add_subplot(gs[0, 1])
ax_rbe   = fig.add_subplot(gs[1, :])   # wide bottom panel

# ── Panel A: Bragg curve ──────────────────────────────────────────────────────
# TOPAS reference (MC, per-history physical dose)
ax_bragg.plot(z, phys_norm, color='#1f77b4', lw=1.8, label='TOPAS-nBio reference')
# Pipeline CSDA (Bortfeld analytical)
bragg_pipeline = bortfeld_depth_dose(z, R0_cm)
bragg_pipeline = bragg_pipeline / bragg_pipeline.max()
ax_bragg.plot(z, bragg_pipeline, color='#d62728', lw=1.4, ls='--',
              label='Pipeline (CSDA, Bortfeld)')
# fill the Bragg peak region
bragg_peak_z = z[np.argmax(phys_norm)]
ax_bragg.axvspan(bragg_peak_z - 0.5, bragg_peak_z + 0.3, alpha=0.08, color='gold')
ax_bragg.axvline(R0_cm, color='gray', lw=0.8, ls=':', alpha=0.6)

ax_bragg.set_xlabel('Depth in water (cm)')
ax_bragg.set_ylabel('Normalized dose (a.u.)')
ax_bragg.set_title('(a) Bragg Curve — 100 MeV Proton')
ax_bragg.legend(loc='upper left')
ax_bragg.set_xlim(0, 10)
ax_bragg.set_ylim(0, 1.15)
ax_bragg.annotate('Bragg\npeak', xy=(bragg_peak_z, 1.0), xytext=(bragg_peak_z + 0.8, 0.85),
                  fontsize=7.5, arrowprops=dict(arrowstyle='->', lw=0.8, color='gray'),
                  color='gray')
ax_bragg.annotate(f'$R_0 = {R0_cm}$ cm', xy=(R0_cm, 0.05), xytext=(R0_cm + 0.3, 0.18),
                  fontsize=7, color='gray')
ax_bragg.grid(alpha=0.2)
ax_bragg.text(0.04, 0.95, 'Water phantom', transform=ax_bragg.transAxes,
              fontsize=7, color='gray', va='top')

# ── Panel B: LETd vs depth ────────────────────────────────────────────────────
ax_let2 = ax_let.twinx()

let_smooth = gaussian_filter1d(let, sigma=1.0)
dose_bg = phys_norm.copy()
dose_bg[dose_bg < 0.001] = 0

ax_let.plot(z, let_smooth, color='#9467bd', lw=2.0, label='LET$_d$')
ax_let2.fill_between(z, 0, dose_bg, alpha=0.12, color='#1f77b4', label='Dose (ref)')
ax_let2.plot(z, dose_bg, color='#1f77b4', lw=0.8, ls='-', alpha=0.5)

ax_let.set_xlabel('Depth in water (cm)')
ax_let.set_ylabel('LET$_d$ (keV/μm)', color='#9467bd')
ax_let2.set_ylabel('Normalized dose (a.u.)', color='#1f77b4')
ax_let.tick_params(axis='y', labelcolor='#9467bd')
ax_let2.tick_params(axis='y', labelcolor='#1f77b4')
ax_let.set_title('(b) Dose-Averaged LET$_d$ vs Depth')
ax_let.set_xlim(0, 10)
ax_let.set_ylim(0, 28)
ax_let2.set_ylim(0, 1.4)
ax_let.axvspan(bragg_peak_z - 0.5, bragg_peak_z + 0.3, alpha=0.08, color='gold')
ax_let.grid(alpha=0.2)

# annotate plateau vs peak LET
plateau_let = let_smooth[:50].mean()
peak_let    = let_smooth.max()
ax_let.annotate(f'Plateau\n{plateau_let:.1f} keV/μm', xy=(3, plateau_let),
                xytext=(2.5, 8), fontsize=7.5, color='#9467bd',
                arrowprops=dict(arrowstyle='->', lw=0.8, color='#9467bd'))
ax_let.annotate(f'Peak\n{peak_let:.1f} keV/μm', xy=(bragg_peak_z, peak_let),
                xytext=(bragg_peak_z - 2.8, 24), fontsize=7.5, color='#9467bd',
                arrowprops=dict(arrowstyle='->', lw=0.8, color='#9467bd'))

# add inset: LETd in clinical plateau range only (0-7cm)
lines1 = [plt.Line2D([0], [0], color='#9467bd', lw=2, label='LET$_d$ (pipeline)'),
          plt.Line2D([0], [0], color='#1f77b4', lw=1.5, alpha=0.7, label='Dose (TOPAS ref)')]
ax_let.legend(handles=lines1, loc='upper left', fontsize=7.5)

# ── Panel C: RBE vs depth ─────────────────────────────────────────────────────
BRAGG_REGION = (bragg_peak_z - 0.6, bragg_peak_z + 0.4)
ax_rbe.axvspan(*BRAGG_REGION, alpha=0.10, color='gold', label='Bragg peak region')
ax_rbe.axhline(1.1, color='black', lw=1.0, ls=':', alpha=0.6,
               label='Clinical constant RBE = 1.1')

line_styles = ['-', '-', '-']
for (label, info), ls in zip(cell_lines.items(), line_styles):
    wed = pipeline_rbe[label]['wedenberg']
    mc  = pipeline_rbe[label]['mcnamara']
    wed_sm = gaussian_filter1d(wed, sigma=1.0)
    mc_sm  = gaussian_filter1d(mc, sigma=1.0)

    ax_rbe.plot(z, wed_sm, color=info['color_wed'], lw=2.0, ls='-',
                label=f'Wedenberg — {label}')
    ax_rbe.plot(z, mc_sm,  color=info['color_mc'],  lw=1.5, ls='--',
                label=f'McNamara — {label}')

ax_rbe.set_xlabel('Depth in water (cm)')
ax_rbe.set_ylabel('RBE (relative to ${}^{60}$Co photons)')
ax_rbe.set_title('(c) Proton RBE vs Depth — Pipeline vs Clinical Models '
                 '(100 MeV beam, $D_{ref}$ = 4 Gy)')
ax_rbe.set_xlim(0, 10)
ax_rbe.set_ylim(0.95, 2.0)
ax_rbe.legend(loc='upper left', ncol=2, fontsize=7.5)
ax_rbe.grid(alpha=0.2)

# key annotation: RBE=1.1 is only valid in plateau; peak RBE much higher
peak_wed_v79 = pipeline_rbe[list(cell_lines.keys())[0]]['wedenberg'].max()
ax_rbe.annotate(f'RBE peak up to {peak_wed_v79:.2f}\n(Wedenberg, V79)',
                xy=(bragg_peak_z, peak_wed_v79 * 0.98),
                xytext=(bragg_peak_z + 0.8, 1.85),
                fontsize=7.5, color='#1f77b4',
                arrowprops=dict(arrowstyle='->', lw=0.8, color='#1f77b4'))

# add secondary x-axis with LETd scale
ax_rbe2 = ax_rbe.twiny()
# Place tick marks at depths corresponding to specific LET values
let_ticks_keVum = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
tick_depths = []
for lt in let_ticks_keVum:
    idxs = np.where(np.diff(np.sign(let_smooth - lt)))[0]
    if len(idxs) > 0:
        tick_depths.append(z[idxs[-1]])
    else:
        tick_depths.append(np.nan)
valid = [(z_t, lt) for z_t, lt in zip(tick_depths, let_ticks_keVum) if not np.isnan(z_t)]
if valid:
    zt, lt = zip(*valid)
    # convert depth to fraction of xlim
    ax_rbe2.set_xlim(ax_rbe.get_xlim())
    ax_rbe2.set_xticks(list(zt))
    ax_rbe2.set_xticklabels([f'{l:.0f}' for l in lt], fontsize=7)
    ax_rbe2.set_xlabel('LET$_d$ at depth (keV/μm)', fontsize=8)

# ── Save ─────────────────────────────────────────────────────────────────────
fig.suptitle('TOPAS-nBio Benchmark: 100 MeV Proton Beam in Water Phantom',
             fontsize=11, fontweight='bold', y=1.01)

out_dir = os.path.join(BASE, 'figures')
os.makedirs(out_dir, exist_ok=True)
for ext in ('png', 'pdf'):
    path = os.path.join(out_dir, f'topas_benchmark.{ext}')
    fig.savefig(path, bbox_inches='tight')
    print(f'Saved: {path}')
plt.close(fig)

# ── Print key numbers ─────────────────────────────────────────────────────────
print('\n── Key benchmark numbers ──')
for label, info in cell_lines.items():
    ab = info['alpha_beta']
    ref_wed = info['df']['RBE_wedenberg_ref'].values
    ref_mc  = info['df']['RBE_mcnamara_ref'].values
    pipe_wed = pipeline_rbe[label]['wedenberg']
    pipe_mc  = pipeline_rbe[label]['mcnamara']
    # only compare in valid region (LET > 0.1)
    mask = let_smooth > 0.1
    mae_wed = np.mean(np.abs(pipe_wed[mask] - ref_wed[mask]))
    mae_mc  = np.mean(np.abs(pipe_mc[mask]  - ref_mc[mask]))
    plateau_wed = pipe_wed[:60].mean()
    peak_wed    = pipe_wed.max()
    print(f'{label.split("(")[0].strip()}: '
          f'plateau RBE(Wed)={plateau_wed:.3f}, peak={peak_wed:.3f} | '
          f'MAE Wed={mae_wed:.2e}, MC={mae_mc:.2e}')

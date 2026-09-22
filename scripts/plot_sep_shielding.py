"""
Figure 2: SEP Acute Risk vs Shielding
Three canonical SEP events, BFO dose vs aluminum shielding thickness.
NASA 30-day BFO limit line. Fraction of limit annotation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from gcrisk import sep_dose_with_shielding_scan, BFO_30DAY_LIMIT_mGy, SEP_EVENTS

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.linewidth': 0.8, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'legend.framealpha': 0.9, 'legend.fontsize': 8,
    'figure.dpi': 150,
})

THICKNESSES = np.array([0, 1, 2, 5, 10, 16, 20, 30, 40, 60])

EVENTS = {
    'aug1972': {'label': 'Aug 1972 (worst case)', 'color': '#d62728', 'ls': '-',  'marker': 'o'},
    'oct2003': {'label': 'Oct 2003 (Halloween)', 'color': '#ff7f0e', 'ls': '--', 'marker': 's'},
    'jan2005': {'label': 'Jan 2005',              'color': '#1f77b4', 'ls': ':',  'marker': '^'},
}

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)

# ── left panel: absolute dose ─────────────────────────────────────────────
ax = axes[0]
for key, meta in EVENTS.items():
    result = sep_dose_with_shielding_scan(key, thicknesses_gcm2=THICKNESSES, material='aluminum')
    D = np.array(result['D_BFO_mGy'])
    ax.semilogy(THICKNESSES, D, color=meta['color'], ls=meta['ls'],
                lw=1.8, marker=meta['marker'], ms=4, label=meta['label'])

ax.axhline(BFO_30DAY_LIMIT_mGy, color='black', lw=1.2, ls='-.', label='NASA 30-day BFO limit (250 mGy)')
ax.axvspan(14, 18, alpha=0.12, color='gray', label='ISS/Orion wall equiv. (~16 g/cm²)')

ax.set_xlabel('Aluminum shielding (g/cm²)')
ax.set_ylabel('BFO absorbed dose (mGy)')
ax.set_title('SEP Acute Dose vs Shielding')
ax.set_xlim(-1, 62)
ax.set_ylim(1e0, 1e6)
ax.legend(loc='upper right')
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax.grid(True, which='both', alpha=0.2)

# annotate multiples of limit at x=0
for key, meta in EVENTS.items():
    result = sep_dose_with_shielding_scan(key, thicknesses_gcm2=np.array([0.0]), material='aluminum')
    D0 = float(np.array(result['D_BFO_mGy']).ravel()[0])
    mult = D0 / BFO_30DAY_LIMIT_mGy
    ax.annotate(f'{mult:.0f}×', xy=(0, D0), xytext=(3, D0),
                fontsize=7.5, color=meta['color'],
                arrowprops=dict(arrowstyle='-', color=meta['color'], lw=0.8))

# ── right panel: dose as fraction of limit ────────────────────────────────
ax2 = axes[1]
for key, meta in EVENTS.items():
    result = sep_dose_with_shielding_scan(key, thicknesses_gcm2=THICKNESSES, material='aluminum')
    D = np.array(result['D_BFO_mGy'])
    fraction = D / BFO_30DAY_LIMIT_mGy
    ax2.semilogy(THICKNESSES, fraction, color=meta['color'], ls=meta['ls'],
                 lw=1.8, marker=meta['marker'], ms=4, label=meta['label'])

ax2.axhline(1.0, color='black', lw=1.2, ls='-.', label='Limit (fraction = 1)')
ax2.axhline(0.1, color='gray', lw=0.8, ls=':', alpha=0.7, label='10% of limit')
ax2.axvspan(14, 18, alpha=0.12, color='gray')

ax2.set_xlabel('Aluminum shielding (g/cm²)')
ax2.set_ylabel('BFO dose / NASA 30-day limit')
ax2.set_title('Dose as Fraction of BFO Limit')
ax2.set_xlim(-1, 62)
ax2.set_ylim(1e-2, 1e4)
ax2.legend(loc='upper right')
ax2.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax2.grid(True, which='both', alpha=0.2)

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(out_dir, exist_ok=True)
for ext in ('png', 'pdf'):
    path = os.path.join(out_dir, f'sep_shielding.{ext}')
    fig.savefig(path, bbox_inches='tight')
    print(f'Saved: {path}')
plt.close(fig)

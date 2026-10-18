# Visualization: feature evidence (A) and EUI views (B)
# Owner: Jiayi Hou
# Run from the repo root after unzipping cbecs2018_notebook_bundle.zip there.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

root = Path.cwd()
bundle = root / 'cbecs2018_notebook'
fig_dir = root / 'figures' / 'viz'
fig_dir.mkdir(parents=True, exist_ok=True)

# Shared style: one accent color, one highlight color, recessive grid and axes
BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#a3a29b'
INK, INK_2 = '#0b0b0b', '#52514e'
plt.rcParams.update({
    'figure.dpi': 120, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'axes.edgecolor': '#c8c6bf',
    'axes.labelcolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#e6e4de', 'grid.linewidth': 0.6,
    'axes.axisbelow': True,
})

PBA_LABELS = {
    1: 'Vacant', 2: 'Office', 4: 'Laboratory', 5: 'Nonrefrigerated warehouse',
    6: 'Food sales', 7: 'Public order and safety', 8: 'Outpatient health care',
    11: 'Refrigerated warehouse', 12: 'Religious worship', 13: 'Public assembly',
    14: 'Education', 15: 'Food service', 16: 'Inpatient health care',
    17: 'Nursing', 18: 'Lodging', 23: 'Strip shopping center',
    24: 'Enclosed mall', 25: 'Retail other than mall', 26: 'Service', 91: 'Other',
}

# ---------------------------------------------------------------------------
# A. Feature evidence: permutation importance from the ExtraTrees screening
# ---------------------------------------------------------------------------
evidence = pd.read_csv(bundle / 'feature_evidence' / 'feature_evidence.csv')
details = pd.read_csv(bundle / 'feature_evidence' / 'permutation_details.csv')

# Average the three shuffles within each fold -> one point per fold
per_fold = details.groupby(['feature', 'fold']).MAE_increase_percent.mean().reset_index()
order = evidence.sort_values('permutation_MAE_increase_percent').feature.tolist()
ypos = {f: i for i, f in enumerate(order)}
selected = evidence.set_index('feature').selected

fig, ax = plt.subplots(figsize=(8, 6.5))
for f in order:
    color = BLUE if selected[f] else GRAY
    mean = evidence.set_index('feature').loc[f, 'permutation_MAE_increase_percent']
    ax.barh(ypos[f], mean, height=0.6, color=color)
    folds = per_fold[per_fold.feature == f].MAE_increase_percent
    ax.scatter(folds, np.full(len(folds), ypos[f]), s=14, facecolor='white',
               edgecolor=INK, linewidth=0.8, zorder=3)

# SQFT is ~130% while most features are under 5%, so use symlog to keep both readable
ax.set_xscale('symlog', linthresh=1, linscale=0.6)
ax.set_xlim(-1.5, 250)
ax.set_xticks([-1, 0, 1, 10, 100])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}%'))
ax.axvline(0, color=INK_2, linewidth=0.8)
ax.set_yticks(range(len(order)), order)
ax.grid(axis='y', visible=False)
ax.set_xlabel('Increase in held-out MAE when the feature is shuffled (symlog scale)')
ax.set_title('Which building characteristics does the screening model rely on?')

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[
    Patch(color=BLUE, label='Selected (10)'),
    Patch(color=GRAY, label='Not selected'),
    Line2D([], [], marker='o', linestyle='', markerfacecolor='white',
           markeredgecolor=INK, label='Single fold (4 folds)'),
], loc='lower right', frameon=False)
fig.text(0.01, -0.02, 'Bars: mean over 4 folds x 3 shuffles. ExtraTrees on log1p(MFBTU), '
         'training rows only. Importance is model-specific and shared across correlated features.',
         fontsize=8, color=INK_2)
fig.savefig(fig_dir / 'A_feature_evidence.png')
plt.show()

# ---------------------------------------------------------------------------
# B. Energy-use intensity (EUI = MFBTU / SQFT, kBtu per sq ft)
# ---------------------------------------------------------------------------
df = pd.read_csv(root / 'cbecs2018_processed.csv')
train = df[df.split == 'train'].copy()
pba_cols = [c for c in train.columns if c.startswith('PBA__code_')]
train['PBA'] = train[pba_cols].idxmax(axis=1).str.replace('PBA__code_', '').astype(int).map(PBA_LABELS)
train['EUI'] = train.MFBTU / train.SQFT
# EIA replaces SQFT for buildings over 1M sq ft with a group average, so their EUI is approximate
train['sqft_masked'] = train.SQFT > 1_000_000

# Tertile cut points on training data define the candidate low/medium/high EUI classes
cuts = train.EUI.quantile([1/3, 2/3]).to_numpy()
train['EUI_class'] = pd.cut(train.EUI, [-np.inf, *cuts, np.inf], labels=['Low', 'Medium', 'High'])
print(f'EUI tertile cut points: {cuts[0]:.1f} and {cuts[1]:.1f} kBtu/sq ft')
print(train.EUI_class.value_counts().sort_index())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={'width_ratios': [1.1, 1.1, 0.9]})

# B1. Distribution with tertile cut points
ax = axes[0]
bins = np.logspace(np.log10(train.EUI.min()), np.log10(train.EUI.max()), 45)
ax.hist(train.EUI, bins=bins, color=BLUE, edgecolor='white', linewidth=0.5)
ax.set_xscale('log')
for c in cuts:
    ax.axvline(c, color=INK, linestyle='--', linewidth=1)
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
ymax = ax.get_ylim()[1]
for label, x, ha in zip(['Low', 'Medium', 'High'], [cuts[0] / 1.15, np.sqrt(cuts[0] * cuts[1]), cuts[1] * 1.15],
                        ['right', 'center', 'left']):
    ax.text(x, ymax * 0.98, label, ha=ha, va='top', color=INK, fontweight='bold',
            fontsize=8 if label == 'Medium' else 10, rotation=90 if label == 'Medium' else 0)
ax.set_xlabel('EUI (kBtu per sq ft; log scale)')
ax.set_ylabel('Buildings')
ax.set_title('EUI distribution and tertile classes')

# B2. EUI vs floor area, flagging masked SQFT values
ax = axes[1]
normal = train[~train.sqft_masked]
masked = train[train.sqft_masked]
ax.scatter(normal.SQFT, normal.EUI, s=6, alpha=0.25, color=BLUE, linewidth=0, label='Reported floor area')
ax.scatter(masked.SQFT, masked.EUI, s=12, alpha=0.8, color=ORANGE, linewidth=0,
           label=f'> 1M sq ft, EIA-masked area (n={len(masked)})')
ax.set_xscale('log'); ax.set_yscale('log')
for c in cuts:
    ax.axhline(c, color=INK, linestyle='--', linewidth=1)
ax.set_xlabel('Floor area (sq ft; log scale)')
ax.set_ylabel('EUI (kBtu per sq ft; log scale)')
ax.set_title('EUI is nearly flat in building size')
ax.legend(loc='lower left', frameon=False, fontsize=8, markerscale=1.5)

# B3. How the switch from total energy to EUI changes numeric associations
ax = axes[2]
num = ['SQFT', 'NWKER', 'WKHRS', 'MONUSE']
corr = pd.DataFrame({
    'MFBTU': train[num + ['MFBTU']].corr(method='spearman').MFBTU[num],
    'EUI': train[num + ['EUI']].corr(method='spearman').EUI[num],
})
names = {'SQFT': 'Floor area', 'NWKER': 'Employees', 'WKHRS': 'Weekly hours', 'MONUSE': 'Months in use'}
for i, f in enumerate(num):
    ax.plot(corr.loc[f, ['EUI', 'MFBTU']], [i, i], color='#c8c6bf', linewidth=2, zorder=1)
ax.scatter(corr.MFBTU, range(len(num)), s=60, color=GRAY, zorder=2, label='Total energy (MFBTU)')
ax.scatter(corr.EUI, range(len(num)), s=60, color=BLUE, zorder=3, label='EUI')
for i, f in enumerate(num):
    ax.text(corr.loc[f, 'EUI'], i - 0.18, f"{corr.loc[f, 'EUI']:.2f}", ha='center', va='bottom', fontsize=8, color=INK)
    ax.text(corr.loc[f, 'MFBTU'], i + 0.18, f"{corr.loc[f, 'MFBTU']:.2f}", ha='center', va='top', fontsize=8, color=INK_2)
ax.axvline(0, color=INK_2, linewidth=0.8)
ax.set_yticks(range(len(num)), [names[f] for f in num])
ax.set_ylim(-0.6, len(num) - 0.3)
ax.invert_yaxis()
ax.set_xlim(-0.2, 1)
ax.grid(axis='y', visible=False)
ax.set_xlabel('Spearman correlation')
ax.set_title('Size matters for totals, not intensity')
ax.legend(loc='lower right', frameon=False, fontsize=8)

fig.suptitle(f'Energy-use intensity, training sample ({len(train):,} buildings, unweighted)',
             x=0.01, ha='left', fontsize=12, fontweight='bold', y=1.03)
fig.tight_layout()
fig.savefig(fig_dir / 'B_eui_overview.png')
plt.show()

# B4. EUI by principal building activity, ordered by median, with class cut points
stats = train.groupby('PBA').EUI.agg(['median', 'size']).sort_values('median')
groups = [train.loc[train.PBA == p, 'EUI'] for p in stats.index]

fig, ax = plt.subplots(figsize=(9, 8))
ax.axvspan(train.EUI.min() / 2, cuts[0], color='#f3f2ee', zorder=0)
ax.axvspan(cuts[1], train.EUI.max() * 2, color='#f3f2ee', zorder=0)
ax.boxplot(groups, orientation='horizontal', widths=0.6, patch_artist=True, showfliers=True,
           boxprops=dict(facecolor='#b7d3f6', edgecolor=BLUE),
           medianprops=dict(color=INK, linewidth=1.5),
           whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE),
           flierprops=dict(marker='o', markersize=2.5, markerfacecolor=GRAY,
                           markeredgecolor='none', alpha=0.6))
ax.set_yticks(range(1, len(stats) + 1), [f'{p} (n={n})' for p, n in zip(stats.index, stats['size'])])
ax.set_xscale('log')
ax.set_xlim(train.EUI.min() / 2, train.EUI.max() * 2)
for c in cuts:
    ax.axvline(c, color=INK, linestyle='--', linewidth=1)
for label, x in zip(['Low', 'Medium', 'High'], [cuts[0] / 8, np.sqrt(cuts[0] * cuts[1]), cuts[1] * 8]):
    ax.text(x, len(stats) + 0.9, label, ha='center', va='bottom', fontweight='bold', color=INK)
ax.set_ylim(0.4, len(stats) + 1.6)
ax.grid(axis='y', visible=False)
ax.set_xlabel('EUI (kBtu per sq ft; log scale)')
ax.set_title('EUI by principal building activity, sorted by median')
fig.text(0.01, -0.01, 'Dashed lines: training tertile cut points that would define low / medium / high EUI classes.',
         fontsize=8, color=INK_2)
fig.tight_layout()
fig.savefig(fig_dir / 'B_eui_by_activity.png')
plt.show()

# Share of each activity falling in each EUI class (useful for the classification option)
print(pd.crosstab(train.PBA, train.EUI_class, normalize='index').round(2)
      .loc[stats.index[::-1]])

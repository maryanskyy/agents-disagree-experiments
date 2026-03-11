#!/usr/bin/env python3
"""
Generate all paper figures from V4 experiment data.
Publication-quality: PDF (vector) + PNG (300 DPI).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

OUT = Path(r"C:\Users\Artem\Desktop\agents-disagree-experiments\paper\figures")
OUT.mkdir(parents=True, exist_ok=True)

# -- Publication style ---------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'mathtext.fontset': 'cm',
})

# -- Colorblind-friendly palette (Wong 2011) ----------------------------
CB_BLUE   = '#0173B2'
CB_ORANGE = '#DE8F05'
CB_GREEN  = '#029E73'
CB_RED    = '#D55E00'
CB_PURPLE = '#CC78BC'
CB_GRAY   = '#949494'
CB_LBLUE  = '#56B4E9'
CB_GOLD   = '#ECE133'

# -- V4 Data ------------------------------------------------------------
CELLS_ORDER = [
    'Diverse-mixed\n+ Judge',
    'Diverse-strong\n+ Judge',
    'Homo-Opus\n+ Judge',
    'Diverse-strong\n+ Vote',
    'Diverse-strong\n+ Synthesis',
]
CELLS = {
    'Diverse-strong\n+ Judge':     {'wr': 0.810, 'ci': (0.768, 0.851)},
    'Homo-Opus\n+ Judge':          {'wr': 0.512, 'ci': (0.500, 0.530)},
    'Diverse-mixed\n+ Judge':      {'wr': 0.929, 'ci': (0.887, 0.964)},
    'Diverse-strong\n+ Vote':      {'wr': 0.496, 'ci': (0.425, 0.563)},
    'Diverse-strong\n+ Synthesis': {'wr': 0.179, 'ci': (0.127, 0.234)},
}

CATEGORY_DATA = {
    'Coding':        {'div_j': 0.833, 'homo_j': 0.500,
                      'div_j_ci': (0.750, 0.917), 'homo_j_ci': (0.500, 0.500)},
    'Creative':      {'div_j': 0.833, 'homo_j': 0.542,
                      'div_j_ci': (0.750, 0.917), 'homo_j_ci': (0.500, 0.625)},
    'Ethics':        {'div_j': 0.792, 'homo_j': 0.500,
                      'div_j_ci': (0.667, 0.917), 'homo_j_ci': (0.500, 0.500)},
    'Math/Logic':    {'div_j': 0.875, 'homo_j': 0.542,
                      'div_j_ci': (0.792, 0.958), 'homo_j_ci': (0.500, 0.625)},
    'Reasoning':     {'div_j': 0.833, 'homo_j': 0.500,
                      'div_j_ci': (0.750, 0.917), 'homo_j_ci': (0.500, 0.500)},
    'Science':       {'div_j': 0.792, 'homo_j': 0.500,
                      'div_j_ci': (0.750, 0.875), 'homo_j_ci': (0.500, 0.500)},
    'Summarization': {'div_j': 0.708, 'homo_j': 0.500,
                      'div_j_ci': (0.583, 0.833), 'homo_j_ci': (0.500, 0.500)},
}

# Per-task forest plot data (diverse_strong_judge vs homo_opus_judge)
FOREST_DATA = [
    ('Streaming dedup',       'Coding',    0.750, 0.500),
    ('Race condition debug',  'Coding',    1.000, 0.500),
    ('Multi-tenant design',   'Coding',    1.000, 0.500),
    ('Code review sec/perf',  'Coding',    0.750, 0.500),
    ('API migration',         'Coding',    0.750, 0.500),
    ('Flaky test fix',        'Coding',    0.750, 0.500),
    ('Polyphonic city',       'Creative',  0.750, 0.500),
    ('Epistolary Mars',       'Creative',  1.000, 0.500),
    ('Poetry cycle',          'Creative',  0.750, 0.500),
    ('Worldbuilding',         'Creative',  0.750, 0.500),
    ('AI rights courtroom',   'Creative',  0.750, 0.750),
    ('Myth retelling',        'Creative',  1.000, 0.500),
    ('Facial recog. transit', 'Ethics',    1.000, 0.500),
    ('AI tutor data',         'Ethics',    1.000, 0.500),
    ('Autonomous weapons',    'Ethics',    0.750, 0.500),
    ('Organ allocation',      'Ethics',    0.500, 0.500),
    ('Ventilator triage',     'Ethics',    0.750, 0.500),
    ('Carbon border adj.',    'Ethics',    0.750, 0.500),
    ('Probability triage',    'Math',      1.000, 0.500),
    ('Integer optimization',  'Math',      1.000, 0.500),
    ('Logic grid contracts',  'Math',      1.000, 0.750),
    ('Bayesian diagnostic',   'Math',      0.750, 0.500),
    ('Scheduling deps',       'Math',      0.750, 0.500),
    ('Game theory split',     'Math',      0.750, 0.500),
    ('Causal policy rev.',    'Reasoning', 0.750, 0.500),
    ('Counterfact. outbreak', 'Reasoning', 1.000, 0.500),
    ('Argument eval. media',  'Reasoning', 1.000, 0.500),
    ('Root cause factory',    'Reasoning', 0.750, 0.500),
    ('Strategic negotiation', 'Reasoning', 0.750, 0.500),
    ('Uncertainty intel',     'Reasoning', 0.750, 0.500),
    ('Heat dome mechanism',   'Science',   0.750, 0.500),
    ('Memory consolidation',  'Science',   0.750, 0.500),
    ('Adaptive clinical',     'Science',   0.750, 0.500),
    ('Battery degradation',   'Science',   0.750, 0.500),
    ('Ecosystem restoration', 'Science',   0.750, 0.500),
    ('Epidemiology models',   'Science',   1.000, 0.500),
    ('Board packet crisis',   'Summary',   0.750, 0.500),
    ('Incident timeline',     'Summary',   0.750, 0.500),
    ('Expert panel compare',  'Summary',   0.750, 0.500),
    ('Customer feedback',     'Summary',   0.500, 0.500),
    ('Legal multi-opinion',   'Summary',   0.500, 0.500),
    ('Policy roundtable',     'Summary',   1.000, 0.500),
]


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', format='pdf')
    fig.savefig(OUT / f'{name}.png', dpi=300)
    plt.close(fig)
    print(f"  OK {name}")


# ================================================================
# FIGURE 1: Selection Bottleneck (2-panel, full-width ~7 in)
# ================================================================
def fig1_selection_bottleneck():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                                    gridspec_kw={'width_ratios': [1, 1.15]})

    s = np.linspace(0, 1, 300)
    mu_best  = 0.65
    oracle_d = 0.93
    mean_d   = 0.45
    s_star   = (mu_best - mean_d) / (oracle_d - mean_d)

    Q_homo    = np.full_like(s, mu_best)
    Q_diverse = s * oracle_d + (1 - s) * mean_d

    # -- Panel (a) --
    mask_l = s <= s_star
    mask_r = s >= s_star
    ax1.fill_between(s[mask_l], Q_diverse[mask_l], Q_homo[mask_l],
                     alpha=0.10, color=CB_RED, linewidth=0)
    ax1.fill_between(s[mask_r], Q_diverse[mask_r], Q_homo[mask_r],
                     alpha=0.10, color=CB_BLUE, linewidth=0)

    ax1.plot(s, Q_homo,    color=CB_GRAY, lw=2.2, label='Homogeneous team')
    ax1.plot(s, Q_diverse, color=CB_BLUE, lw=2.2, label='Diverse team')
    ax1.axvline(x=s_star, color=CB_RED, ls='--', lw=1.2, alpha=0.7)
    ax1.plot(s_star, mu_best, 'o', color=CB_RED, ms=7, zorder=5)

    ax1.text(s_star + 0.04, mu_best - 0.06, r'$s^*$', fontsize=13,
             fontweight='bold', color=CB_RED)
    ax1.text(s_star / 2, 0.40, 'Diversity\nhurts', ha='center', fontsize=8,
             color=CB_RED, fontstyle='italic', alpha=0.85)
    ax1.text((1 + s_star) / 2, 0.86, 'Diversity\nhelps', ha='center',
             fontsize=8, color=CB_BLUE, fontstyle='italic', alpha=0.85)

    ax1.set_xlabel('Selector quality  $s$')
    ax1.set_ylabel('Output quality  $Q(T, s)$')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.32, 0.98)
    ax1.legend(loc='lower right', framealpha=0.9, edgecolor='none')
    ax1.set_title('(a)  Selection quality threshold',
                  fontsize=10, fontweight='bold')

    # -- Panel (b): V4 empirical operating points --
    ax2.plot(s, Q_homo,    color=CB_GRAY, lw=2.2)
    ax2.plot(s, Q_diverse, color=CB_BLUE, lw=2.2)
    ax2.axvline(x=s_star, color=CB_RED, ls='--', lw=0.8, alpha=0.3)

    ops = [
        (0.00, 0.179, 'Synthesis\n(WR = .18)', 's', CB_RED),
        (0.10, 0.496, 'Vote\n(WR = .50)', 'D', CB_ORANGE),
        (0.75, 0.810, 'Judge\n(WR = .81)', '^', CB_BLUE),
    ]
    for sx, wr, lbl, mkr, col in ops:
        ax2.plot(sx, wr, marker=mkr, ms=10, color=col, zorder=6,
                 markeredgecolor='white', markeredgewidth=0.8)
        ha = 'left' if sx < 0.5 else 'right'
        xoff = 0.05 if sx < 0.5 else -0.05
        ax2.annotate(lbl, xy=(sx, wr),
                     xytext=(sx + xoff, wr + 0.05),
                     fontsize=7.5, color=col, fontweight='bold',
                     ha=ha, va='bottom')

    # Homo baseline — position label far right to avoid Vote collision
    ax2.axhline(y=0.512, color=CB_GRAY, ls=':', lw=1.2, alpha=0.5)
    ax2.text(0.95, 0.43, 'Homo baseline\n(WR = .51)', fontsize=6.5,
             color=CB_GRAY, ha='right', va='top', fontstyle='italic')

    ax2.set_xlabel('Selector quality  $s$')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0.05, 0.98)
    ax2.set_title('(b)  V4 empirical operating regimes',
                  fontsize=10, fontweight='bold')

    plt.tight_layout(w_pad=2.5)
    save(fig, 'fig1_selection_bottleneck')


# ================================================================
# FIGURE 2: Factorial bar chart (5 cells, horizontal for readability)
# ================================================================
def fig2_factorial_heatmap():
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    names = CELLS_ORDER
    wrs   = [CELLS[n]['wr'] for n in names]
    ci_lo = [CELLS[n]['ci'][0] for n in names]
    ci_hi = [CELLS[n]['ci'][1] for n in names]
    yerr_lo = [w - lo for w, lo in zip(wrs, ci_lo)]
    yerr_hi = [hi - w for w, hi in zip(wrs, ci_hi)]

    # Horizontal bars for better label readability
    colors = [CB_LBLUE, CB_BLUE, CB_GRAY, CB_ORANGE, CB_RED]
    # Clean single-line names for horizontal layout
    clean = [n.replace('\n', ' ') for n in names]

    y = np.arange(len(names))
    bars = ax.barh(y, wrs, height=0.6, color=colors, edgecolor='white',
                   linewidth=0.8, zorder=3)
    ax.errorbar(wrs, y, xerr=[yerr_lo, yerr_hi], fmt='none',
                ecolor='#333333', capsize=4, capthick=1.2, linewidth=1.2,
                zorder=4)

    # Chance line
    ax.axvline(x=0.5, color=CB_GRAY, ls=':', lw=1, alpha=0.5, zorder=1)

    # Value labels to right of bars
    for i, (v, bar) in enumerate(zip(wrs, bars)):
        weight = 'bold' if v > 0.7 or v < 0.25 else 'normal'
        xpos = v + yerr_hi[i] + 0.015
        ax.text(xpos, i, f'{v:.3f}', ha='left', va='center',
                fontsize=9, fontweight=weight, color='#222222')

    # Annotations
    ax.annotate('Weak-model surprise',
                xy=(0.929, 0), xytext=(0.75, -0.7),
                fontsize=7, color=CB_LBLUE, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=CB_LBLUE, lw=1),
                ha='center')
    ax.annotate('Synthesis failure',
                xy=(0.179, 4), xytext=(0.40, 4.7),
                fontsize=7, color=CB_RED, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=CB_RED, lw=1),
                ha='center')

    ax.set_yticks(y)
    ax.set_yticklabels(clean, fontsize=8.5)
    ax.set_xlabel('BT-Corrected Win Rate')
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-1.0, len(names) - 0.3)
    ax.invert_yaxis()

    # Minimal legend via text
    ax.text(0.98, len(names) - 0.5,
            '95% CI shown\n$N$ = 42 tasks per cell',
            fontsize=6.5, ha='right', va='top', color='#666666',
            fontstyle='italic')

    plt.tight_layout()
    save(fig, 'fig2_factorial_heatmap')


# ================================================================
# FIGURE 3: Diversity benefit across 7 task categories
# ================================================================
def fig3_team_size_scaling():
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    cats   = list(CATEGORY_DATA.keys())
    n_cats = len(cats)

    div_wr  = [CATEGORY_DATA[c]['div_j'] for c in cats]
    homo_wr = [CATEGORY_DATA[c]['homo_j'] for c in cats]
    div_ci  = [CATEGORY_DATA[c]['div_j_ci'] for c in cats]
    homo_ci = [CATEGORY_DATA[c]['homo_j_ci'] for c in cats]

    x = np.arange(n_cats)
    w = 0.30

    # Homo bars
    ax.bar(x - w/2, homo_wr, w, color=CB_GRAY, edgecolor='white',
           label='Homo-Opus + Judge', zorder=3)
    yerr_h_lo = [wr - ci[0] for wr, ci in zip(homo_wr, homo_ci)]
    yerr_h_hi = [ci[1] - wr for wr, ci in zip(homo_wr, homo_ci)]
    ax.errorbar(x - w/2, homo_wr, yerr=[yerr_h_lo, yerr_h_hi], fmt='none',
                ecolor='#555555', capsize=3, capthick=1, zorder=4)

    # Diverse bars
    ax.bar(x + w/2, div_wr, w, color=CB_BLUE, edgecolor='white',
           label='Diverse-strong + Judge', zorder=3)
    yerr_d_lo = [wr - ci[0] for wr, ci in zip(div_wr, div_ci)]
    yerr_d_hi = [ci[1] - wr for wr, ci in zip(div_wr, div_ci)]
    ax.errorbar(x + w/2, div_wr, yerr=[yerr_d_lo, yerr_d_hi], fmt='none',
                ecolor='#333333', capsize=3, capthick=1, zorder=4)

    # Delta annotations (above the diverse bar, offset to avoid error bars)
    for i in range(n_cats):
        delta = div_wr[i] - homo_wr[i]
        ypos = div_ci[i][1] + 0.035   # above the CI cap
        ax.text(i + w/2, ypos, f'+.{int(delta*1000):03d}',
                fontsize=7, color='#222222', fontweight='bold',
                ha='center', va='bottom')

    ax.axhline(y=0.5, color=CB_GRAY, ls=':', lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('Win Rate')
    ax.set_ylim(0.38, 1.05)
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='none', fontsize=7.5)

    plt.tight_layout()
    save(fig, 'fig3_team_size_scaling')


# ================================================================
# FIGURE 4: Forest plot (per-task effects, 42 tasks in 7 categories)
# ================================================================
def fig4_forest_plot():
    categories_order = ['Coding', 'Creative', 'Ethics', 'Math',
                        'Reasoning', 'Science', 'Summary']
    # Distinct shapes + colors for categories (colorblind-safe)
    cat_style = {
        'Coding':    CB_BLUE,
        'Creative':  CB_PURPLE,
        'Ethics':    CB_ORANGE,
        'Math':      CB_GREEN,
        'Reasoning': CB_LBLUE,
        'Science':   CB_GRAY,
        'Summary':   CB_RED,
    }

    # Group tasks by category, sort by delta desc within each
    grouped = []
    for cat in categories_order:
        tasks = [(t, c, d, h) for t, c, d, h in FOREST_DATA if c == cat]
        tasks.sort(key=lambda x: x[2] - x[3], reverse=True)
        grouped.extend(tasks)

    n = len(grouped)
    deltas = [d - h for _, _, d, h in grouped]
    names  = [t for t, _, _, _ in grouped]
    cats   = [c for _, c, _, _ in grouped]

    # Approximate CIs: each task has ~4 pairwise comparisons
    ci_half = []
    for _, _, d, h in grouped:
        se_d = np.sqrt(d * (1 - d) / 4) if 0 < d < 1 else 0.05
        se_h = np.sqrt(h * (1 - h) / 4) if 0 < h < 1 else 0.05
        ci_half.append(1.96 * np.sqrt(se_d**2 + se_h**2))

    fig, ax = plt.subplots(figsize=(4.0, 7.5))

    y_pos = np.arange(n)

    # Alternating category bands
    current_cat = None
    band_toggle = False
    for i, (t, c, d, h) in enumerate(grouped):
        if c != current_cat:
            current_cat = c
            band_toggle = not band_toggle
            start = i
        if band_toggle:
            ax.axhspan(i - 0.5, i + 0.5, color='#f5f5f5', zorder=0)

    # Category separators
    current_cat = None
    cat_ranges = {}
    for i, (t, c, d, h) in enumerate(grouped):
        if c != current_cat:
            if current_cat is not None:
                ax.axhline(y=i - 0.5, color='#cccccc', lw=0.5, zorder=1)
            cat_ranges.setdefault(c, [i, i])
            current_cat = c
        cat_ranges[c][1] = i

    # Plot effect sizes
    for i in range(n):
        col = cat_style[cats[i]]
        ax.plot(deltas[i], y_pos[i], 'o', color=col, ms=4.5, zorder=5)
        ax.plot([max(deltas[i] - ci_half[i], -0.15),
                 min(deltas[i] + ci_half[i], 0.85)],
                [y_pos[i], y_pos[i]], '-', color=col, lw=1.2, zorder=4)

    # Zero line
    ax.axvline(x=0, color='black', lw=0.8, zorder=2)

    # Overall effect diamond
    overall_delta  = 0.298
    overall_ci_lo  = 0.250
    overall_ci_hi  = 0.345
    diamond_y = n + 1.5
    diamond = plt.Polygon([
        (overall_ci_lo, diamond_y),
        (overall_delta, diamond_y + 0.45),
        (overall_ci_hi, diamond_y),
        (overall_delta, diamond_y - 0.45),
    ], closed=True, facecolor=CB_BLUE, edgecolor='#222222', lw=1.2, zorder=6)
    ax.add_patch(diamond)
    ax.text(overall_ci_hi + 0.04, diamond_y,
            f'Overall: +{overall_delta:.3f}  [{overall_ci_lo:.3f}, {overall_ci_hi:.3f}]',
            fontsize=6.5, fontweight='bold', va='center', color='#222222')

    # Separator before overall
    ax.axhline(y=n + 0.3, color='#999999', lw=0.8, zorder=1)

    # Task labels on y-axis
    ax.set_yticks(list(y_pos) + [diamond_y])
    ax.set_yticklabels(names + ['Overall'], fontsize=6)

    ax.set_xlabel(r'$\Delta$ Win Rate  (Diverse+Judge $-$ Homo+Judge)', fontsize=9)
    ax.set_xlim(-0.18, 0.80)
    ax.set_ylim(-0.8, diamond_y + 1.5)
    ax.invert_yaxis()

    # Category legend — use colored category labels on right margin instead
    # This avoids any legend-data overlap in a dense forest plot
    for cat in categories_order:
        idxs = [i for i, (_, c, _, _) in enumerate(grouped) if c == cat]
        mid_y = (idxs[0] + idxs[-1]) / 2
        ax.text(0.73, mid_y, cat, fontsize=5.5, fontweight='bold',
                color=cat_style[cat], va='center', ha='left',
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          ec='none', alpha=0.8))

    # Summary stats — below diamond
    n_positive = sum(1 for d in deltas if d > 0)
    n_zero = sum(1 for d in deltas if d == 0)
    ax.text(0.30, diamond_y + 1.3,
            f'{n_positive}/42 positive, {n_zero} ties;  '
            f'g = 2.71, p < 1e-14',
            fontsize=6.5, color='#444444', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='#f0f0f0',
                      ec='#cccccc', lw=0.5))

    plt.tight_layout()
    save(fig, 'fig4_forest_plot')


# ================================================================
if __name__ == '__main__':
    print("Generating V4 figures...")
    fig1_selection_bottleneck()
    fig2_factorial_heatmap()
    fig3_team_size_scaling()
    fig4_forest_plot()
    print(f"\nAll figures saved to {OUT}")

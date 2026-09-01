"""
agu_style.py
============
Shared Matplotlib styling for the Water Resources Research figures of

    Patil, Dallison & Jahanshahi,
    "Storage controls on the pace of flood-drought transitions across
     Great Britain under a warming climate."

Encapsulates the AGU/Wiley graphics requirements so that every figure script
(fig01_studyarea.py, fig02_continuum.py, fig03_mechanism.py,
 fig04_corroboration.py, fig05_projection.py, figS1_timescales.py) produces
submission-ready output without repeating the boilerplate.

AGU figure requirements enforced here
-------------------------------------
* Width 50-170 mm (1-col 50-85 mm; 2-col 105-170 mm); height <= 228 mm.
* In-figure text 8 pt at final print size (6 pt sub/superscript); nothing < 6 pt.
* Vector PDF for publication; a 600-dpi PNG preview is written alongside for drafting.
* Colour-blind-safe palette (Okabe-Ito) per AGU's accessible-colour guidance.
* Fonts embedded as TrueType (Type-42) so the PDF is editable/portable.

Dependencies: matplotlib, numpy (both present in the JASMIN `jaspy` env).
"""
from __future__ import annotations

import matplotlib as mpl

# --------------------------------------------------------------------------- #
# Physical sizing                                                             #
# --------------------------------------------------------------------------- #
MM = 1.0 / 25.4                     # millimetres -> inches
W_1COL = 85 * MM                    # maximum single-column width (in)
W_1P5COL = 120 * MM                 # a convenient 1.5-column width (in)
W_2COL = 170 * MM                   # maximum two-column width (in)
H_MAX = 228 * MM                    # maximum height (in)

# --------------------------------------------------------------------------- #
# Colour-blind-safe palette (Okabe & Ito, 2008)                               #
# --------------------------------------------------------------------------- #
OKABE_ITO = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "purple":  "#CC79A7",
    "grey":    "#999999",
}

# Semantic roles used consistently across figures
C_BASELINE = OKABE_ITO["grey"]      # baseline period
C_FUTURE   = OKABE_ITO["vermil"]    # representative future (RCP8.5)
C_FAST     = OKABE_ITO["skyblue"]   # upper zone / fast flow (Q0+Q1)
C_SLOW     = OKABE_ITO["blue"]      # lower zone / baseflow (Q2)
C_TAIL     = OKABE_ITO["orange"]    # the censored slow tail (> 90 d)

# One colour per RCP, ordered by forcing (light -> dark)
RCP_COLORS = {
    "rcp26": OKABE_ITO["skyblue"],
    "rcp45": OKABE_ITO["green"],
    "rcp60": OKABE_ITO["orange"],
    "rcp85": OKABE_ITO["vermil"],
}
RCP_LABELS = {
    "rcp26": "RCP2.6", "rcp45": "RCP4.5",
    "rcp60": "RCP6.0", "rcp85": "RCP8.5",
}


def set_style() -> None:
    """Apply AGU-compliant rcParams. Call once at the top of each figure script."""
    mpl.rcParams.update({
        # --- fonts: 8 pt body, nothing below 6 pt ---
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":        8,
        "axes.titlesize":   8,
        "axes.labelsize":   8,
        "xtick.labelsize":  7,
        "ytick.labelsize":  7,
        "legend.fontsize":  7,
        "figure.titlesize": 8,
        "mathtext.default": "regular",
        # --- lines & ticks: legible after reduction (>= 0.5 pt rule) ---
        "axes.linewidth":   0.6,
        "lines.linewidth":  1.0,
        "patch.linewidth":  0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction":  "out",
        "ytick.direction":  "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        # --- spines: drop the chartjunk top/right ---
        "axes.spines.top":   False,
        "axes.spines.right": False,
        # --- legend ---
        "legend.frameon":   False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.labelspacing":  0.3,
        # --- output: vector PDF, embedded TrueType fonts, high-dpi raster ---
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
        "svg.fonttype":     "none",
        "savefig.dpi":      600,
        "figure.dpi":       150,
        "savefig.bbox":     "tight",
        # 0.02 in was too tight: matplotlib's tight bounding box slightly
        # under-measures a text artist's true inked extent (mathtext labels
        # worst -- fig05's "Δ$P_s$ (mm)" had its closing parenthesis
        # sliced off at the right edge), so a hairline pad leaves nothing to
        # absorb the error. 0.06 in (~1.5 mm) clears it with margin to spare
        # and costs ~1% in figure width.
        "savefig.pad_inches": 0.06,
    })


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.03,
                weight: str = "bold") -> None:
    """Place a multipart panel letter (a, b, c ...) at the axes' upper left."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
            ha="right", va="bottom", fontweight=weight, fontsize=8)


def legend_below(ax, *args, pad_mm: float = 2.0, **kwargs):
    """Attach a legend centred below `ax`, clearing its tick labels and x-axis
    label by `pad_mm` millimetres. Returns the legend.

    Below-axes placement is the house rule for any legend that would otherwise
    risk sitting on top of data (a different inside-axes corner only moves the
    risk around, and the guess breaks as soon as the data changes). The catch
    is the offset: `bbox_to_anchor` is in axes-fraction units, so a hand-picked
    value has to clear the tick labels *and* the axis-label text, and the
    figure has to be re-rendered to find out whether it does. This measures the
    x-axis' drawn extent instead and converts the requested millimetre gap into
    the right fraction, so the legend sits a fixed, predictable distance below
    the panel regardless of axes height or label depth.
    """
    kwargs.setdefault("loc", "upper center")
    leg = ax.legend(*args, **kwargs)
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.xaxis.get_tightbbox(fig.canvas.get_renderer())
    y_px = (bb.y0 if bb is not None else ax.get_window_extent().y0) \
        - pad_mm * MM * fig.dpi
    y = ax.transAxes.inverted().transform((0, y_px))[1]
    leg.set_bbox_to_anchor((0.5, y), transform=ax.transAxes)
    return leg


def equalise_panel_gaps(fig, groups) -> None:
    """Redistribute horizontal whitespace so adjacent panels look equally
    spaced.

    `groups` is a list of axes groups, left to right -- one group per visual
    panel, listing every axes that belongs to it (e.g. `[[axA], [axB],
    [axC, cax]]`, so a panel and its attached colorbar move together).

    Why this exists rather than just tuning `wspace`: `wspace` spaces the
    *cells* of the gridspec, but what the eye reads is the gap between the
    panels' rendered content, and the two are not the same. A panel carrying
    a y-axis label and tick labels overflows its cell on the left, while a
    map with `set_axis_off()` and `set_aspect("equal")` under-fills its cell
    (matplotlib shrinks the box to satisfy the aspect and anchors it to one
    side -- to the 'E' side once a colorbar has been attached, dumping all of
    the slack on its left). Equal cell spacing therefore renders as visibly
    unequal panel spacing. This measures the drawn tight bounding box of each
    group and translates the groups so the gaps between those boxes are
    equal, holding the leftmost and rightmost edges fixed so the figure's
    overall extent (and every panel's size) is unchanged.

    Call it after all panels are drawn and immediately before `save()`.
    """
    from matplotlib.transforms import Bbox

    if len(groups) < 2:
        return
    fig.canvas.draw()                       # positions/aspects must be final
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [Bbox.union([ax.get_tightbbox(renderer) for ax in grp])
             .transformed(inv) for grp in groups]

    left, right = boxes[0].x0, boxes[-1].x1
    widths = [b.width for b in boxes]
    gap = (right - left - sum(widths)) / (len(boxes) - 1)

    x = left
    for grp, box in zip(groups, boxes):
        dx = x - box.x0
        if abs(dx) > 1e-9:
            for ax in grp:
                # the *original* position, not the aspect-adjusted one: a pure
                # translation of it translates the adjusted box identically,
                # and leaves apply_aspect() free to redo its own thing at the
                # next draw.
                p = ax.get_position(original=True)
                ax.set_position([p.x0 + dx, p.y0, p.width, p.height])
        x += box.width + gap


def save(fig, outstem, png: bool = True) -> None:
    """
    Save `fig` as a vector PDF (for submission) and, optionally, a 600-dpi PNG
    (for pasting into the draft). `outstem` is a path without extension.
    """
    from pathlib import Path
    outstem = Path(outstem)
    outstem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outstem.with_suffix(".pdf"))
    if png:
        fig.savefig(outstem.with_suffix(".png"), dpi=600)
    print(f"  wrote {outstem.with_suffix('.pdf')}"
          + (f" and {outstem.with_suffix('.png').name}" if png else ""))

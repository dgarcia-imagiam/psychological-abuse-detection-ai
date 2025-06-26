from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def create_empty_compare_llm_dataframe(names: List[str]):
    df = pd.DataFrame(
        data=np.zeros((len(names), len(names)), dtype=int),
        index=names,
        columns=names
    )
    np.fill_diagonal(df.values, -1)

    return df


def create_compare_llm_figure(df: pd.DataFrame):
    """
    Produce a square, color‑coded matrix plot for a DataFrame whose
    rows/columns share the same set of labels.

    Color scheme
    ------------
        -2 → dark gray
        -1 → light gray
         0 → red
         1 → orange
         2 → green
      other → black

    Parameters
    ----------
    df : pandas.DataFrame
        A square DataFrame of integers.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing a single Axes image.
    """
    # ---- 1. Map values to *codes* (0‒5) ------------------------------------
    mapping = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}            # all others ⇒ 5
    code_matrix = np.full(df.shape, 5, dtype=int)         # default = 5 (“other”)
    for val, code in mapping.items():
        code_matrix[df.values == val] = code

    # ---- 2. Build a discrete ListedColormap --------------------------------
    colors = ["dimgray",            # code 0  -> -2
              "lightgrey",          # code 1  -> -1
              "red",                # code 2  ->  0
              "orange",             # code 3  ->  1
              "green",              # code 4  ->  2
              "black"]              # code 5  -> other
    cmap  = ListedColormap(colors,   name="custom_value_map")
    norm  = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, cmap.N)

    # ---- 3. Create the plot -------------------------------------------------
    fig, ax = plt.subplots(figsize=(len(df.columns) * 0.5,
                                    len(df.index)   * 0.5))
    n = len(df)  # square → rows == cols
    im = ax.imshow(
        code_matrix,
        cmap=cmap, norm=norm,
        interpolation="none",  # no anti-aliasing
        aspect="equal",
        extent=(-0.5, n-0.5, n-0.5, -0.5)
    )

    # ---- 4. Ticks / labels --------------------------------------------------
    # the image now spans exactly one integer unit per pixel
    ax.set_xticks(np.arange(n))  # major ticks at centres
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(df.columns, rotation=90, va="top")
    ax.set_yticklabels(df.index)

    # grid on every edge (-0.5 … n-0.5 inclusive)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1, snap=True)
    ax.tick_params(which="minor", length=0)

    # Tighten layout, detach, return
    fig.tight_layout()
    plt.close(fig)      # prevent automatic display in notebooks
    return fig


def get_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a one-column DataFrame called ``score`` whose index matches *df*.

    Rules
    -----
    • For each row, sum every value ≥ 0.
    • If a row contains *no* non-negative numbers, set its score to –1.

    Parameters
    ----------
    df : pandas.DataFrame
        Any numeric DataFrame.

    Returns
    -------
    pandas.DataFrame
        Index = df.index, single column ``score`` (dtype:int/float).
    """
    # sum of non-negative values
    scores = df.where(df >= 0, 0).sum(axis=1)

    # rows with *only* negatives → score = –1
    all_neg_mask = ~(df >= 0).any(axis=1)
    scores[all_neg_mask] = -1

    return scores.to_frame("score")


def get_scores_many(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine several one-column ``score`` DataFrames.

    For every index label (row name)

    1. Sum its ``score`` values across *all* input DataFrames.
    2. If every value contributing to that row is < 0, set the total to –1.
    3. After the totals are computed, drop any row whose final score is < 0.
    4. Return the result sorted by descending score.

    Parameters
    ----------
    dfs : list[pandas.DataFrame]
        Each DataFrame must have a single column called ``score``.

    Returns
    -------
    pandas.DataFrame
        One column ``score``, index = union of all row labels.
    """
    if not dfs:
        return pd.DataFrame(columns=["score"])

    # ---------- 1. Align rows & collect all scores --------------------------
    # Concatenate *vertically* → we get duplicates of each row, one per DF.
    stacked = pd.concat(dfs, axis=0)            # index may repeat

    # ---------- 2. Compute the sum per row ----------------------------------
    # groupby(level=0) because the *row labels* form the index level
    sums  = stacked.groupby(level=0)["score"].sum()

    # ---------- 3. Detect “all-negative” rows -------------------------------
    # For each label, were *all* contributing scores < 0 ?
    all_negative = (
        stacked["score"]
        .lt(0)                                  # True where score < 0
        .groupby(level=0)
        .all()                                  # True only if all True
    )

    # overwrite totals that are “all negative” with –1
    sums.loc[all_negative] = -1

    # ---------- 4. Final post-processing ------------------------------------
    result = sums.to_frame("score")

    # drop rows with negative totals (this also removes the –1 rows)
    result = result[result["score"] >= 0]

    # sort descending
    result = result.sort_values("score", ascending=False)

    return result


def normalize_scores(
        df: pd.DataFrame,
        column: str = "score",
        target_total: float = 100.0,
        decimals: int | None = 2
    ) -> pd.DataFrame:
    """
    Return a copy of *df* in which *column* is rescaled so its values sum
    to *target_total*.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table with a numeric column to normalise.
    column : str, default "score"
        Name of the numeric column to scale.
    target_total : float, default 100.0
        The desired sum of the column after scaling.
    decimals : int | None, default 2
        Round the rescaled values to this many decimal places.
        Pass *None* for no rounding at all.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the same index/columns, but a normalised
        *column*.  The dtype becomes float (after scaling).
    """
    total = df[column].sum()
    if total == 0:
        raise ValueError(f"Cannot normalise: the sum of '{column}' is 0.")

    factor = target_total / total
    out = df.copy()
    out[column] = df[column] * factor

    if decimals is not None:
        out[column] = out[column].round(decimals)

        # optional: tweak the last value so rounding errors don’t spoil 100
        diff = target_total - out[column].sum()
        if diff:                             # non-zero float is True
            # add 'diff' to the largest row (least likely to flip sign)
            idx_max = out[column].idxmax()
            out.at[idx_max, column] += diff

    return out


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize


def create_compare_llm_barplot_figure(
    df: pd.DataFrame,
    column: str = "score",
    decimals: int = 2,
    *,
    pad: float = 0.02,
    fig_width: float = 6.0,
) -> plt.Figure:
    """
    Create a horizontal bar plot whose colours fade from red → orange → green.

    Colour map
    ----------
    * min value  (v_min)         → red
    * midpoint   ((v_min+v_max)/2) → orange
    * max value  (v_max)         → green
    Values in between are linearly interpolated.

    Bars are annotated with the numeric value, right-aligned just beyond
    each bar.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a numeric column *column*; its index supplies the labels.
    column : str, default "score"
        Name of the column to plot.
    decimals : int, default 2
        Number of decimal places in the value annotations.
    pad : float, default 0.02
        Fraction of (v_max – v_min) added when positioning the annotations.
    fig_width : float, default 6.0
        Width of the figure in inches; height scales automatically.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with a single Axes containing the bar plot.
    """
    # ------- 0.  Extract & validate -----------------------------------------
    if column not in df.columns:
        raise KeyError(f"column '{column}' not found in DataFrame")

    scores = df[column].astype(float)
    if scores.empty:
        raise ValueError("DataFrame contains no rows to plot")

    v_min, v_max = scores.min(), scores.max()

    # ------- 1.  Build colour map -------------------------------------------
    if v_min == v_max:
        # all bars equal → use green for everything
        bar_colors = ["green"] * len(scores)
        norm = None
    else:
        cmap = LinearSegmentedColormap.from_list(
            "red_orange_green", [(0.0, "red"), (0.5, "orange"), (1.0, "green")]
        )
        norm = Normalize(vmin=v_min, vmax=v_max)
        bar_colors = cmap(norm(scores.values))

    # ------- 2.  Create figure & axes ---------------------------------------
    fig_height = 0.5 * len(scores) + 1.0      # heuristic for tidy spacing
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # ------- 3.  Draw horizontal bars ---------------------------------------
    y_pos = np.arange(len(scores))
    ax.barh(y_pos, scores.values, color=bar_colors)

    # ------- 4.  Axis & tick formatting -------------------------------------
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index)
    ax.invert_yaxis()                          # highest score at top

    right_lim = v_max * (1 + pad) if v_max > 0 else v_max - pad * abs(v_max)
    ax.set_xlim(left=0, right=right_lim)
    ax.set_xlabel(column.capitalize())

    ax.spines[["top", "right"]].set_visible(False)

    # ------- 5.  Annotate values --------------------------------------------
    offset = pad * (v_max - v_min)
    for y, v in zip(y_pos, scores.values):
        ax.text(v + offset, y, f"{v:.{decimals}f}", va="center")

    fig.tight_layout()
    plt.close(fig)        # prevent automatic display in notebooks
    return fig

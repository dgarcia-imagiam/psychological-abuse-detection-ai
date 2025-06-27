from typing import List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, Normalize, to_rgba
from functools import reduce
from padai.utils.pandas import iqr_bounds


def create_empty_compare_llm_dataframe(names: List[str]):
    df = pd.DataFrame(
        data=np.zeros((len(names), len(names)), dtype=int),
        index=names,
        columns=names
    )
    np.fill_diagonal(df.values, -1)

    return df


def create_compare_llm_figure(df: pd.DataFrame, title: Optional[str] = None, default_min_max: Optional[Tuple[float, float]] = (0., 2.)):
    """
    Produce a square, color‑coded matrix plot for a DataFrame whose
    rows/columns share the same set of labels.

    Color scheme
    ------------
    *  -2 → dark-grey
    *  -1 → light-grey
    *   >=0 → red → orange → green (min … mid … max)
    *  any other negative value → black

    Parameters
    ----------
    df : pandas.DataFrame
        A square DataFrame of integers.

    title : str | None, default None
        If given, placed above the figure; if None, no title is shown.

    default_min_max: tuple[float, float], default None
        If given, values to use when vmin == vmax

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing a single Axes image.
    """
    values = df.values

    # build a code matrix
    grad_steps = 256  # samples in the red-orange-green ramp
    sentinel = 2 + grad_steps  # index reserved for “other”
    code = np.full(values.shape, sentinel, dtype=int)

    code[values == -2] = 0  # dark-grey
    code[values == -1] = 1  # light-grey

    # map every value ≥ 0 into [2 … 2+grad_steps-1]
    nonneg_mask = values >= 0
    if nonneg_mask.any():
        vmin = values[nonneg_mask].min()
        vmax = values[nonneg_mask].max()

        if vmin == vmax:
            assert vmin >= default_min_max[0]
            assert vmax <= default_min_max[1]

            vmin = default_min_max[0]
            vmax = default_min_max[1]

        if vmin == vmax:  # all equal → map directly to green
            code[nonneg_mask] = 2 + grad_steps - 1
        else:
            scale = (values[nonneg_mask] - vmin) / (vmax - vmin)
            code[nonneg_mask] = 2 + np.round(scale * (grad_steps - 1)).astype(int)

    # build the matching color list and normalizer
    ramp = LinearSegmentedColormap.from_list(
        "red-orange-green", ["red", "orange", "green"], N=grad_steps
    )

    colours = (
            ["dimgray", "lightgrey"]  # 0, 1
            + [ramp(i) for i in range(grad_steps)]  # 2 … 2+grad_steps-1
            + ["black"]  # sentinel
    )
    cmap = ListedColormap(colours, name="custom_llm_map")
    norm = BoundaryNorm(np.arange(len(colours) + 1) - 0.5, cmap.N)

    # create the plot
    n = len(df)
    fig, ax = plt.subplots(figsize=(n * 0.5, n * 0.5))
    ax.imshow(
        code,
        cmap=cmap,
        norm=norm,
        interpolation="none",
        aspect="equal",
        extent=(-0.5, n - 0.5, n - 0.5, -0.5),
    )

    # ticks, grid, layout
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(df.columns, rotation=90, va="top")
    ax.set_yticklabels(df.index)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1, snap=True)
    ax.tick_params(which="minor", length=0)

    if title is not None:
        fig.suptitle(title, fontsize=16)

    fig.tight_layout()
    plt.close(fig)
    return fig


def get_row_scores(df: pd.DataFrame) -> pd.DataFrame:
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


def get_row_scores_many(dfs: List[pd.DataFrame]) -> pd.DataFrame:
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


def create_compare_llm_barplot_figure(
    df: pd.DataFrame,
    column: str = "score",
    decimals: int = 2,
    *,
    pad: float = 0.02,
    fig_width: float = 6.0,
    title: Optional[str] = None,

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
    title : str | None, default None
        If given, placed above the figure; if None, no title is shown.

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

    if title is not None:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()
    plt.close(fig)        # prevent automatic display in notebooks
    return fig


def get_scores(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        raise ValueError("dfs list is empty")

    # sum of values ≥ 0
    sum_ = reduce(lambda a, b: a.add(b.clip(lower=0), fill_value=0), dfs)

    # element-wise minimum
    min_ = reduce(lambda a, b: a.where(a < b, b), dfs)

    # mask: True where *all* values are negative
    invalid = reduce(lambda a, b: a & b, [df.lt(0) for df in dfs])

    # substitute
    return sum_.mask(invalid, min_)


def get_average_scores(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Element-wise average of multiple DataFrames, ignoring negatives.

    For every cell across *dfs*:

    • If at least one value ≥ 0 is present,
      return the arithmetic mean of the non-negative values only.

    • Otherwise (all values < 0),
      return the minimum of those negatives.

    The result uses float dtype because of the division step.
    """
    if not dfs:
        raise ValueError("dfs list is empty")

    # sum of non-negative values
    sum_nonneg = reduce(
        lambda a, b: a.add(b.clip(lower=0), fill_value=0),
        dfs,
    )

    # count of non-negative contributors
    counts = sum((df >= 0).astype(int) for df in dfs)

    # preliminary average (will be NaN where counts == 0)
    avg = sum_nonneg.div(counts).where(counts > 0)

    # cells where *all* values are negative
    min_negatives = reduce(lambda a, b: a.where(a < b, b), dfs)
    invalid_mask = counts.eq(0)          # True where every value was < 0

    # substitute minima into “all-negative” cells
    result = avg.mask(invalid_mask, min_negatives)

    return result


def mse_nonneg(df1: pd.DataFrame, df2: pd.DataFrame) -> Union[float, np.float64]:
    """
    Scalar MSE over cells where *both* DataFrames have values ≥ 0.
    """
    if not (df1.index.equals(df2.index) and df1.columns.equals(df2.columns)):
        raise ValueError("df1 and df2 must have identical index and columns")

    mask = (df1 >= 0) & (df2 >= 0)
    sq_err = (df1 - df2).pow(2).where(mask)      # invalid → NaN

    # flatten to 1-D and take one mean  → every surviving cell counts once
    mse = sq_err.stack().mean()  # stack() drops NaNs by default

    return mse


def barplot_with_outliers(
    df: pd.DataFrame,
    *,
    decimals: int = 2,
    k: float = 1.5,
    cmap_norm = None,      # optional, see note
    title: Optional[str] = None,

) -> plt.Figure:
    """
    Horizontal bar-plot of a single numeric column.

    Parameters
    ----------
    df : pandas.DataFrame
        Index supplies the labels; must have exactly one numeric column.
    decimals : int, default 2
        Round the value annotations.
    k : float, default 1.5
        Tukey fence multiplier; 1.5 → “normal” outliers, 3 → “extreme”.
    cmap_norm : callable | None
        Optional function that receives the value Series and returns an
        array-like of face colours; ignored for outliers (they are red).
    title : str | None, default None
        If given, placed above the figure; if None, no title is shown.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if df.shape[1] != 1:
        raise ValueError("DataFrame must have exactly one column")
    values = df.iloc[:, 0].astype(float)
    labels = df.index

    # 1 — identify outliers (IQR rule)
    lower, upper = iqr_bounds(values, k=k)
    is_outlier = (values < lower) | (values > upper)

    # 2 — colour map: red for outliers, else optional gradient or grey
    if cmap_norm:
        base_colours = np.asarray(cmap_norm(values))
    else:
        base_colours = np.repeat([to_rgba("steelblue")], len(values), axis=0)
    base_colours[is_outlier.values] = to_rgba("red")

    # 3 — plot
    y_pos = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(values) + 1))
    ax.barh(y_pos, values.values, color=base_colours)

    # axis / labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()                      # smallest value at bottom
    ax.set_xlabel(df.columns[0])

    # annotate values
    pad = 0.01 * (values.max() - values.min())
    for y, v in zip(y_pos, values):
        ax.text(v + pad, y,
                f"{v:.{decimals}f}",
                va="center", ha="left",
                fontsize=9)

    # tidy look
    ax.spines[["right", "top"]].set_visible(False)

    if title is not None:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()
    plt.close(fig)
    return fig


def get_mode_scores(
    dfs: List[pd.DataFrame],
    *,
    random_state: Optional[int] = None,      # set a seed for reproducible ties
) -> pd.DataFrame:
    """
    Element-wise *mode* across several equally-shaped DataFrames.

    • If at least one value in the cell is ≥ 0, return the **mode**
      of those non-negatives.  When several values tie for the mode,
      choose *one at random* (controlled by `random_state`).

    • If **all** values in the cell are < 0, return the minimum
      of those negatives (unchanged from the average-helper behaviour).
    """
    if not dfs:
        raise ValueError("dfs list is empty")

    # Stack the values into a 3-D array for fast slicing
    arr = np.stack([df.values for df in dfs], axis=0)
    n_rows, n_cols = arr.shape[1:]

    rng = np.random.default_rng(random_state)
    out = np.empty((n_rows, n_cols), dtype=arr.dtype)

    for i in range(n_rows):
        for j in range(n_cols):
            cell_vals = arr[:, i, j]

            nonneg = cell_vals[cell_vals >= 0]
            if nonneg.size:                          # at least one ≥ 0
                vals, counts = np.unique(nonneg, return_counts=True)
                max_count = counts.max()
                tied = vals[counts == max_count]     # all equally common
                out[i, j] = rng.choice(tied)         # pick one at random
            else:
                out[i, j] = cell_vals.min()          # all-negative fallback

    first = dfs[0]
    return pd.DataFrame(out, index=first.index, columns=first.columns)


def different_nonneg(
    df1: pd.DataFrame,
    df2: pd.DataFrame

) -> Union[int, np.integer]:
    """
    Count the cells where ``df1`` and ``df2`` differ, *restricted* to the
    positions where both values are ≥ 0.

    Returns
    -------
    int  (or numpy integer)
        Number of cells that satisfy:
            (df1 >= 0) & (df2 >= 0) & (df1 != df2)
    """
    if not (df1.index.equals(df2.index) and df1.columns.equals(df2.columns)):
        raise ValueError("df1 and df2 must have identical index and columns")

    valid   = (df1 >= 0) & (df2 >= 0)      # keep only non-negative pairs
    diff    = df1.ne(df2)                  # element-wise inequality
    mismask = diff & valid                 # True where *both* conditions hold

    return int(mismask.values.sum())       # faster than mismask.sum().sum()
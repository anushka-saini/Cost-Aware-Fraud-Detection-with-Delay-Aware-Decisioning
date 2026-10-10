"""Metrics, bootstrap intervals, the cost function and threshold sweeps."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud.config import get_settings

TIERS = ("ALLOW", "REVIEW", "BLOCK")


# ---------------------------------------------------------------------------
# PR-AUC and bootstrap
# ---------------------------------------------------------------------------

def pr_auc(y_true, scores) -> float:
    return float(average_precision_score(np.asarray(y_true), np.asarray(scores)))


def _bootstrap_indices(n, n_iterations, seed):
    rng = np.random.RandomState(seed)
    for _ in range(n_iterations):
        yield rng.randint(0, n, n)


class _FastAP:
    """Average precision on bootstrap resamples without re-sorting.

    Scores are sorted once. A resample is then just a count per row, and the
    weighted average precision is a pair of cumulative sums. Tied scores are
    grouped exactly as scikit-learn does (one threshold per distinct score).
    """

    def __init__(self, y_true, scores):
        y_true, scores = np.asarray(y_true, dtype=np.float64), np.asarray(scores, dtype=np.float64)
        self.order = np.argsort(-scores, kind="stable")
        self.y = y_true[self.order]
        sorted_scores = scores[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]   # last row of each tie group

    def __call__(self, counts) -> float:
        """``counts[i]`` = how many times original row i appears in the resample."""
        w = np.asarray(counts, dtype=np.float64)[self.order]
        tp = np.cumsum(w * self.y)[self.ends]
        total = np.cumsum(w)[self.ends]
        if tp[-1] == 0:
            return float("nan")
        keep = total > 0
        tp, total = tp[keep], total[keep]
        precision = tp / total
        return float(np.sum(np.diff(np.r_[0.0, tp]) * precision) / tp[-1])


def _resample_counts(n, n_iterations, seed):
    for idx in _bootstrap_indices(n, n_iterations, seed):
        yield np.bincount(idx, minlength=n)


def _bootstrap_args(n_iterations, seed):
    settings = get_settings()
    return n_iterations or settings.n_bootstrap, settings.seed if seed is None else seed


def bootstrap_pr_auc(y_true, scores, n_iterations=None, seed=None) -> np.ndarray:
    """PR-AUC on resamples (with replacement) of the evaluation set.

    Same resampling scheme as the original notebooks (``RandomState(seed)``
    drawing row indices), so intervals are comparable with the old ones.
    """
    n_iterations, seed = _bootstrap_args(n_iterations, seed)
    ap = _FastAP(y_true, scores)
    out = np.array([ap(c) for c in _resample_counts(len(ap.y), n_iterations, seed)])
    return out[~np.isnan(out)]


def pr_auc_with_ci(y_true, scores, n_iterations=None, seed=None) -> dict:
    boot = bootstrap_pr_auc(y_true, scores, n_iterations, seed)
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"pr_auc": pr_auc(y_true, scores), "ci_low": float(low), "ci_high": float(high),
            "n_bootstrap": int(len(boot))}


def paired_bootstrap_difference(y_true, scores_a, scores_b, n_iterations=None, seed=None) -> dict:
    """PR-AUC(a) - PR-AUC(b) on the same resamples.

    ``significant`` is True only when the 95% interval of the difference
    excludes zero. Use this, not a comparison of point estimates, before
    calling one model better than another.
    """
    n_iterations, seed = _bootstrap_args(n_iterations, seed)
    ap_a, ap_b = _FastAP(y_true, scores_a), _FastAP(y_true, scores_b)
    diffs = np.array([ap_a(c) - ap_b(c) for c in _resample_counts(len(ap_a.y), n_iterations, seed)])
    diffs = diffs[~np.isnan(diffs)]
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "difference": pr_auc(y_true, scores_a) - pr_auc(y_true, scores_b),
        "ci_low": float(low), "ci_high": float(high),
        "p_two_sided": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))),
        "significant": bool(low > 0 or high < 0),
    }


def precision_at_k(y_true, scores, k: float) -> dict:
    """Precision among the top ``k`` fraction of scores, with its ceiling."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    n_top = max(1, int(round(len(scores) * k)))
    top = np.argpartition(-scores, n_top - 1)[:n_top]
    return {"k": k, "precision": float(y_true[top].mean()),
            "max_possible": float(min(1.0, y_true.sum() / n_top))}


# ---------------------------------------------------------------------------
# Tiers and cost
# ---------------------------------------------------------------------------

def assign_tier(prob: float, t_review: float, t_block: float) -> str:
    if prob >= t_block:
        return "BLOCK"
    if prob >= t_review:
        return "REVIEW"
    return "ALLOW"


def cost_report(y_true, probs, amounts, t_review, t_block, cost_false_block=None, cost_review=None) -> dict:
    """Outcome of applying the two thresholds.

    Cost: a fraud that lands in ALLOW costs its amount; a legitimate
    transaction that lands in BLOCK costs ``cost_false_block``; every
    transaction sent to REVIEW costs ``cost_review``.
    """
    settings = get_settings()
    cost_false_block = settings.cost_false_block if cost_false_block is None else cost_false_block
    cost_review = settings.cost_review if cost_review is None else cost_review
    y = np.asarray(y_true).astype(bool)
    probs, amounts = np.asarray(probs), np.asarray(amounts, dtype=np.float64)

    block = probs >= t_block
    review = (probs >= t_review) & ~block
    allow = ~block & ~review
    n_fraud, n_legit = int(y.sum()), int((~y).sum())
    tp_block, fp_block = int((block & y).sum()), int((block & ~y).sum())
    amount_missed = float(amounts[allow & y].sum())
    precision = tp_block / (tp_block + fp_block) if tp_block + fp_block else 0.0
    recall = tp_block / n_fraud if n_fraud else 0.0
    return {
        "t_review": float(t_review), "t_block": float(t_block),
        "total_cost": amount_missed + cost_false_block * fp_block + cost_review * int(review.sum()),
        "fraud_allowed": int((allow & y).sum()),
        "fraud_allowed_amount": amount_missed,
        "fraud_reviewed": int((review & y).sum()),
        "fraud_blocked": tp_block,
        "legit_reviewed": int((review & ~y).sum()),
        "legit_reviewed_pct": float((review & ~y).sum() / n_legit) if n_legit else 0.0,
        "legit_blocked": fp_block,
        "block_precision": float(precision),
        "block_recall": float(recall),
        "block_f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "n_fraud": n_fraud, "n_legit": n_legit,
    }


class CostCurve:
    """Exact cost for any pair of thresholds, without rescanning the data.

    The cost splits into a part that depends only on the review threshold
    (missed-fraud amount + review volume) and a part that depends only on the
    block threshold (false blocks - reviews avoided), so both are precomputed
    over the sorted scores.
    """

    def __init__(self, y_true, probs, amounts, cost_false_block=None, cost_review=None):
        settings = get_settings()
        self.cost_false_block = settings.cost_false_block if cost_false_block is None else cost_false_block
        self.cost_review = settings.cost_review if cost_review is None else cost_review
        y = np.asarray(y_true).astype(bool)
        probs, amounts = np.asarray(probs, dtype=np.float64), np.asarray(amounts, dtype=np.float64)
        order = np.argsort(probs, kind="stable")
        self.sorted_probs = probs[order]
        self.n = len(probs)
        # prefix sums over ascending score: index i = "everything below position i"
        self._fraud_amount_below = np.r_[0.0, np.cumsum(np.where(y[order], amounts[order], 0.0))]
        self._fraud_below = np.r_[0, np.cumsum(y[order])]
        self._legit_below = np.r_[0, np.cumsum(~y[order])]
        self.n_fraud, self.n_legit = int(y.sum()), int((~y).sum())

    def _pos(self, t):
        return np.searchsorted(self.sorted_probs, t, side="left")

    def cost(self, t_review, t_block):
        """Vectorised over arrays of thresholds (requires t_review <= t_block)."""
        r, b = self._pos(np.asarray(t_review)), self._pos(np.asarray(t_block))
        legit_blocked = self.n_legit - self._legit_below[b]
        return (self._fraud_amount_below[r] + self.cost_false_block * legit_blocked
                + self.cost_review * (b - r))

    def fraud_allowed(self, t_review):
        return self._fraud_below[self._pos(np.asarray(t_review))]

    def legit_reviewed(self, t_review, t_block):
        r, b = self._pos(np.asarray(t_review)), self._pos(np.asarray(t_block))
        return self._legit_below[b] - self._legit_below[r]


def candidate_thresholds(probs, max_candidates=4000) -> np.ndarray:
    """Thresholds worth trying: the distinct score values (thinned by quantile
    when there are too many), so no cost change between scores is skipped."""
    unique = np.unique(np.asarray(probs, dtype=np.float64))
    if len(unique) > max_candidates:
        unique = np.unique(np.quantile(unique, np.linspace(0, 1, max_candidates), method="nearest"))
    return unique


def shift_odds(threshold, factor):
    """Multiply a probability threshold's odds by ``factor`` (0 and 1 stay put)."""
    t = np.asarray(threshold, dtype=np.float64)
    inside = (t > 0) & (t < 1)
    safe = np.where(inside, t, 0.5)
    odds = safe / (1 - safe) * factor
    return np.where(inside, odds / (1 + odds), t)


def sweep_thresholds(y_true, probs, amounts, odds_margin=1.0, block_grid=None, review_grid=None,
                     cost_false_block=None, cost_review=None) -> dict:
    """Thresholds that minimise cost on the given data.

    With ``odds_margin = k > 1`` the choice is the pair whose *worst* cost,
    when each threshold's odds are scaled anywhere between 1/k and k, is
    lowest: a point on a plateau is preferred over a slightly cheaper one next
    to a cliff. The margin is relative because calibrated thresholds at a
    0.07% base rate are tiny; an absolute +/-0.02 would swamp them.
    ``odds_margin=1`` is the plain minimum.
    """
    curve = CostCurve(y_true, probs, amounts, cost_false_block, cost_review)
    review_grid = candidate_thresholds(probs) if review_grid is None else np.asarray(review_grid)
    block_grid = candidate_thresholds(probs) if block_grid is None else np.asarray(block_grid)

    def part(grid, fn):
        if odds_margin <= 1:
            return fn(grid)
        factors = np.exp(np.linspace(-np.log(odds_margin), np.log(odds_margin), 9))
        return np.max([fn(shift_odds(grid, f)) for f in factors], axis=0)

    # cost = [missed amount below r - review_cost * r] + [false blocks above b + review_cost * b]
    review_part = part(review_grid, lambda t: curve._fraud_amount_below[curve._pos(t)]
                       - curve.cost_review * curve._pos(t))
    block_part = part(block_grid, lambda t: curve.cost_review * curve._pos(t)
                      + curve.cost_false_block * (curve.n_legit - curve._legit_below[curve._pos(t)]))

    # Best review threshold at or below each block threshold (running minimum).
    order_r = np.argsort(review_grid)
    review_sorted, review_cost_sorted = review_grid[order_r], review_part[order_r]
    running_best = np.minimum.accumulate(review_cost_sorted)
    running_arg = np.zeros(len(review_sorted), dtype=int)
    best_so_far = 0
    for i in range(len(review_sorted)):
        if review_cost_sorted[i] < review_cost_sorted[best_so_far]:
            best_so_far = i
        running_arg[i] = best_so_far

    idx = np.searchsorted(review_sorted, block_grid, side="right") - 1
    valid = idx >= 0
    total = np.full(len(block_grid), np.inf)
    total[valid] = block_part[valid] + running_best[idx[valid]]
    best_block = int(np.argmin(total))
    t_block = float(block_grid[best_block])
    t_review = float(review_sorted[running_arg[idx[best_block]]])
    return {
        "t_review": t_review, "t_block": t_block, "odds_margin": float(odds_margin),
        "objective": float(total[best_block]),
        "cost_at_choice": float(curve.cost(t_review, t_block)),
    }


def threshold_robustness(y_true, probs, amounts, t_review, t_block, deltas=(0.02, 0.05), odds_factors=(0.5, 2.0),
                         cost_false_block=None, cost_review=None) -> list[dict]:
    """Cost when one threshold is moved and the other held fixed.

    Two kinds of move: an absolute shift of +/- delta in probability, and a
    relative one that scales the threshold's odds by a factor.
    """
    base = cost_report(y_true, probs, amounts, t_review, t_block, cost_false_block, cost_review)
    moves = [("absolute", d, lambda t, d=d: t + d) for d in sorted({-d for d in deltas} | set(deltas))]
    moves += [("odds_factor", f, lambda t, f=f: float(shift_odds(t, f))) for f in odds_factors]
    rows = []
    for which in ("t_review", "t_block"):
        for kind, amount, move in moves:
            r = move(t_review) if which == "t_review" else t_review
            b = move(t_block) if which == "t_block" else t_block
            r, b = float(np.clip(r, 0.0, 1.0)), float(np.clip(b, 0.0, 1.0 + 1e-9))
            if r > b:
                continue
            report = cost_report(y_true, probs, amounts, r, b, cost_false_block, cost_review)
            rows.append({
                "moved": which, "kind": kind, "amount": float(amount), "t_review": r, "t_block": b,
                "total_cost": report["total_cost"], "fraud_allowed": report["fraud_allowed"],
                "legit_reviewed_pct": report["legit_reviewed_pct"], "legit_blocked": report["legit_blocked"],
                "cost_change": report["total_cost"] - base["total_cost"],
            })
    return rows


# ---------------------------------------------------------------------------
# Other diagnostics
# ---------------------------------------------------------------------------

def recall_by_amount_quartile(y_true, probs, amounts, t_review, t_block, bins) -> list[dict]:
    y = np.asarray(y_true).astype(bool)
    view = pd.DataFrame({"amount": np.asarray(amounts)[y], "prob": np.asarray(probs)[y]})
    view["bucket"] = pd.cut(view["amount"], bins=bins, labels=[f"Q{i + 1}" for i in range(len(bins) - 1)])
    rows = []
    for bucket, group in view.groupby("bucket", observed=True):
        rows.append({"quartile": str(bucket), "n_fraud": int(len(group)),
                     "recall_at_review": float((group["prob"] >= t_review).mean()),
                     "recall_at_block": float((group["prob"] >= t_block).mean())})
    return rows


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index, with bin edges from the reference quantiles."""
    reference, current = np.asarray(reference, dtype=np.float64), np.asarray(current, dtype=np.float64)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature: compare the share of each distinct value instead
        edges = np.unique(np.r_[reference, current])
        if len(edges) < 2:
            return 0.0
        edges = np.r_[edges, edges[-1] + 1]
    edges = edges.astype(np.float64)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_share = np.histogram(current, bins=edges)[0] / len(current)
    ref_share, cur_share = np.clip(ref_share, 1e-6, None), np.clip(cur_share, 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))

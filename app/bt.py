"""Bradley–Terry 模型（含编辑左栏偏置）的核心数值计算。

模型设定：
    logit(P(左栏胜)) = score(左模型) - score(右模型) + bias(编辑)
其中参照模型的分数固定为 0（其列不出现在设计矩阵中），
每个编辑有一个偏置列（该行恒为 +1）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.linalg import null_space as _null_space
from scipy.optimize import minimize
from scipy.special import expit

# (left_model_id, right_model_id, editor_id)
Row = tuple[int, int, int]


@dataclass(frozen=True)
class ParamSpace:
    """参数向量的列布局：先模型分（参照模型除外），后编辑偏置。"""

    reference_model_id: int | None
    score_col: dict[int, int]  # model_id -> 列号（不含参照模型）
    bias_col: dict[int, int]  # editor_id -> 列号
    n_params: int


def build_param_space(
    model_ids: Iterable[int],
    editor_ids: Iterable[int],
    reference_model_id: int | None,
) -> ParamSpace:
    score_col: dict[int, int] = {}
    for m in sorted(model_ids):
        if m == reference_model_id:
            continue
        score_col[m] = len(score_col)
    offset = len(score_col)
    bias_col = {e: offset + j for j, e in enumerate(sorted(editor_ids))}
    return ParamSpace(reference_model_id, score_col, bias_col, offset + len(bias_col))


def row_vector(left_model_id: int, right_model_id: int, editor_id: int, ps: ParamSpace) -> np.ndarray:
    """单场对比在设计矩阵中的行向量。"""
    x = np.zeros(ps.n_params)
    col = ps.score_col.get(left_model_id)
    if col is not None:
        x[col] += 1.0
    col = ps.score_col.get(right_model_id)
    if col is not None:
        x[col] -= 1.0
    col = ps.bias_col.get(editor_id)
    if col is not None:
        x[col] += 1.0
    return x


def design_matrix(rows: Iterable[Row], ps: ParamSpace) -> np.ndarray:
    rows = list(rows)
    X = np.zeros((len(rows), ps.n_params))
    for i, (left, right, editor) in enumerate(rows):
        X[i] = row_vector(left, right, editor, ps)
    return X


def matrix_rank(X: np.ndarray) -> int:
    if X.shape[0] == 0 or X.shape[1] == 0:
        return 0
    return int(np.linalg.matrix_rank(X))


def null_space(X: np.ndarray) -> np.ndarray:
    """X 的右零空间（形状 p × (p - rank)），即尚未被数据确定的方向。"""
    p = X.shape[1]
    if p == 0:
        return np.zeros((0, 0))
    if X.shape[0] == 0:
        return np.eye(p)
    return _null_space(X)


@dataclass
class FitResult:
    theta: np.ndarray
    neg_log_likelihood: float
    success: bool
    message: str


def fit_bradley_terry(X: np.ndarray, y: np.ndarray, l2: float = 0.1) -> FitResult:
    """L2 正则的 Bradley–Terry 最大似然拟合。

    最小化  sum(softplus(eta) - y * eta) + l2/2 * ||theta||^2 ，
    其中 eta = X @ theta，y=1 表示左栏胜。
    """
    y = np.asarray(y, dtype=float)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = X @ theta
        nll = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
        nll += 0.5 * l2 * float(theta @ theta)
        grad = X.T @ (expit(eta) - y) + l2 * theta
        return nll, grad

    res = minimize(objective, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
    return FitResult(res.x, float(res.fun), bool(res.success), str(res.message))

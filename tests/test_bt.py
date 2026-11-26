"""bt 模块单元测试：设计矩阵、秩、L2 正则拟合。"""
import numpy as np
import pytest
from scipy.special import expit

from app.bt import (
    build_param_space,
    design_matrix,
    fit_bradley_terry,
    matrix_rank,
    null_space,
    row_vector,
)

# 模型 1(参照), 2, 3；编辑 10, 20
PS = build_param_space([1, 2, 3], [10, 20], reference_model_id=1)


def test_param_space_layout():
    assert PS.score_col == {2: 0, 3: 1}
    assert PS.bias_col == {10: 2, 20: 3}
    assert PS.n_params == 4


def test_row_vector_with_reference_on_left():
    # 左=参照模型(1)：其列不存在，只给右模型 -1
    x = row_vector(1, 2, 10, PS)
    assert x.tolist() == [-1.0, 0.0, 1.0, 0.0]


def test_row_vector_with_reference_on_right():
    x = row_vector(3, 1, 20, PS)
    assert x.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_row_vector_without_reference():
    x = row_vector(2, 3, 10, PS)
    assert x.tolist() == [1.0, -1.0, 1.0, 0.0]


def test_design_matrix_and_rank():
    rows = [(1, 2, 10), (2, 3, 10), (1, 3, 20), (3, 2, 20)]
    X = design_matrix(rows, PS)
    assert X.shape == (4, 4)
    assert matrix_rank(X) == 4  # 满列秩


def test_rank_deficient_and_null_space():
    rows = [(1, 2, 10), (2, 3, 10)]
    X = design_matrix(rows, PS)
    assert matrix_rank(X) == 2
    N = null_space(X)
    assert N.shape == (4, 2)  # 4 参数 - 秩 2 = 2 个未确定方向
    assert np.allclose(X @ N, 0.0)


def test_empty_design_matrix():
    X = design_matrix([], PS)
    assert X.shape == (0, 4)
    assert matrix_rank(X) == 0
    assert null_space(X).shape == (4, 4)


def test_fit_recovers_known_parameters():
    rng = np.random.default_rng(42)
    true = np.array([1.2, -0.8, 0.5, -0.3])  # s2, s3, b10, b20
    pairs = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]
    rows, y = [], []
    for editor in (10, 20):
        for left, right in pairs:
            for _ in range(100):
                rows.append((left, right, editor))
                eta = row_vector(left, right, editor, PS) @ true
                y.append(float(rng.random() < expit(eta)))
    X = design_matrix(rows, PS)
    fit = fit_bradley_terry(X, np.array(y), l2=1e-4)
    assert fit.success
    assert np.allclose(fit.theta, true, atol=0.2)


def test_l2_shrinks_scores():
    # 可分数据（左全胜）：无正则时 MLE 发散，L2 使解有限且随强度收缩
    rows = [(2, 3, 10)] * 5 + [(1, 3, 20)] * 5
    X = design_matrix(rows, PS)
    y = np.ones(10)
    weak = fit_bradley_terry(X, y, l2=1e-3)
    strong = fit_bradley_terry(X, y, l2=100.0)
    assert np.isfinite(weak.theta).all()
    assert np.linalg.norm(strong.theta) < np.linalg.norm(weak.theta)

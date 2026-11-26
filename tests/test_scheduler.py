"""scheduler 模块单元测试：候选生成、最小补发集合、无解报告。"""
import numpy as np

from app.bt import build_param_space, design_matrix, matrix_rank
from app.scheduler import MAX_CANDIDATES, build_plan

# 模型 1(参照), 2, 3；编辑 10, 20 → 4 个参数
PS = build_param_space([1, 2, 3], [10, 20], reference_model_id=1)
EDITORS = [10, 20]
MODELS = [1, 2, 3]


def test_plan_fills_deficiency_with_minimal_set():
    received = [(1, 2, 10), (2, 3, 10)]  # 秩 2，缺 2
    free = [f"MS-{i}" for i in range(5)]
    plan = build_plan(received, [], free, EDITORS, MODELS, PS)
    assert plan.feasible
    assert plan.deficiency == 2
    assert len(plan.selected) == 2  # 最小集合 = 缺口数
    # 选出的场次确实能补满
    rows = received + [(c.left_model_id, c.right_model_id, c.editor_id) for c in plan.selected]
    assert matrix_rank(design_matrix(rows, PS)) == PS.n_params
    # 一篇稿子至多一个候选
    assert len({c.manuscript_id for c in plan.selected}) == len(plan.selected)


def test_pending_rows_count_toward_rank():
    received = [(1, 2, 10), (2, 3, 10)]
    pending = [(1, 3, 20)]  # 待交也计入排程 → 只缺 1
    free = [f"MS-{i}" for i in range(5)]
    plan = build_plan(received, pending, free, EDITORS, MODELS, PS)
    assert plan.feasible
    assert plan.deficiency == 1
    assert len(plan.selected) == 1


def test_full_rank_needs_no_plan():
    received = [(1, 2, 10), (2, 3, 10), (1, 3, 20), (3, 2, 20)]
    plan = build_plan(received, [], ["MS-1"], EDITORS, MODELS, PS)
    assert plan.feasible
    assert plan.selected == []


def test_infeasible_reports_max_achievable_rank():
    received = [(1, 2, 10)]  # 秩 1，缺 3
    free = ["MS-1", "MS-2"]  # 只有 2 篇空闲稿子 → 最多补 2
    plan = build_plan(received, [], free, EDITORS, MODELS, PS)
    assert not plan.feasible
    assert plan.selected == []
    assert plan.max_achievable_rank == 3  # 1 + 2


def test_no_free_manuscripts():
    received = [(1, 2, 10)]
    plan = build_plan(received, [], [], EDITORS, MODELS, PS)
    assert not plan.feasible
    assert plan.max_achievable_rank == 1


def test_candidate_pool_capped_at_16():
    # 20 个模型 → 19 个分数参数 + 1 个编辑偏置 = 20 个参数，缺 19 > 16
    models = list(range(1, 21))
    ps = build_param_space(models, [10], reference_model_id=1)
    received = [(1, 2, 10)]
    free = [f"MS-{i:02d}" for i in range(30)]
    plan = build_plan(received, [], free, [10], models, ps)
    assert plan.deficiency == 19
    assert plan.candidates_considered == MAX_CANDIDATES
    assert not plan.feasible
    assert plan.max_achievable_rank == 1 + MAX_CANDIDATES


def test_candidates_use_available_directions():
    # 编辑 20 从未出现 → 其偏置列全零，候选必须包含编辑 20 的场次
    received = [(1, 2, 10), (2, 3, 10), (1, 3, 10)]
    plan = build_plan(received, [], ["MS-1"], EDITORS, MODELS, PS)
    assert plan.feasible
    assert any(c.editor_id == 20 for c in plan.selected)

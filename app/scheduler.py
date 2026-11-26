"""补评排程：实收+待交答卷不满秩时，挑选最少的补评场次。

规则（与需求对应）：
- 待交（pending）场次计入排程基矩阵——假定它们会收齐；
- 每篇原稿最多补一场：候选场次一篇稿子至多一个；
- 候选场次至多 16 个，从中贪心选出使秩补满的最小集合；
- 无解（补不满）时报告最多能达到的秩。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .bt import ParamSpace, Row, design_matrix, matrix_rank, null_space, row_vector

MAX_CANDIDATES = 16
_PROJ_TOL = 1e-9


@dataclass
class Candidate:
    manuscript_id: str
    editor_id: int
    left_model_id: int
    right_model_id: int
    projection: float  # 行向量在当前零空间上的投影长度（越大越能补秩）
    row: np.ndarray = field(repr=False)


@dataclass
class Plan:
    feasible: bool
    rank: int  # 实收+待交的当前秩
    n_params: int
    deficiency: int
    candidates_considered: int  # 进入候选池的场次数（<= MAX_CANDIDATES）
    selected: list[Candidate]
    max_achievable_rank: int  # 补发候选场次后最多能达到的秩


def _best_row_for_manuscript(
    manuscript_id: str,
    editor_ids: Iterable[int],
    model_ids: Iterable[int],
    null_basis: np.ndarray,
    ps: ParamSpace,
) -> Candidate | None:
    """为一篇稿子搜索 (编辑, 左模型, 右模型)，使行向量在零空间上的投影最大。"""
    best: Candidate | None = None
    nt = null_basis.T  # (d, p)
    for editor_id in editor_ids:
        for left_id in model_ids:
            for right_id in model_ids:
                if left_id == right_id:
                    continue
                row = row_vector(left_id, right_id, editor_id, ps)
                projection = float(np.linalg.norm(nt @ row))
                if best is None or projection > best.projection + 1e-12:
                    best = Candidate(manuscript_id, editor_id, left_id, right_id, projection, row)
    return best


def build_plan(
    received_rows: Iterable[Row],
    pending_rows: Iterable[Row],
    free_manuscript_ids: Iterable[str],
    editor_ids: Iterable[int],
    model_ids: Iterable[int],
    ps: ParamSpace,
    max_candidates: int = MAX_CANDIDATES,
) -> Plan:
    """构造补发计划。

    基矩阵 = 实收答卷 + 待交场次；目标是补发最少的场次使收齐后满列秩。
    """
    editor_ids = sorted(editor_ids)
    model_ids = sorted(model_ids)
    base = design_matrix(list(received_rows) + list(pending_rows), ps)
    rank0 = matrix_rank(base)
    p = ps.n_params
    if p == 0 or rank0 >= p:
        return Plan(True, rank0, p, 0, 0, [], rank0)

    # 顺序构造候选池（至多 max_candidates 个，每篇稿子至多一个）：
    # 每个候选都针对"基矩阵 + 已选候选"的零空间挑选，保证互不冗余。
    current = base
    current_rank = rank0
    pool: list[Candidate] = []
    for manuscript_id in sorted(free_manuscript_ids):
        if current_rank == p or len(pool) == max_candidates:
            break
        basis = null_space(current)
        cand = _best_row_for_manuscript(manuscript_id, editor_ids, model_ids, basis, ps)
        if cand is None or cand.projection <= _PROJ_TOL:
            continue
        pool.append(cand)
        current = np.vstack([current, cand.row])
        current_rank = matrix_rank(current)

    # 从候选池贪心选出最小补发集合：每选一行秩至多 +1，故 deficiency 行即最优
    cur = base
    cur_rank = rank0
    selected: list[Candidate] = []
    for cand in pool:
        if cur_rank == p:
            break
        augmented = np.vstack([cur, cand.row])
        new_rank = matrix_rank(augmented)
        if new_rank > cur_rank:
            selected.append(cand)
            cur = augmented
            cur_rank = new_rank

    if cur_rank == p:
        return Plan(True, rank0, p, p - rank0, len(pool), selected, p)

    # 无解：报告候选全部收齐后最多能达到的秩
    return Plan(False, rank0, p, p - rank0, len(pool), [], current_rank)

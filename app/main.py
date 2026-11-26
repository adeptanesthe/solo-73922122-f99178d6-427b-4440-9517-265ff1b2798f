"""译文两两盲评平台 API。

Bradley–Terry 模型分离模型水平与编辑左栏偏爱：
    logit(P(左栏胜)) = score(左模型) - score(右模型) + bias(编辑)
第一个登记的模型为参照（分数固定为 0）。
实收答卷的设计矩阵满列秩时用 L2 正则拟合并给出排名；
否则从至多 16 个候选场次中选出最少的一组补发。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Union

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import models, schemas
from .bt import build_param_space, design_matrix, fit_bradley_terry, matrix_rank
from .db import Base, engine, get_db
from .scheduler import MAX_CANDIDATES, build_plan


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(engine)
    yield


app = FastAPI(
    title="译文两两盲评平台",
    description=__doc__,
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------- 工具函数

def _require_model(db: Session, model_id: int) -> None:
    if db.get(models.Model, model_id) is None:
        raise HTTPException(404, f"模型 {model_id} 不存在")


def _require_editor(db: Session, editor_id: int) -> None:
    if db.get(models.Editor, editor_id) is None:
        raise HTTPException(404, f"编辑 {editor_id} 不存在")


def _get_match(db: Session, match_id: int) -> models.Match:
    match = db.get(models.Match, match_id)
    if match is None:
        raise HTTPException(404, f"场次 {match_id} 不存在")
    return match


def _param_space(db: Session):
    model_ids = sorted(db.scalars(select(models.Model.id)).all())
    editor_ids = sorted(db.scalars(select(models.Editor.id)).all())
    reference = model_ids[0] if model_ids else None  # 固定第一个登记的模型为 0 分
    return build_param_space(model_ids, editor_ids, reference), model_ids, editor_ids, reference


def _matches_with_status(db: Session, statuses: tuple[str, ...]) -> list[models.Match]:
    stmt = select(models.Match).where(models.Match.status.in_(statuses)).order_by(models.Match.id)
    return list(db.scalars(stmt).all())


def _rows_of(matches: list[models.Match]) -> list[tuple[int, int, int]]:
    return [(m.left_model_id, m.right_model_id, m.editor_id) for m in matches]


def _free_manuscripts(db: Session) -> list[str]:
    """空闲原稿 = 无待交场次 且 未被补发过（每篇原稿最多补一场）。

    已撤回的补发不占名额（撤回后可重派）。
    """
    all_ms = set(db.scalars(select(models.Match.manuscript_id)).all())
    occupied = set(
        db.scalars(select(models.Match.manuscript_id).where(models.Match.status == models.PENDING)).all()
    )
    supplemented = set(
        db.scalars(
            select(models.Match.manuscript_id).where(
                models.Match.origin == models.ORIGIN_PLAN,
                models.Match.status.in_((models.PENDING, models.COMPLETED)),
            )
        ).all()
    )
    return sorted(all_ms - occupied - supplemented)


def _build_plan(db: Session):
    ps, model_ids, editor_ids, _ = _param_space(db)
    received = _matches_with_status(db, (models.COMPLETED,))
    pending = _matches_with_status(db, (models.PENDING,))
    plan = build_plan(
        _rows_of(received),
        _rows_of(pending),
        _free_manuscripts(db),
        editor_ids,
        model_ids,
        ps,
    )
    return ps, received, pending, plan


def _plan_out(plan) -> schemas.PlanOut:
    if plan.feasible:
        if plan.selected:
            note = f"需补发 {len(plan.selected)} 场；与待交场次一起收齐后即可满秩"
        else:
            note = "待交场次收齐后即可满秩，无需补发新场次"
    else:
        note = "候选场次不足以补足秩，暂无法给出排名"
    return schemas.PlanOut(
        feasible=plan.feasible,
        max_candidates=MAX_CANDIDATES,
        candidates_considered=plan.candidates_considered,
        n_selected=len(plan.selected),
        matches=[
            schemas.PlannedMatch(
                manuscript_id=c.manuscript_id,
                editor_id=c.editor_id,
                left_model_id=c.left_model_id,
                right_model_id=c.right_model_id,
            )
            for c in plan.selected
        ],
        max_achievable_rank=None if plan.feasible else plan.max_achievable_rank,
        note=note,
    )


# ---------------------------------------------------------------- 根路径

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/docs")


# ---------------------------------------------------------------- 登记

@app.post("/models", response_model=schemas.NamedOut, status_code=201, tags=["登记"])
def create_model(body: schemas.NamedIn, db: Session = Depends(get_db)):
    """登记参评模型。第一个登记的模型是参照模型，分数固定为 0。"""
    if db.scalar(select(models.Model).where(models.Model.name == body.name)):
        raise HTTPException(409, f"模型名 '{body.name}' 已存在")
    obj = models.Model(name=body.name)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@app.get("/models", response_model=list[schemas.NamedOut], tags=["登记"])
def list_models(db: Session = Depends(get_db)):
    return list(db.scalars(select(models.Model).order_by(models.Model.id)).all())


@app.post("/editors", response_model=schemas.NamedOut, status_code=201, tags=["登记"])
def create_editor(body: schemas.NamedIn, db: Session = Depends(get_db)):
    if db.scalar(select(models.Editor).where(models.Editor.name == body.name)):
        raise HTTPException(409, f"编辑名 '{body.name}' 已存在")
    obj = models.Editor(name=body.name)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@app.get("/editors", response_model=list[schemas.NamedOut], tags=["登记"])
def list_editors(db: Session = Depends(get_db)):
    return list(db.scalars(select(models.Editor).order_by(models.Editor.id)).all())


# ---------------------------------------------------------------- 答卷与场次

@app.post("/responses", response_model=schemas.MatchOut, status_code=201, tags=["答卷"])
def register_response(body: schemas.ResponseIn, db: Session = Depends(get_db)):
    """登记一份已有答卷（直接计入实收，用于拟合）。"""
    _require_editor(db, body.editor_id)
    _require_model(db, body.left_model_id)
    _require_model(db, body.right_model_id)
    match = models.Match(
        manuscript_id=body.manuscript_id,
        editor_id=body.editor_id,
        left_model_id=body.left_model_id,
        right_model_id=body.right_model_id,
        status=models.COMPLETED,
        winner=body.winner,
        origin=models.ORIGIN_MANUAL,
    )
    db.add(match)
    db.commit()
    db.refresh(match)
    return match


@app.post("/matches", response_model=schemas.MatchOut, status_code=201, tags=["场次"])
def create_match(body: schemas.PairIn, db: Session = Depends(get_db)):
    """人工发出一场盲评（待交）。发出即占用原稿：同一原稿同时至多一场待交。"""
    _require_editor(db, body.editor_id)
    _require_model(db, body.left_model_id)
    _require_model(db, body.right_model_id)
    occupied = db.scalar(
        select(models.Match).where(
            models.Match.manuscript_id == body.manuscript_id,
            models.Match.status == models.PENDING,
        )
    )
    if occupied is not None:
        raise HTTPException(409, f"原稿 '{body.manuscript_id}' 已有待交场次 {occupied.id}，被占用")
    match = models.Match(
        manuscript_id=body.manuscript_id,
        editor_id=body.editor_id,
        left_model_id=body.left_model_id,
        right_model_id=body.right_model_id,
        status=models.PENDING,
        origin=models.ORIGIN_MANUAL,
    )
    db.add(match)
    db.commit()
    db.refresh(match)
    return match


@app.get("/matches", response_model=list[schemas.MatchOut], tags=["场次"])
def list_matches(status: str | None = Query(None), db: Session = Depends(get_db)):
    stmt = select(models.Match).order_by(models.Match.id)
    if status:
        stmt = stmt.where(models.Match.status == status)
    return list(db.scalars(stmt).all())


@app.post("/matches/{match_id}/submit", response_model=schemas.MatchOut, tags=["场次"])
def submit_match(match_id: int, body: schemas.SubmitIn, db: Session = Depends(get_db)):
    """交回答卷。已撤回的场次其答卷一律无效。"""
    match = _get_match(db, match_id)
    if match.status == models.WITHDRAWN:
        raise HTTPException(409, f"场次 {match_id} 已撤回，其答卷无效")
    if match.status != models.PENDING:
        raise HTTPException(409, f"场次 {match_id} 已交回，不能重复提交")
    match.status = models.COMPLETED
    match.winner = body.winner
    db.commit()
    db.refresh(match)
    return match


@app.post("/matches/{match_id}/withdraw", response_model=schemas.MatchOut, tags=["场次"])
def withdraw_match(match_id: int, db: Session = Depends(get_db)):
    """撤回待交场次：释放原稿，可重派；此后该场次的答卷无效。"""
    match = _get_match(db, match_id)
    if match.status != models.PENDING:
        raise HTTPException(409, f"只能撤回待交场次（当前状态：{match.status}）")
    match.status = models.WITHDRAWN
    match.winner = None
    db.commit()
    db.refresh(match)
    return match


# ---------------------------------------------------------------- 排名与排程

@app.get("/ranking", response_model=Union[schemas.RankingOk, schemas.RankingInsufficient], tags=["排名"])
def ranking(l2: float = Query(0.1, gt=0.0, description="L2 正则强度"), db: Session = Depends(get_db)):
    """实收答卷满列秩则拟合并给出排名；否则返回补发计划（或无解时的最高秩）。"""
    ps, model_ids, editor_ids, reference = _param_space(db)
    received = _matches_with_status(db, (models.COMPLETED,))
    pending = _matches_with_status(db, (models.PENDING,))
    X = design_matrix(_rows_of(received), ps)
    rank = matrix_rank(X)

    if ps.n_params > 0 and rank < ps.n_params:
        _, _, _, plan = _build_plan(db)
        return schemas.RankingInsufficient(
            rank=rank,
            n_params=ps.n_params,
            deficiency=ps.n_params - rank,
            n_responses=len(received),
            n_pending=len(pending),
            plan=_plan_out(plan),
        )

    # 满列秩：L2 正则拟合
    if ps.n_params == 0:
        theta = np.zeros(0)
        nll = None
        success = None
    else:
        y = np.array([1.0 if m.winner == "left" else 0.0 for m in received])
        fit = fit_bradley_terry(X, y, l2)
        theta, nll, success = fit.theta, fit.neg_log_likelihood, fit.success

    model_rows = list(db.scalars(select(models.Model).order_by(models.Model.id)).all())
    editor_rows = list(db.scalars(select(models.Editor).order_by(models.Editor.id)).all())

    scored = []
    for m in model_rows:
        score = 0.0 if m.id == reference else float(theta[ps.score_col[m.id]])
        scored.append((m, score))
    scored.sort(key=lambda t: (-t[1], t[0].id))

    biases = sorted(
        (schemas.EditorBias(editor_id=e.id, name=e.name, bias=float(theta[ps.bias_col[e.id]])) for e in editor_rows),
        key=lambda b: (-b.bias, b.editor_id),
    )

    return schemas.RankingOk(
        reference_model_id=reference,
        l2=l2,
        rank=rank,
        n_params=ps.n_params,
        n_responses=len(received),
        n_pending=len(pending),
        neg_log_likelihood=nll,
        optimizer_success=success,
        models=[
            schemas.ModelScore(model_id=m.id, name=m.name, score=score, rank=i + 1)
            for i, (m, score) in enumerate(scored)
        ],
        editors=biases,
    )


@app.post("/schedule/apply", response_model=schemas.ApplyOk, status_code=201, tags=["排名"])
def apply_plan(db: Session = Depends(get_db)):
    """按当前补发计划发出场次（origin=plan，状态 pending，发出即占用原稿）。"""
    ps, _, _, _ = _param_space(db)
    received = _matches_with_status(db, (models.COMPLETED,))
    X = design_matrix(_rows_of(received), ps)
    rank = matrix_rank(X)
    if ps.n_params == 0 or rank == ps.n_params:
        raise HTTPException(409, "实收答卷的设计矩阵已满列秩，无需补发场次")

    _, _, _, plan = _build_plan(db)
    if not plan.feasible:
        raise HTTPException(
            409,
            detail={
                "message": "候选场次不足以补足秩，无法补发",
                "rank": plan.rank,
                "n_params": plan.n_params,
                "max_achievable_rank": plan.max_achievable_rank,
            },
        )
    if not plan.selected:
        raise HTTPException(409, "待交场次收齐后即可满秩，无需补发新场次")

    created = []
    for c in plan.selected:
        match = models.Match(
            manuscript_id=c.manuscript_id,
            editor_id=c.editor_id,
            left_model_id=c.left_model_id,
            right_model_id=c.right_model_id,
            status=models.PENDING,
            origin=models.ORIGIN_PLAN,
        )
        db.add(match)
        created.append(match)
    db.commit()
    for match in created:
        db.refresh(match)
    return schemas.ApplyOk(
        rank_before=plan.rank,
        n_params=plan.n_params,
        deficiency=plan.deficiency,
        created=[schemas.MatchOut.model_validate(m) for m in created],
    )


# ---------------------------------------------------------------- 状态

@app.get("/manuscripts", response_model=list[schemas.ManuscriptOut], tags=["状态"])
def list_manuscripts(db: Session = Depends(get_db)):
    matches = list(db.scalars(select(models.Match)).all())
    by_ms: dict[str, list[models.Match]] = {}
    for m in matches:
        by_ms.setdefault(m.manuscript_id, []).append(m)
    out = []
    for manuscript_id in sorted(by_ms):
        group = by_ms[manuscript_id]
        if any(m.status == models.PENDING for m in group):
            status = "occupied"
        elif any(m.origin == models.ORIGIN_PLAN and m.status == models.COMPLETED for m in group):
            status = "supplemented"
        else:
            status = "free"
        out.append(schemas.ManuscriptOut(manuscript_id=manuscript_id, status=status, n_matches=len(group)))
    return out


@app.get("/status", response_model=schemas.StatusOut, tags=["状态"])
def status(db: Session = Depends(get_db)):
    ps, _, _, reference = _param_space(db)
    received = _matches_with_status(db, (models.COMPLETED,))
    pending = _matches_with_status(db, (models.PENDING,))
    withdrawn = _matches_with_status(db, (models.WITHDRAWN,))
    rank_received = matrix_rank(design_matrix(_rows_of(received), ps))
    rank_with_pending = matrix_rank(design_matrix(_rows_of(received) + _rows_of(pending), ps))
    return schemas.StatusOut(
        n_models=len(ps.score_col) + (1 if ps.reference_model_id is not None else 0),
        n_editors=len(ps.bias_col),
        reference_model_id=reference,
        matches={
            models.PENDING: len(pending),
            models.COMPLETED: len(received),
            models.WITHDRAWN: len(withdrawn),
        },
        n_params=ps.n_params,
        rank_received=rank_received,
        rank_with_pending=rank_with_pending,
        full_rank=ps.n_params > 0 and rank_received == ps.n_params,
    )


@app.post("/admin/reset", tags=["管理"])
def reset(db: Session = Depends(get_db)):
    """清空全部数据（演示/调试用）。"""
    db.query(models.Match).delete()
    db.query(models.Editor).delete()
    db.query(models.Model).delete()
    db.commit()
    return {"status": "reset"}

"""Pydantic 请求/响应模型。"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Winner = Literal["left", "right"]


# ---------- 登记 ----------

class NamedIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class NamedOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str


# ---------- 场次 / 答卷 ----------

class PairIn(BaseModel):
    manuscript_id: str = Field(min_length=1, max_length=100)
    editor_id: int
    left_model_id: int
    right_model_id: int

    @model_validator(mode="after")
    def _distinct_models(self):
        if self.left_model_id == self.right_model_id:
            raise ValueError("left_model_id 与 right_model_id 必须是不同模型")
        return self


class ResponseIn(PairIn):
    """登记一份已有答卷（直接成为实收场次）。"""

    winner: Winner


class SubmitIn(BaseModel):
    winner: Winner


class MatchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    manuscript_id: str
    editor_id: int
    left_model_id: int
    right_model_id: int
    status: str
    winner: str | None
    origin: str
    created_at: datetime | None


# ---------- 排名 ----------

class ModelScore(BaseModel):
    model_id: int
    name: str
    score: float
    rank: int


class EditorBias(BaseModel):
    editor_id: int
    name: str
    bias: float


class RankingOk(BaseModel):
    status: Literal["ok"] = "ok"
    reference_model_id: int | None
    l2: float
    rank: int
    n_params: int
    n_responses: int
    n_pending: int
    neg_log_likelihood: float | None
    optimizer_success: bool | None
    models: list[ModelScore]
    editors: list[EditorBias]


class PlannedMatch(BaseModel):
    manuscript_id: str
    editor_id: int
    left_model_id: int
    right_model_id: int


class PlanOut(BaseModel):
    feasible: bool
    max_candidates: int
    candidates_considered: int
    n_selected: int
    matches: list[PlannedMatch]
    max_achievable_rank: int | None = None
    note: str


class RankingInsufficient(BaseModel):
    status: Literal["insufficient_rank"] = "insufficient_rank"
    rank: int
    n_params: int
    deficiency: int
    n_responses: int
    n_pending: int
    plan: PlanOut


class ApplyOk(BaseModel):
    status: Literal["sent"] = "sent"
    rank_before: int
    n_params: int
    deficiency: int
    created: list[MatchOut]


# ---------- 状态 ----------

class ManuscriptOut(BaseModel):
    manuscript_id: str
    status: str  # free | occupied | supplemented
    n_matches: int


class StatusOut(BaseModel):
    n_models: int
    n_editors: int
    reference_model_id: int | None
    matches: dict[str, int]
    n_params: int
    rank_received: int
    rank_with_pending: int
    full_rank: bool

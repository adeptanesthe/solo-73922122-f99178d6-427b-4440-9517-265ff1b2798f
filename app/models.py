"""ORM 模型。"""
from __future__ import annotations

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, func

from .db import Base

# 场次状态
PENDING = "pending"  # 已发出、待交回
COMPLETED = "completed"  # 答卷已收（实收）
WITHDRAWN = "withdrawn"  # 已撤回（其答卷一律无效）

# 场次来源
ORIGIN_MANUAL = "manual"  # 人工登记/发出
ORIGIN_PLAN = "plan"  # 排程器补发


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)


class Editor(Base):
    __tablename__ = "editors"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)


class Match(Base):
    """一场盲评：一篇原稿、一位编辑、左右两个模型；交回后记录胜方。"""

    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    manuscript_id = Column(String(100), index=True, nullable=False)
    editor_id = Column(Integer, ForeignKey("editors.id"), nullable=False)
    left_model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    right_model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    status = Column(String(16), nullable=False, default=PENDING)
    winner = Column(String(8), nullable=True)  # "left" | "right" | None
    origin = Column(String(8), nullable=False, default=ORIGIN_MANUAL)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

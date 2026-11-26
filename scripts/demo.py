#!/usr/bin/env python3
"""端到端演示：登记 → 录答卷 → 不满秩 → 补评排程 → 收回答卷 → 得到排名。

用法：
    python3 scripts/demo.py [base_url]     # 默认 http://localhost:8000

对应 README 中 /docs 的手工演示流程，可反复运行（每次先重置数据）。
"""
from __future__ import annotations

import json
import sys

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
c = httpx.Client(base_url=BASE, timeout=30.0)


def step(title: str) -> None:
    print(f"\n=== {title} ===")


def show(resp: httpx.Response) -> dict:
    body = resp.json()
    print(f"{resp.request.method} {resp.request.url.path} -> {resp.status_code}")
    print(json.dumps(body, ensure_ascii=False, indent=2)[:1200])
    resp.raise_for_status()
    return body


step("0. 重置演示数据")
show(c.post("/admin/reset"))

step("1. 登记参评模型（第一个登记的是参照模型，分数固定为 0）")
models = {}
for name in ["Alpha", "Beta", "Gamma", "Delta"]:
    models[name] = show(c.post("/models", json={"name": name}))["id"]

step("2. 登记编辑")
editors = {}
for name in ["Eva", "Frank"]:
    editors[name] = show(c.post("/editors", json={"name": name}))["id"]

step("3. 登记已有答卷（8 份；新模型 Delta 还没有答卷）")
seed = [
    ("MS-001", "Eva", "Alpha", "Beta", "right"),
    ("MS-002", "Eva", "Beta", "Gamma", "left"),
    ("MS-003", "Eva", "Alpha", "Gamma", "left"),
    ("MS-004", "Eva", "Gamma", "Alpha", "left"),
    ("MS-005", "Frank", "Alpha", "Beta", "right"),
    ("MS-006", "Frank", "Beta", "Gamma", "left"),
    ("MS-007", "Frank", "Alpha", "Gamma", "right"),
    ("MS-008", "Frank", "Gamma", "Alpha", "right"),
]
for ms, ed, lm, rm, winner in seed:
    show(
        c.post(
            "/responses",
            json={
                "manuscript_id": ms,
                "editor_id": editors[ed],
                "left_model_id": models[lm],
                "right_model_id": models[rm],
                "winner": winner,
            },
        )
    )

step("4. 请求排名 → 不满秩，返回补发计划（最少的补评场次）")
r = show(c.get("/ranking"))
assert r["status"] == "insufficient_rank", r
assert r["plan"]["feasible"] and r["plan"]["n_selected"] == 1, r

step("5. 发出补评场次（占用原稿）")
applied = show(c.post("/schedule/apply"))
match_id = applied["created"][0]["id"]

step("6. 待交计入排程、但不用于拟合：仍不满秩，却无需再补发")
r = show(c.get("/ranking"))
assert r["status"] == "insufficient_rank" and r["plan"]["n_selected"] == 0, r

step("7. 收回补评答卷（左栏 Alpha 胜）")
show(c.post(f"/matches/{match_id}/submit", json={"winner": "left"}))

step("8. 再次请求排名 → 满秩，L2 正则拟合出模型分与编辑偏置")
r = show(c.get("/ranking"))
assert r["status"] == "ok", r
order = [row["name"] for row in r["models"]]
biases = {row["name"]: row["bias"] for row in r["editors"]}
print("模型排名:", " > ".join(order))
print("编辑偏置:", {k: round(v, 3) for k, v in biases.items()})
assert order == ["Beta", "Alpha", "Gamma", "Delta"], order
assert biases["Eva"] > 0 > biases["Frank"], biases

step("9. 撤回演示：发出 → 撤回 → 旧答卷无效 → 原稿释放可重派")
m = show(
    c.post(
        "/matches",
        json={
            "manuscript_id": "MS-009",
            "editor_id": editors["Eva"],
            "left_model_id": models["Alpha"],
            "right_model_id": models["Gamma"],
        },
    )
)
mid = m["id"]
show(c.post(f"/matches/{mid}/withdraw"))
err = c.post(f"/matches/{mid}/submit", json={"winner": "left"})
print(f"撤回后提交 -> {err.status_code}（旧答卷无效）")
assert err.status_code == 409
show(
    c.post(
        "/matches",
        json={
            "manuscript_id": "MS-009",
            "editor_id": editors["Frank"],
            "left_model_id": models["Beta"],
            "right_model_id": models["Delta"],
        },
    )
)

step("10. 总体状态")
show(c.get("/status"))
print("\n演示完成：补评答卷收回后成功得到排名。")

"""API 端到端测试：登记 → 答卷 → 排程 → 补评收回 → 排名。"""

def _seed(client, model_names=("Alpha", "Beta", "Gamma", "Delta"), editor_names=("Eva", "Frank")):
    models = {n: client.post("/models", json={"name": n}).json()["id"] for n in model_names}
    editors = {n: client.post("/editors", json={"name": n}).json()["id"] for n in editor_names}
    return models, editors


def _full_flow_seed(client):
    """8 份答卷覆盖 Alpha/Beta/Gamma 与两位编辑；新模型 Delta 无答卷 → 缺 1。

    隐含真值：Beta > Alpha > Gamma > Delta，Eva 偏爱左栏(+)，Frank 反之(-)。
    """
    m, e = _seed(client)
    responses = [
        ("MS-001", e["Eva"], m["Alpha"], m["Beta"], "right"),
        ("MS-002", e["Eva"], m["Beta"], m["Gamma"], "left"),
        ("MS-003", e["Eva"], m["Alpha"], m["Gamma"], "left"),
        ("MS-004", e["Eva"], m["Gamma"], m["Alpha"], "left"),
        ("MS-005", e["Frank"], m["Alpha"], m["Beta"], "right"),
        ("MS-006", e["Frank"], m["Beta"], m["Gamma"], "left"),
        ("MS-007", e["Frank"], m["Alpha"], m["Gamma"], "right"),
        ("MS-008", e["Frank"], m["Gamma"], m["Alpha"], "right"),
    ]
    for manuscript_id, editor_id, left, right, winner in responses:
        r = client.post(
            "/responses",
            json={
                "manuscript_id": manuscript_id,
                "editor_id": editor_id,
                "left_model_id": left,
                "right_model_id": right,
                "winner": winner,
            },
        )
        assert r.status_code == 201, r.text
    return m, e


def test_register_and_validate(client):
    m, e = _seed(client, ("Alpha", "Beta"), ("Eva",))
    assert client.get("/models").status_code == 200
    # 重名 409
    assert client.post("/models", json={"name": "Alpha"}).status_code == 409
    assert client.post("/editors", json={"name": "Eva"}).status_code == 409
    # 左右同一模型 422
    r = client.post(
        "/responses",
        json={
            "manuscript_id": "MS-X",
            "editor_id": e["Eva"],
            "left_model_id": m["Alpha"],
            "right_model_id": m["Alpha"],
            "winner": "left",
        },
    )
    assert r.status_code == 422
    # 不存在的模型/编辑 404
    r = client.post(
        "/responses",
        json={
            "manuscript_id": "MS-X",
            "editor_id": e["Eva"],
            "left_model_id": m["Alpha"],
            "right_model_id": 999,
            "winner": "left",
        },
    )
    assert r.status_code == 404


def test_full_flow_plan_apply_submit_rank(client):
    m, e = _full_flow_seed(client)

    # 1) 不满秩：返回补发计划
    r = client.get("/ranking").json()
    assert r["status"] == "insufficient_rank"
    assert r["n_params"] == 5 and r["rank"] == 4 and r["deficiency"] == 1
    plan = r["plan"]
    assert plan["feasible"] and plan["n_selected"] == 1
    assert plan["candidates_considered"] <= 16
    # Delta 无答卷 → 补评必须涉及 Delta
    pm = plan["matches"][0]
    assert pm["manuscript_id"] == "MS-001"
    assert pm["editor_id"] == e["Eva"]
    assert {pm["left_model_id"], pm["right_model_id"]} == {m["Alpha"], m["Delta"]}

    # 2) 发出补评场次
    applied = client.post("/schedule/apply")
    assert applied.status_code == 201, applied.text
    created = applied.json()["created"]
    assert len(created) == 1
    match_id = created[0]["id"]
    assert created[0]["status"] == "pending" and created[0]["origin"] == "plan"

    # 3) 待交计入排程但不用于拟合：拟合秩仍缺，但无需再补发
    r = client.get("/ranking").json()
    assert r["status"] == "insufficient_rank"
    assert r["n_pending"] == 1
    assert r["plan"]["feasible"] and r["plan"]["n_selected"] == 0
    # 重复 apply → 409
    assert client.post("/schedule/apply").status_code == 409

    # 4) 收回补评答卷（Alpha 胜 Delta）
    r = client.post(f"/matches/{match_id}/submit", json={"winner": "left"})
    assert r.status_code == 200

    # 5) 满秩：得到排名
    r = client.get("/ranking").json()
    assert r["status"] == "ok"
    assert r["rank"] == 5 and r["n_responses"] == 9
    scores = {row["name"]: row["score"] for row in r["models"]}
    assert scores["Alpha"] == 0.0  # 参照模型固定为 0
    order = [row["name"] for row in r["models"]]
    assert order == ["Beta", "Alpha", "Gamma", "Delta"]
    assert [row["rank"] for row in r["models"]] == [1, 2, 3, 4]
    biases = {row["name"]: row["bias"] for row in r["editors"]}
    assert biases["Eva"] > 0 > biases["Frank"]  # Eva 偏爱左栏，Frank 反之


def test_manuscript_occupation_and_withdraw(client):
    m, e = _seed(client, ("Alpha", "Beta"), ("Eva",))
    body = {"manuscript_id": "MS-1", "editor_id": e["Eva"], "left_model_id": m["Alpha"], "right_model_id": m["Beta"]}
    r = client.post("/matches", json=body)
    assert r.status_code == 201
    match_id = r.json()["id"]
    # 发出即占用：同一原稿不能再发
    assert client.post("/matches", json=body).status_code == 409
    # 撤回后可重派
    assert client.post(f"/matches/{match_id}/withdraw").status_code == 200
    r = client.post("/matches", json=body)
    assert r.status_code == 201
    match_id = r.json()["id"]
    # 交回后既不能重复提交也不能撤回
    assert client.post(f"/matches/{match_id}/submit", json={"winner": "left"}).status_code == 200
    assert client.post(f"/matches/{match_id}/submit", json={"winner": "left"}).status_code == 409
    assert client.post(f"/matches/{match_id}/withdraw").status_code == 409


def test_withdrawn_match_response_is_invalid(client):
    m, e = _seed(client, ("Alpha", "Beta"), ("Eva",))
    r = client.post(
        "/matches",
        json={"manuscript_id": "MS-1", "editor_id": e["Eva"], "left_model_id": m["Alpha"], "right_model_id": m["Beta"]},
    )
    match_id = r.json()["id"]
    assert client.post(f"/matches/{match_id}/withdraw").status_code == 200
    # 旧答卷无效
    r = client.post(f"/matches/{match_id}/submit", json={"winner": "left"})
    assert r.status_code == 409
    # 撤回的场次不进入拟合
    status = client.get("/status").json()
    assert status["matches"]["withdrawn"] == 1
    assert status["matches"]["completed"] == 0


def test_one_supplement_per_manuscript(client):
    m, e = _seed(client, ("Alpha", "Beta"), ("Eva",))
    client.post(
        "/responses",
        json={
            "manuscript_id": "MS-1",
            "editor_id": e["Eva"],
            "left_model_id": m["Alpha"],
            "right_model_id": m["Beta"],
            "winner": "left",
        },
    )
    # 缺 1 → 在 MS-1 上补一场
    r = client.get("/ranking").json()
    assert r["status"] == "insufficient_rank" and r["plan"]["n_selected"] == 1
    assert r["plan"]["matches"][0]["manuscript_id"] == "MS-1"
    applied = client.post("/schedule/apply").json()
    match_id = applied["created"][0]["id"]
    client.post(f"/matches/{match_id}/submit", json={"winner": "right"})
    assert client.get("/ranking").json()["status"] == "ok"
    # 新登记一位编辑 → 又缺 1，但 MS-1 已补过一场，无空闲原稿 → 无解
    client.post("/editors", json={"name": "Frank"})
    r = client.get("/ranking").json()
    assert r["status"] == "insufficient_rank"
    assert not r["plan"]["feasible"]
    assert r["plan"]["max_achievable_rank"] == 2
    assert client.post("/schedule/apply").status_code == 409


def test_infeasible_reports_max_achievable_rank(client):
    m, e = _seed(client, ("Alpha", "Beta", "Gamma"), ("Eva",))
    client.post(
        "/responses",
        json={
            "manuscript_id": "MS-1",
            "editor_id": e["Eva"],
            "left_model_id": m["Alpha"],
            "right_model_id": m["Beta"],
            "winner": "left",
        },
    )
    # 参数 3 个（Beta、Gamma、Eva 偏置），秩 1，只有 1 篇空闲原稿 → 最多到秩 2
    r = client.get("/ranking").json()
    assert r["status"] == "insufficient_rank"
    assert r["deficiency"] == 2
    assert not r["plan"]["feasible"]
    assert r["plan"]["max_achievable_rank"] == 2
    r = client.post("/schedule/apply")
    assert r.status_code == 409
    assert r.json()["detail"]["max_achievable_rank"] == 2


def test_withdraw_frees_planned_manuscript(client):
    m, e = _seed(client, ("Alpha", "Beta"), ("Eva",))
    client.post(
        "/responses",
        json={
            "manuscript_id": "MS-1",
            "editor_id": e["Eva"],
            "left_model_id": m["Alpha"],
            "right_model_id": m["Beta"],
            "winner": "left",
        },
    )
    applied = client.post("/schedule/apply").json()
    match_id = applied["created"][0]["id"]
    # 撤回补发场次 → 原稿释放，可再次排程
    client.post(f"/matches/{match_id}/withdraw")
    r = client.get("/ranking").json()
    assert r["plan"]["feasible"] and r["plan"]["n_selected"] == 1
    manuscripts = {row["manuscript_id"]: row["status"] for row in client.get("/manuscripts").json()}
    assert manuscripts["MS-1"] == "free"


def test_admin_reset(client):
    _seed(client, ("Alpha",), ("Eva",))
    assert client.post("/admin/reset").status_code == 200
    assert client.get("/models").json() == []
    assert client.get("/status").json()["n_params"] == 0

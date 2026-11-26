# 译文两两盲评平台

编辑部把同一篇原稿的译文两两盲评。平台用 **Bradley–Terry 模型**把"模型水平"和
"编辑偏爱左栏的习惯"分离开：

```
logit(P(左栏胜)) = score(左模型) − score(右模型) + bias(编辑)
```

- 每个模型一个分数，**第一个登记的模型为参照，分数固定为 0**（保证可识别）；
- 每位编辑一个偏置，> 0 表示偏爱左栏，< 0 表示偏爱右栏；
- **实收答卷的设计矩阵满列秩**时，用 L2 正则（岭正则）做最大似然拟合并给出排名；
- 不满秩时，从**至多 16 个候选场次**中选出**最少的一组**补发：与待交场次一起收齐后
  恰好补足秩；候选不够时报告**最多能达到的秩**。

## 快速开始

```bash
docker compose up --build
# 打开 http://localhost:8000/docs
```

本地开发（不用 Docker）：

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
python3 -m pytest tests -q          # 运行测试
python3 scripts/demo.py             # 对运行中的服务做端到端演示
```

## 数据与规则

- **登记**：`POST /models`、`POST /editors`；`POST /responses` 登记已有答卷
  （原稿号、编辑、左右模型、胜方），直接计入实收。
- **场次**：`POST /matches` 发出一场盲评（待交）。**发出即占用原稿**：同一原稿
  同时至多一场待交，重复发出返回 409。
- **交回**：`POST /matches/{id}/submit` 记录胜方；只有交回的答卷才用于拟合。
- **撤回**：`POST /matches/{id}/withdraw` 撤回待交场次 → 原稿释放、可重派；
  此后该场次的**旧答卷一律无效**（提交返回 409，且不参与拟合）。
- **补评**：`GET /ranking` 不满秩时返回补发计划；`POST /schedule/apply` 按计划
  发出（`origin=plan`）。**每篇原稿最多补一场**：已被补发过（补发场次待交或已交回）
  的原稿不再进入候选；撤回的补发不占名额。
- **排程**：待交场次计入排程基矩阵（假定会收齐），但不用于拟合。

## /docs 演示：收回补评答卷后得到排名

下面每步对应 Swagger UI 里的一个接口，点 **Try it out** 填入 JSON 执行即可。
`scripts/demo.py` 把同一流程自动化了，可对照。

1. `POST /admin/reset` —— 清空演示数据（可选）。
2. `POST /models` 依次登记 `Alpha`、`Beta`、`Gamma`、`Delta`
   （`Alpha` 最先登记，是参照模型，分数固定为 0）。记下返回的 id，下文假设为 1–4。
3. `POST /editors` 登记 `Eva`、`Frank`（id 假设为 1、2）。
4. `POST /responses` 登记 8 份已有答卷（新模型 Delta 还没有任何答卷）：

   | manuscript_id | editor_id | left | right | winner |
   |---|---|---|---|---|
   | MS-001 | 1 (Eva) | 1 Alpha | 2 Beta | right |
   | MS-002 | 1 (Eva) | 2 Beta | 3 Gamma | left |
   | MS-003 | 1 (Eva) | 1 Alpha | 3 Gamma | left |
   | MS-004 | 1 (Eva) | 3 Gamma | 1 Alpha | left |
   | MS-005 | 2 (Frank) | 1 Alpha | 2 Beta | right |
   | MS-006 | 2 (Frank) | 2 Beta | 3 Gamma | left |
   | MS-007 | 2 (Frank) | 1 Alpha | 3 Gamma | right |
   | MS-008 | 2 (Frank) | 3 Gamma | 1 Alpha | right |

   每份的 JSON 形如：
   ```json
   {"manuscript_id": "MS-001", "editor_id": 1, "left_model_id": 1, "right_model_id": 2, "winner": "right"}
   ```
5. `GET /ranking` —— 返回 `insufficient_rank`：参数 5 个（Beta/Gamma/Delta 分数 +
   两位编辑偏置），实收答卷的秩只有 4（Delta 的分数列全零）。`plan` 里给出最少的
   补发场次：1 场，涉及 Delta（如 `MS-001, Eva, Alpha|Delta`）。
6. `POST /schedule/apply` —— 发出该补评场次（状态 `pending`，占用 MS-001）。
   此时再 `GET /ranking` 仍显示不满秩（待交不用于拟合），但计划变为
   `n_selected = 0`：待交收齐即可，无需再补发。
7. `POST /matches/{id}/submit` —— 收回补评答卷，提交 `{"winner": "left"}`。
8. `GET /ranking` —— 满秩，返回拟合结果。本例得到：

   ```
   Beta(≈3.3) > Alpha(0, 参照) > Gamma(≈−0.2) > Delta(≈−0.7)
   Eva 偏置 ≈ +1.8（偏爱左栏），Frank 偏置 ≈ −1.7（偏爱右栏）
   ```

   数值随 `l2` 正则强度（默认 0.1，可用 `GET /ranking?l2=...` 调整）略有变化，
   但排名与偏置符号稳定。

撤回流程也可在 /docs 演示：`POST /matches` 发出一场 → `POST /matches/{id}/withdraw`
撤回 → 对该场次 `submit` 返回 409（旧答卷无效）→ 同一原稿可重新 `POST /matches`。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/models` `/editors` | 登记参评模型 / 编辑 |
| GET | `/models` `/editors` | 列表 |
| POST | `/responses` | 登记已有答卷（计入实收） |
| POST | `/matches` | 发出一场盲评（待交，占用原稿） |
| GET | `/matches` | 场次列表（可按 `status` 过滤） |
| POST | `/matches/{id}/submit` | 交回答卷 |
| POST | `/matches/{id}/withdraw` | 撤回待交场次 |
| GET | `/ranking` | 满秩→排名与偏置；不满秩→补发计划/最高可达秩 |
| POST | `/schedule/apply` | 按补发计划发出场次 |
| GET | `/manuscripts` `/status` | 原稿占用情况 / 总体状态 |
| POST | `/admin/reset` | 清空数据（演示用） |

## 设计说明

- **秩与补发**：参数列 = 非参照模型分数 + 编辑偏置。实收（拟合用）与实收+待交
  （排程用）分别构造 0/±1 设计矩阵，用 SVD 求秩与零空间。候选场次按"在零空间上
  投影最大"逐篇贪心生成（每篇原稿至多一个、至多 16 个、互不冗余），再贪心选出
  最小补发集合——每行至多贡献 1 秩，故选满 deficiency 行即最优；补不满时报告
  候选全部收齐后能达到的最高秩。
- **拟合**：`scipy.optimize`（L-BFGS）最小化负对数似然 + `l2/2·||θ||²`，
  解析梯度；L2 保证可分离数据下解仍有限唯一。
- **存储**：SQLAlchemy + SQLite（Compose 中挂卷 `/data` 持久化），
  `DATABASE_URL` 可换成其他数据库。

## 已知限制

- 参照模型固定为第一个登记的模型，暂不支持指定；
- 已交回的答卷不能撤回或作废（只有待交场次可撤回）；
- 候选场次只在已有原稿中生成，不登记新原稿；
- 单进程 SQLite，未做并发写优化与置信区间输出。

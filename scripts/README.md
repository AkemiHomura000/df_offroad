# scripts/

项目脚本入口。先 `source ddv2_env.sh`（训练/推理）或 `source verti_bench_env.sh`（Chrono 仿真）。

推荐流水线（Verti，当前分支 `exp/lidar-elevation-bev`）：

```
采集专家 → 建 elevation 数据集 → IL 微调 →（可选 GRPO）→ 开环评测 / 闭环批量评测
```

本分支默认将 DDV2 的 `lidar_feature` 从点云**密度** BEV 换成**高程** BEV（`[-3,4]` m → `[0,1]`）。
回退密度：传感器设 `lidar_bev_mode="density"`，训练用 `DATA_DIR=datasets/verti_ddv2_density`。

`status_feature` 默认方案 B：自车系目标点编码进前 4 维（`gx,gy,‖g‖ / 32`），替代 L/S/R/U one-hot。
重写已有数据集 status（不重采）：
`python scripts/bridge/rewrite_verti_status_goal.py --data_dir datasets/verti_ddv2_elevation --src log/verti_expert`

---

## Shell 启动器（`*.sh`）

### `bridge/` — 仿真与闭环

| 脚本 | 作用 |
|------|------|
| `run_collect_expert.sh` | Verti-Bench 专家轨迹采集（MPPI/PID 等） |
| `run_verti_ddv2.sh` | 单场景闭环：起 plan server + Chrono（可开渲染） |
| `run_eval_verti_ddv2_batch.sh` | 无渲染批量闭环评测，写出 `scene.json` / `summary.json` |

常用环境变量：`CKPT` `PLAN_ANCHOR` `WORLDS` `TASKS` `OUT` `WORLD` `RENDER`

### `train/` — 训练与开环评测

| 脚本 | 作用 |
|------|------|
| `start_finetune_verti.sh` | nohup 启动 Verti elevation IL 微调（默认 10 ep） |
| `start_finetune_rellis.sh` | nohup 启动 RELLIS IL 微调 |
| `start_verti_grpo.sh` | nohup 启动 VertiPDM-GRPO |
| `start_eval_openloop.sh` | nohup 开环 ADE/FDE 评测（可对比 baseline） |
| `run_finetune_rellis.sh` | IL 微调实际入口（被 start_* 调用；RELLIS/Verti 共用） |
| `run_eval_rellis.sh` | 开环评测实际入口 |
| `run_verti_grpo.sh` | GRPO 实际入口 |

常用环境变量：`OUT` `DATA_DIR` `CKPT`/`INIT_CKPT` `PLAN_ANCHOR` `EPOCHS` `BATCH` `LR` `RESUME` `BASELINE` `MAX_SAMPLES`

示例：

```bash
# Verti IL
EPOCHS=20 OUT=log/verti_il_finetune bash scripts/train/start_finetune_verti.sh

# 从 last.ckpt 再训到总 20 ep（在已有 OUT 上）
RESUME=last EPOCHS=20 OUT=log/verti_il_finetune_v3_density \
  bash scripts/train/run_finetune_rellis.sh

# 开环：GRPO vs IL
CKPT=log/verti_grpo_v1/ckpts/last.ckpt \
BASELINE=log/verti_il_finetune_v3_density/ckpts/best.ckpt \
DATA_DIR=datasets/verti_ddv2_density MAX_SAMPLES=100 \
OUT=log/verti_grpo_vs_il_eval bash scripts/train/start_eval_openloop.sh

# 闭环 100 场景
WORLDS=all TASKS=1 OUT=log/verti_ddv2_eval_100 \
  bash scripts/bridge/run_eval_verti_ddv2_batch.sh
```

### `rellis/` — RELLIS 数据准备

| 脚本 | 作用 |
|------|------|
| `extract_rellis3d.sh` | 解压/抽取 RELLIS-3D 相机与激光 |
| `process_rellis_full.sh` | 按序列抽取 → 建 npz → 可删 raw 省盘 |

---

## Python 入口（按目录）

### `bridge/`

| 文件 | 作用 |
|------|------|
| `collect_expert_verti.py` | 专家采集核心逻辑 |
| `build_verti_ddv2_dataset.py` | 专家 ep → dens/elevation 训练样本（默认 `--status_nav goal`） |
| `rewrite_verti_status_goal.py` | 已有数据集 status 重写为 ego 目标点（方案 B，不重采） |
| `build_verti_metric_cache.py` | 为 GRPO 建 VertiMetricCache |
| `verti_metric_cache.py` / `verti_pdm_scorer.py` | 越野 metric / VertiPDM 打分 |
| `ddv2_plan_server.py` / `ddv2_plan_client.py` / `ddv2_plan_protocol.py` | 闭环规划 TCP 服务 |
| `eval_verti_ddv2_batch.py` | 批量闭环评测（无 GUI） |
| `viz_verti_expert.py` | 专家轨迹可视化 |
| `nuplan_min_stub.py` | 精简环境缺 nuplan 时的 stub |

### `train/`

| 文件 | 作用 |
|------|------|
| `train_rellis_il.py` | IL 微调（RELLIS/Verti 数据集通用） |
| `eval_rellis_il.py` | 开环评测 + 对比图 |
| `train_verti_grpo.py` / `verti_grpo_utils.py` | Verti GRPO 训练 |
| `sanity_verti_pdm.py` | VertiPDM 可分性自检 |
| `patch_verti_pdm_backend.py` | 可选：把官方 RL head 接到 VertiPDM |

### `rellis/`

| 文件 | 作用 |
|------|------|
| `build_rellis_ddv2_dataset.py` | RELLIS → DDV2 样本 |
| `generate_rellis_anchors.py` | k-means 轨迹 anchors |
| `fix_rellis_ego_coords.py` / `resplat_rellis_lidar.py` | 坐标/激光修正 |
| `viz_infer_rellis.py` | 多模态推理可视化面板 |
| `export_rellis_videos.py` | 导出视频 |

---

## 已删除的冗余脚本

历史版本启动器已合并为上表通用脚本，例如：

- `start_finetune_v4/v5/v6/v7_*.sh` → `start_finetune_rellis.sh`
- `start_finetune_verti_v1/v2/v3_*.sh` → `start_finetune_verti.sh`
- `start_eval_verti_*.sh` / `start_eval_verti_grpo_vs_il_100.sh` → `start_eval_openloop.sh`
- `start_verti_grpo_v1.sh` → `start_verti_grpo.sh`
- `check_traj_cmd.sh`、`run_chrono_ddv2_bridge_demo.sh`（一次性调试/演示）

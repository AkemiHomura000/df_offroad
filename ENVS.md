# 环境说明（df_offroad）

Miniconda/Anaconda: `$HOME/miniconda3` 或 `$HOME/anaconda3`（见 `*_env.sh`）

| 环境 | 用途 | 启用 |
|------|------|------|
| `chrono9` | Verti-Bench / PyChrono 仿真 | `source verti_bench_env.sh` |
| `ddv2` | DiffusionDriveV2 推理/训练（无 NAVSIM） | `source ddv2_env.sh` |

闭环时：Chrono 用 `chrono9`，模型用 `ddv2`（进程间通信；勿混装）。

---

## chrono9

- Python 3.9 · PyChrono 9.0.1 · numpy 1.24.0 · torch 2.7.1+cu118 · cudatoolkit 11.8
- 数据: `CHRONO_DATA_DIR=.../verti_bench/envs/data/`
- 代码路径: `PYTHONPATH` 含仓库根（`verti_bench_env.sh` / 脚本内按 `__file__` 解析；本机或 `~/df_offroad` 均可）
- **Ubuntu 20.04 源码编**：源码 `downloads/chrono`，构建 `downloads/chrono_build`（勿用 `~/Documents`）。`verti_bench_env.sh` 若发现 `downloads/chrono_build/bin/pychrono` 会自动前置到 `PYTHONPATH`/`LD_LIBRARY_PATH`。勿 `pip install pychrono`。
- 仿真: `cd verti_bench && python setup.py`
- DDV2 传感器: `python setup.py ... ddv2_sensors=true`（可选 `ddv2_dump_dir=...`）
  - 模块: `verti_bench/envs/utils/ddv2_sensors.py`
  - 输出: `camera_feature (3,256,1024)` · `depth_feature (1,256,1024)` · `lidar_feature (1,256,256)` · `status_feature (8,)`
  - **默认** `camera_feature` = 深度 `/depth_max` 再复制成 3 通道（`camera_feature_source="depth"`）；`="rgb"` 可回退 RGB
  - 采集另存 `cam_front` RGB + `cam_depth`（米，同视角同分辨率）；建库 `--camera_mode depth|rgb`（默认 depth → `datasets/verti_ddv2_*_depth`）
  - `lidar_feature` 默认 = **高程 BEV**（底盘系 max-z ∈ `[-3,4]` m → `[0,1]`；`lidar_bev_mode="density"` 可回退点数直方图）
  - `status_feature` 默认 = **方案 B**（8 维）：`[gx,gy]/32` + `[roll,pitch]/π` + `vx,vy,ax,ay`；`status_nav_mode="command"` 可回退 L/S/R/U
  - 相机: 前视 RGB + Depth（同挂点）`1024×256`，HFOV 100°，挂点约 `(1.2,0,1.6)`，俯仰 −8°，`depth_max=40` m；LiDAR BEV ±32 m
- MPPI/TAL/TNT 等原依赖 `grid_map_msgs`（ROS2: `ros-humble-grid-map-msgs`；ROS1: `ros-noetic-grid-map-msgs`）。无 ROS 时 `verti_bench_env.sh` / `collect_expert_verti.py` 会自动启用 `verti_bench/third_party/ros_msg_stubs`。`system=ddv2` / `pid` 本就不依赖 ROS。

---

## ddv2

- Python 3.9 · torch 2.0.1+cu118 · torchvision 0.15.2 · PL 2.2.1
- 精简依赖: `DiffusionDriveV2/requirements-ddv2-lean.txt`
- **未装**: nuplan、geopandas、ray、官方 `requirements.txt` 全量、OpenScene 数据
- `PYTHONPATH` → `DiffusionDriveV2/`（仅代码导入；不跑 NAVSIM）
- 原版 agent 仍可能 `import nuplan`，接 Chrono 前需改接口

---

## Chrono ↔ DDV2

闭环入口见 `scripts/bridge/run_verti_ddv2.sh`（及批量评测 `run_eval_verti_ddv2_batch.sh`）。脚本说明见 `scripts/README.md`。

## Verti-Bench + DDV2 规划（替换 A*）

`system=ddv2`：Chrono 传感器 → TCP → FT DiffusionDrive → 局部轨迹 → PID 跟踪。高层目标仍通过 `status` 的 one-hot 命令（朝向 goal）。

运行时会弹出 **DDV2 live** 窗口（相机 / LiDAR BEV+轨迹 / GPU与RTT耗时曲线），Irrlicht 里用球体标出预测航点；终端打印 `GPU=..ms RTT=..ms`。关闭可视化：`LIVE_VIZ=false DRAW_WORLD=false`。

```bash
# 一键：起 plan server + Verti-Bench
bash scripts/bridge/run_verti_ddv2.sh
# 可选: CKPT=... WORLD=1 RENDER=false SPEED=3 MAX_TIME=30 bash scripts/bridge/run_verti_ddv2.sh

# 或分两终端：
# T1
source ddv2_env.sh
python scripts/bridge/ddv2_plan_server.py --ckpt log/rellis_il_finetune_v3/ckpts/best.ckpt --port 8765
# T2
source verti_bench_env.sh
cd verti_bench && python setup.py system=ddv2 world_id=1 vehicle=hmmwv render=true
```

## RELLIS-3D → DDV2 数据集

```bash
# 全量：解压 → npz → 删 raw（按序列）
bash scripts/rellis/process_rellis_full.sh

# 或仅构建（需已解压序列）
python scripts/rellis/build_rellis_ddv2_dataset.py --max_samples 400
```

输出：`datasets/ddv2_rellis_full/{train,val}/*.npz` + `meta.json`  
字段：`camera_feature (3,256,1024)` · `lidar_feature (1,256,256)` · `status (8,)` · `trajectory (8,3)`  
轨迹：由 `poses.txt` 在 10 Hz 下取未来 4 s、间隔 0.5 s，变到当前车体坐标。

## RELLIS IL 微调（对齐官方 DiffusionDrive / TransfuserAgent）

原仓库训练分两段：
1. **IL**（`agent=diffusiondrive` / `TransfuserAgent`）：`LossComputer`（focal cls + L1 reg）×`trajectory_weight`，AdamW + WarmupCosLR，lr=6e-4，100 epoch  
2. **RL**（`agent=diffusiondrivev2_rl_agent`）：在 IL ckpt 上 GRPO，需 PDMS —— RELLIS 暂不做  

本仓脚本对齐第 1 段，并可选 `anchor_keep` 约束未匹配模态（缓解 RELLIS 多模态折线）：

```bash
# → log/rellis_il_finetune_v2
bash scripts/train/run_finetune_rellis.sh
# 纯官方损失（不加 anchor_keep）: ANCHOR_KEEP=0 bash scripts/train/run_finetune_rellis.sh

# 评测
CKPT=log/rellis_il_finetune_v2/ckpts/best.ckpt bash scripts/train/run_eval_rellis.sh
```

---

## 目录

```
df_offroad/
  verti_bench/              # Chrono 仿真
  DiffusionDriveV2/         # 模型代码
  datasets/                 # 训练数据集（与 log 分离）
    ddv2_rellis_full/       # 全量 RELLIS→DDV2
    ddv2_rellis_demo/       # 小样本 demo
  scripts/
    rellis/                 # 解压 / 转 npz
    bridge/                 # Chrono↔DDV2 桥接
  verti_bench_env.sh
  ddv2_env.sh
  ENVS.md
  log/                      # 运行日志、bridge 临时产物
```

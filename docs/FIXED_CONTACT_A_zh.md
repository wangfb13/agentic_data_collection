# 白瓶固定接触参考与 a 轴恢复

本模式从主策略自然接近后的双侧轻触开始。白瓶放到另一个位置时，仍使用 11 月 10 日现场接受的同一套参考。局部观测只用左右指 DD 和力，主任务可以继续用自己的相机。

## 参考和运行契约

| 项目 | 本版定义 |
|---|---|
| 标准夹法 | `white_bottle_a_contact_target_20261110_v1`。白瓶侧面标称抓取高度，使用当天操作员接受的双指 PROBE 和当时的左右指顺序、预处理、轻触协议、工具轴。原始记录在 `outputs/white_a_current_x5_20261110/session_meta.json` 的 `initial_centroids`，接受时刻为 `1794277714320418341`。版本化资产可独立加载。 |
| 传感器背景 | 启动准备时全开空载扣零，文件身份保存在窗口里。本操作只更新 DD 和力的背景。 |
| 偏移 | `offset_a_mm` 是手相对于标准夹法沿工具 +a 的有符号位移。c、l 未训练，返回空。特征是原 ContactPolicy 双指 20 维描述符。拟合目标为负实测手偏移，推理取负号还原偏移。 |
| 偏移伺服 | `offset_servo` 检查双侧稳定、边缘、特征支持域、训练标签范围及不确定度；满足条件时使用 `clip(-gain * offset_a_mm)`。增益、死区与限幅作用于控制。无接触或贴边直接返回 Unknown。 |
| 图策略 | `graph_policy` 使用重训的触觉几何、双指六维力、噪声、支持域、价值和边动作模型。图动作的单位是 mm，图模式的偏移预测为空。动作来源一次选定。 |
| 目标与退出 | 两种模式共用 `forcepolicy_handoff_l1_floor0330`。稳定双侧新窗口在支持域内，图的退出和 Ready 概率条件通过，才可向 ForcePolicy 移交。通过模型的 Ready 表示预计可交接，实际 L1 用后续力控事件核验。允许非零偏移已 Ready；小偏移且未 Ready、拒判、失触和步数耗尽均以未完成结束。 |
| 执行 | 执行器收到 `tcp_translation_a`、`frame=tcp_at_action_start`、版本、参考 ID、观测 ID/时间和有符号动作量。核对控制权、新鲜度、双侧接触，冻结 TCP 旋转。张开、退让、平移、再接近、重新轻触，按同一份起始坐标解释全部路段。平均法向力、逐指力上限、全程工作区、方向、速度、超时及步数由执行器检查。回执从接触到重新接触测量净修正。 |
| 主任务 | 接管前允许主模型接近。正式 ForcePolicy 调节等待位置判断。已验证的 Ready 通过 V4 原有交接契约交还控制；ForcePolicy 就绪后主模型使用新历史继续推理。 |
| 记录和评估 | 表中 `fixed_contact_a_v1` 窗口分存 `supervision`、`offset_prediction` 和动作回执；自然部署没有偏移真值。当前 TCP 仅作运动记录、坐标转换和执行核验。严格重复条件、独立试次、摆放/方位分组分别记录。 |

## 离线操作

在 6f5090 仓库根目录，已有本轮生成的资产可以直接执行回放。当前发行目录的默认 ForcePolicy 目标对应历史白瓶闭合下限 0.330。

```bash
cd /home/dm/DM_das/bottle_grasp_experiment
.venv/bin/python -m skills.contact_position_a.local_replay
.venv/bin/python tests/local_contact_v4_smoke.py
.venv/bin/python -m skills.contact_position_a.local_deploy \
  --action-source offset_servo --offline-fixture --out outputs/local_contact_a_v1/check_offset
```

最后一条通过同一台 6f5090 的 V4 启动器执行 `task-check`。图对照用 `--action-source graph_policy` 和另一个输出目录。测试工位会在开启硬件之前被拒绝。重新训练可运行下面两条，原始 campaign 保持只读。

```bash
.venv/bin/python -m skills.contact_position_a.local_assets --out outputs/local_contact_a_v1
.venv/bin/python -m skills.contact_position_a.local_replay --assets outputs/local_contact_a_v1
```

偏移和图训练的源数据有不同用途。历史 A 目录通过清单绑定到操作员接受的标准夹法、原记录摘要和摆放组，标签取该次会话内的实测 TCP 投影。11 月 10 日 fast 会话有继承同一次已接受 T0 的记录。物体被推动、未完成、缺少实测、贴边等不作偏移监督。动作和结果可以进入图训练；成功 B 的动作不作为几何标签。新位置的历史 A 若要接入，需要提供当地标准夹法与固定参考一致的证据和独立实测来源，按整组摆放及方位留出。只有一个摆放组时只能留出试次组。旧扫描、旧模型和旧包继续走各自的入口。

## 现场验收

1. 为换过的机器人准备独立工作区。复制 `skills/contact_position_a/station.template.json`。用当前机器人的基座坐标填写已实测的 TCP 安全盒及允许的各个接近旋转矩阵。对声明的整个盒和方向容差，张至最开的完整夹爪在平移、退让、再接近时都须避开桌面和固定障碍。这里的 mm 是 TCP 坐标，这个区域在一次工位配置后可容纳多个瓶位。旧机 T0 和新录的一点 TCP 都不足以确定这个区域。
2. 核对当前工具的 TCP 定义、左右指序列号、胶面版本、空载背景、法向受力符号与轻触协议，在工位文件记录对应版本和现场核验。局部轴来自现场实测 TCP，首版只支持原有侧抓接触方式。若新机器人用了不匹配的触觉通道或不同协议，应先检查已知接触的特征分布。
3. 固定模型、参考和退出目标。先在相同接近方向换瓶位，由主模型自然接近；再在已测量的允许方位选另一方向。每个摆放与方位写入 `--placement-group`，只有严格相同的重复设置才复用 `--repeat-setup`。这两个字段不参与模型推理。首次使用现场工位文件仍先执行检查。

```bash
.venv/bin/python -m skills.contact_position_a.local_deploy \
  --action-source offset_servo --station /实际路径/当前工位.json \
  --placement-group p2_angle1 --out outputs/local_contact_a_v1/site_p2_angle1
```

4. 工位核验及 `task-check` 通过后，由现场操作员在同一条命令末尾增加 `--real`，按提示输入 `MOVE`。准备阶段夹爪全开空载，无人遥操。每次部署从当前双指观测开始。真机比较两种来源时使用相同摆位、同一退出目标和同一组现场条件。
5. 每组记录未调整即 Ready、实际调用、重新接触后的 Ready、拒判原因、实际动作次数、交接 L1、主任务结果、人工干预和总交互数。分别统计需要恢复条件下的成功率和全试次结果。要报新位置的毫米误差，需要独立测量的偏移真值；计划值、模型估计和已执行修正量均不能代替。

## 本次离线数据

偏移估计训练 17 个窗口，按试次分组留出 5 个，MAE 1.26 mm，P90 2.16 mm。训练实测偏移覆盖约 −5.57 至 +4.53 mm。它们来自同一历史摆放和接近方向。真实离线回放与假执行器测试分开保存。

| 对同一批 48 个已记录窗口的训练内回放 | Ready | 发纠正动作 | Unknown |
|---|---:|---:|---:|
| 偏移伺服 | 28 | 0 | 20 |
| 图策略 | 45 | 0 | 3 |

偏移伺服的 20 个拒判中，不确定度超限 10 个、超训练偏移范围 4 个、贴边或描述子无效 3 个、触觉 OOD 3 个。历史 ForcePolicy 终端证据有 18 次 L1 成功、1 次失败，图的 48 个支持节点均归为 Ready。图策略另有 3 个窗口未通过共同的接触质量检查。零次纠正不能验证恢复效果。假硬件使用受控的“需要动作 → Ready”判断覆盖动作执行和交接契约；这不构成真机效果证据。

训练、回放与 V4 预检结果在 `outputs/local_contact_a_v1/`。几何模型为 `offset.json`，图模型为 `graph/models/bundle.joblib`，逐窗口结果为 `recorded_window_replay.json`。如需验证接管分界，优先记录匹配协议下、可复现的未 Ready 首触，以及同条件纠正后是否达到同一个 L1 目标。保留失触、分布外和力控未就绪的原始原因。

## 改动位置

| 仓库内文件 | 职责 |
|---|---|
| `skills/contact_position_a/contact_reference.py`、`white_reference.json`、`white_A_sources.json` | 固定参考、已确认的样本、兼容条件及旧 A 标签来源 |
| `bottle_scan/contact_policy.py`、`skills/contact_position_a/local_assets.py` | 按每批自己的实测几何标签训练固定参考偏移模型；离线重训纯触觉/力的图模型和支持域 |
| `bottle_scan/state_graph/skill.py`、`skills/contact_position_a/local_decision.py` | 共用 Ready 与独立动作来源；偏移分支可独立于图动作模型运行 |
| `skills/contact_position_a/local_motion.py`、`local_backend.py`、`action.py` | TCP 快照、工作区与新鲜度、实测净修正、V4 物理交接契约 |
| `skills/contact_position_a/runtime.py`、`scheduler.py`、`observe.py`、`bottle_scan/state_graph/facts.py` | 可选视觉、无绝对 T0 的窗口、标签来源与覆盖、重复分组 |
| `skills/contact_position_a/deploy.py`、`local_deploy.py`、`station.template.json`、`local_replay.py` | 旧入口 post_condition 修正、新入口、工位模板、历史窗口回放 |
| `tests/test_fixed_contact_reference.py`、`tests/local_contact_v4_smoke.py` | 无 T0、无腕图、无真值、符号、拒判、动作净位移与外部后端交接 |
| `skills/contact_position_a/v4_fixed_reference.patch` | 已应用到 6f5090 外部 V4。涉及 `runtime/bottle_scan/contact_scheduler.py`、`contact_skill_backend.py`、`contact_recovery_backend.py`、`integration/launch_contact_starvla.py` 和 `run_on_5090.sh`。 |

外部运行目录为 `/home/dm/DM_das/contact_scheduler_20261104_experiment_b_v4_single_open_20261110`。其他安装可在同版本目录执行 `git apply --check /绝对路径/v4_fixed_reference.patch`，确认适配后再应用。在本机可用 `git apply --reverse --check` 检查补丁已经安装。现场工位测量、真机恢复和新位置绝对精度尚待验收。

## 已执行的离线检查

- 主仓库全量回归 1484 项通过、4 项跳过。新模式的 15 个单元用例另外复跑通过。
- 真实 V4 类与假硬件的专项脚本 2 项通过，覆盖观测、局部动作、重触回执、交接契约校验，以及冻结模型在元数据改变时的输出一致性。
- 外部 V4 调度、后端和 StarVLA 集成的 pytest 回归 178 项通过。容器默认 Python 缺少 pytest，另外在测试进程内加入已有的 pytest 包后，容器 unittest 套件 144 项通过；没有修改运行环境。
- 两种动作来源的 V4 容器 task-check 均通过。上述检查全部离线。真实机械臂、传感器及下游控制器的现场交接仍待验证。

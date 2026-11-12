# a 轴主模型—恢复 skill—主模型完整链路（2026-11-08）

本次交付只包含代码和离线测试，没有启动机器人、相机、夹爪或实机推理。
本指南中的 `run --real` 命令留给现场逐次执行。

## 复用与改动

统一入口位于 `/home/dm/DM_das/bottle_grasp_experiment`：

```bash
cd /home/dm/DM_das/bottle_grasp_experiment
bash run_on_5090.sh a-axis --help
```

它调用已有 `/home/dm/DM_das/contact_scheduler_20261104_experiment_b_v4`，不会新建第二套运动控制器。
仍使用已有 `261010_starvlapi_two_bottle_pick_unknown_orange/pretrained_model` 主模型和冻结的
接触模型 `6fd653518e46eaefca86`。没有重训、修改权重、修改 T0、放宽动作边界或伪造物理验收。
需要使用其它已有主模型时，可在所有 check/run 命令上显式添加 `--checkpoint /绝对路径/pretrained_model`；
每次先 check，并核对输出实际选择。主模型匹配哪种瓶子和具体任务仍须现场核验。

新增 `--route main`（默认）：准备时空载归零，随后主模型先运行；在原参考位置附近形成合格双侧接触后，
调度器暂停主模型，清空动作队列，skill 根据触觉判断是否需要恢复。恢复完成后重置策略和处理器，
积累新的 30 帧历史，经过原有稳力交接，再让主模型从新观测继续。每个 episode 最多接管一次。
既有 ForcePolicy 稳力交接仍被复用，这次没有做新力控技能学习。

保留 `--route pilot`：执行原来受控偏移和轻触前奏，主模型从开始就暂停。
默认 skill 结束后停止；显式 `--after-skill handoff` 才允许继续主模型。
pilot 适合先确认接触恢复与交接，不证明“主模型先运行再被打断”。

修复了暂停耗尽主模型任务时间、发送前停止/超时复检和旧汇总器错误。
主任务时间预算仅扣实际主循环活动时间，恢复和交接暂停时间单列；skill 和交接自己的墙钟超时仍生效。

## 现场前提

本版本是固定工作站、固定姿态下的 a 轴开发实验，不需要视觉定位或新手眼标定。
保留现有运动、力和开度检查。主模型整段轨迹沿用原推理程序的保护范围，本次没有增加全场景碰撞规划。

1. 使用原训练参考瓶位与匹配的瓶子，瓶子保持桌面支撑。a 轴是原 T0 局部轴，不能当成世界竖直方向。
2. 初始夹爪全开、两触觉面空载；既有机器人、相机、触觉服务由你按原流程准备好。
3. 右手新传感器配置为 `left: X26040505`、`right: X26040305`。配置中的物理左右对应、受压符号、
   旧恢复模型在新传感器上的效果仍未重新验收。先用现有只读观测核对两指按压对应、Fz 方向及空载稳定性。
4. 旧开度下限、瓶径、T0 与本次安装必须兼容。若瓶型或安装改变，不要通过重写旧 T0/证书绕过检查。
5. 正式部署证书仍为空，因此运行保持明确的 `development`、`verified=false`。现场结果用于建立新证据。
6. 不要同时运行另一套主模型推理/采集进程。当前流程复用主推理的相机和单一动作发送路径；
   恢复 backend 不另外订阅相机。

自然接管仅支持原 T0 附近的 a 方向区域、离轴约 2 mm 内、姿态约 1 度内的合格双侧接触，
最终以已有更严格检查为准。没有触发时，记录为“未调用”，不能解释成 skill 恢复成功。
如果主模型本来就在其它位置或姿态抓取，应先解决任务与旧 skill 适用范围的兼容问题。

## 第一步：纯离线检查

```bash
cd /home/dm/DM_das/bottle_grasp_experiment
bash run_on_5090.sh a-axis check
```

检查主策略 checkpoint、冻结 skill、原训练来源、协议、传感器配置、依赖及受审源文件哈希。
默认 `main` 路径的 `experiment_b_offset_mm` 应为 `null`，`after_skill` 应为 `handoff`。
该入口不启动传感器或机器人。继承了 `CONTACT_EXPERIMENT_B_OFFSET_MM` 的终端会被拒绝，
应在专用于本次实验的终端清除该变量，再检查；不要将其设成 0 来假装自然入口。

## 第二步：先确认接触恢复，停止在交回之前

以下命令会运动，只在现场前提确认后执行。每次独立运行、结束后人工检查并复位，不用 shell 循环盲跑。
先 0，再 +5，再 -5 mm；每次修改 trial-id 和 offset，保留失败记录。

```bash
CONTACT_TABLETOP_CONFIRMED=1 \
bash run_on_5090.sh a-axis run --real --route pilot --offset-mm 0 \
  --trial-id pilot_zero \
  --log-dir /home/dm/DM_das/bottle_grasp_experiment/outputs/a_axis_pilot
```

观察：轻触后是否判对需要恢复；需要时是否按“开爪—退让—a 轴修正—重新接触”执行；
是否取得两个独立稳定观测窗后退出。0 mm 也可能被判错，不能提前删除该次试验。
未完成、超时或故障不允许后续夹紧/提起。此阶段不要求主模型继续，也不将停止当成任务完成。

## 第三步：一次受控恢复后的主模型交接

第二步现场通过后再执行。该命令允许主模型继续，可能提起和搬运物体。

```bash
CONTACT_TABLETOP_CONFIRMED=1 \
bash run_on_5090.sh a-axis run --real --route pilot --offset-mm 5 \
  --after-skill handoff --trial-id pilot_handoff_plus5 \
  --log-dir /home/dm/DM_das/bottle_grasp_experiment/outputs/a_axis_handoff
```

确认恢复成功后主模型从当前状态继续，没有回放暂停前动作、突然回到旧抓取位置或重新夹偏。
恢复失败时应保持并停止交接，不自动提起或继续任务。

## 第四步：主模型先执行的完整主任务

这是用户要求的完整链路。这里不施加程序化偏移，也不执行 B 前奏。
初始仍需全开空载归零；之后让原主模型自然接近瓶子。

```bash
CONTACT_TABLETOP_CONFIRMED=1 \
bash run_on_5090.sh a-axis run --real \
  --trial-id main_task_001 \
  --log-dir /home/dm/DM_das/bottle_grasp_experiment/outputs/a_axis_main
```

应观察到：

1. `main_started`，且有 `main_action_sent` 的 `before_skill` 记录。
2. 形成合格接触后出现 `main_paused`，reason 为 `contact_skill_takeover`。
3. `skill_acquired` 后记录 invocation、实际恢复回执、两个新观测窗和 `skill_exit`。
4. 成功后依次完成新历史预热、稳力交接和 `main_resumed`。
5. 出现 `main_execution_resumed`、`main_action_sent` 的 `after_skill`，现场看到主模型继续。
6. 人工确认主任务实际结果，例如是否将目标瓶子放到规定区域；不要用第 4 步的许可代替第 5 步，
   也不要用发送动作代替实际任务成功。

主模型的任务文本、episode 有效执行时长沿用现有入口（默认任务为药瓶分类，时长由原配置/环境决定）。
键盘停止和异常会保留原因；判断是否完成以预先约定的物理目标为准。
本次离线覆盖了不适用、过期观测、恢复耗尽、发送失败和停止请求；现场不需要故意制造碰撞或掉落来测试。

## 第五步：离线汇总和独立任务结果

每次运行自动生成唯一 `episode_*` 目录，旧结果不会被覆盖。

```bash
bash run_on_5090.sh a-axis summary \
  /home/dm/DM_das/bottle_grasp_experiment/outputs/a_axis_main
```

关键字段：

- `skill_completed`：接触恢复是否完成。
- `main_resume_permission`：是否获准交回主模型。
- `main_execution_resumed`：是否有成功发送的恢复后主模型动作；这不等于实测物理运动。
- `handoff_chain_observed`：恢复到主模型交接链路的记录是否齐全。
- `full_chain_observed`：自然主任务中，接管前后都有主模型动作，并完整经过成功恢复和交接。
- `task_success`：独立人工标注的整任务结果；未标注时为 `null`。
- `timing`：控制循环墙钟、暂停、主任务活动时间；不含之前的准备，准备成本在原 backend 记录中保留。
- `recording_status/parse_error/failure`：缺失、截断及故障；不能以成功试验替换删除。

将下面路径中的 `episode_实际编号` 换成该次 summary 显示的完整目录，再根据现场观察填写：

```bash
bash run_on_5090.sh a-axis review \
  /home/dm/DM_das/bottle_grasp_experiment/outputs/a_axis_main/episode_实际编号 \
  --outcome success --note '视频00:40：恢复后主模型将目标瓶子放入约定区域并释放'
```

失败用 `--outcome failure` 并写实际现象；不确定用 `unknown`。
标注只写独立 `task_outcome.json`，绑定该 episode 的事件哈希，不改原始数据、不覆盖旧判断。
完整日志结束后才能标注成功；标注后再修改日志会使标注失效。
瓶子移位时，还应按原接触评估工具的 `manual_review.json` 约定保留现场事实，
避免把“TCP 回到了 T0”误当成瓶子接触已恢复。

## 这轮现场的通过标准

先取得正常接触、正偏、负偏的 pilot 记录，再取得成功交接记录，最后取得自然主任务全链路记录。
自然任务如果没有触发接管，应保留为未调用，不能算入恢复链路成功样本。
成功退出后必须有新主动作和现场任务结果；失败退出不得继续主模型。
几次冒烟只表示联调通过。固定采集与 Agent 采集的等预算效果对照、独立重复评测仍是之后的主实验工作。

## 本次离线验证记录

- 6f5090 临时副本的调度、backend、真实主循环假硬件集成与结果工具测试：160 passed，6 subtests passed。
- 药瓶仓库全库回归：1127 passed、13 skipped、553 subtests passed；唯一失败为隔离副本缺失
  旧 `geometry_candidates.json` 测试夹具。复制该只读夹具后，原失败用例 1 passed，合计1128项通过。
- 实际 `daimon_ros1_das` 容器使用已有 `.venv_robot` 运行核心 unittest：133 tests，OK。
  容器没有 pytest，未为此次测试修改生产依赖；纯离线结果工具的 pytest 用例已在宿主覆盖。
- 部署后 `bash run_on_5090.sh a-axis check`：PASS，明确 `CHECK_ONLY=1`，无相机连接或机器人命令。
- 主策略源文件重新审查：唯一差异为默认 checkpoint 从 261026 改为 261010；动作发送路径未变。
  更新了调度器受审指纹和参考源码快照，没有修改原 StarVLA 仓库中的启动脚本或权重。
- 针对性覆盖：新观测续推理、旧队列清除、5秒恢复不耗尽0.19秒主任务测试预算、force-only不冒充主模型动作、
  发送失败、推理期间ESC、过期交接、恢复未完成、日志截断和人工标注哈希失效。
- 改动前文件备份于两个项目各自 `backups/a_axis_chain_20261108_165045_987610/`。

所有上述控制路径均使用假硬件；已有模型只进行离线加载、来源和协议检查。本次新增真实机器人交互为0。

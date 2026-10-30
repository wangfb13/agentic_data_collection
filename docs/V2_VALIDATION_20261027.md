# 药瓶抓取实验 V2 离线验收记录（2026-10-27）

本记录区分软件功能、模拟流程和记录容量。单元/专项测试与集成 smoke 已有通过证据；默认力控和开度各 10 次模拟完成。**本机 1920×1080 双路随机 JPEG、30 fps 压力测试触发 128 MiB 队列满；5090 的同级 2 秒补测通过，但结束时仍有约 104.9 MB 积压，尚不能宣称持续吞吐就绪。** 本次没有执行任何实机机械臂或夹爪运动。

对应 [V2 需求](V2_REQUIREMENTS.md)、[变更记录](../CHANGELOG.md) 和 [现场完整指南](ADVANCED.md)。以下路径以项目目录为根；Lenovo 本机代码/证据目录为 `/home/lenovo/wfb/bottle_grasp_experiment`，5090 项目目录为 `/home/dm/DM_das/bottle_grasp_experiment`。软件复制不等于现场运动验收，历史第一版证据不代替本版验证。

## 1. 测试与证据清单

| 项目 | 已观察结果 | 证据 |
|---|---|---|
| D2 集成阶段 | 299 项测试通过，包含当时已完成的 D3 组件用例 | [测试日志](../outputs/v2_d2_validation/tests_299_pass.log) |
| 开度/力控及相关安全模块 | 125 项通过 | [125 项日志](../outputs/v2_opening_force_validation/tests_125.log) |
| 运行状态与准备工作流专项 | 41 项通过 | [工作流日志](../outputs/v2_validation/workflow_tests_03.log) |
| commissioning / 报告登记专项 | 25 项通过 | [验收和登记日志](../outputs/v2_validation/commissioning_registration_tests.log) |
| 开度 PASS 持久化专项 | 9 项模块测试通过；CSV、清理、最终报告失败均不得提交确认 | [日志](../outputs/v2_validation/opening_commit_tests.log) |
| 最终全量 discovery | `393` 项，`全部通过` | [全量日志](../outputs/v2_validation/all_tests_final.log) |
| 集成 smoke | 8 个接触案例、6 个视觉点；固定参考及视觉无闭合检查通过 | [acceptance.json](../outputs/v2_final_smoke/acceptance.json) |
| 默认力控模拟 | 10 次，`passed=true`，`failures=[]`，`operator_confirmation=SIMULATION_ONLY` | [报告](../outputs/v2_force_validation_paced2/force_report.json)、[CSV](../outputs/v2_force_validation_paced2/force_report.csv) |
| 默认开度模拟 | 10 次，`completed=true`、`cleanup_complete=true`，`operator_confirmation=SIMULATION_ONLY` | [报告](../outputs/v2_opening_final/opening_report.json)、[CSV](../outputs/v2_opening_final/opening_report.csv) |
| 接触三点 dry-run | 合成候选 3、接受 2、几何拒绝 1；未放宽碰撞检查 | [dry-run](../outputs/v2_validation/final_dry_contact.log) |
| 视觉 dry-run | 合成候选 400，接受 400 | [dry-run](../outputs/v2_validation/final_dry_visual.log) |
| 停止模拟 | 软件停止 / SIGKILL 各 3 次，6 次通过；simulated=true、operator_confirmed=false | [报告](../outputs/v2_validation/final_dry/stop_demo.json) |
| 多 session 分析 | 两个合成 session 共 28 条汇总、44 条描述子，合并及分 session 输出，warnings=[] | [分析报告](../outputs/v2_validation/multi_session/analysis_report.json) |
| force / opening dry-run | 默认 10 次序列、闭合延迟/过冲/停滞分辨率检查输出，无设备连接 | [force](../outputs/v2_opening_force_validation/force_dry_run.log)、[opening](../outputs/v2_opening_force_validation/opening_dry_run.log) |
| 本机全分辨率记录压力 | 周期隔离通过，容量失败，整体失败 | [native_load](../outputs/v2_validation/recording_load_review_01/native_load/report.json)、[slow_disk](../outputs/v2_validation/recording_load_review_01/slow_disk/report.json) |

先行全量候选日志为 [386 项通过](../outputs/v2_validation/all_tests_final_candidate.log)，之后增加配置绑定和持久化回归；最终计数仍以最后一轮日志为准。`recording.stress` 为纯合成压力参数，不绑定物理报告；传感器新鲜度 `recording.source_max_age_s` 等真实采集参数改变会使旧报告失效。

不同专项存在重叠，不能相加得到“总测试数”。早期失败日志保留作为排查记录，最终结论以对应最终日志为准。模拟报告不能通过真实报告登记或正式运行门禁。

## 2. 已覆盖的关键行为

- 抓取中心纯平移 G 的平移等价性、三点目标、倾斜时中心固定、侧抓/顶抓视觉轨道和新旧标签；G 的来源/形状与几何门禁。
- 无待做项在硬件加载前返回；首段 T_check 显示起终点及路径后要求 MOVE，取消未开始的动作不补发保持或开夹。
- 扣零要求开度、DD 空载和 100 个新源力帧；首次 EMPTY，过大波动/接触/来源过期拒绝。四个命令话题全部发布者查询失败或有外部发布者时拒绝。
- 控制器来源、单位和力定义显式区分；原控制器交接、来源目标/阈值、几何闭合下限、退出稳定及归属检查。验证报告缺 PASS、模拟、绑定变化或测量未达标不能授权正式扫描。
- 软件停止路径与定位工具中断共用；SIGKILL 使用无 ROS 假子进程验证记录逻辑，父进程不偷偷补发保持。真实最后目标证据缺失时拒绝生成有效停止结论。
- PROBE 连续稳定窗口/超时标记、HOLD 中段窗口、实际帧和力、信噪比/缺口；未知像素间距保留 px，未知 CoP 为 NaN。
- LOWER 原高度下限包含最终反馈，即使 settle_s=0 也不能漏检；空中错误保留恢复锁。T0 微调重新表达瓶体偏移，桌面瓶子在基座中保持固定。
- 非实验臂停放位/关节记录、CHECK 守卫、双臂图像关联；两腕固定参考和换瓶复位不更新原模板。
- 独立写盘的满队列、磁盘异常、写盘进程退出、未完成记录及恢复；窗口模式胶面图按最终描述子选择；DD 编码和来源保留。
- 多 session 分析按各自 T0 抓取中心偏移、合并与分组输出、关键配置不一致警告；完整/简化/视觉验收与源码哈希绑定，模拟来源拒绝。
- 离线 register-report 先验证全部报告，再只改两个报告路径；原快照备份、报告/配置 SHA 审计、progress/plan/commissioning 保持，验证期间报告变化及写失败不会默默授权。

## 3. smoke 和 dry-run 的具体范围

`v2_final_smoke` 覆盖 normal、no_contact、stall、force_timeout、dropped、slip，以及轻/重区块后续 normal，共 8 个接触案例；另有 6 个视觉点。结果表明故障分支保留预期标记，不能把所有 `completed` 解释成成功抓取。

分析生成 14 条汇总、22 条窗口描述子、1775 条位姿误差记录，`warnings=[]`。固定参考未改变，视觉分支未发闭合指令。该 smoke 使用合成传感数据和几何，验证软件流程，不验证真实机器人动力学或真实视觉精度。

三点 dry-run 中 +20 mm 深入点因几何/路径碰撞被剔除，零偏移和越盖点接受。其结论是检查确实拒绝了这个合成场景中的不可行目标，不是三点实机验收通过。输出同时包含每点抓取中心基座位置及相对 T0 中心偏移。

force/opening dry-run 的合成例子打印检测延迟 0.1 s、过冲估计 0.2 mm（上限 0.5 mm）、停滞窗口行程 0.6 mm（所需至少 0.25 mm）。这些值来自演示配置，不能抄入真实配置；真实参数必须使用实际 DD/力帧率与开度分辨率。

默认 10 次 force/opening 模拟使用固定模拟生产节拍，防止虚拟时钟瞬间生成大量数据；这不是控制周期隔离的证据。周期结论仅取下一节真实墙钟压力测量。

## 4. 记录压力：本机队列满，5090 短时通过但持续吞吐未证实

此测试使用合成计时循环与真正的异步记录 API，时间来自 `time.monotonic`。图像为与实机分辨率一致的 **1920×1080 随机合成画面**，预编码为 JPEG quality 95 后按两路 30 fps 提交；每帧 2,432,613 字节。并提交两指 96×128×3 的 float32 解码 DD 和 uint16 编码 DD（27 Hz），以及两指六维力（30 Hz）。它没有读取真实相机或控制机器人。

状态机/插值/力控三个合成计时循环设为 200 Hz，允许写盘负载增加的 p95 周期不超过 10 ms。有界队列 134,217,728 字节（128 MiB）。

| 测试 | 观测时长 / 注入写延迟 | 状态机 p95 增量 | 插值 p95 增量 | 力控 p95 增量 | 周期隔离 / 记录容量 / 整体 |
|---|---|---:|---:|---:|---|
| native_load | 2 s / 每消息 0.1 ms | 0.0269 ms | 0.0558 ms | 0.0098 ms | true / **false** / **false** |
| slow_disk | 1 s / 每消息 50 ms | 0.0169 ms | 0.0448 ms | 0.2260 ms | true / **false** / **false** |

两项均触发 `Recording byte queue full; acquisition incomplete, no blocking or silent drop`。队列峰值约 131.9 MB，下一条数据无法容纳时拒绝继续记录。报告中的已提交负载速率不是持续写盘吞吐；容量失败后不能拿它宣称“已支持双路 30 fps 全量记录”。

这份证据说明本次环境下队列满时没有用等待磁盘来拉长三个测试循环到配置限值之外；**它不证明当前配置能持续容纳这一级别的图像负载，也不证明实机 ROS 控制循环硬实时。** 原 existing 控制器内部同步 CSV 不在此测试范围。

现场正式长时间记录前，需在目标部署环境用实际图像文件大小、真实源帧率及目标记录模式重新测试并处理容量瓶颈。不能取消队列上限、隐藏错误或静默丢图来把报告改成通过；仅扩大缓冲区也不能证明持续吞吐成立。当前证据必须如实保留整体失败状态。

### 5090 同级负载补测

在远端单元测试和 smoke 结束后单独运行；配置除 duration_s=2、write_delay_s=0.0001 外沿用默认合成压力参数。报告为 loop_period_bound_passed=true、recording_capacity_passed=true、passed=true，495 条记录全部写完，manifest complete=true。提交延迟 p95=0.662 ms、max=0.993 ms，三个计时循环 p95 增量未超过 10 ms 限值。

峰值待写约 105.31 MB，2 秒生产结束仍积压约 104.87 MB，之后 close 正常冲刷。该结果只证明本次 2 秒负载可被缓冲并最终保存，**不能证明持续双路 30 fps 的稳态写入吞吐**，也不撤销本机失败证据。没有实际相机、ROS 或机器人调用。见 [报告](../outputs/v2_validation/remote_stress/report.json)、[日志](../outputs/v2_validation/remote_stress/record_stress_native_2s.log)、[配置](../outputs/v2_validation/remote_stress/record_stress_native_config.yaml)、[完成状态](../outputs/v2_validation/remote_stress/recording_status.json)。

## 5. 仍需操作者完成的实测与真实 PASS

| 待测/待验 | 当前要求 |
|---|---|
| T0、自由位、G 与出处、左臂停放、桌面/瓶体/夹爪几何 | 由你用定位工具、只读记录、CAD/卡尺取得；保留来源，不用合成示例替换 |
| a/c/l、两指/法向/面内映射、两腕安装与可见性 | 先实际确定 declared，再由 prepare 保存 verified 复核 |
| 开度映射、真实分辨率、DD/力帧率、扣零噪声、力/速度/停止限值 | 测量并填配置；必需 TODO/null 仍拒绝实机 |
| DD 像素间距与 N/CoP | SDK 单位可先使用；未知可选像素间距/CoP 明确 unavailable，使用 px 和 NaN，不虚构物理值 |
| 固定模板与阈值 | T_check 框两侧轮廓，沿 a 移瓶约 2 mm，分别取得两路阈值，复位后不更新初始模板 |
| **停止 PASS** | 操作者启动 stop-test 实机各 3 次，检查额外位移/最后目标，测量通过并输入 PASS |
| **力控 PASS** | 操作者在 T0 启动指定控制器默认 10 次，核对峰值/稳定/保持并输入 PASS；正式 contact 必需 |
| **commissioning PASS** | 首次完整三点；同硬件绑定的新 session 可规定的 lite，纯视觉可规定的前 8 点；先数据检查再人工 PASS |
| 开度重复性 | 首次实机单步默认 10 次，观察后 PASS；卡尺拟合仅保存建议，不自动改配置 |
| 实际记录容量 | 本机压力触发队列满，5090 2 秒通过但有积压；需按实际负载检验持续吞吐，不能称已就绪 |

三类正式门禁为停止、力控和 commissioning：正式 visual 不夹紧，仍需停止与对应 commissioning；正式 contact 需要全部三类。opening-repeat 的首次单步确认另外保留。任何 `SIMULATION_ONLY` 结果都不能替代上述真实证据。

重测用新报告文件，真实通过后用指南的 `register-report` 离线登记；它不会代替现场输入 PASS，不修改测量/几何/验收阈值，也不清除恢复锁。物理参数改变后按新配置、新 session 和当前绑定重新验收。

## 6. 最终本机验证结论

- 最终全量计数：`393`。
- 最终全量结论：`全部通过`。
- 专项、模拟及压力失败证据不随总测试通过而改写。实机参数与现场验收仍按第 5 节执行。

## 7. 5090 部署与复验

- 部署目录：`/home/dm/DM_das/bottle_grasp_experiment`；Lenovo 本机同版副本：`/home/lenovo/wfb/bottle_grasp_experiment`。源文件通过 Python 读取明文、SSH 传输并逐文件 SHA-256 校验，避免本机透明加密文件直接复制造成损坏。
- 首次替换前的旧版备份：`outputs/v2_deploy_backup/20261027_194031`。原 `single_gripper_force_control.py` SHA-256 保持 `e7e79669db47d0fbd7d0dea1e17229638ea7808cb0917bd3904bbcfc66ebdc18`；未修改机器人底层服务。
- 现有 `daimon_ros1_das` 容器中，以统一启动器调用 Python 3.12.13，**393 项测试全部通过**：[远端日志](../outputs/v2_delivery_20261027/remote_unit_tests.log)。
- 同一容器的纯模拟 smoke 完成 **8 个接触案例 + 6 个视觉点**，分析 14 条汇总、22 条描述子、1775 条位姿记录，warnings=[]；固定模板及视觉无闭合检查通过：[远端 acceptance](../outputs/v2_delivery_20261027/remote_smoke/acceptance.json)、[日志](../outputs/v2_delivery_20261027/remote_smoke.log)。
- 最后持久化修复后再次完成默认 10 次开度模拟，报告及清理均完整：[报告](../outputs/v2_opening_final/opening_report.json)、[日志](../outputs/v2_validation/final_opening.log)。

本次只更新项目文件并运行离线测试、dry-run、模拟和分析，没有执行实机运动或用模拟报告授权实机。

## 8. 后续现场诊断修正

首次真实 doctor 发现系统关节输出被误判为控制竞争；已按实际 ROS 图区分输入和系统输出，并补上 `/joint_control` 检查。原版本 393 项结果保留为该次交付证据；修正后的新测试和只读复查见 [发布者检查修正记录](OWNERSHIP_FIX_20261027.md)。

后续更新：两指 DD 缺失已定位并恢复，完整 doctor 现已通过，详见 [DD 服务恢复记录](DD_RECOVERY_20261027.md)。前文的失败状态为当时的诊断证据，保留以便追溯。

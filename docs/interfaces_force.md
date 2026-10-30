# 5090 法向力与夹爪接口只读审计

审计仓库：`6f5090:/home/dm/DM_das/starvla-feat-sparky_x5_starvla`。
只读检查了祖先目录、`scripts`、`src` 的 `AGENTS.md`，未发现适用文件；未启动、导入、运行机器人/夹爪/传感器程序，也未修改既有 controller 或 policy。

## 可由代码确认的接口

| 项目 | 证据（相对于仓库） | 结论 |
|---|---|---|
| 位置读/写 | `src/lerobot/robots/spark3_pose_gripper/schema.py:186`；`spark3_pose_gripper.py:58,83,90,97` | 每侧 ROS Float64：`/{left,right}_gripper_state` 和 `/{left,right}_gripper_command`。状态直接读取 `msg.data`；指令在通用驱动中被夹紧到配置范围。实验必须在调用前拒绝非法值，不能依赖底层夹紧。 |
| 归一化/counts | `scripts/single_gripper_force_control.py:19,20,1513,1735` | 状态×1000 为 counts，指令=counts/1000；增加张开，减小闭合。不是毫米。既有脚本写有操作员确认的 counts 下限 300，不能据此认定本实验物理 `w_min`。 |
| 速度、电流 | `spark3_pose_gripper.py:47–127` | 本适配层仅位置；未找到针对这只夹爪的速度/电流读写接口。不得用虚构值伪装实测，也不能把位置差分称为硬件实测速度。需要补齐驱动/topic、单位与失效行为。 |
| 独立已验证反馈脚本 | `scripts/single_gripper_force_control.py:1735–1751,1754–2022` | 软件死区 + 固定位置步进 + 等待新双指数据，不是 PID，也没有电机原生力模式。目标来自启动配置 `target_force`，没有公开的实时 set-target/enter/exit 服务。 |
| 独立脚本力定义 | `single_gripper_force_control.py:203–225,1567–1611` | SDK raw 模式 `mean(sign_i*Fz_i)`，不再扣零、不做 N/gf 换算。gf tag 模式 `mean(sign_i*(Fz_i-zero_i))*gf_scale`；这里每指阈值检查独立于均值。SDK 数值不能直接宣称 N。 |
| 发指令前重检 | `single_gripper_force_control.py:2094–2149,2199–2205` | 单一命令源锁和 ROS publisher 检查，检测位置变化、时效、边界后才发单步。锁只能协调此脚本，不能原子阻止其他程序。 |
| 生产集成控制器 | `shfile/infer_261011_right10_forcepolicy_impl.sh:965–1245` | 内嵌 `RightNormalForceController`，通过原 robot action 通道替换夹爪位置指令；启动参数 `RIGHT_FORCE_TARGET`；`update(latch,raw_observation,samples,force_reference)` 由 grasp latch 进入，unlatched 时 `reset()` 并返回位置指令。仍是软件一步一 count 的死区闭环。 |
| 集成力定义及接触丢失 | 上述 shell `:1044–1067,1227–1239` | 用经过该流静态/自动扣零后的 `sum(abs(Fz_left),abs(Fz_right))`；持续丢失会重启接触搜索。它与独立脚本均值/符号定义不等价，且自动重试与本实验 dropped/暂停需求不同，不能未经适配直接用于实验。 |
| ForcePolicy | `src/forcepolicy/runtime.py:65–82,131–184`；`inference.py:18–33,93–143`；`control_bridge.py:5–45` | 学习模型输出候选绝对力参考，不驱动硬件。`aggregate_nt` 是两指绝对 Fz 之和与切向模长之和；模型需要 checkpoint；limiter 做幅度/变化率约束。主任务 fixed 模式不应加载学习模型。可选 forcepolicy 需另外明确授权、模型与参考定义，不能假称已接通。 |

## 启动与退出副作用

- `spark3_pose_gripper.py:282–339` 的 `connect()` 注册机械臂/双夹爪 publisher、订阅器并连接相机；若 `trigger_initial_pose_service` 非空会调用初始化姿态服务（`:332,365–377`），可能运动。该值默认 `None`（`config_spark3_pose_gripper.py:38`），实机适配必须显式验证仍为空，禁止使用未知启动入口。
- `spark3_pose_gripper.py:599–631` 的 disconnect 仅注销 ROS/相机资源，不停止电机、不撤销既有目标。
- 独立脚本 `single_gripper_force_control.py:39–44,2000–2022` 在超时/异常/Ctrl+C 停止增加指令，但最后一个目标仍可能执行；不发送电机 stop、卸载、开爪或归零。退出程序不等于停止运动，`force target=0` 也不等于退出反馈控制。
- 集成控制器初始化可能将开度置为预设接近起点（shell `:1100–1117`），失败会打开回接近开度（`:1186–1198`），接触丢失会重启闭合；这些行为不能直接替代本实验“抬起后异常保持夹爪等待人”的规定。
- `src/forcepolicy/runtime.py` 本身 start/stop 只启动/停止推理线程；`inference.py` 加载 checkpoint 有模型反序列化，主任务无需也不应调用。

## 必须保持 TODO 的实机门禁

1. **物理开度 mm ↔ counts**：至少两个已测开度/counts 标定点，确认是否线性、机械死区和软下限；不得把 `w_min=直径−余量` 直接当 counts 或归一化值。
2. **N 单位/符号**：两指压缩符号与 Fz 对应、完整六轴坐标、静态零点、SDK 原始值到 N/N·mm 或 N·m 的依据。控制力究竟每指、均值、总和必须明确并贯穿 F_probe/F_target/F_max。
3. **力控制器公共生命周期**：保持既有算法/参数不改，但需要经人工验证的外部适配 API：设置目标、开始闭环、停止发力控位置命令、切回位置模式。并验证下一个位置命令不会与 controller 竞争。
4. **真实停止与保持**：明确机械臂 stop/halt 接口及完成确认；抬起后冻结夹爪当前状态的真实接口。单纯终止进程或 unregister 不能证明停止/保持。
5. **夹爪速度/电流**：真实状态/topic/API、单位、时间戳与关闭时反应。未知字段应 NaN + availability metadata；实机依赖若未验证则拒绝启动。

以上 TODO 未解决时，第一版只能执行离线模拟、规划和分析；用户亲自手动启动也不能绕过未完成的实机接口门禁。

## 离线分析约定

`analyze.py SESSION` 只读数据，不导入 ROS/机器人模块。读 `/streams/{dd_left,dd_right,wrench_left,wrench_right,tcp_command,tcp_measured}/` 的 `data,monotonic_ns,metadata_json`。试验阶段来自 `meta.json` 中 `stages=[{name,start_ns,end_ns}]`。参考来自 `session_meta.json.references=[{path,monotonic_ns}]`，NPZ 含 `left_dd,right_dd`。先按单调时钟选择绝对时间最近参考（等距选较早），PROBE 取阶段末帧，HOLD 取阶段时间中点最近帧；输出引用文件与帧时间便于追溯。

DD 中 depth 通道必须显式配置 `analysis.depth_channel` 或 `tactile.depth_channel`；`pixel_pitch_mm` 是 `[u,v]`，`u` 为列，`v` 为行，mm 原点在图像中心。正残差加权，负残差置零仅发生在分析副本，原始文件不变。物理面积、质心需要真实像素间距。`analysis.centroid_to_al.{left,right}` 是 2×2 矩阵，把 `[u_mm,v_mm]` 映射到 `[a_mm,l_mm]`，未知时拟合结果为 NaN。模板匹配像素偏移不是 DD 像素间距标定。

CoP 使用局部六维力 `Fx,Fy,Fz,Mx,My,Mz`，显式 `force_unit=N`、`moment_unit=Nm|Nmm`、`cop_plane_z_mm`。在作用面 z=s 下，`u=(s Fx−My)/Fz`、`v=(Mx+s Fy)/Fz`；未知单位/原点或法向力接近零时返回 NaN。它是作用面上合力作用点估计，非直接等同图像原点下质心；比较前还需坐标原点映射。

二维热图仅统计 `c=0` 且两倾角为零的平移试验；操作员 x 标记保留 CSV，排除比率分母和拟合；每格同时输出样本数。命令/实测位姿按“实测时间之前最近命令”配对，超过 `max_pose_command_age_s` 排除并记录配对数量，避免使用未来目标。

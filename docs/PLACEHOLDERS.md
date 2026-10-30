# V2 实测占位符登记表

现场顺序见 [完整操作指南](ADVANCED.md)。用 `move-tcp --real` 选方向/填距离，到位后用只读 `record-pose` 记录，再告诉我“这是占位符 1”，即可把实际记录替换到对应配置。不需要手动控制机器人走准 5 mm；prepare 会发送三个方向的 5 mm 检查位移。

实机使用 `config.ros.yaml`。TODO 和 null 仍阻止实机启动；演示值不属于测量。仅明确设计为可选的 DD 像素间距、未启用的 mm 漂移阈值、不可计算 CoP 接触平面可以使用 `unavailable`，不能把其它 TODO 改成这个词。

## 位置、几何与方向

| 编号 | 现场取得什么 | 配置或记录 |
|---|---|---|
| 1：理想抓取 T0 | 右夹爪已张开，瓶身在两指传感面之间、瓶子由桌面支撑，读取实际 TCP 完整 4×4 位姿 | `T0`；`measurements/T0_initial.json`；prepare 默认微调后的最终 T0 以 session_meta 为准 |
| 2：自由空间位 | 远离桌面、瓶子、左臂和障碍物；可做三个方向 +5 mm/返回以及配置方向 50 mm 停止测试 | `preparation.axis_test_pose`；`measurements/axis_test_pose.json` |
| 3：工作空间与物体几何 | 桌面 z、允许空间、瓶各段半径/轴向边界、瓶轴偏移、手指/手掌外形和软垫范围 | `safety.*`、`geometry.bottle.*`、`geometry.gripper.*` |
| 4：a/c/l 映射 | 在 TCP 中指尖方向 a、两指连线 c、另一个正交方向 l 的单位向量与符号；侧抓瓶轴 l，顶抓 a | `axes.*`、`hardware.axis_mapping_declared`；prepare 写 `axis_verified` |
| 5：两指与触觉映射 | 两指序列号/身份，法向轴/符号，DD 通道/阈值，u/v 到 a/l 方向，传感器到 TCP 旋转 | `tactile.*`、`analysis.centroid_to_al.*`、公开 ROS 接收转换；`sensor_mapping_declared` |
| 6：固定双腕模板 | T_check 两路包含瓶身左右轮廓边缘的 ROI；沿 a 移瓶约 2 mm 后每路位移和灵敏度；复位结果 | `check.*`；session 固定参考、position_calibration；`camera_mounts_declared` |
| 7：实验限值 | 实测允许速度/超时/力阈值/开度/偏移范围与姿态容差 | `motion.*`、`force.*`、`gripper.*`、`design.*` |
| 8：抓取中心 G | 两指传感面中心连线中点在 TCP 中的 xyz，单位 mm；CAD/URDF 版本或实测依据 | `tool.grasp_center_in_tcp` 为纯平移 4×4，旋转 I；`tool.grasp_center_evidence` |
| 9：左臂停放位 | 左腕可看到瓶子和右夹爪，左臂不占右臂路径；记录实际 TCP；prepare 再只读取得七关节角 | `measurements/parked_left.json`；session `nonexperiment_arm_park`；`park_guard` 容差默认 1 mm/0.5° |
| 10：高度交叉核对 | 最终 T0 抓取中心到桌面的卡尺高度，以及允许计算值/实测值差值 | prepare 现场输入；`preparation.height_crosscheck_tolerance_mm`、session `grasp_center_height` |

位姿交接至少包含：臂侧、来源时间和 topic/API、基座 frame、TCP 定义、平移单位、完整旋转表示、夹爪实际开度、瓶子类型及标记位置。`record-pose` JSON 已包含其中大部分，另外补充卡尺与摆放说明即可。归一化开度不是 mm。

瓶子和夹爪几何保留相对于原 T0 TCP 的定义：`center_offset_mm` 是瓶轴原点，分段上下界相对于该原点；手指、手掌偏移也是 TCP 坐标。G 用于运动的抓取中心和标签，不把现有碰撞几何自动平移到新原点。prepare 微调后会重表达瓶轴偏移，使瓶体在基座位置保持固定。

## V2 新增测量参数

| 组 | 要填或核实的值 | 依据/生成方式 |
|---|---|---|
| 扣零守卫 | `zero_guard.force_std_max`、`dd_contact_threshold`、`dd_contact_max_fraction`、`offset_delta_warn`；100 新帧及超时 | 真空载噪声、DD 背景、重连零点差值；首次现场输入 EMPTY |
| 闭合品质 | `close.observed_force_fps`、`observed_dd_fps`、`width_resolution_mm`、`max_overtravel_mm`；`detection_frames` 默认 2 | 真实帧时间戳/缺口、开度量化及允许行程；不用配置标称 FPS 代替测量 |
| 开度映射 | `hardware.gripper_width_calibration`，至少两点严格递增且覆盖命令范围 | 按现场既有夹爪方式取得归一化读数与卡尺 mm；opening-repeat 进一步测重复性 |
| 力控定义 | `force.controller`、`target_definition`、`unit`；接触、目标、容差、最大力；validation 的目标/峰值/稳定时间/波动限值 | 当前默认 bridge_loop + mean + sdk；不同信号来源的 SDK 不自动等价 |
| 停止测试 | `stop_test.direction_base`、触发时间、停稳时间窗、位置/角度噪声、观察超时、`max_extra_mm` | 自由位方向/余量；操作者允许的停止指标。默认路径 50 mm、每类 3 次 |
| 窗口取值 | `probe.settle_s`、`timeout_s`、`window_frames`、`hold.window_s` | 默认稳定 0.2 s、超时 3 s、5 帧、HOLD 中段 0.5 s；核对来源帧率是否支持 |
| 滑动阈值 | 未知间距用 `tactile.centroid_drift_px`；已知对应网格间距时可用 mm | 真实 DD 质心波动与允许漂移，不能直接沿用另一分辨率 |
| 卸载判断 | `lower.unload_ratio` 默认 0.2；`vertical_mapping_confirmed` | 竖直映射尚未确认保持 false，使用切向合力幅值；确认后须有坐标依据 |
| 图像/写盘 | 两腕真实 fps、`estimated_bytes_per_image`、`sample_hz`、`record.gel_raw`、有界 writer 配置 | 实测帧率/文件大小和独立墙钟压力测试；默认 raw=windows |

## SDK、像素和 CoP 的正确填写

- SDK 路线：`force.unit: sdk`，记录/图表注明 SDK 单位，不要求虚构 N 系数。兼容的旧 `_N` 配置键仍按 force.unit 解释；换 N 必须提供当前来源的转换和依据。
- DD 间距未知：`tactile.pixel_pitch_mm: unavailable`；`centroid_drift_mm: unavailable`；填写真实 px 漂移阈值。描述子使用像素、斜率 px/mm；知道对应当前处理后网格间距时才添加 mm 输出。
- CoP：力矩/接触面没有物理标定时 `moment_unit: sdk`、`cop_plane_z_mm: unavailable`，CoP 为 NaN。不要填 0 代替未知距离。
- 39030 来源：当前接受处理后 DD，服务名称、版本可用信息、分辨率、dtype、量化方式写入运行来源。服务未传版本时保留 unknown，不编造。

## 哪些是配置声明，哪些由程序验收

`axis_mapping_declared`、`sensor_mapping_declared`、`camera_mounts_declared` 表示你已经通过定位、按压和画面检查确定映射，配置中才能填 true。prepare 再次现场复核后写 `axis_verified`、`sensor_verified`、`camera_mounts_verified`。

stop-test、validate-force 的真实报告必须经你查看结果输入 PASS；三点验收生成摘要，数据通过后也需 PASS。不能把报告路径填进配置就认为已经通过，更不能手工修改 `commissioning_passed`。

停止/力控报告默认分别为 `outputs/validation/stop_report.json` 和 `outputs/validation/force_report.json`。这些路径不是物理未知量，不填 TODO；先准备 session，再生成报告，正式运行会检查真实性、确认和绑定。重测用新的 `--output` 路径保留旧证据；真实测量通过并输入 PASS 后，用 `register-report --session SESSION --stop REPORT / --force REPORT` 登记对应报告。该离线入口完整复用正式门禁，只变报告路径，备份原 snapshot 并追加路径/SHA 审计；不能用它修改测量或验收阈值。实际参数改变仍需新配置/新 session 及重新验收。

## 固定参考与换瓶

T_check 是最终 T0 沿局部 −a 后退默认 80 mm。两腕固定参考只在 session 开始建立。每次 RETREAT、人工复位或 light/heavy 切换都与它比较；复位后不更新参考。首次约 2 mm 标定沿 a 方向，分别记录两路灵敏度。

主动改变瓶位、重新记录 T0 或改变左臂停放位时，建立新 session 和参考。两瓶放同一标记位置、原参考能通过时，T0 和参考保持不变。触觉空载参考的周期更新不等于更新腕部模板。

## 建议的测量交接格式

```text
占位符编号：
测量文件/来源：
实测值：
单位、基座/TCP/传感器坐标系：
测量方法或 CAD/URDF 版本：
当前臂侧、瓶子和实际开度：
方向/符号的观察结果：
仍未知的字段：
```

位置记录只是完整参数的一部分。填完后按指南 dry-run → prepare → stop-test → validate-force → opening-repeat → 三点 PASS → 正式采集，不因定位工具已能发 TCP 就略过实验门禁。

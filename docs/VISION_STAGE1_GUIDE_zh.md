# 第一阶段：遥操作示教边界与视觉引导开爪定位

当前药瓶优先使用[无标记方案](MARKERLESS_BOTTLE_GUIDE_zh.md)。本页后续 ArUco 字段与标记安装说明仅适用于旧标记配置；示教和逐步开爪验收流程继续复用。无标记配置需要重新建立瓶子参考及标定绑定，不能沿用旧标记位姿哈希。

本阶段把“已标定图像 → 瓶子相对位置 → 示教范围内的开爪移动”接入现有软件保护和运动执行器。代码与离线测试不代表现场标定或验收已完成。

**本次开发没有连接真机、没有启动机器人服务、没有发布运动指令。** 下面带 `--real` 的操作是后续现场操作说明，并未在开发中执行。

## 1. 当前能力与边界

- 使用固定在瓶子非接触区域的、已知尺寸和 ID 的 **ArUco 标记**定位目标。这是标记辅助的瓶子定位，不是无标记瓶身识别，也没有新增深度相机。
- 已知标记到瓶子的刚体关系、相机内参和相机安装外参后，输出瓶子在机器人基座坐标系中的位姿；位置单位为毫米。
- 目标是恢复示教时“瓶子—张开夹爪”的相对位置。首版只执行平移，不根据视觉结果自动转动夹爪。目标姿态变化超过示教容差时拒绝动作。
- 每次只执行计划中的第一小步，随后重新取像、读取位姿、定位并规划。观察不到目标、时间过期、平面姿态歧义、路径越界或碰撞检查不通过时，不执行该步。
- 不夹紧、不轻触、不提起、不训练新的恢复模型。视觉对准结果不等于接触成功或技能部署验收通过。
- 原有 a/l 接触采集命令不自动改为视觉模式。新入口为 `vision-stage1`，经已有 `runtime.Runner` 执行；底层几何函数、健康反馈检查、命令所有权和现场配置检查继续复用。

## 2. 文件与数据关系

| 内容 | 作用 |
|---|---|
| `bottle_scan/vision_localization.py` | ArUco 检测、PnP 多解筛查、相机到基座的坐标变换、时间与观测有效性检查 |
| `bottle_scan/vision_workspace.py` | 生成带内容哈希的示教范围，沿用工具包络、桌面、瓶体和路径检查 |
| `bottle_scan/vision_motion.py` | 由新观测计算相对目标，检查整段拟议路径，每次执行一小步并记录实际位移 |
| `vision_stage1.py` / `bottle_scan/vision_stage1.py` | 纯离线生成/预览、只读示教、显式现场逐步验收入口 |

工作空间文件保存原站点几何配置及示教结果。示教范围只能缩小已有工作空间和单步限制，不能降低保守桌面边界；夹爪开度使用原先标定的全开宽度。文件包含内容哈希，改变数据后应重新生成版本，不应原地绕过校验。

相机标定文件保存内参、畸变、外参、标记尺寸和 ID、标记到瓶子的关系及证据说明。`evidence` 字段是标定记录，不是自动认可的真机证书；填写 `simulated: false` 也不能替代现场测量。

现场入口强制检查 `calibration.evidence.station_hardware_hash` 与当前已完成的站点文件 `hardware_hash` 一致。该绑定应在对应站点完成标定与独立检查后记录，不能仅复制哈希就认为相机已经校准。

示教文件中的 `teaching.reference_calibration_sha256` 必须绑定生成瓶子参考时的规范化标定内容。`teach` 自动填入；从保存记录离线 `build-workspace` 时，应在 `teaching` 或 `reference.constraints` 中保留这个原始字段。真实运动计划要求该值与当前标定完全一致，缺失或更换标定后继续使用旧参考都会被拒绝。调整相机外参、标记尺寸、标记安装变换、图像参数或标定记录后，应重新校验并生成相应版本和参考，不能只替换哈希。

## 3. 现场必须先完成的标定

### 3.1 站点、工具和机器人

继续完成原站点流程要求的机器人身份、TCP、夹爪开度映射、工具尺寸、桌面、命令所有权与停止/运动检查。视觉链路需要真正的毫米开度和工具包络；不能用归一化夹爪数值假装毫米。

确认现有几何模型中的手指、手掌及相机外形覆盖实际工具。旧 CAD 估计的夹持中心不能被称为实测结果；若其误差影响目标或包络，先补测。定位规划在保持夹爪姿态不变时，会使用 `tool.grasp_center_in_tcp` 补偿 TCP 到实际夹持中心的偏置，使小角度视觉变化不会直接把夹持中心带偏；这一补偿的准确性依赖该偏置的实际测量。原有软件保护不能证明全臂无碰撞或硬件级停止能力，现场仍需验证实际工作区域及控制器行为。

### 3.2 相机内参与安装外参

1. 在实际使用的分辨率、镜头设置下标定相机内参和畸变，保留原始标定图像、误差报告及配置版本。更换分辨率、裁剪、缩放或镜头设置后重新检查。
2. 腕部相机使用 `camera_mount: eye_in_hand` 和 `T_tcp_camera_mm`。明确 `mount_tcp_side` 是 `left` 还是 `right`，外参必须对应**该相机所在手臂的 TCP 定义**。不能把左腕相机的外参乘到右臂 TCP 上。
3. 本阶段真实入口只开放 `eye_in_hand`，按左右腕实时 TCP 计算相机位姿。`camera_mount: fixed` 和 `T_base_camera_mm` 仅用于离线几何验证；当前没有验收固定相机的现场绑定，腕部相机临时停住不能当固定相机使用。
4. 用未参与标定的位姿或已知目标进行独立检查，确认转换后的三维方向和尺度。不能只用同一组图像的重投影误差证明机器人坐标正确。
5. 对图像与位姿的时间关联进行检查。当前记录中的主机接收时间并不证明硬件曝光同步；首轮应在机器人停稳后取像，并现场量化相机延迟、位姿延迟与容许时间差。

变换记号 `T_A_B` 表示把 B 坐标中的点转换到 A 坐标中。腕部相机使用：

```text
T_base_camera = T_base_tcp_of_mounting_arm × T_tcp_camera
T_base_bottle = T_base_camera × T_camera_marker × T_marker_bottle
```

所有变换的平移分量和标记尺寸都使用 mm，旋转部分为正交旋转矩阵，最后一行为 `[0, 0, 0, 1]`。

### 3.3 标记、瓶子与目标关系

- 将标记刚性固定在不会接触夹爪、不会遮挡抓取区域的位置，打印后实测黑色标记方形边长，不能用纸张外框尺寸。
- 确认 ID 与字典一致；视野内不能有两个相同目标 ID。标记不能在瓶身上弯曲、松动或相对瓶子滑动。
- 定义瓶子坐标系并测量 `T_marker_bottle_mm`，保留测量依据。改变安装位置后更新版本。
- 从实际视角检验标记可见性。正视或远距离的小标记可能存在平面姿态歧义，程序会拒绝；应调整已允许的观察姿态或安装方式，而不是降低门禁来强行得到动作。

### 3.4 旧瓶体几何模型的参考关系

`reference_tcp_mm` 是这次示教的**张开、无接触、已通过几何检查的目标 TCP**；不是允许挤压瓶子的接触姿态。

`reference_bottle_pose_mm` 是同一瓶子状态下由视觉测得的瓶子位姿。

`geometry_reference_tcp_mm` 是旧几何模块构造瓶体时所用的参考 T0，在该 `reference_bottle_pose_mm` 下的对应基座位姿。它不一定等于当前目标 TCP，不能为了消除碰撞报错而随手填写。程序用以下关系更新碰撞模型：

```text
当前几何参考 = 当前视觉瓶子位姿 × inverse(示教时瓶子位姿) × 示教时几何参考
```

如果使用旧 session 的 T0，必须确认瓶子相对基座未发生变化，或者有已验证的变换将旧参考转换到本次状态。参考错位会同时影响目标与碰撞检查。现场 `teach --limits` 强制要求显式提供 `geometry_reference_tcp_mm`，不会从新的良好抓取位置推断旧几何参考。若初始参考 TCP 本身就是原 T0 且瓶子状态一致，可以明确使用相同矩阵；纯离线 API 的默认值不能替代这个确认。

## 4. 相机标定文件结构

下面是**待填写结构**，含 `TODO` 的文件会被拒绝，不能用于现场执行。每个矩阵都要填写为真实数字数组。程序只支持当前定义的针孔相机模型；畸变数组长度为 4/5/8/12/14，不能把鱼眼模型的参数直接填入。

```json
{
  "schema": "vision_calibration_v1",
  "unit": "mm",
  "camera_name": "right_wrist",
  "camera_mount": "eye_in_hand",
  "mount_tcp_side": "right",
  "image_size_wh": "TODO_实际宽高两个整数",
  "camera_matrix": "TODO_实测3x3内参矩阵",
  "distortion_coefficients": "TODO_匹配成像模型的畸变数组",
  "T_tcp_camera_mm": "TODO_实测4x4相机到对应TCP变换",
  "T_marker_bottle_mm": "TODO_实测4x4瓶子到标记变换",
  "marker_dictionary": "DICT_4X4_50",
  "marker_id": "TODO_实际打印ID整数",
  "marker_length_mm": "TODO_实测黑色方形边长",
  "max_reprojection_error_px": "TODO_由独立图像验证确定",
  "max_observation_age_s": "TODO_由静止取像与延迟验证确定",
  "max_pose_skew_s": "TODO_由图像位姿时间关联验证确定",
  "min_marker_area_px2": "TODO_由有效观测范围验证确定",
  "pose_ambiguity_ratio": 1.5,
  "pose_ambiguity_gap_px": 0.1,
  "min_depth_mm": "TODO_允许的最近标记深度",
  "max_depth_mm": "TODO_允许的最远标记深度",
  "simulated": false,
  "evidence": {
    "station_hardware_hash": "TODO_完成标定验证后绑定的当前真实站点hardware_hash",
    "intrinsics": "TODO_原始图像和标定报告路径/哈希",
    "extrinsics": "TODO_手眼测量及独立验证记录",
    "marker_to_bottle": "TODO_测量记录",
    "timing": "TODO_图像位姿时间验证记录"
  }
}
```

`pose_ambiguity_ratio` 和 `pose_ambiguity_gap_px` 上面的数值是代码默认候选筛查规则，**不是已验证的物理定位精度**。现场应报告使用这些规则后的拒绝率与独立定位误差。不要为了降低拒绝率无依据放松规则。

固定相机配置中删除 `mount_tcp_side` 和 `T_tcp_camera_mm`，改为实测 `T_base_camera_mm`。标定内容哈希由程序计算；如果保存了 `calibration_sha256`，它必须与规范化内容一致。

## 5. 运行流程

以下命令在 `/home/dm/DM_das/bottle_grasp_experiment` 中执行，所有输出使用新文件名，不覆盖旧版本。实际测量文件的路径由现场人员选择。

### 5.1 先离线检查与预览

```bash
.venv/bin/python vision_stage1.py --help
.venv/bin/python -m pytest tests/test_vision_localization.py tests/test_vision_workspace.py tests/test_vision_motion.py -q
```

从已保存的教学数据生成工作空间，以下两个输入选一种：

```bash
.venv/bin/python vision_stage1.py build-workspace \
  --config station_profile.yaml --teaching measurements/vision_teaching.json \
  --out outputs/vision_workspace_v1.json

.venv/bin/python vision_stage1.py build-workspace \
  --config station_profile.yaml --captures measurements/vision_captures.json \
  --out outputs/vision_workspace_from_captures_v1.json
```

配置可以是原几何配置，或者包含 `config` 的站点文件。`--captures` 使用五个唯一角色：`lower_corner`、`upper_corner`、`reference`、`observation`、`bottle_reference`。前四种记录包含 `tcp_mm`；瓶子记录包含 `pose_mm`。`reference.constraints` 包含下面介绍的非位姿约束。

已有图像与对应位姿可纯离线预览：

```bash
.venv/bin/python vision_stage1.py preview \
  --workspace outputs/vision_workspace_v1.json \
  --calibration measurements/vision_calibration_v1.json \
  --image measurements/vision_reference.png \
  --pose measurements/vision_reference_pose.json \
  --out outputs/vision_preview_v1.json
```

位姿文件字段为 `T_base_tcp_mm`、`current_tcp_mm`、`image_timestamp_ns`、`pose_timestamp_ns`、`now_ns`、`pose_tcp_side`。`T_base_tcp_mm` 对应相机安装所在手臂；`current_tcp_mm` 对应本次准备移动的机械臂。两者在双腕设置下可能不同。

离线 `now_ns` 应采用记录时同一单调时钟中的评估时刻，不能把历史图像重新标成“当前实时图像”。`preview` 只生成结果，不创建控制连接、不执行动作。模拟数据必须保持 `simulated: true`；不能改成 false 用于现场。

### 5.2 现场只读示教

准备 `limits` JSON，其非位姿字段包括：

```json
{
  "max_translation_step_mm": "TODO_不大于原站点单步限制",
  "orientation_tolerance_deg": "TODO_不大于原站点旋转限制",
  "max_total_translation_mm": "TODO_本轮累计平移预算且不小于单步",
  "table_z_mm": "TODO_不低于原站点保守桌面上界",
  "geometry_reference_tcp_mm": "TODO_与本次瓶子参考匹配的4x4旧几何T0",
  "clear_space_confirmed": true
}
```

`clear_space_confirmed` 只在你确认**整个区域内部**的条件后设为 true，不是模板应自动勾选的选项。全开宽度来自站点配置；上下边界由示教的完整工具包络加现有几何余量计算。

```bash
bash run_on_5090.sh vision-stage1 teach --real \
  --station station_profile.yaml \
  --calibration measurements/vision_calibration_v1.json \
  --limits measurements/vision_limits_v1.json \
  --out outputs/vision_workspace_live_v1.json
```

示教输出是 `--out` 指定的工作空间 JSON；图像和捕获记录保存在同名追加 `.evidence` 的目录中。后续现场 `commission --workspace` 应指向本次生成的 JSON，而不是上面离线示例的文件。

按提示遥操作到下界角、上界角、开爪参考位置和观察位置，全程保持夹爪完全张开，每次停止遥操作后读取实际反馈。程序对全部四个机器人示教位置检查全开开度，并比较每次捕获前后的实际位姿；捕获期间移动超过原位置/角度容差会拒绝该次示教。参考位置同步保存标记图像与位姿，计算瓶子参考。示教程序只读传感器，不替你移动、不闭爪、不初始化姿态。它不能通过两个角点证明内部无障碍。

若需要单独保存 TCP 测量，可以复用原有只读入口：

```bash
bash run_on_5090.sh record-pose --arm right \
  --label vision_geometry_reference --output measurements/vision_geometry_reference.json
```

该文件里的 `T_base_tcp_mm` 需要按目标字段语义使用，不能把记录文件本身当作教学矩阵直接传入。

### 5.3 现场逐步运动验收

仅在前述标定、原站点检查及下面验收准备完成后运行：

```bash
bash run_on_5090.sh vision-stage1 commission --real \
  --station station_profile.yaml \
  --workspace outputs/vision_workspace_live_v1.json \
  --calibration measurements/vision_calibration_v1.json \
  --out outputs/vision_commission_v1 \
  --max-steps 10 --tolerance-mm 1
```

`--tolerance-mm 1` 是命令参数示例，不表示当前相机或机器人已经具备 1 mm 的精度；应根据独立测量结果选择并记录可实现的阈值。

每步先显示下一步 TCP 目标和最终目标，再要求显式输入 `MOVE`。确认后重新取像、读取实际位姿并重新规划；机器人或视觉目标相对预览超出容差时取消运动。取像前、发指令前和运动过程中均检查另一只机械臂的停靠状态；已发布运动后继续检查开爪、接触、健康反馈和位姿跟踪。每次动作后必须使用晚于动作完成的新图像。在两个时间分离的新观测中达到视觉位置阈值后结束，不进入闭合或提起阶段。

`observation_tcp_mm` 是示教并验证的观察位置记录；第一阶段不会因为看不见目标就自主移动去找目标。需要改变观察位置时，由现场人员按既有流程完成，再重新取像。现场被拒绝的记录应保留，不能直接改 JSON 清除拒绝原因。

## 6. 现场验收：必须能证明什么

预先写下允许误差、试验点、重复次数和通过标准，再执行；不要看完结果后调整标准。建议按以下顺序验收并保留图像、实际位姿及输出报告。

| 验收项 | 操作与通过依据 |
|---|---|
| 坐标与尺度 | 在多个独立已知位置放置目标，比较视觉估计和独立测量；确认方向、左右腕对应及 mm 尺度，记录误差分布 |
| 相机安装与时间 | 保持瓶子不动、改变相机所在机械臂的允许位姿，瓶子基座估计应在预定误差内稳定；检查停稳取像时的时间差 |
| 多方向开爪定位 | 从不同允许起点对同一参考关系定位，验证前后、上下及组合偏差的实际运动方向；先小位移，再扩大到预定范围 |
| 固定世界边界 | 让目标落在示教空间之外，计划必须拒绝；目标移动不能带着工作空间一起移动 |
| 工具与路径 | 检查完整工具包络、桌面/瓶体与路径限制；对离线构造的“端点可行、中间碰撞”路径应拒绝。不要用实机碰撞来测试软件 |
| 观测失效 | 遮挡标记、换错 ID、出现重复目标 ID、读取过期记录时拒绝新动作。计划接受不表示可以忽略运动期间实际反馈 |
| 姿态与模型范围 | 瓶子姿态改变超过已允许范围时拒绝，不将角度误差偷偷变成平移或自主旋转 |
| 预算与结束 | 达到视觉误差阈值后停止；超过步数/累计位移预算、失败或反馈异常后退出并记录，不能把中途停止记成成功 |

视觉定位误差、机器人跟踪误差和目标允许误差应共同决定验收阈值。仅重投影误差小、离线测试通过、模板匹配成功或 `aligned` 标志为真，都不能单独证明接触技能可靠。

最终报告至少保留：站点/标定/工作空间版本与哈希、真实与模拟标记、成功与拒绝次数、各方向误差、实际运动回执、失效原因以及尚未完成的检查。完成这一阶段后，才进入视觉触觉联合采集、修正模型训练和 Agent 自主实验设计阶段。

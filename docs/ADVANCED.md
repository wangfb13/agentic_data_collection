# 药瓶抓取实验 V2：现场完整操作指南

本文按默认“右臂抓瓶、左臂停放后提供左腕视角”写。每个 `--real` 命令都由你在现场手动启动；没有 `--real` 的定位预览、dry-run、simulate、分析和备份不会让机器人运动。下面的数值位置只从你的实测记录取得，不提供可误当实机位置的演示坐标。

如果下一轮改由左臂抓瓶，需要单独核对 `hardware.arm_side`、TCP、夹爪/触觉端点和几何，重新生成配置与验收；不能只把某条定位命令改为 `--arm left` 就复用右臂实验参数。此时原右臂成为需停放的非实验臂。

全流程为：连接与只读检查 → 选择左臂停放位、右臂自由位和 T0 → 填实测配置 → dry-run → prepare → 停止测试 → 力控验证 → 开度重复性 → 三点单步验收 → 正式接触/视觉扫描 → 分析与备份。首次准备需要的尺寸、方向、开度和阈值登记在 [占位符表](PLACEHOLDERS.md)。

## 1. 打开正确终端，确认脚本环境

如果当前在 Lenovo 本机终端：

```bash
ssh 6f5090
cd /home/dm/DM_das/bottle_grasp_experiment
```

如果已经看到 `root@DM-26ZJ-017:/home/dm/DM_das/bottle_grasp_experiment#`，且在现有 ROS 容器中，就直接从下条命令开始。统一启动器自动辨认容器；容器里不需要 Docker 命令。

```bash
bash run_on_5090.sh doctor --config config.ros.yaml
```

这一步只读。检查终端和 `outputs/live_diagnostics/latest.json`：实际 TCP/开度状态是否新鲜，两路腕部图像、两指 DD/六维力是否可用，控制输入及系统关节输出有哪些发布者。

检查覆盖 `/wbc_goal_pose`、`/joint_control`、`/joint_commands`、`/left_gripper_command`、`/right_gripper_command`。`doctor` 只读，不创建运动发布者；正式实验连接后才发布 TCP/夹爪目标。无需为了让 doctor 通过再启动一个发布者。

当前系统中 `/daimon_hw_interface_node` 接收 `/wbc_goal_pose`，并在 `/joint_commands` 上发布关节数据给 `/wbc_state_publisher`。不要关闭这个底层节点。公开 ROS 配置的 `infrastructure_output_policy: daimon_public_ros_v1` 仅在每次读取的发布/订阅关系符合已核对结构时，将这一个“话题＋节点”标注为系统输出；完整发布者仍显示。它在其它命令输入上发消息、同话题出现未知发布者、订阅关系改变或查询失败，仍拒绝启动。通用模板与模拟配置保持 `strict`。

其它遥操、推理或夹爪控制进程若占用命令输入，先在你原来的终端正常结束，再开始实验。程序不会替你杀掉进程；也不要把系统节点加入任意白名单。`doctor` 的通过只证明本次来源检查，不能代替标定与运动验收。

现场此时应为：瓶子放在桌面的标记位置；夹爪空载，按你已有的夹爪操作方式张开。`move-tcp` 本身不控制夹爪。后续软件扣零要求它已经处于实测 `w_open`，不能靠扣零步骤自动打开一个未知状态的夹爪。

### 1.1 如果六维力正常，但两指 DD 没有数据

先看 doctor 的 `checks.dd`。如果两指 `dd_count=0`，同时报 `dd_left, dd_right` 缺失，核对右 X5 的触觉服务。2026-10-27 本次现场问题是旧 `demo_all_sensors_broadcast_server_with_wrench.py` 重新运行，它只提供 39010/39020；已有的 `starvla-tactile-dd.service` 未启动。现已恢复，完整 doctor 通过，见 [DD 恢复记录](DD_RECOVERY_20261027.md)。

以后重新启动设备时，触觉启动项应使用带 DD 的服务。不要再启动旧的不带 DD 的程序，也不能让两个程序同时占用同一组传感器。若旧程序已运行，先在其原触觉服务终端用 Ctrl-C 正常结束；保留腕部相机和机器人底层控制节点。确认两指胶面空载，再从 5090 宿主机终端登录右 X5：

```bash
ssh cftc@192.168.9.11
systemctl --user start starvla-tactile-dd.service
systemctl --user status starvla-tactile-dd.service --no-pager
```

这组命令只启动触觉数据服务。若提示端口占用，不要继续启动第二个 SDK owner；先确认旧触觉进程已退出。状态应为 active/running，触觉端口应包含 39010、39020、39030。输入 `exit` 返回 5090，再回容器运行原 doctor 命令。

DD 服务当前为手动启动；启动时会固定新的胶面参考。因此不要在试验进行中切换服务。服务重启后用新 session 执行 prepare，重新记录空载参考与软件零点，不能直接沿用旧 session 的传感器参考。未实测参数和现场验收仍按后续步骤完成。

`move-tcp` 的实测到位容差默认为 2 mm / 1°，控制器判定与最终检查使用同一组数值。可用 `--pose-tolerance-mm` 和 `--pose-tolerance-deg` 显式调整；这两个参数不改变目标位姿，也不改变现场确认、运动距离限制或速度。移动结束会打印实测误差，日志保留所用容差。

## 2. 用选方向/填数定位，不需要手动走准 5 mm

先熟悉只读菜单：

```bash
bash run_on_5090.sh move-tcp --config config.ros.yaml --arm right
```

菜单中的 `b:x+`/`b:z-` 等是基座方向，`t:x+`/`t:z-` 等是当前 TCP 方向；选择后输入距离，单位 mm。`xyz` 输入基座坐标的 X Y Z，保持姿态；`file` 载入已记录位姿；`r` 刷新，`q` 退出。默认只显示目标，不发运动。

由你准备执行时，加 `--real`：

```bash
bash run_on_5090.sh move-tcp --config config.ros.yaml --arm right --real
```

例如你已确认基座 +Z 是安全的上方，可选 `b:z+`，填 `5`。程序显示实际起点和目标，输入 `MOVE` 后才向公开 ROS 发布 TCP 目标，由底层运控解析执行。也可使用单次命令：

```bash
bash run_on_5090.sh move-tcp --config config.ros.yaml --arm right --direction z+ --distance-mm 5 --frame base --real
```

`a/c/l` 是你配置的抓取方向，不自动等于 TCP 的 x/y/z。先确定映射再选择方向。此独立定位工具限制单次距离、速度和反馈新鲜度，但不规划整臂绕障；你需要确认整段直线路径。目标过远或中间有障碍时分段定位，不通过扩大限值来跨过障碍。每次实际到位的证据保存在 `outputs/tcp_moves/`，不能只凭“命令已发送”判定到位。

### 2.1 左臂停放位

只在任何实验程序都退出后，操作左臂：

```bash
bash run_on_5090.sh move-tcp --config config.ros.yaml --arm left --real
```

把左腕相机移到能看清瓶身及右夹爪的位置，同时给右臂全流程留出空间。用 `q` 退出，记录当前位置：

```bash
bash run_on_5090.sh record-pose --arm left --label parked_left --output measurements/parked_left.json
```

左腕相机确实在左臂上；它在本实验中兼作固定外部视角，用于事后回看倾倒、推瓶、滑动。prepare 还会从实时反馈正式记录左 TCP 和七个关节角；这份 JSON 是前期交接记录。之后左臂保持不动。默认右臂实验程序不会下发左臂运动，发现停放位变化超过 1 mm / 0.5° 就停止。

### 2.2 右臂自由空间位：占位符 2

用右臂 `move-tcp --real` 到远离桌面、瓶子、左臂和其它障碍物的自由空位。这里需容纳准备时的三个方向各 +5 mm 及返回，也需容纳停止测试配置方向的 50 mm 运动及可能的停止余量。记录：

```bash
bash run_on_5090.sh record-pose --arm right --label placeholder_2_free --output measurements/axis_test_pose.json
```

将 JSON 内 `T_base_tcp_mm` 的完整 4×4 矩阵填入 `preparation.axis_test_pose`。这不是 T0。不要自己手动移动三次 5 mm；prepare 会自动发送这些 TCP 小位移，每次由你观察方向后输入 `yes`。

### 2.3 理想抓取位 T0：占位符 1

保持右夹爪张开，用方向菜单或 XYZ 定位，让瓶身目标高度位于两指传感面之间、瓶子仍由桌面支撑，形成你希望作为零偏移的理想抓取状态。确认手掌和手指外壳不碰瓶。退出定位工具，记录：

```bash
bash run_on_5090.sh record-pose --arm right --label placeholder_1_T0 --output measurements/T0_initial.json
```

将 `T_base_tcp_mm` 填入 `T0`。有卡尺实际开度时，可在记录命令加 `--width-mm 实测数字`；这个参数只是登记人工测量，不会打开夹爪。位姿在基座 `base_link` 中，平移单位 mm，TCP 为实际 `right_gripper_base_link`。不要把末端法兰位置直接当成此 TCP。

## 3. 填实测配置并固定本轮的定义

在 5090 项目目录编辑 `config.ros.yaml`，保留自己每次实测的依据；可用 `nano config.ros.yaml`。未测量值继续保留 TODO。若要我替换值，提供“占位符编号 + JSON 路径 + 物理测量与单位”，不要只报一个无坐标系的三元组。

### 3.1 G、桌面、瓶子和夹爪几何

`tool.grasp_center_in_tcp` 是 TCP 到抓取中心的固定变换 G。2026-10-28 已按你的确认，在 `config.ros.yaml` 中使用**两指灰色指尖短段中心的中点**作初始近似：TCP 坐标 `[0, 0, 117]` mm，旋转块保持单位阵。这里的 +Z 是 `right_gripper_base_link` 的 Z，不能当成基座竖直方向；TCP 原点及记录的 T0 均不因此改变。这个值现在不需要你再量。

模型推导原值为 116.9894 mm，按近似位置取整为 117 mm。配置保留 `grasp_center_approximate: true`、`grasp_center_measured: false` 和来源，不将其称为已实测胶面中心。详细计算见 [抓取中心近似说明](GRASP_CENTER_APPROXIMATION_20261028.md)。原图中心用于触觉图像居中；真实传感器的凸起仍未包含在旧 CAD 中，碰撞包络不会由此自动变为已确认。

以后若得到更准确的安装位置，只需将对应 TCP 下的 x/y/z 填入 G 的最后一列并更新依据；保持单位 mm、旋转块为单位阵。若 CAD 给的是带旋转的传感器坐标系，先把中心转换到 TCP，不直接填整套传感器 frame。

接触目标采用 `T0·G·Trans(δ)·Rot(θ)·G⁻¹`，纯平移与原公式一致，倾斜围绕抓取中心。视觉的瓶子坐标系为 `B=T0·G`；侧抓瓶轴沿 l、顶抓沿 a。新视觉标签 `inv(T_measured·G)·B` 与旧标签 `inv(T_measured)·T0` 同时保存。

需要实测：桌面基座 z、工作空间、瓶身/瓶颈/瓶盖各段半径和轴向上下界、瓶轴相对原 T0 TCP 的偏移、夹爪指体/手掌外形及偏移、软垫可接触范围。几何仍以实际 TCP 定义，不因 G 改变而迁移已有碰撞模型。prepare 微调 T0 后会将瓶轴偏移重表达到新 T0，保持桌面瓶体在基座中的实际位置不变。

名义部位的计算为：把目标抓取中心变换到原 T0 TCP 下，减去 `geometry.bottle.center_offset_mm`，投影到瓶轴，再和 body/neck/cap 的 `axial_min_mm`、`axial_max_mm` 比较，范围外为 empty。该分类是计划标签，不是视觉识别结论。

### 3.2 力、开度、DD 和缺标定项

当前配置明确使用 `force.controller: bridge_loop`、`force.target_definition: mean`、`force.unit: sdk`。这表示使用现有 39020 信号、软件扣零后的 SDK 单位，目标按两指均值计算；每指、均值、总和仍全部记录。接触阈值与 F_max 逐指检查。所有力阈值、目标和验收限值必须在这个实际信号定义下测定。兼容旧配置的 `_N` 键名不意味着当前数值已标定为 N，实际单位以 `force.unit` 为准。

若改为 `unit: N`，需要对应来源的换算系数及依据，同时保持 `units.force`、`tactile.force_unit` 和接收转换配置一致。不要把某个旧脚本的 gf 估计或另一来源 SDK 数值直接作为本信号的 N。

开度需要至少两个实测“归一化读数—mm”点，覆盖将要命令的范围，填入 `hardware.gripper_width_calibration`；填 `gripper.open_mm`、`min_mm`、容差、闭合速度、开度分辨率。开度重复性测试稍后验证这个设置，不替代首次运行前必需的基本物理标定。

没有 DD 网格物理像素间距时，将 `tactile.pixel_pitch_mm` 明确登记为 `unavailable`，使用实测 `tactile.centroid_drift_px`。不使用的 `centroid_drift_mm` 可同样登记 `unavailable`；以后得到对应当前 96×128 网格的间距再填写 `[u_mm_per_px, v_mm_per_px]` 和 mm 漂移阈值。未标定力矩/接触平面时 `cop_plane_z_mm: unavailable`，CoP 输出 NaN。这里仅表示相应可选物理量不可计算，不能把其它必需实测字段改成 unavailable 绕过门禁。

DD、按压方向、六轴法向符号及两指对应仍需实测。当前 39030 是已有服务缩放、量化后的数据，保存编码数组、解码值和来源；不是未处理 SDK 原始浮点 DD。`record.gel_raw` 默认 `windows` 保存 PROBE/HOLD 窗口内 39010 胶面原图，也可选 `off` 或 `full`。改变记录方式会影响存储量和硬件配置哈希。

### 3.3 必需的参数和声明

按 [占位符表](PLACEHOLDERS.md) 填完以下组：速度/超时/路径容差、力限值、开度与分辨率、实测 DD/力帧率、扣零噪声限值、停止测试方向与停止余量、模板搜索半径/质量阈值、实验设计范围和预计存储量。

闭合检查使用较慢的 DD/力实测帧率：`检测延迟 = close.detection_frames / min(DD_fps, force_fps)`，默认两帧；`过冲行程估计 = 闭合速度 × 检测延迟` 不得超过 `close.max_overtravel_mm`。同时 `闭合速度 × 停滞窗口 ≥ 5 × 开度分辨率`。若失败，应核实测量并调整可验证的闭合参数，不能跳过检查。

三个配置声明只有做过前期观察后才能设为 true：

| 配置项 | 你已经实际确定的内容 | prepare 保存的现场复核 |
|---|---|---|
| `hardware.axis_mapping_declared` | 用 move-tcp 确定 a/c/l 方向与符号 | `axis_verified`、逐轴观察记录 |
| `hardware.sensor_mapping_declared` | 手动按压确定两指身份和面内方向 | `sensor_verified`、DD 质心观察 |
| `hardware.camera_mounts_declared` | 两路腕部安装和可重复观测关系明确 | `camera_mounts_verified`、模板及标定证据 |

配置中的声明和 session 的验证结果不同；不要手工把 session 的 verified/PASS 字段改成 true。`check.threshold_px` 在准备前可保持 `calibrate_at_session_start`，prepare 会写入两路实测标定阈值；其它 TODO 仍拒绝实机。

## 4. 先生成 dry-run，核对计划和路径

```bash
bash run_on_5090.sh prepare --config config.ros.yaml --dry-run
bash run_on_5090.sh contact --config config.ros.yaml --dry-run --commissioning --output outputs/preflight_v2_three_points
bash run_on_5090.sh contact --config config.ros.yaml --dry-run --output outputs/preflight_v2_contact
bash run_on_5090.sh visual --config config.ros.yaml --dry-run --output outputs/preflight_v2_visual
bash run_on_5090.sh stop-test --config config.ros.yaml --dry-run
bash run_on_5090.sh validate-force --config config.ros.yaml --dry-run
```

先把 TODO 列表补齐，再看计划中的 T0/T_check、抓取中心位置与相对偏移、碰撞/路径拒绝原因、闭合过冲和停滞分辨率、总时长与存储量。被拒绝的点不裁剪到边界，也不靠减少安全余量强行通过。三点中的任一点被拒绝，就不能把另两点当成完整三点验收。

`--demo`/`--simulate` 使用独立合成几何。想先熟悉交互之外的数据流程，可运行 README 的演练命令；不要给实机命令加这两个参数。

## 5. 建立第一个真实 session：prepare

现场开始位置：左臂已停放且退出定位程序；右臂在记录的理想 T0；右夹爪已张开至实测 w_open，传感面不碰瓶，瓶子在原标记位置。选新的 session 名称，已有目录不复用。

```bash
bash run_on_5090.sh prepare --config config.ros.yaml --real --single-step --session outputs/real_001
```

程序的实际顺序如下，按终端提示逐项做：

1. 先读取当前实际 TCP 为本 session T0；检查开度和新鲜度。首次没有空载参考时，确认传感面确实空载后输入 `EMPTY`。随后取得至少 100 个新的独立力帧，检查波动后才应用软件扣零；已有空载参考的重连还会检查 DD 无接触印迹。零点和相对上次变化写入 `session_meta.json/zero_runs`。
2. 只读记录非实验左臂的 TCP 与关节角。确认左视角和右臂运动空间后按提示回车。
3. 显示当前位姿 → T_check 的起终点和路径检查。输入 `MOVE` 才开始这一段；即使关闭单步也必须确认。T_check 是 T0 沿 −a 后退默认 80 mm、姿态不变。
4. 默认回到 T0 进行轻触微调。每轮 CLOSE/PROBE 后显示两指质心残差，先松开再允许位移；输入 `a+`、`a-`、`l+`、`l-` 每次移动 1 mm。回车接受当前 T0，不需要你手动走准 1 mm。明确要跳过时，在启动命令加 `--no-tune`。残差保存 px；已知间距时同时保存 mm。
5. 程序显示由 `T0·G` 与桌面 z 算得的抓取中心高度。按卡尺实测输入 mm，差值超过配置容差会警告；先检查 G、桌面 z 和所用 TCP，不能把卡尺值填成计算值只为消除警告。
6. 张开夹爪，到 `axis_test_pose`。自动沿 +a、+l、+c 各走 5 mm，并逐次返回自由位；方向确实符合定义才输入 `yes`。你只观察确认，不用手动完成这些精确小位移。
7. 在自由位按提示分别按压两指的指尖侧和另一个面内侧，读取 DD 质心，确认方向后输入 `yes`，每次移开手再继续。
8. 回到 T_check，采集配置时长的空载参考，默认 10 s；保存两路原尺寸腕部参考图，进入 ROI 框选。框要包含瓶身左右轮廓边缘，不能只框标签纹理。
9. 保持瓶子不动拍重复图估计噪声；按提示把瓶子**沿接近方向 a** 故意移动约 2 mm。放稳后回车，两路分别显示像素偏移及 px/mm 灵敏度，设置各自阈值。
10. 把瓶子放回最初标记位置，回车与最初参考比较。任一路不通过就继续复位、复查，直到通过；参考图不更新。准备完成时右臂停在 T_check、夹爪张开，左臂仍为停放位。

如果在 SSH 中没有 OpenCV 图形窗口，用预先指定的 ROI 文件方式：

```bash
bash run_on_5090.sh prepare --config config.ros.yaml --real --single-step --session outputs/real_001 --rois-json measurements/real_001_rois.json
```

文件可暂不存在。程序保存本次原尺寸图到 `outputs/real_001/references/bottle_position_preview/` 后会等待；在另一终端把两张 PNG 下载到本机，查看原尺寸像素，再创建 JSON。格式为每路 `[左上角x, 左上角y, 宽, 高]`，整数像素，坐标必须来自本次原图。文件内容应为如下结构，其中说明文字须换成实际数值后才是有效 JSON：

```text
{"left_wrist": [实际x, 实际y, 实际宽, 实际高],
 "right_wrist": [实际x, 实际y, 实际宽, 实际高]}
```

创建完成后回到等待终端按回车。不能用任意示例 ROI 或缩小预览图上的坐标。

### 5.1 保存微调后的 T0，供下一步定位

prepare 会改变 session 的 T0，不能继续使用微调前 `measurements/T0_initial.json` 返回力控位置。下面只读取 session 并导出一个 move-tcp 可载入的位姿文件，不发送运动：

```bash
bash run_on_5090.sh python - <<'PY'
import json
from pathlib import Path
import yaml
session = Path('outputs/real_001')
meta = json.loads((session / 'session_meta.json').read_text())
cfg = yaml.safe_load((session / 'config_snapshot.yaml').read_text())
side = cfg['hardware']['arm_side']
pose = {'arm': side, 'base_frame': 'base_link',
        'tcp_frame': side + '_gripper_base_link', 'translation_unit': 'mm',
        'T_base_tcp_mm': meta['T0'], 'source': str(session / 'session_meta.json')}
path = Path('measurements/real_001_T0.json')
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('x') as stream:
    json.dump(pose, stream, indent=2)
print(path)
PY
```

之后 session 自带的 `config_snapshot.yaml` 是这组实验的配置，T0 和像素阈值已经更新。修改根目录 YAML 不会改动已建 session；不要直接改 snapshot 绕过已有计划或报告绑定。

## 6. stop-test：自由空间停止验收

开始前瓶子仍在桌面标记处、左臂不动、右夹爪张开且空载。用已记录的自由位定位；每次确认路径，长路径分段处理：

```bash
bash run_on_5090.sh move-tcp --config config.ros.yaml --arm right --pose-file measurements/axis_test_pose.json --real
bash run_on_5090.sh stop-test --config outputs/real_001/config_snapshot.yaml --dry-run
bash run_on_5090.sh stop-test --config outputs/real_001/config_snapshot.yaml --real
```

默认报告写 `outputs/validation/stop_report.json`，由 `stop_test.report_path` 配置。也可以在 prepare 前为这一轮配置唯一报告路径。已有报告绝不覆盖；再次验收用新的 `--output` 路径保存证据，通过后按第 7.2 节用 `register-report` 离线登记，不直接编辑 session 快照。

程序测试“配置实验速度、配置方向的 50 mm 自由空间运动”，软件停止 3 次、真实 SIGKILL 子进程 3 次。每一次都会显示当前起点和目标，输入 `MOVE` 才执行。报告记录请求停止/停稳时间、额外位移；SIGKILL 还核对最后实际发布的目标和最终位置。

每次停下后程序**不自动返回**。按提示，在第二个 5090 终端用同一条 move-tcp 位姿文件命令回到 `axis_test_pose`，退出定位工具，再回测试终端按回车。SIGKILL 阶段父进程只读观测，不偷偷发送保持命令来改变测量结果。停止或观测行为异常时按现场底层停止/急停流程处理，不能继续按 MOVE。

六次都满足配置限值后，核对报告并输入 `PASS`。它只证明本次自由位、方向、速度下的结果，不证明全工作空间硬件停止能力。正式 contact/visual 要求真实、当前配置匹配且经你确认的停止报告；模拟报告不能使用。

结束位置是最后一次测试的实际停住处。下一步请自行用 move-tcp 回到微调后的 T0。

## 7. validate-force：T0 力控验证

```bash
bash run_on_5090.sh move-tcp --config config.ros.yaml --arm right --pose-file measurements/real_001_T0.json --real
bash run_on_5090.sh validate-force --session outputs/real_001 --dry-run
bash run_on_5090.sh validate-force --session outputs/real_001 --real
```

现场必须是：右夹爪张开、两指围着瓶身，实际 TCP 到达 session T0，瓶子仍在桌面且对准标记，左臂未动。此工具不移动任何机械臂，也不提瓶。若当前位置不等于 T0，会要求你先定位。

扣零检查通过后输入 `MOVE`，按单步提示执行默认 10 次 `OPEN → CLOSE → PROBE → SQUEEZE → HOLD → RELEASE`。每次记录接触后的每指力峰值、进入目标力带时间、稳态均值与波动、最大力。全部满足配置的验收限值后，核对结果输入 `PASS`。正常结束仍在 T0、夹爪张开。

默认报告 `outputs/validation/force_report.json`，旁边有同名 CSV 及 `_data/` 原始记录。正式 contact 要求所选控制器、单位、来源和当前配置匹配的真实 PASS 报告。报告已存在时程序保留旧证据并拒绝覆盖；重测使用新的 `--output` 路径，并按第 7.2 节登记通过的报告。

### 7.1 需要与原有控制器对比时

`force.controller: existing` 是显式可选路线。它调用原 `single_gripper_force_control.py`，不修改原文件。原控制器使用 39010 胶面原图经其 SDK/保存参考计算；当前 bridge_loop 使用 39020 六维力。两者的 SDK 数值不能自动等价。

仅在 `config.existing.example.yaml` 所列宿主监督器命令、原脚本/标定 SHA、原参考、允许位置和退出稳定检查都已配置并验证后，另建 existing 配置和 prepared session，再运行对应 validate-force。`force.existing.control_limits` 的最大力、接触和目标容差按 39010 控制来源另行测定；全局力限值/接触/容差仍守护 39020 评估来源；SQUEEZE 反馈目标是 `force.validation.evaluation_target`，PROBE 反馈目标是另行实测的 `force.existing.evaluation_probe_target`。命令目标仍按原控制器的 39010 定义，不能把两者 SDK 幅值互相代用。原控制器目标通过完整启动/退出/确认交接切换，退出本身不能被当作已经停止或张开。未部署完整 existing 路线时继续使用明确配置的 bridge_loop，不自动回退。

两份已完成报告可以离线对比，不连接机器人：

```bash
bash run_on_5090.sh validate-force --compare outputs/validation/force_report.json outputs/validation/force_existing_report.json --output outputs/validation/force_comparison.csv
```

按 evaluation_signal、unit 和目标定义解释比较结果。原控制器内部同步 CSV 未改动，独立写盘进程的压力测试不能替它证明控制周期。

### 7.2 报告失败或需要重测：保存新文件，再离线登记

先结束采集、确认 session 没有活动/未完成试验。旧报告及原始记录都保留；需要重测哪一项就执行对应一项，不必重复另一项。测试仍必须由你现场启动，位置要求不变。

停止测试先回自由位，报告文件名用未使用过的新名字：

```bash
bash run_on_5090.sh move-tcp --config config.ros.yaml --arm right --pose-file measurements/axis_test_pose.json --real
bash run_on_5090.sh stop-test --config outputs/real_001/config_snapshot.yaml --real --output outputs/validation/real_001_stop_retry_01.json
```

力控测试先回 session 实际 T0：

```bash
bash run_on_5090.sh move-tcp --config config.ros.yaml --arm right --pose-file measurements/real_001_T0.json --real
bash run_on_5090.sh validate-force --session outputs/real_001 --real --output outputs/validation/real_001_force_retry_01.json
```

对应测试测量合格、你已输入 PASS 后，再运行下列**离线登记**。只重测一项时只保留该项参数：

```bash
bash run_on_5090.sh register-report --session outputs/real_001 --stop outputs/validation/real_001_stop_retry_01.json --force outputs/validation/real_001_force_retry_01.json
```

登记入口不连接 ROS、不移动机器人。它复用正式运行的完整报告校验，拒绝模拟、未确认、测量失败、过期源码/配置绑定或不同控制器报告；多个报告全部通过后才更新 locator。它只修改 session 快照中的 `stop_test.report_path`、`force.validation.report_path`，保留原配置逐字节备份到 `validation_registration_backups/`，在 meta 追加旧/新路径、报告 SHA、配置 SHA 和审计状态。硬件、测量、几何、计划和已有 commissioning PASS 不变。

这是更新已验证报告路径的专用入口，不是重新认可测量的开关。若要调整验收阈值、速度、控制器或实际标定，先改真实配置并建立新 session、重新做相应验收；不能把新报告登记到参数不匹配的旧 session。活动/未完成试验会阻止登记，需先按第 12 节完成或处理恢复。

## 8. opening-repeat：接触开度重复性

仍在微调后的 T0，瓶子在桌面，夹爪张开。首次实机运行必须单步：

```bash
bash run_on_5090.sh opening-repeat --session outputs/real_001 --dry-run
bash run_on_5090.sh opening-repeat --session outputs/real_001 --real --single-step
```

默认 10 次，使用与扫描相同的 CLOSE+PROBE，然后松开；不提瓶、不移动臂。输出每指接触归一化开度、分布、来源时间及最小非零相邻反馈差 `observed_min_nonzero_delta_normalized`。这个差值是本次观察到的分辨率估计，不等于已证明硬件编码器分辨率。结果与 `close.width_resolution_mm` 及已有 mm 映射交叉核对。若测得新映射/限值需要更改配置，应新建 session，并让报告和验证重新匹配，不能继续沿用旧哈希。

可在停止实验、按现场方法取得卡尺对照后，将 `normalized,width_mm` 两列实测值存成 CSV。离线拟合命令：

```bash
bash run_on_5090.sh opening-repeat --manual-pairs measurements/opening_pairs.csv --output outputs/validation/opening_fit.json
```

不要填演示值；拟合不会自动改硬件配置。报告默认路径为 `outputs/validation/opening_report.json`。看完完整摘要输入 `PASS`；该 session 首次成功单步验收后，配置未变的后续重复才允许省略 `--single-step`。异常、中断或模拟记录不能解除首次单步要求。

## 9. 首次三点单步验收：看到摘要后输入 PASS

运行前夹爪张开、空载，瓶子在原标记处，左臂在原停放位。程序会先做扣零、只读停放位检查，然后显示当前位置 → T_check 的路径；输入 `MOVE` 后再进入实验流程。

```bash
bash run_on_5090.sh contact --dry-run --session outputs/real_001 --commissioning
bash run_on_5090.sh contact --real --session outputs/real_001 --commissioning --single-step
```

三点固定为零偏移、沿 +a 深入、越过瓶盖。观察每个阶段实际动作是否和计划一致，特别是越过瓶盖点应进入无双指接触分支，不能提瓶。任何安全剔除、未完成、异常、掉落/滑动标记或未稳定 PROBE 都不能伪装成通过。

程序写 `commissioning_full_summary.json` 和 `.md`，同时显示验收摘要：最大力与 F_max、接触峰值、PROBE 稳定性与信噪比、零偏移两指质心、越盖无接触分支、到位误差、控制周期、CHECK。数据检查通过且现场一致后才输入 `PASS`，然后写 `commissioning_passed`。不是“三条记录跑完就自动通过”。

若摘要未通过，保留当前数据定位原因，修改物理参数或配置后按需要建立新 session、重新验收。不能编辑 progress.json 的通过标记。

## 10. 正式 contact：先小批次，再续跑

```bash
bash run_on_5090.sh contact --dry-run --session outputs/real_001
bash run_on_5090.sh contact --real --session outputs/real_001 --resume --single-step --limit 5
```

小批次无问题后按需要继续：

```bash
bash run_on_5090.sh contact --real --session outputs/real_001 --resume --limit 20
```

一次完整接触试验的实际流程为：T_check 检查 → 预备位 → 目标抓取位 → CLOSE → PROBE → SQUEEZE → 提起 → HOLD → LOWER → 松开 → RETREAT → T_check 固定参考检查 → 人工标注。目标或路径不安全时写 skipped_safety，不发送那个目标。

PROBE 要连续在目标力带内默认 0.2 s，超时标 `probe_not_settled` 后继续，分析会带该标记。描述子取稳定窗口最后默认 5 个有效 DD 帧的中位数；HOLD 取中段默认 0.5 s，保留实际帧索引、力和缺口。HOLD 任一指低于接触阈值标 `dropped`，任一指 DD 质心漂移超过当前 px/mm 阈值标 `slip_suspected`。

LOWER 监测竖直切向力和；坐标映射未确认时用切向合力幅值。降至 HOLD 均值的默认 20% 或回到提起前高度就停止，不下降到原高度以下；记录检测卸载高度差。

人工标注仍由你肉眼判断：回车正常，`t` 倾倒、`m` 移位、`d` 掉落、`s` 滑动/旋转、`x` 作废，并可写备注。自动标记和人工标注分别保存。作废不删除原始数据。

每次 RETREAT 后到 T_check，任一路图像偏移超阈值或模板质量不合格就暂停。你复位后重新拍照，直到两路都通过；始终和 session 开始时的参考比较。换 light/heavy 时同样在 T_check 检查。两瓶放同一标记位置则 T0、G、腕部参考不变；不要换瓶时重拍模板“接受”偏差。

`--limit` 限制本次尝试数，不是保证成功提起的次数；完整计划仍保存。已完成 ID、作废与安全剔除不会重复。待做项为零时直接退出，连硬件都不连接。触觉空载参考按配置周期更新，和永不随复位更新的腕部模板是两回事。

## 11. 正式 visual 和新的 session

同一已通过 contact 三点验收的 session：

```bash
bash run_on_5090.sh visual --dry-run --session outputs/real_001
bash run_on_5090.sh visual --real --session outputs/real_001 --resume --single-step --limit 8
bash run_on_5090.sh visual --real --session outputs/real_001 --resume
```

当前视觉计划使用 `design.bottles[0]`，默认 light；若刚做完 heavy，按提示换回计划指定瓶子并通过固定参考检查。视觉始终张开夹爪，到点停稳后保存两腕图像；意外接触会停止。每张图关联右 TCP、左 TCP、B 和观测时间。right_wrist 为眼在手上；left_wrist 为停放臂上的固定外部视角，不能把右腕的眼在手标签误当成左相机外参。

视觉正常结束后，经几何/路径检查回 T_check。检查不通过就停在原地并提示人工处理，不强行返回。启动前仍需真实停止报告；视觉不执行夹紧，正式视觉本身不要求力控报告。

### 11.1 新瓶位、新 T0 或主机重启后的新 session

主动更换瓶位、重新记录 T0，或主机重启后，建立新 session，重新 prepare 和固定参考。已有一个真实完整三点 PASS session、硬件/桥接配置与源码哈希相同时，可以采用 lite：

```bash
bash run_on_5090.sh prepare --config config.ros.yaml --real --single-step --session outputs/real_002
bash run_on_5090.sh contact --dry-run --session outputs/real_002 --commissioning-lite
bash run_on_5090.sh contact --real --session outputs/real_002 --commissioning-lite --single-step --verified-session outputs/real_001
```

新 session 做一次零偏移单步试验并核对摘要输入 PASS；之后仍需当前匹配的停止/力控报告，才能正式 contact。原验收来源必须是完整三点 PASS，不能用模拟 session，也不能用一个 lite session 继续作为完整验证来源。

纯视觉新 session 同样先 prepare，已有匹配的完整验证来源时，前 8 个可行点单步验收：

```bash
bash run_on_5090.sh visual --dry-run --session outputs/real_002 --commissioning
bash run_on_5090.sh visual --real --session outputs/real_002 --commissioning --single-step --verified-session outputs/real_001
bash run_on_5090.sh visual --real --session outputs/real_002 --resume
```

此路线不要求本 session 做接触三点验证。硬件/桥接相关配置或代码变化导致哈希不匹配时，恢复完整三点验收；程序会说明不匹配，不提供跳过开关。原验收 session、配置和源码证据应一起保存。

## 12. 停止、故障与续跑

正常分批停止：用 `--limit`，等当前试验完成、回到 T_check、写完进度再退出。Ctrl-C、关闭 SSH 或杀进程不是硬件急停。软件停止仅在有新鲜实测反馈且 ROS 命令可发送时尝试发布当前位置保持；无新鲜反馈时不能伪造保持目标。

提起前故障会按当前状态尝试软件停止和受检查约束的释放/后撤；不是所有故障都能完成恢复。LIFT/HOLD/LOWER 可能仍悬空时，程序保持并设置 `requires_operator_recovery`，不会自动开夹让瓶子落下。先按现场已经验证的方法恢复；程序提示恢复锁时新建 session，不能只编辑 JSON 清锁继续。

普通未完成批次可继续：

```bash
bash run_on_5090.sh contact --dry-run --session outputs/real_001
bash run_on_5090.sh contact --real --session outputs/real_001 --resume --limit 20
```

未完成旧尝试保留在 `interrupted/`，已完成 ID 不重复。主机重启改变单调时钟域、修改 T0/瓶位/固定参考、修改锁定计划或存在恢复锁时，不直接续跑旧 session。每次启动需要重新扣零检查和首段 MOVE，不因 `--resume` 略过。

报错后先看 `last_run_error.json`、对应条目的 `meta.json` 和终端信息。给我反馈“session 路径、trial/point ID、阶段、错误和现场实际状态”，能最快定位问题。

## 13. 分析：单 session 与多 session

采集进程退出后再分析，避免把仍在写入的记录当成完整数据：

```bash
bash run_on_5090.sh analyze outputs/real_001 --output outputs/real_001/analysis
bash run_on_5090.sh analyze outputs/real_001 outputs/real_002 --output outputs/combined_real_001_002
```

SESSION 是位置参数，不是 `--session`。多 session 必须指定 `--output`。合并结果带 `session` 列，各自偏移相对各自 T0 抓取中心；同时在 `per_session/` 下输出分 session 拟合和热图。G、工具定义、DD 处理方式、力单位/定义不一致会警告，不能把警告忽略后解释为同条件对照。

| 输出 | 先看什么 |
|---|---|
| `analysis_report.json` | 记录数量、缺标定/来源/跨 session 不一致警告 |
| `summary.csv` | 标签、两腕 CHECK 位移、到位误差、循环周期均值/p95/max、DD/力缺口、接触峰值、PROBE SNR/未稳定标记、卸载高度差 |
| `descriptors.csv` | 两指 PROBE/HOLD 窗口中位描述子、帧来源、实际力、质心 px 和可用 mm |
| `linear_fits.csv`、质心图 | 质心—偏移斜率、R²、样本数；未知间距时为 px/mm |
| `heatmap_*.png`、`heatmap_cells.csv` | 结果率与每格样本数；空格不是零失败率 |
| `pose_errors.csv`、阶段误差图表 | 实际 TCP 与命令的平移/旋转误差 |
| `per_session/` | 多 session 分组结果，用来检查合并效应 |

每个 trial 的 `streams.h5` 保存独立来源事件和时间戳，分析按已提交边界读取；`images/` 保存图像和来源时间。DD 处理来源含服务名称/版本、分辨率、dtype 和量化方式；源端未发送版本时明确记 unknown，不编造版本。主机接收时间与最近观测位姿不是曝光硬件同步。

主要目录：

```text
outputs/real_001/
  config_snapshot.yaml
  session_meta.json
  progress.json
  trial_index.csv
  commissioning_full_summary.json
  commissioning_full_summary.md
  plans/
  references/bottle_position/          固定双腕模板
  references/bottle_position_preview/  本次原尺寸 ROI 图
  references/calibration_observations/  约 2 mm 标定证据
  references/unloaded_.../             空载 DD/力参考
  checks/                             每次固定模板检查
  trials/                             完整数据、meta、图像
  interrupted/                        中断尝试（如有）
  analysis/
outputs/validation/                   停止、力控、开度报告和原始记录
```

写盘使用独立进程及有界队列，满队列/写盘故障会报错、保留未完成证据，不阻塞控制循环等磁盘。可以在不采集时做离线墙钟压力测试：

```bash
bash run_on_5090.sh record-stress --config config.ros.yaml --output outputs/writer_stress_01
```

该工具只产生合成数据，比较本机空载与写盘负载下的真实时间周期，不是仿真时间“变快”的证明，也不保证硬实时。原 existing 控制器的内部同步 CSV 仍需其实际力控验收单独观察。 记录压力测试中，本机的双路 1920×1080 随机 JPEG、30 fps 负载触发 128 MiB 队列满；5090 的同级 2 秒补测通过，但生产结束仍有约 104.9 MB 待写，随后才写完。两处循环隔离均通过，持续吞吐仍需按实际图像大小和帧率验证；证据见 [V2 验收记录](V2_VALIDATION_20261027.md)。

## 14. 收工与 Lenovo 本地备份

先正常结束采集，确认现场瓶子和机器人状态，并按已有流程停放/处理。分析和备份不会替你把机械臂移到安全位。

以下在 **Lenovo 本机终端**执行，备份整个 session 以及外部验收报告：

```bash
mkdir -p /home/lenovo/wfb/bottle_data_backup/real_001
mkdir -p /home/lenovo/wfb/bottle_data_backup/validation
rsync -av --partial 6f5090:/home/dm/DM_das/bottle_grasp_experiment/outputs/real_001/ /home/lenovo/wfb/bottle_data_backup/real_001/
rsync -av --partial 6f5090:/home/dm/DM_das/bottle_grasp_experiment/outputs/validation/ /home/lenovo/wfb/bottle_data_backup/validation/
rsync -avcn 6f5090:/home/dm/DM_das/bottle_grasp_experiment/outputs/real_001/ /home/lenovo/wfb/bottle_data_backup/real_001/
rsync -avcn 6f5090:/home/dm/DM_das/bottle_grasp_experiment/outputs/validation/ /home/lenovo/wfb/bottle_data_backup/validation/
```

后两条使用校验和且只预览差异；稳定数据正常应无待更新文件。不要在源目录还在写入时把这一结果当成完整快照。命令没有删除远端数据。保留本机代码和文档目录 `/home/lenovo/wfb/bottle_grasp_experiment`，并把实际使用的测量 JSON、配置、源码版本/校验记录与数据一起归档；不能只下载图表。

下一次先做只读 doctor，再按是否改 T0/瓶位、是否重启、硬件哈希是否相同，选择旧 session 正常续跑或新 session 的 prepare + lite/完整验收。

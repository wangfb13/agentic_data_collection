# 历史：第一版严格接口设计与早期说明

本页保留第二版文档重写前的 README，仅供追溯。以下的启动条件、输出和行为可能已过时，不是当前操作步骤。当前只使用公开 ROS 路线，见 [README](../README.md) 和 [完整操作指南](ADVANCED.md)。历史模板文件 `config.yaml`、`hardware_bridge_template.py`、`docs/HARDWARE_CONTRACT.md` 保留供代码追溯，现场不以它们启动实机。

---

# 药瓶抓取实验

部署目录：`6f5090:/home/dm/DM_das/bottle_grasp_experiment`。

**现场请从 [完整操作指南](ADVANCED.md) 开始。** 它逐步写明命令、机械臂实际位置、框选、2 mm 标定、三点验证、正式采集、换瓶、续跑、分析和本机备份。公开 ROS 接入使用 `config.ros.yaml` 和 `bash run_on_5090.sh`，默认右臂；本机副本在 `/home/lenovo/wfb/bottle_grasp_experiment`。

`bash run_on_5090.sh ...` 可在 **5090 宿主机或现有 `daimon_ros1_das` 容器内**执行。在容器内直接运行，不需要退出容器或安装 Docker；在宿主机上则自动进入该容器。两者都使用项目 `.venv_robot`，不依赖当前激活的 conda 环境。

2026-10-27 按操作员要求接入现有 ROS 位姿/夹爪命令和 X5 数据流，并接受现有处理后的 DD。右 X5 已启用既有 `starvla-tactile-dd.service`；两指 DD、六维力和胶面原图已收到，腕部相机服务未改。接入采用软件插值与实测位置保持，不宣称底层硬件急停或断网保护；电流未提供时记录缺失值，速度标为位置差分估计。未运行实机运动或夹爪开合测试。实际位姿、几何、mm/N 标定仍需填写，不能用演示值替代。

明天先看已经生成的 `outputs/acceptance_demo/analysis/`（图表与 CSV）、`outputs/acceptance_demo/acceptance.json`（模拟验收断言）和 `docs/VALIDATION.md`（验收记录）。两套完整 dry-run 位于 `outputs/dry_contact/`、`outputs/dry_visual/`。

这版提供接触扫描、非接触视觉扫描、准备流程和分析程序；可以生成离线试验计划、运行模拟状态机、保存模拟数据并画出结果。`config.yaml` 保留最初严格接口模板，`config.ros.yaml` 是当前选择的公开 ROS 路线。实验准备和采集在实际配置中仍有 TODO 时拒绝运动；`doctor` 和 `record-pose` 可以先直接读取当前状态。独立的 `move-tcp` 可用于实测前定位：选方向和距离、输入坐标或读取已记录位姿，默认预览，显式 `--real` 并逐次输入 `MOVE` 才执行。它不控制夹爪、不读取触觉、不自动通过实验门禁。

所有硬件运动只能由操作员本人在终端手动启动，现场守在急停旁。本次开发和验证不需要、也不会通过这些示例连接机器人。现有机器人驱动、力控制器和 policy 保持独立。

## 1. 程序与约定

| 文件 | 用途 |
|---|---|
| `prepare.py` | 新建 session，准备 T0、腕部参考与空载触觉参考；模拟模式使用合成场景 |
| `collect_contact.py` | 接触偏移扫描的规划和状态机 |
| `collect_visual.py` | 张开夹爪的非接触视觉扫描 |
| `analyze.py` | 离线读取 session，输出描述子、拟合、热图和 CSV |
| `config.ros.yaml`、`bridge_ros_public.py` | 当前公开 ROS/X5 接入及待填实测配置 |
| `run_on_5090.sh` | 自动识别宿主机/容器，使用现有 ROS 容器环境统一启动；默认只读 doctor |
| `robot_doctor.py`、`record_pose.py` | 只读检查数据源；登记当前实际 TCP |
| `move_tcp.py` | 按方向/毫米数、基座 XYZ 或已记录位姿下发公开 ROS TCP 目标，等待实测到位 |
| `config.yaml` | 保留最初严格接口模板，统一使用 mm、度、N、秒 |
| `bottle_scan/backend.py` | 硬件能力契约和独立模拟后端 |
| `hardware_bridge_template.py`、`docs/HARDWARE_CONTRACT.md` | 尚未绑定的实机接口与必须验证的能力 |
| `docs/PLACEHOLDERS.md` | 明天实操时逐项提供测量值的登记表 |
| `docs/interfaces_force.md` | 现有力控制器与夹爪接口的证据、单位差异、退出限制 |
| `docs/interfaces_sensors.md` | 触觉、相机、原始数据链路的证据与待确认项 |

只有两路腕部图像；配置、采集及存储中不包含第三视角相机。TCP 偏移采用右乘 `T0 @ Trans(delta) @ Rot(...)`，平移偏移在 T0 的局部轴中定义，旋转绕 TCP 原点；提起/放下沿基座竖直方向。

`--demo` 使用一整套独立合成几何，**其通过的碰撞检查不能证明真实场景安全**。`--simulate` 使用虚拟时间和合成传感器，不发送 ROS、电机或相机指令。图像、力、开度、电流、DD 均标记为模拟数据，不能用于论文中的实测结论。

## 2. 在 5090 上运行离线示例

以下命令在 5090 宿主机或现有 ROS 容器内均可执行，使用项目独立 Python 环境，不修改已有机器人容器环境。示例 session 目录必须尚不存在；重复运行时换新目录名以保留已有数据。

```bash
cd /home/dm/DM_das/bottle_grasp_experiment
```

先生成完整试验列表和安全筛选结果：

```bash
bash run_on_5090.sh contact --dry-run --demo --output outputs/dry_contact
bash run_on_5090.sh visual --dry-run --demo --output outputs/dry_visual
```

dry-run 不下发运动，会列出试验/候选点、被拒绝的原因、估计时长与存储量。估算与采样率、JPEG 大小、等待操作员的时间有关；合成配置下的估算仅用于检查流程。非接触视觉默认生成 400 个候选位姿，安全筛选后保留数可能少于 400。

不带 `--simulate` 或 `--real` 时默认 dry-run。真实配置仍有 TODO 时，先列出缺项并退出；不会用猜测值计算真实轨迹。可先运行 `bash run_on_5090.sh prepare --dry-run` 查看缺项。

新建接触模拟 session，运行六种结果分支，再分析：

```bash
bash run_on_5090.sh python smoke_test.py --session outputs/acceptance_new
```

这个离线验收脚本会准备参考、运行 8 次接触试验（覆盖正常、无双侧接触、无接触卡停、夹紧力未达到、HOLD 接触丢失、HOLD 质心漂移）、6 个视觉点，再生成分析图表和 `acceptance.json`。它只能使用模拟后端，且会断言关键结果与预期一致。合成数据使用缩小网格和更快的虚拟运动以缩短软件验收时间；实机默认设计仍保留在配置中。

单独调用采集入口也可以：

```bash
bash run_on_5090.sh prepare --simulate --small-demo --session outputs/demo_contact_new
bash run_on_5090.sh contact --simulate --session outputs/demo_contact_new --limit 6
bash run_on_5090.sh analyze outputs/demo_contact_new
```

`--limit` 只缩短本次采集，不改变保存的完整计划。人工注入 `--scenarios normal,no_contact,stall,force_timeout,dropped,slip` 可测试模拟分支，但各点仍受真实状态机的几何否决，不能保证每个随机点都进入 HOLD。

视觉扫描建议使用单独 session，便于区分数据：

```bash
bash run_on_5090.sh prepare --simulate --session outputs/demo_visual
bash run_on_5090.sh visual --simulate --session outputs/demo_visual --limit 8
```

视觉点的 `meta.json` 保存实测位姿标签 `label_B_in_tcp = inv(T_measured) @ T0` 和两路图像路径。`analyze.py` 的抓取结果热图针对接触试验；视觉数据主要通过图像和上述位姿标签使用。

离线测试：

```bash
bash run_on_5090.sh python -m unittest discover -s tests -v
```

## 3. 实验准备与固定腕部参考

以下是流程概要；现场按 [完整操作指南](ADVANCED.md) 的准确交互次序执行（空载参考在腕部框选之前）。本 README 后面保留的严格接口示例用于说明第一版设计，不替代当前公开 ROS 启动入口。

实机桥接完成后，每个新 session 的操作顺序如下；当前应先通过上述模拟流程确认格式。

1. 用 `bash run_on_5090.sh move-tcp --real` 选方向/填数，将已张开的夹爪定位到理想抓取位姿，再用 `record-pose` 读取实际 TCP 为 T0。先确认 TCP 定义、基座坐标、单位和四元数顺序，再写入配置或 session。
2. 用 `move-tcp` 选方向/填数到周围留有余量的自由空位，记录 `measurements/free_pose.json`，再通过 `--pose-file measurements/T0.json` 返回 T0。后续 `+a/+l/+c` 各 5 mm 由 prepare 自动执行，你只观察确认，不需要遥操走准距离。按测量清单确认 a/c/l 方向、工作空间、桌面、瓶子与夹爪几何，以及两指传感器的方向；基座/TCP 的 XYZ 与 a/c/l 不自动对应。
3. 夹爪张开，移动到 `T_check`：T0 沿其局部 `−a` 后退 `check.retreat_mm`，默认 80 mm。两路腕部相机各拍一张参考图，操作员在每张图中框选瓶身。
4. 首次阈值标定时，按提示将瓶子故意移动约 2 mm；分别记录两路模板匹配的像素偏移。按 `check.calibration_fraction` 生成检查阈值，再把瓶子放回原标记位置并复查。两路阈值可以不同。这是瓶位检查标定，不是 DD 的 mm/px 标定。
5. 在无接触状态记录 10 秒空载 DD 和六维力；每指接触阈值为 `max(5 × 空载法向力标准差, force.contact_min_N)`。之后按 `design.baseline_every`，默认每 30 次试验，重新记录触觉空载参考。

初始 `check.threshold_px: calibrate_at_session_start` 表示准备阶段需要现场标定，不能作为采集阶段的数值阈值。准备程序把两路标定结果写入 session 的配置快照，后续采集要求它们已是有效数值。

ROI 弹窗需要可用图形桌面/`DISPLAY` 和带 GUI 的 OpenCV；纯 SSH 或 headless 环境不会自动显示框选窗口。此时可先在两路**对应原尺寸参考图**上人工确定矩形，再通过 `prepare.py --rois-json rois.json ...` 提供，格式为 `{"left_wrist":[x,y,width,height],"right_wrist":[x,y,width,height]}`，单位像素。不要用缩放截图坐标或示例矩形代替实测 ROI。

可选 `prepare.py --tune` 在 T0 探测接触并显示两指质心，`a+/a-/l+/l-` 每次微调 1 mm，回车确认。每次位移前先释放夹爪；其真实运行同样要求硬件门禁和单步确认。

**触觉空载参考可以定期更新；腕部瓶位模板在整个 session 内固定。** 每次 RETREAT 后返回 T_check，任一路模板偏移超阈值或匹配质量不足，就暂停并提示复位；重新拍照仍与 session 开始的参考比较，直到通过。按回车只触发重拍，不会更新模板或放宽阈值。

轻瓶换重瓶也执行同样检查。两瓶在同一标记位置时沿用 T0 和腕部参考图。若外观、遮挡或照明变化导致模板无法匹配，应先检查成因；程序不会自动把失败图像当作新参考。只有明确改变瓶位并重新记录 T0 时才建立新参考，第一版建议新建 session，保留旧 session 的追溯关系。

## 4. 接触扫描操作

默认按瓶子分块，块内固定随机种子打乱，轻/重瓶分别各偏移两次，每 10 次插入零偏移。主网格、闭合轴偏移、两种倾斜和越过瓶盖的范围均以配置与实测几何为准。安全检查失败的点记录原因并跳过，不把目标截断到边界后执行。

状态顺序为：`MOVE_PRE → APPROACH → CLOSE → PROBE → SQUEEZE → LIFT → HOLD → LOWER → RELEASE → RETREAT → CHECK → LABEL`。CLOSE 的无双侧接触和卡停分支、SQUEEZE 的力未达到分支会转入 RELEASE，跳过提起。当前固定力实验契约使用每指法向力 N；已有控制器的“均值”和“总和”定义不同，必须在真实桥接中明确转换和控制含义。

HOLD 自动标记独立于人工判断：任一指法向力低于该指接触阈值标记 `dropped`；任一指 DD 质心相对 HOLD 初始有效质心的漂移超过 `tactile.centroid_drift_mm` 标记 `slip_suspected`。它们是观测标记，不等于人工确认掉落/滑动。

| 键 | 人工标注 |
|---|---|
| 回车 | 正常 |
| `t` | 瓶子倾倒 |
| `m` | 瓶子被推动、位置改变 |
| `d` | 提起过程中掉落 |
| `s` | 明显滑动或转动 |
| `x` | 本次作废 |

每次可补充文字备注。`t/m/d` 后暂停复位并重做固定参考检查。`x` 数据保留，分析时不计入成功率分母和拟合。

首次实机运行必须使用 `--commissioning --single-step`，仅包含零偏移、沿手指方向最深点、沿瓶轴越过瓶盖点这三项。每个阶段开始前都需操作员确认；碰撞检查仍具有否决权，不为完成三点而绕过安全限制。桥接能力、TODO、dry-run 与首次验证门禁通过后，才具备进入完整实机扫描的条件。

`force.squeeze_mode` 默认 `fixed`。配置中的 `forcepolicy` 仅保留为未来扩展选择，第一版明确拒绝该模式，不加载学习模型。

下面是**硬件桥接与全部实测门禁完成之后，由现场操作员手动执行**的命令形式。当前第一版尚不具备执行条件；`--real` 不会绕过 TODO 或硬件能力检查：

```bash
bash run_on_5090.sh python prepare.py --config config.measured.yaml --dry-run
bash run_on_5090.sh python collect_contact.py --config config.measured.yaml --dry-run --commissioning --output outputs/real_preflight
bash run_on_5090.sh python prepare.py --config config.measured.yaml --real --single-step --session outputs/real_session
bash run_on_5090.sh python collect_contact.py --real --session outputs/real_session --commissioning --single-step
```

准备结束后，接触采集启动前仍会根据 session 中实际记录的 T0 重建并打印 dry-run 计划。三点完成后进入完整扫描需要新的手动启动；已有进度使用 `--resume`，不重新执行完成项。视觉实机采集也要求先通过该三点验证。

## 5. 异常、恢复与续跑

提起前异常：请求停止机械臂，然后张开夹爪，并在安全检查通过时后撤。提起开始前即锁定“可能在空中”状态；提起后异常：请求停止机械臂，保持夹爪当前状态，报警等待人工处理，不自动松开或后撤。

这描述的是状态机的**接口契约**。当前已有底层代码的退出/注销并不保证电机停止，最后一个位置目标仍可能执行。因此，在可靠 stop、保持、看门狗和退出行为验证之前，真实模式保持禁用。

基础 `config.yaml` 的 `hardware.bridge_module` 指向 `hardware_bridge_template`，其中严格硬件能力位仍为 false。当前 `config.ros.yaml` 显式选择 `bridge_ros_public`，按操作员同意的软件 ROS 语义运行，保留其能力范围和数据来源；不会把公开话题发布伪装成严格硬件保证。

每次完成写入 `progress.json` 和 `trial_index.csv`；已完成项不会覆盖。中断试验的旧文件保留到 `interrupted/` 后才允许新尝试。若进度标记要求人工恢复，必须先确认机械臂和瓶子实际状态；不能以重启程序代替处理空中异常。更改试验计划、T0 或几何后应新建 session，避免把不同比较基准拼接在同一组数据里。

模拟续跑需要显式 `--resume`；实机已有进度且未给该选项时会询问。若记录仍标记可能在空中，第一版直接拒绝续跑，要求人工恢复并新建 session。主机重启使单调时钟域改变时也要新建 session 和参考。

## 6. 数据和分析输出

```text
SESSION/
  config_snapshot.yaml
  session_meta.json
  progress.json
  trial_index.csv
  references/              # 原始空载数据、平均参考、固定腕部模板
  plans/                   # 固定计划
  checks/                  # 每次瓶位检查图像、偏移和判定
  trials/trial_XXXX/
    meta.json
    streams.h5
    images/                # 仅两路腕部 JPEG 及时间戳
  analysis/
```

H5 每个流在 `/streams/名称/` 下保存 `data`、`monotonic_ns`、`device_timestamp`、`sequence`、`metadata_json`；读取时以 `committed_count` 为提交边界。DD 保留原始 dtype/shape，使用无损压缩，分析处理不回写原始数据。主机单调时钟作为各流时间基准，设备时间另存；真实桥接必须连续排空每个原始采集事件，不能把 latest-only 快照重复写入伪装成完整流。

主要分析文件：

- `summary.csv`：每试验一行，偏移、瓶子、自动结果、人工标注、每指提起前后法向力变化及描述子；安全剔除项保留行但不进入接触统计。
- `descriptors.csv`：PROBE 末帧与 HOLD 中点的两指接触总量、面积、质心、边缘占比、二阶矩和 CoP；含原帧时间与采用的参考文件。
- `linear_fits.csv` 与 `centroid_vs_*.png`：两个面内方向、两指分别拟合，给出斜率、R²、样本数。
- `heatmap_*.png`、`heatmap_cells.csv`：分瓶二维热图和每格有效数量；仅比较闭合偏移及倾角为零的平移试验。
- `contact_opening_and_timing.png`：接触开度与瓶轴偏移、接触起始时差与闭合偏移。
- `zero_offset_drift.png`：零偏移描述子随时间变化。
- `pose_errors.csv`、`pose_error_by_stage.csv/png`：按阶段统计平移/旋转误差。
- `analysis_report.json`：处理数量、参考选择规则及缺失标定说明。

参考按主机单调时间选择最近一次空载参考，等距时选较早者。DD 面积/质心需要真实像素间距；a/l 投影需要实测传感器方向；CoP 需要确认 N、力矩单位和作用面相对传感器原点的位置。缺少这些标定时输出 NaN 与说明，不伪造物理单位。命令/实测位姿按实测之前最近命令配对，超过允许时差的样本不进入误差统计。

## 7. 明天的实际交接

第一版六维力 CoP 公式要求已确认的法向为传感器 z 轴；其他法向不会套用错误公式，会输出 NaN 和说明，需接入相应的接触面坐标变换后扩展。

按 [完整操作指南](ADVANCED.md) 和 [PLACEHOLDERS.md](docs/PLACEHOLDERS.md) 提供“占位符编号 + 实测信息”。可以先用 `bash run_on_5090.sh move-tcp --real` 选方向或填写目标定位，再用 `bash run_on_5090.sh record-pose` 保存实际 TCP，不凭截图猜坐标。自由空位只需留出余量，三个方向各 5 mm 由 prepare 自动执行。当前公开 ROS 桥已接入；补齐位置、尺寸和单位标定后，重新做真实配置 dry-run，再由你本人启动三点单步验证。

## 第二版第一组变化

接触倾斜和视觉公转改为以抓取中心为基准。`tool.grasp_center_in_tcp` 必须填写实测/CAD/URDF 支持的固定变换，并写明 `tool.grasp_center_evidence`；抓取中心坐标轴保持 TCP 轴向，因此 G 的旋转块为单位阵，纯平移及三点目标不变。真实值保持 TODO，不使用演示值。

采集先计算待做项，无待做项不连接硬件。首次到 T_check 必须预览路径并输入 `MOVE`；视觉结束经检查回到 T_check。连接不自动扣零，首次空载需输入 `EMPTY`，每次都检查开度、DD 参考及新鲜力帧波动。doctor 和实机入口会列出双臂/双夹爪命令发布者，有其它发布者或查询失败即拒绝。

详细变更与每组验证见 [CHANGELOG](CHANGELOG.md)，第二版原始要求见 [V2_REQUIREMENTS](docs/V2_REQUIREMENTS.md)。

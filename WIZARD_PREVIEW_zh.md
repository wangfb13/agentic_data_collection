# setup + run 完整终端交互示例（历史模拟阶段）

实机向导现已接入，当前启动方法见 [首次设置](docs/FIRST_SETUP.md)。以下仍是初次交付的模拟记录，不作实机验收。

这是 **2026-10-28 实际执行的模拟演练输出**，用于你确认交互。`--rehearsal` 自动代填演示回答，包含 MOVE、PASS 和数字标签；这些不是你已经作出的现场确认。向导暂时只允许 `--simulate`，真实适配器未开放，没有启动 5090 实机动作。操作者随后要求将本版同步到 5090；实际日志仍保留最初在 Lenovo 上生成的路径，5090 的启动方法见 [首次设置](docs/FIRST_SETUP.md)。

模拟 doctor/模型/尺寸/示教均明确标为合成输入；N、G=0、示教坐标仅属于演示夹具，不覆盖真实 SDK 单位与 G≈[0,0,117] mm。所有试验阶段、几何检查、轻触、恢复、HDF5/JPEG 记录和分析实际在模拟后端执行，下面不是预先编写的伪日志。

## 一、一次性 setup：全部输出

命令（在 Lenovo 项目副本执行）：

```bash
PYTHONPATH=.deps python3 wizard.py setup --simulate --rehearsal --profile outputs/wizard_delivery_20261028/review/station_profile.yaml
```

```text
模拟交互预览 · SIMULATION_ONLY · 不连接 ROS、不驱动实机
setup：每步完成即保存；同一条命令可以续接。
[1/7] doctor 通过（模拟输入；未检测真实网络）。
[2/7] 模拟模型：a=TCP+X，c=TCP+Y，l=TCP+Z，G=[0,0,0] mm。
  手指 50×8×20 mm，手掌 16×110×24 mm；这些仅是模拟夹具，实机已有 G≈[0,0,117] mm 未被覆盖。
轴向、抓取中心与外包络确认一次；回车接受此模拟模型： [回车]
[3/7] 输入卡尺尺寸（本次演示使用合成夹具的默认值）。
瓶身直径 高度（mm）： 36.0 65.0
瓶颈直径 高度（mm）： 22.0 15.0
瓶盖直径 高度（mm）： 28.0 15.0
夹爪张开宽度（mm）： 90.0
  开度配对：2 个不同宽度点；不足以检验线性，已记录。
[4/7] 示教桌面与工作空间。
  示教点均为模拟夹具数据；实机版将读取机器人反馈。安全余量由原几何模块统一计入一次。
把最低指尖移到桌面上方约 10 mm，按回车记录： [回车]
将工具外包络边界示教到安全工作空间下界角，按回车记录： [回车]
将工具外包络边界示教到安全工作空间上界角，按回车记录： [回车]
  桌面 z=0.0 mm；安全余量 1.0 mm，统一计入一次。
[5/7] 预填来源：bottle_scan/config.py 的 demo_config/force_defaults（模拟）。
  单位=N，F_target 定义=mean，目标=2.0，每指上限=4.0。
  真实公开 ROS 配置仍使用 SDK 单位，本次不会改成 N。
确认以上模拟力设置，按回车： [回车]
[6/7] 自由空间停止测试：软件停止×3、SIGKILL×3（仅模拟子进程）。
  6/6 通过；报告：/home/lenovo/wfb/bottle_grasp_experiment/outputs/wizard_delivery_20261028/review/station_profile_setup/stop_report.json
核对停止测试摘要；输入 PASS： PASS
[7/7] 到 T_check 并设置一次右腕检查区域。
首段移动目标 TCP xyz(mm)：[220.0, 0.0, 70.0]
确认现场安全，输入 MOVE 启动本段及后续自动流程： MOVE
右腕原尺寸图：/home/lenovo/wfb/bottle_grasp_experiment/outputs/wizard_delivery_20261028/review/station_profile_setup/right_wrist_full.png
框选瓶身区域，输入 x y w h（像素；左腕只记录）： 92 42 61 101
setup 完成：/home/lenovo/wfb/bottle_grasp_experiment/outputs/wizard_delivery_20261028/review/station_profile.yaml
硬件哈希（模拟）：d1a1bea46d0846fc；实机放行=否。
```

## 二、日常 run --mode contact：全部输出

```bash
PYTHONPATH=.deps python3 wizard.py run --simulate --rehearsal --mode contact --profile outputs/wizard_delivery_20261028/review/station_profile.yaml --session outputs/wizard_delivery_20261028/review/contact_session
```

为了完整展示异常交互，本次小型演练在第 4、5、6 次正式试验中分别注入合成掉落、滑动、零偏移质心漂移。普通交互模拟不注入这些故障。计划是 8 次正式试验和 1 个被几何否决的候选，另有 3 个验收点；正式实验的完整设计保留在高级模块，接实机时再接入。

```text
模拟交互预览 · SIMULATION_ONLY · 不连接 ROS、不驱动实机
请把瓶子放到标记处，把夹爪张开并移到瓶子旁大致的抓取位置，然后按回车。 [回车]
session：/home/lenovo/wfb/bottle_grasp_experiment/outputs/wizard_delivery_20261028/review/contact_session；初始 T0 已自动读取（模拟反馈）。
自动准备：每次移动前松开夹爪；方向标定与居中全程不提瓶。
首段移动目标 TCP xyz(mm)：[300.0, 0.0, 70.0]
确认现场安全，输入 MOVE 启动本段及后续自动流程： MOVE
  方向标定：T0、a±5、l±5 mm 完成；双指响应矩阵有效。
准备完成：T0 双指最大残差=0.001 等效 mm；右腕阈值=1.41 px；区域预览=/home/lenovo/wfb/bottle_grasp_experiment/outputs/wizard_delivery_20261028/review/contact_session/references/right_wrist_roi.png
核对准备摘要，按回车确认： [回车]
计划：候选 9，可执行 8，剔除 1；预计 3.4 分钟，0.38 GiB。
主要剔除原因：[('palm:bottle:body', 1)]
估计不含标注、复位、验收和其他等待；完整 dry-run 已保存。
确认计划，按回车继续： [回车]
验收：此哈希首次使用，零偏移/最深 +a/越瓶盖 3 点逐阶段确认。
[MOVE_PRE] 阶段确认，回车继续： [回车]
[APPROACH] 阶段确认，回车继续： [回车]
[CLOSE] 阶段确认，回车继续： [回车]
[PROBE] 阶段确认，回车继续： [回车]
[SQUEEZE] 阶段确认，回车继续： [回车]
[LIFT] 阶段确认，回车继续： [回车]
[HOLD] 阶段确认，回车继续： [回车]
[LOWER] 阶段确认，回车继续： [回车]
[RELEASE] 阶段确认，回车继续： [回车]
[RETREAT] 阶段确认，回车继续： [回车]
[CHECK] 阶段确认，回车继续： [回车]
试验 commission_0001 · 验收 1/3 · light · 偏移(a,c,l)=[0.0, 0.0, 0.0]
  自动：双侧接触=True 目标力=True 提起=True 掉落=False 疑似滑动=False 瓶位=通过
回车/1 正常 | 2 倾倒 | 3 被推动 | 4 掉落 | 5 滑动/转动 | 0 作废 | n 备注 | u 撤销上一标签 | p 暂停 | q 本次后安全退出
标签： 1
[MOVE_PRE] 阶段确认，回车继续： [回车]
[APPROACH] 阶段确认，回车继续： [回车]
[CLOSE] 阶段确认，回车继续： [回车]
[PROBE] 阶段确认，回车继续： [回车]
[SQUEEZE] 阶段确认，回车继续： [回车]
[LIFT] 阶段确认，回车继续： [回车]
[HOLD] 阶段确认，回车继续： [回车]
[LOWER] 阶段确认，回车继续： [回车]
[RELEASE] 阶段确认，回车继续： [回车]
[RETREAT] 阶段确认，回车继续： [回车]
[CHECK] 阶段确认，回车继续： [回车]
试验 commission_0002 · 验收 2/3 · light · 偏移(a,c,l)=[5.0, 0.0, 0.0]
  自动：双侧接触=True 目标力=True 提起=True 掉落=False 疑似滑动=False 瓶位=通过
回车/1 正常 | 2 倾倒 | 3 被推动 | 4 掉落 | 5 滑动/转动 | 0 作废 | n 备注 | u 撤销上一标签 | p 暂停 | q 本次后安全退出
标签： 1
[MOVE_PRE] 阶段确认，回车继续： [回车]
[APPROACH] 阶段确认，回车继续： [回车]
[CLOSE] 阶段确认，回车继续： [回车]
[RELEASE] 阶段确认，回车继续： [回车]
[RETREAT] 阶段确认，回车继续： [回车]
[CHECK] 阶段确认，回车继续： [回车]
试验 commission_0003 · 验收 3/3 · light · 偏移(a,c,l)=[0.0, 0.0, 71.0]
  自动：双侧接触=False 目标力=False 提起=False 掉落=False 疑似滑动=False 瓶位=通过
回车/1 正常 | 2 倾倒 | 3 被推动 | 4 掉落 | 5 滑动/转动 | 0 作废 | n 备注 | u 撤销上一标签 | p 暂停 | q 本次后安全退出
标签： 1
验收摘要：3/3 完成；稳定轻触、力限值、越瓶盖无接触和最终瓶位检查：通过
输入 PASS 才继续（本次只能登记模拟验收）： PASS
请放置 light 瓶到同一标记处，回车后自动检查；T0 和原始参考不变： [回车]
试验 trial_0001 · 1/9 · light · 偏移(a,c,l)=[0.0, 0.0, 0.0]
  自动：双侧接触=True 目标力=True 提起=True 掉落=False 疑似滑动=False 瓶位=通过
回车/1 正常 | 2 倾倒 | 3 被推动 | 4 掉落 | 5 滑动/转动 | 0 作废 | n 备注 | u 撤销上一标签 | p 暂停 | q 本次后安全退出
标签： 1
试验 trial_0002 · 2/9 · light · 偏移(a,c,l)=[0.0, 0.0, 0.0]
  自动：双侧接触=True 目标力=True 提起=True 掉落=False 疑似滑动=False 瓶位=通过
回车/1 正常 | 2 倾倒 | 3 被推动 | 4 掉落 | 5 滑动/转动 | 0 作废 | n 备注 | u 撤销上一标签 | p 暂停 | q 本次后安全退出
标签： 1
零偏移后备检查：前 3 次有效双指质心基准已锁定。
试验 trial_0003 · 3/9 · light · 偏移(a,c,l)=[0.0, 0.0, 0.0]
  自动：双侧接触=True 目标力=True 提起=True 掉落=False 疑似滑动=False 瓶位=通过
回车/1 正常 | 2 倾倒 | 3 被推动 | 4 掉落 | 5 滑动/转动 | 0 作废 | n 备注 | u 撤销上一标签 | p 暂停 | q 本次后安全退出
标签： 1
试验 trial_0004 · 4/9 · light · 偏移(a,c,l)=[-5.0, 0.0, -10.0]
  自动：双侧接触=True 目标力=True 提起=True 掉落=True 疑似滑动=False 瓶位=通过
回车/1 正常 | 2 倾倒 | 3 被推动 | 4 掉落 | 5 滑动/转动 | 0 作废 | n 备注 | u 撤销上一标签 | p 暂停 | q 本次后安全退出
标签： 4
请把瓶子复位到标记处；回车后自动检查： [回车]
首段移动目标 TCP xyz(mm)：[222.0, 0.0, 69.0]
确认现场安全，输入 MOVE 启动本段及后续自动流程： MOVE
请放置 heavy 瓶到同一标记处，回车后自动检查；T0 和原始参考不变： [回车]
试验 trial_0005 · 5/9 · heavy · 偏移(a,c,l)=[5.0, 0.0, 10.0]
  自动：双侧接触=True 目标力=True 提起=True 掉落=False 疑似滑动=True 瓶位=通过
回车/1 正常 | 2 倾倒 | 3 被推动 | 4 掉落 | 5 滑动/转动 | 0 作废 | n 备注 | u 撤销上一标签 | p 暂停 | q 本次后安全退出
标签： 5
试验 trial_0006 · 6/9 · heavy · 偏移(a,c,l)=[0.0, 0.0, 0.0]
  自动：双侧接触=True 目标力=True 提起=True 掉落=False 疑似滑动=False 瓶位=通过
回车/1 正常 | 2 倾倒 | 3 被推动 | 4 掉落 | 5 滑动/转动 | 0 作废 | n 备注 | u 撤销上一标签 | p 暂停 | q 本次后安全退出
标签： 1
零偏移漂移报警：任一指 a/l 超过 1.5 mm；请检查瓶子是否仍在标记处。
1 继续 / 2 已复位后继续 / q 安全退出： 2
首段移动目标 TCP xyz(mm)：[222.0, 0.0, 69.0]
确认现场安全，输入 MOVE 启动本段及后续自动流程： MOVE
试验 trial_0007 · 7/9 · heavy · 偏移(a,c,l)=[-5.0, 0.0, 10.0]
  自动：双侧接触=True 目标力=True 提起=True 掉落=False 疑似滑动=False 瓶位=通过
回车/1 正常 | 2 倾倒 | 3 被推动 | 4 掉落 | 5 滑动/转动 | 0 作废 | n 备注 | u 撤销上一标签 | p 暂停 | q 本次后安全退出
标签： 1
试验 trial_0008 · 8/9 · heavy · 偏移(a,c,l)=[5.0, 0.0, -10.0]
  自动：双侧接触=True 目标力=True 提起=True 掉落=False 疑似滑动=False 瓶位=通过
回车/1 正常 | 2 倾倒 | 3 被推动 | 4 掉落 | 5 滑动/转动 | 0 作废 | n 备注 | u 撤销上一标签 | p 暂停 | q 本次后安全退出
标签： 1
本轮结束：已到 T_check，夹爪张开；断点已保存。
分析：已完成 8 条，安全剔除 1 条。
标签比例：正常 6/8 (75%)；掉落 1/8 (12%)；滑动/转动 1/8 (12%)
拟合斜率：heavy/PROBE/left/a=0.1238 px/mm；light/PROBE/left/a=0.1963 px/mm；heavy/HOLD/left/a=0.6125 px/mm；light/HOLD/left/a=0.1577 px/mm
结果目录：/home/lenovo/wfb/bottle_grasp_experiment/outputs/wizard_delivery_20261028/review/contact_session/analysis
零偏移质心时间图：/home/lenovo/wfb/bottle_grasp_experiment/outputs/wizard_delivery_20261028/review/contact_session/analysis/zero_offset_centroid_drift.png
```

## 三、再次运行同一 session，无待做项：全部输出

命令与上一条相同。

```text
模拟交互预览 · SIMULATION_ONLY · 不连接 ROS、不驱动实机
没有待做项；直接退出，没有创建后端，没有发送运动。
```

## 四、额外演练

还实际执行了“已验证哈希下新建 session，完成一次后退出 → 指定 session 续跑剩余项”和 visual 0°/90° 两个朝向块：

- [已验证哈希：先做一次并保存断点](outputs/wizard_delivery_20261028/review/resume_first_transcript.txt)
- [从断点完成剩余项，保留参考、不重新标定](outputs/wizard_delivery_20261028/review/resume_rest_transcript.txt)
- [visual 完整终端输出](outputs/wizard_delivery_20261028/review/visual_transcript.txt)
- [模拟验证汇总](outputs/wizard_delivery_20261028/review/verification.json)

## 五、请确认的交互

重点看准备、计划各确认一次是否合适，首次验收的逐阶段提示，以及每次只显示数字标签菜单的形式。你确认后才接真实 doctor/TF/读数/运动适配器，继续保留全部原有门禁和未完成的实机项。

日常步骤见 [一页操作](OPERATOR_GUIDE_zh.md)，首次设置见 [setup 附录](docs/FIRST_SETUP.md)，故障恢复见 [恢复页](docs/RECOVERY.md)，现有详细现场命令见 [ADVANCED](docs/ADVANCED.md)，本轮与实机工作的边界见 [实施范围](docs/WIZARD_SCOPE_20261028.md)。

验证结果：最终 **423 项离线测试通过**；[测试日志](outputs/wizard_delivery_20261028/final_all_tests.txt)。上面的会话目录已经完成，重跑相同命令会跳过已保存步骤或显示“没有待做项”。如需亲自体验完整交互，先按 [setup 附录](docs/FIRST_SETUP.md) 使用默认新 profile，然后执行日常 run，不指定已有 session 即自动新建。模拟 profile、停止报告和 PASS 记录均不能用于真实硬件。

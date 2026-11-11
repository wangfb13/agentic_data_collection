# RSI 日常操作：积累并验证采集方法

这个入口把现有 E1/E2 执行器与方法记忆、ContactPolicy 训练和等预算对照连接起来。Agent 可以根据过去的改进经历提出新采集规则；程序对照验证后，才决定保留哪个方法。方法和策略分别保存，失败经验也会保留。

旧的 `e1e2`、`agentic run/resume`、setup/run 向导继续使用原入口。已经运行的 `outputs/agentic_planning01` 不会被自动转成 RSI。RSI 使用新输出目录，可以把兼容的旧 E1/E2 数据作为共同起点。

## 1. 复用已有数据启动精简采集

在 5090 的实验目录执行：

```bash
cd /home/dm/DM_das/bottle_grasp_experiment
bash run_on_5090.sh rsi run --real \
  --seed-campaign outputs/rsi_contact01 \
  --out outputs/rsi_efficient01
```

这里 `outputs/rsi_contact01` 是旧任务，新任务写入 `outputs/rsi_efficient01`。程序要求旧任务已暂停且没有活动批次；自动导入其原始种子和已完成的 A 采集批次，**不把 B 评价或最终验证加入训练**。旧文件保持原位；元数据、标签及实测 TCP 数据会计算哈希，改变后不能静默续跑。该检查不对全部图像和 DD 数据计算哈希。T0、机械臂、触觉和力协议不兼容时会停止。

若只使用单个 E1/E2 会话，改用 `--seed-session outputs/e1e2_A_formal01`。该参数可以重复填写；每项必须指向含 `session_meta.json` 的单个会话或批次。两种参数可同时使用，重复路径会去重。

新建时既不传 `--seed-session` 也不传 `--seed-campaign`，程序会先做配置中的共同 A 采集，再训练初始策略：

```bash
bash run_on_5090.sh rsi run --real --out outputs/rsi_from_start01
```

等价入口是 `bash run_on_5090.sh agentic rsi run ...`。这不会改变旧 `agentic run` 的含义。

现场仍由你摆瓶、确认开爪、标注和复位。采用旧数据时，使用种子会话的 T0；不会因为你当前停在另一个位置就悄悄重定义 T0，也不会从任意当前位置自动规划回程。按终端说明恢复到原 T0；程序只允许已有执行器核对过的安全退让位衔接。停止其它遥操/推理后，再确认首段运动的 `MOVE`。

## 2. 启动后会依次做什么

1. 读取兼容的种子数据，或先完成共同 A 采集。检查力、DD、位姿与记录是否满足原执行器要求。
2. 训练初始 ContactPolicy，并做按完整 session 或完整试验分组的留出检查。样本不足或误差超标时暂停，不把未通过的策略用于 B。
3. 从长期方法库检索相关方法、有效证据和反例；把实际输入及 Agent 的提案写入 `prompt_buffer/`。
4. Agent 提出一个新方法，或提出复用已有方法，或者解释为何应暂停。新方法必须能用已实现的特征与规则表达。
5. 形成旧方法 `old` 与候选方法 `new` 两个分支。它们共享相同起点数据，各自分配同样的新采集预算，按交错顺序执行 A 采集。已有记录的点获得较低的覆盖补充优先级；每分支每个名义点的扫描上限按整轮累计，零点监测另计。
6. 两个分支各自训练 ContactPolicy，在预先确定、相同的测点与瓶子上交错执行小样本 B 筛查。筛查只决定是否值得补充，不能晋升方法或证明两方法一样。
7. 若筛查成功率差的绝对值达到预先配置的 0.2，程序在相同测点和预算下，用**全新的独立 B 试验**正式比较。只有独立确认满足固定评估要求且候选有足够优势，才替换当前方法；否则保留旧方法并保存结果。递归模式下一轮可检索筛查和比较经历。
8. 达到本次轮数/预算限制时，冻结当前策略并执行预先配置的最终 A/B 验证。最终验证数据不再反馈训练或方法晋升。Agent 选择复用当前方法时仍执行等预算的同方法重复对照，检查稳定性，不把它算作发现了更好的新方法。

方法分支 `old/new` 和机器人条件 `A/B` 是两套不同的标记：前者比较“怎样采集更有效”，后者表示“不纠正/使用触觉恢复”。

运行时你会看到类似 `RSI collection / new` 或 `RSI evaluation / old` 的阶段提示。每个试验的开合、轻触、提起、保持、放下和退让仍由现有 E1/E2 状态机完成。

## 3. 标注、暂停和续跑

每个试验结束仍使用终端菜单：

| 输入 | 含义 |
| --- | --- |
| 回车或 `1` | 正常 |
| `2` | 倾倒 |
| `3` | 推动 |
| `4` | 掉落 |
| `5` | 滑动 |
| `0` | 作废 |
| `n` | 备注 |
| `q` | 在当前试验结束后的菜单退出 |

按菜单提示检查瓶子是否仍在标记处，必要时复位。不要把“自动流程走完”当成人工正常标签；成功统计还会检查提起、HOLD 与掉落状态和记录完整性。

`q` 是正常暂停方式。下次直接执行：

```bash
bash run_on_5090.sh rsi resume --real --out outputs/rsi_efficient01
```

续跑读取原目录中冻结的配置、方法、模型、计划和 T0。**不要再加 `--seed-session`、`--seed-campaign`、`--provider`、`--memory-mode`、`--task-id` 或旧入口的 `--planning agent`。** 已完成试验保留，未完成尝试按原执行器规则归档到 `interrupted/` 后处理；额外尝试会占用预算。已结束且没有待做工作的 RSI 会话直接退出，不连接硬件。旧 `rsi_contact01` 的续跑仍使用创建时的旧协议；新精简协议需要新目录。

Ctrl+C 沿用现有软件停止规则：尚未提起时停止机械臂并开爪；已经提起时保持夹爪，现场处理。逐指超力的既有开爪规则仍优先。软件停止不替代现场急停。

如果提示存在未处理的空中/恢复状态，先在现场把瓶子安全放回支撑面、确认夹爪完全张开并恢复原 T0，再执行：

```bash
bash run_on_5090.sh rsi recover --out outputs/rsi_efficient01
```

核对提示后输入 `RECOVERED`。这个命令只记录人工恢复声明，不连接机器人、不替你执行移动；随后再 `resume --real`，首段动作仍需 `MOVE`。不要用 `recover` 绕过尚未处理的实际状态。

## 4. 查看进度、结果和长期经验

以下命令不执行机器人运动：

```bash
bash run_on_5090.sh rsi status --out outputs/rsi_efficient01
bash run_on_5090.sh rsi report --out outputs/rsi_efficient01
bash run_on_5090.sh rsi memory
```

`status` 显示当前阶段、活动批次、方法/策略版本和最近错误。`report` 更新并打印 `outputs/rsi_efficient01/REPORT.md` 的位置；采集进程仍持有该会话锁时，可以先直接查看已经生成的报告，避免另开写入进程。`memory` 列出默认方法库的版本与证据状态；使用其它方法库配置时传 `--config 你的配置.yaml`。

主要结果这样读：

- `new_better`：本次固定预算和预定评估下，候选方法的优势达到当前程序的保守判据；支持在本任务范围保留它。
- `old_better`：证据支持旧方法更好；保存候选的失败经验。
- `inconclusive`：区间重叠、优势不足或样本不足，不能据此说两种方法一样好，也不能自动晋升候选。
- `invalid`：起点、预算、覆盖或记录完整性等对照条件不满足；不能把这次对照用于晋升。
- `pending`：方法尚无足够晋升证据；不是已经学到可迁移规律。
- `current_task_valid` / `cross_task_valid`：分别有当前任务的支持性证据/至少两个不同显式问题签名的支持性证据。仅更换任务 ID 不算新问题；仍须检查实际问题是否不同且可比、适用范围与反例。程序的这个状态本身不证明一般化 RSI。
- `contested`：同时存在支持和反例，应先看条件差异；不能只看以前成功的那次。

达到时间、试验或轮数上限，表示本次预算结束。它不等于“所有问题都解决了”或“已经没有值得补的数据”。

## 5. 单独训练 ContactPolicy

可以先离线检查旧数据能否训练出通过门禁的策略：

```bash
bash run_on_5090.sh rsi train \
  --session outputs/e1e2_A_formal01 \
  --out outputs/contactpolicy_from_A.json
```

多份兼容数据重复传 `--session`。这个命令不需要 `--real`，只读取本地数据、训练并写模型文件；模拟数据才加 `--simulate`。它不会替换正在运行会话的策略。

查看 `usable_for_B`、`validation` 和模型文件的 `blocking_reasons`。`usable_for_B: true` 表示通过当前离线几何预测检查，真正能否提高抓取成功率仍需独立实机 B 评估。

新模型 JSON 同时保存 `skill_conditions`：何时调用、何时成功退出、实际容差和条件版本。最终报告也直接展示这三项。外部调度器可调用无运动副作用的 `ContactPolicy.should_start()` / `should_exit()`，输入现有 PROBE 结果和经过检查的上下文；接口、完整条件和旧模型兼容方式见 [ContactPolicy 调用与退出](CONTACT_POLICY_CONDITIONS.md)。独立训练请使用新的 `--out` 文件名，不覆盖历史模型。

当前 ContactPolicy 是小型 MLP 集成，默认输入两指当前 PROBE 描述子，包括质心、外框、面积与主轴角；可以配置数值 DD 的 4×4 深度池化。**当前模型没有接入六维力、腕部图像或时间历史作为学习输入，也没有实现 LeRobot 数据导出。** 这些数据的原始记录继续保存，但“已保存”不代表“已作为模型输入”。

## 6. 三种记忆方式与新任务

新建时用 `--memory-mode recursive|fixed|none`。具体含义和比较注意事项见 [RSI 设计说明](RSI_DESIGN.md)。日常运行用默认 `recursive`。

要检验上次经验能否帮助新的问题，使用**新输出目录**、明确的问题描述，并保留相同的方法库路径，例如：

```bash
bash run_on_5090.sh rsi run --real \
  --seed-session outputs/另一个兼容的起点会话 \
  --task-id bottle_problem_02 \
  --problem '替换成这次实际改变的问题或失败场景' \
  --memory-mode recursive \
  --out outputs/rsi_contact02
```

示例里的起点会话要替换成实际目录。不同硬件协议、模拟/实机证据不会被当成同一适用范围的成功经验。给同一个问题换一个 `task-id` 不构成迁移能力证据。

## 7. 默认工作量和配置

新 `rsi run` 默认使用 `config.rsi.efficient.yaml`，底层安全与运动设置来自 `config.e1e2.yaml`。当前默认研究范围为 a 轴 −10～+10 mm；候选网格基础间隔 2 mm；没有种子时，共同 A 起点为 −10、−5、0、+5、+10 mm，各 6 次，轻重瓶各半。若明确需要旧协议，新建时传 `--config config.rsi.yaml`。

每轮每分支新增 12 次 A 扫描，按 4 次一小批选点；每个名义偏移默认最多 4 次扫描（整轮累计，轻/重瓶平衡），零点监测另计。Agent 若有证据说明需要更多重复，必须在提案的 `repeat_justification` 中记录理由；默认绝对上限为每点 8 次，分支总次数不增加。旧方法没有该说明时仍按 4 次限制。对每个候选点，程序在 Agent 方法评分上加入同样的已有覆盖补充项：`0.5 / (1 + 邻域已有 A 次数)`，邻域默认 ±1 mm；这是启发式，不表示这些观测完全相同。筛查每分支 10 次 B（5 个偏移，每种瓶子各一次）；筛查差异达到 0.2 时，另做每分支 20 次全新 B 正式确认。

按默认零点监测计算，仅筛查的一轮通常 **48 次**，触发独立确认通常 **92 次**；最终 A/B 验证另需 44 次。分组留出仍按完整 session 分开，新模式仅用样本数量平衡训练/留出规模，不能证明跨日泛化。筛查样本少，可能漏掉真正的改进；不晋升也不等于两种方法相同。默认最多 3 轮、400 次新增尝试和 8 小时；程序预留独立确认的最坏情况预算，实际次数以 `rsi.json` 为准，中断也计数。

如需改变工作量，先复制并编辑 YAML，再用 `--config 新文件.yaml` 新建一个输出目录。不要在已经冻结的对照中途改变预算、评估点或判据。缩小评估样本会更容易得到 `inconclusive`；不能为得到“有效”而事后降低标准。

## 8. 原始数据与提示词保存在哪里

完整目录解释见 [RSI 设计说明](RSI_DESIGN.md#数据与证据目录)。最常用的位置：

```text
outputs/rsi_efficient01/REPORT.md
outputs/rsi_efficient01/rsi.json
outputs/rsi_efficient01/prompt_buffer/
outputs/rsi_efficient01/cycles/001/protocol.json
outputs/rsi_efficient01/cycles/001/screening.json
outputs/rsi_efficient01/cycles/001/comparison.json
outputs/rsi_efficient01/models/<model_id>.json
outputs/rsi_efficient01/batches/<batch_id>/trials/<trial_id>/streams.h5
outputs/rsi_method_memory/methods/<method_id>.json
outputs/rsi_method_memory/experiences/<experience_id>.json
```

真实采集继续保存服务提供的处理后 DD、六维力、扣零法向力、开度与指令、TCP 指令/实测位姿、阶段、重抓和人工标签。腕部图像依然是阶段关键帧，胶面原图依然是选定时刻关键帧；不是新增连续视频录像。处理后 DD 也不应标成 SDK 原始 DD。

方法库保存经验和证据引用；原始试验文件保留在采集目录里。备份时应同时保存旧种子会话、新 RSI 输出和方法库，不能只备份 `REPORT.md`。

## 9. 常见暂停

| 提示 | 下一步 |
| --- | --- |
| Agent 网络请求失败 | 先恢复笔记本代理/到 5090 的现有隧道，用 `bash run_on_5090.sh agentic auth test` 检查；之后 `rsi resume`。网络失败不会悄悄改用本地方法。 |
| ContactPolicy 样本不足或验证不合格 | 看模型 `blocking_reasons`；先检查可用、独立、实测定位的样本数量与质量，再决定补采或重建任务。不要改 `usable_for_B`。 |
| 方法提案未通过协议 | 查看 `prompt_buffer` 中的驳回理由。程序最多允许三次格式修订；不能表达的新感知或动作能力应暂停。 |
| 不能凑满等预算计划 | 方法的间隔、去重复距离或单点次数限制互相冲突；本次计划不执行，不偷偷补随机点。 |
| 原始种子或已冻结配置变化 | 保留原实验，核对修改来源；需要新设置时新建任务。不要手改断点文件绕过。 |
| DD 缺失、力/开度/移动异常 | 沿用原 E1/E2 故障处理；先确认真实状态与服务，再续跑。学习层不会关闭这些保护。 |

本功能的离线测试通过与否、是否完成实机验证，应以交付记录和你现场采集的证据为准；示例命令本身不代表已经执行了实机实验。

初始策略训练不足时会暂停；可以用更多兼容种子新建会话，或不传种子执行共同起点采集。某轮方法的新增数据导致训练门禁未通过，程序保存未通过的模型和原因，保留原策略，交回 Agent 提下一种方法；不会反复对同一数据盲目续训，也不会伪造该方法的实机失败率。

新问题可在新建命令加入 `--problem '真实改变的问题或失败场景描述'`。同一个问题应保持描述一致；只有不同显式问题签名的支持性对照，才允许标记 `cross_task_valid`。仅改 `--out` 或 `--task-id` 不会晋升。描述还需与实际实验一致，系统不能凭文字证明任务真的改变。

精简模式只设一个采集预算检查点；筛查结果在 `cycles/NNN/screening.json`，不能据一个点声称学习曲线。触发独立确认后会保存对应 `comparison.json`；旧模式的多检查点曲线位于 `learning_curve.json` 和 `learning_curve.png`，其中的 80% 首次达到预算不是成功率置信保证或理论最少数据量。

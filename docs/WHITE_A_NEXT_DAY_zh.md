# 白瓶 a 轴：明天完成的位置恢复与主任务续接实验

目标：主 policy 在抓准白瓶时，已有 forcepolicy 可以帮助完成任务。本实验只解决抓偏，固定理想 T0 对应的双指目标接触区域。先确认接触位置，再交已有 forcepolicy 调整夹持力，最后由主 policy 完成任务。无需重新训练 forcepolicy，也不需要视觉标记、l 轴或花瓶任务。

本次软件准备只做代码和离线测试，没有启动真机。实际效果必须通过下面的真实结果验证；代码通过测试不等于新 skill 已训练好。

## 已经接通的接口

1. 新 `learned_a_skill_v1` 同时带调用头、恢复动作模型、退出头，能导出给当前 V4 主入口。
2. 初始接触会让主模型暂停以获得规定的轻触观测。这是观测准备，不自动算一次位置恢复调用。
3. 学到的调用条件判定是否需要恢复。正常则不平移重抓；未知、适用范围外或条件矛盾时不假装正常。
4. 成功退出要求目标位置成立、稳定双侧接触和新观测；返回明确的 `before_squeeze` 状态。合同声明下一控制模块是 `existing_forcepolicy`，主模型恢复还需等力控准备完成。
5. 控制层接收状态、清掉旧动作和历史缓存、收集新观测，再走原力控交接。不会向未经训练的 VLA 增加它不认识的输入字段。
6. 新版主入口在位置确认前禁止正式 forcepolicy 接管。轻触阶段保持原有的 PROBE 控制；后台算出力预测不等于已经执行夹持力控制。
7. 最终报告分别记录：需要调用、实际调用、实际非零位移、位置恢复成功、力控交接、主模型重新发出动作、独立人工确认任务成功。

范围是固定桌面、固定瓶位、当前传感器、a 轴、每 episode 一次接触位置恢复。多 skill 路由和多次任务阶段接管不属于明天的验收。

## 旧数据与时间安排

旧白瓶 A 共 26 次，均保存了一个有效双指接触窗口；采集 wall time 合计约 43.7 分钟，中位数约 85.8 秒/次，不包括全部场外准备。旧 -5/0/+5 mm 三组触觉质心有明显差异，但偏移命令绝不是正常/需恢复标签。

旧 A 可以保留作动作几何监督；按新目标人工复核的接触窗口也可训练调用/退出头。审核工具只生成未标注模板和描述子示意图，不按偏移、旧模型结论或提起结果自动填标签。审核视图与原数据不允许同时当作两份独立试验训练。

建议以一个完整实验日为预算：

| 阶段 | 建议用时 | 完成条件 |
|---|---:|---|
| 现场恢复与目标确认 | 30–45 分钟 | 理想 T0、两瓶标记、空载扣零、稳定轻触及目标区域明确 |
| 复核旧数据／补 A | 60–120 分钟 | 正常与确实需 a 修正的两类样本，以及达标/未达标窗口齐全 |
| 首轮训练、B、Agent 补采 | 60–120 分钟 | 至少出现真实非零恢复；不确定与错误按证据诊断并补采 |
| 冻结独立 B | 45–75 分钟 | 同一模型达到冻结的接触 skill 指标 |
| 接主入口并验证整段任务 | 60–90 分钟 | 位置恢复→原 forcepolicy→主 policy，真实任务结果独立记录 |

这不是保证几点成功的承诺。若所有样本都标正常、夹爪无法建立规定轻触、B 始终拒绝或动作后仍不能退出，就停在对应问题上，不降低标准凑一个成功视频。

## 上午先做什么

- 恢复已确认的理想 T0、桌面/底盘和瓶子标记。若参考位姿改变或需要改夹爪协议，必须用现有流程建立新参考会话，并在新 campaign 中使用它。
- 两指全开空载后扣零，检查两指触觉和力都是新数据；已有右手 DD 服务及机器人底层服务按原现场流程启动，不重复启动另一份触觉 owner。
- 正常目标定义为“与理想 T0 的双指目标接触区域一致、稳定、不靠边”。不是“碰到任意胶面”。人工先确认几个参考接触样本，再审核其它窗口。
- 位置 skill 只需要规定的轻触力。旧 E1/E2 后续固定夹紧阶段的 `force_not_reached` 单独记录，不能反推“位置不对”。但若连轻触都未建立，不能拿该窗口充当正常接触训练。
- 0.336 是旧记录的闭合下限；不能因为写在配置里就视为重新验证。若改变它，0.334/0.336 数据分开处理，不混成同一协议。

## 采集与训练

代码部署完成后才初始化明天的新 campaign。不要修改旧 campaign 的冻结配置或 source hash；旧离线检查目录不是明天的正式数据目录。

仓库：

```bash
cd /home/dm/DM_das/bottle_grasp_experiment
```

先复核旧 A。已生成的 `outputs/white_a_contact_review_20261109/index.html` 可以直接在浏览器打开；全部标签仍为空。灰色十字来自旧记录 `00023`，只是候选参考，需与你现场确认的理想 T0 接触区域核对。图是质心和包围盒示意，不是原始触觉图；如果不能据此判断，保留未知并补采现场可确认的窗口。

逐条填写首次是否需要恢复、每个窗口是否达到目标；最后接触是否达标与提起成功分开。填写审核人、确认声明后，页面下载 `annotations.json`。把该文件放到 6f5090 的仓库输出目录，例如 `outputs/white_a_contact_annotations_20261110.json`，再执行下面的离线转换：

```bash
bash run_on_5090.sh skill-label-review apply \
  --review outputs/white_a_contact_review_20261109 \
  --annotations outputs/white_a_contact_annotations_20261110.json \
  --out outputs/white_a_contact_review_view_20261110
```

这会新建带人工标签的训练视图，不改原录制、不增加试验次数。不要直接使用包内尚未填写的 `annotations.json`；原数据和其审核视图不能同时作为两份样本。审核后的类别覆盖和分组验证是否够用，由训练检查决定；无需为凑数量编造正常或异常标签。

复制并审核 `config.white_bottle_a_loop.yaml` 为现场配置。目标文字可按实际确认的区域补充；只有现场确认参考场景和夹爪协议后，才在初始化前设置 `physical_setup_confirmed: true`。若协议没变可用已审核数据视图作 seed；不要同时传入原会话与其审核视图。

```bash
bash run_on_5090.sh skill-loop init \
  --config config.white_bottle_a_loop.tomorrow.yaml \
  --out outputs/white_a_joint_20261110
```

有合规审核视图时，在 `init` 中添加 `--seed-session <审核视图目录>`；没有则从新的 A 开始。默认新 A 是 -5、-2.5、0、+2.5、+5 mm，每档两瓶共 8 次，40 次正式扫描加 4 次零点监测。它是冷启动模板，不是要求旧 26 次全部重做。

完成现场准备后才执行：

```bash
bash run_on_5090.sh skill-loop run \
  --out outputs/white_a_joint_20261110 --real --allow-agent
```

保留逐阶段运动确认、换瓶与人工标签。程序自动在批次之间训练、验证、调用 Agent 设计补 A 并更新三部分。正常接触不能硬标“需恢复”，未达标窗口和未知也必须保留。若首轮就满足指标，正常进入冻结独立验证，不为展示 Agent 补采而故意制造失败。

默认开发 B 为 12 次扫描（另有零点监测），独立 B 为 24 次扫描（另有零点监测）。需要至少 3 次有实际位移与新触觉回执的成功位置恢复；只有零偏移正常例无法验证恢复作用。详细比例标准见 `A_AXIS_JOINT_LOOP_zh.md`。

## 导出到你当前使用的主入口

只有独立 B 通过、campaign 为 `complete` 时才允许下面的离线导出。导出会重新核验原始 B 记录，不仅相信状态文件中的“完成”字样。

```bash
bash run_on_5090.sh primary-skill-export export \
  --campaign outputs/white_a_joint_20261110 \
  --out outputs/white_a_primary_20261110

bash run_on_5090.sh a-axis check \
  --skill-config outputs/white_a_primary_20261110/scheduler.json
```

该配置明确绑定新模型、新目标和参考状态；没有 `--skill-config` 时仍会用旧默认模型，因此不能漏掉。

## 主任务验证顺序

先做一个 0 偏移对照：目标位置正确，应不执行 a 轴重抓；位置确认后才交原 forcepolicy，再由主 policy 继续。随后用已经采集并验证的正负偏移测实际纠偏。以下命令均是现场动作命令，本次开发未执行。

```bash
CONTACT_TABLETOP_CONFIRMED=1 bash run_on_5090.sh a-axis run \
  --real --route pilot --offset-mm 0 --after-skill handoff \
  --skill-config outputs/white_a_primary_20261110/scheduler.json \
  --trial-id white_a_zero_light_01
```

只有在瓶子支撑/标记、空载起点和整段动作路径都已现场核对时才设置 `CONTACT_TABLETOP_CONFIRMED=1`。这个 0 偏移入口会定位到参考位，不是保持当前 TCP 不动。需要失败时只停在 skill 端检查，可明确用 `--after-skill stop`，不会让主模型继续。

偏移试验把 `--offset-mm` 换成已经验证的 `5` 或 `-5`，并换新的 `--trial-id`；是否应调用仍由实际接触判断。两只瓶子分别做正常对照和有实际恢复的试验。

受控前奏验证了位置恢复和续接，还需要至少分别观察轻/重白瓶的自然主任务入口：

```bash
CONTACT_TABLETOP_CONFIRMED=1 bash run_on_5090.sh a-axis run \
  --real --route main \
  --skill-config outputs/white_a_primary_20261110/scheduler.json \
  --trial-id white_a_main_light_01
```

自然入口让主 policy 自己接近，不能混入 `CONTACT_EXPERIMENT_B_OFFSET_MM`。它要验证“主模型开始→接触判断→实际恢复（若需要）→forcepolicy→主模型继续”。受控前奏没有主模型最初接近这一段，不能混称自然入口全链验证。

每次结束后，使用终端打印出的真实 episode 目录独立登记结果：

```bash
bash run_on_5090.sh a-axis review <实际episode目录> \
  --outcome success --note '根据实际观察填写：瓶子、是否实际纠偏、夹持及放置结果、视频编号'

bash run_on_5090.sh a-axis summary <实际episode目录>
```

失败填 `failure`，看不清填 `unknown`。不要直接复制示例文字作成功依据。报告里的 `main_execution_resumed` 只证明恢复后发出了动作；`task_success` 来自独立观察；`learned_recovery_to_task_success_observed` 还要求真实非零纠偏和完整主任务续接证据。

## 明天“做完”的判据

- 正常接触：不执行不必要的 a 位移，能进入原力控与后续任务。
- 抓偏接触：调用新位置 skill，发生可核对的实际 a 修正，重新观察后达到目标才退出。
- 退出失败、未知或超时：不能提前放行正式 forcepolicy 或主模型后续动作。
- 成功退出：先交原 forcepolicy，力控准备完成后主模型用新观测续接，独立记录任务成功。
- 至少有正常对照、正负方向的真实恢复、两种重量、自然主任务入口的重复记录，保留全部失败。单条成功视频证明链路能运行，不证明成功率已经提高；要报告“提高多少”，需同条件的主 policy+forcepolicy 原基线对照并统计全部结果。

本实验完成的是 a 轴位置恢复的可复现试验。接下来再讨论 l 轴、学习夹持力或擦花瓶。

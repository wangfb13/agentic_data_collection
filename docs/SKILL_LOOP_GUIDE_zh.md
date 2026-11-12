# 药瓶完整恢复 skill 与证据探索闭环（2026-11-06）

本次所有改动、训练、测试及结果都在 5090 的 /home/dm/DM_das/bottle_grasp_experiment。
没有启动真实机器人，也没有新增机器人交互。现有未提交改动保留；起点 Git 为
128b7a2b69711c03ac92d15f4a1d88d15d6ab9c9，起点源码/未提交补丁/状态保存在
outputs/skill_baseline_20261105/。旧 e1e2、agentic、rsi 默认行为保留，新 skill 显式启用。

## 1. 三个输出与控制权

可加载包：outputs/skill_offline_20261106_v2/bottle_skill/，包含 skill.json、model.json、PROVENANCE.json。

- invocation：needs_recovery（需要/正常/未知）、applicable/verified（已验证适用范围）、
  candidate_supported（候选模型观测支持）、invoke。需要恢复与已验证适用不是同一件事。
- recovery：已有 ContactPolicy 纯触觉 ensemble MLP 输出 a 轴 mm 修正；打开、退让、平移、重新接近、
  重新轻触。实测动作回执和动作后新观测是必需的；动作/观测有上限，沿用原力、开度、运动及命令所有权保护。
- exit：目标为参考接触区内、非边缘、稳定的双侧接触。残余修正低于原 servo.tol_mm，
  力稳定至少 0.2 s，并至少两个时间分离的新鲜观测窗。success/completed=true 才允许主策略继续夹紧/提起；
  incomplete/completed=false 表示未完成，交给主策略重规划/操作员。完成接触恢复不等于已经完成提起任务。

控制位置：主策略到安全预抓位 → adapter.initial_probe 首次轻触 → skill.start 调用判断 →
skill 拥有有限重抓循环，每次动作后观测 → 显式 handoff → 主策略后续操作。
初始已经正常也需要目标确认，不会为了调用而强行修正；暂不适用、未知、过期、超时/耗尽不会称作成功。

部署示例（不创建任何硬件连接）：
```python
from bottle_scan.recovery_skill import load_skill_bundle
skill = load_skill_bundle("outputs/skill_offline_20261106_v2/bottle_skill")
result = skill.run(adapter)  # adapter 实现 initial_probe/execute/observe/handoff
```
adapter.execute 返回 applied、实测 action_mm、completed_ns；观测与回执使用同一时钟。
模型输入是允许列表内的触觉描述子和必要观测上下文；计划偏移、真实误差、试验编号不能成为运行输入。
本次包没有真实前瞻证书，默认 deployment 会拒绝未经验证的调用。
development 仅用于明确的 B 实验，输出仍标记 verified=false，不能解释为正式部署验收。

## 2. 今天实际验证了什么

最终报告：outputs/skill_offline_20261106_v2/REPORT_zh.md 和 report.json。

310 次历史尝试，122 条合格 A；候选池105、开发12、按整 session 保留5。
B 开发33；历史保留B100（正式scan92）。原始数据未被修改，训练视图仅引用原试验。
三分支使用同一 ContactPolicy、种子、动作空间和24/48/72采样预算；本地证据规则称
agent_evidence，不冒充实际调用了远程 LLM。原有 LLM RSI 方法规划仍可独立使用。

|72 条 A 预算|A 保留 MAE mm|P90 mm|
|---|---:|---:|
|fixed|1.300|3.376|
|simple_feedback|0.804|2.095|
|agent_evidence|0.591|1.270|

保留集只有5条，不能据此宣称统计优势。24条预算时三组MAE都是0.467 mm，学习曲线并非单调改善。
共同 MAE<=1/P90<=2 几何门槛首次均在24条达到；加上共同开发开销是69条历史回放样本。
完整 skill 的同等能力交互量目前全部 not_established，见 sample_efficiency.json。

B_formal01 的正式scan超出原2 mm横轴一致性监督门禁，调用真值保留未知。
历史92条B的75%实际成功率来自旧控制器，不能归因于新skill。
新skill双窗口退出在旧数据没有前瞻证据；缺少证据保持unknown。
历史RSI validation可能已经用于旧方法比较/后续A设计；本次严格不指导补采，但并非从未曝光的盲测。

重新运行离线对照（新目录，拒绝覆盖）：
```bash
cd /home/dm/DM_das/bottle_grasp_experiment
.venv/bin/python skill_offline.py --out outputs/skill_offline_new --epochs 500
.venv/bin/python -m pytest tests -q
```
使用现有 NumPy 学习器而未替换为不同 GPU 学习器，保证对照一致。
宿主 .venv 已补测试缺少的 pyzmq 27.3.0。全库 pytest 应指定 tests，避免把 outputs 历史源码快照当成同名测试。
早期失败日志保留；最终全套回归947 passed、13 skipped、391 subtests passed，见regression_delivery.log；最后元数据修复另见final_metadata_regression.log。

## 3. a 轴下一轮 A—训练—B—反馈

以下 plan/train/evaluate/feedback 都不连接硬件。collect --real 会运动，只在现场启用机器人后执行。
run_on_5090.sh 使用现有 ROS 容器和原硬件配置，不启动机器人服务。原 MOVE 确认、标签和异常恢复仍保留。

离线查看计划：
```bash
bash run_on_5090.sh skill-collect plan --cond A --out outputs/skill_A_preview.json
```

新增A（保留原固定T0、协议；必要时现场输入原开度下限）：
```bash
bash run_on_5090.sh skill-collect collect --real --cond A --role exploration \
  --reference-session outputs/e1e2_A_formal01 --out outputs/skill_A01
bash run_on_5090.sh skill-collect train \
  --session outputs/e1e2_A_formal01 --session outputs/skill_A01 \
  --out outputs/skill_model01.json
```
训练保留全部原始状态，模型拟合仅用兼容、可靠的观测；最终独立验证不会进入训练。
已有 ContactPolicy include_successful_B 接口可显式使用开发B的实测成功转移，默认关闭；
几何监督及成功演示均不声称是最优动作。

今天导出的 a 模型可直接用于明确的开发B：
```bash
bash run_on_5090.sh skill-collect collect --real --cond B --role development \
  --mode development --fit outputs/skill_offline_20261106_v2/bottle_skill/model.json \
  --out outputs/skill_B_dev01
bash run_on_5090.sh skill-feedback evaluate --session outputs/skill_B_dev01 \
  --role development --out outputs/skill_B_dev01_report.json
bash run_on_5090.sh skill-feedback feedback --report outputs/skill_B_dev01_report.json \
  --config config.skill.bottle.yaml --exploration-config config.skill.exploration.yaml \
  --out outputs/skill_next01 --budget 12
bash run_on_5090.sh skill-collect plan --cond A \
  --selection outputs/skill_next01/selection.json --out outputs/skill_next01_preview.json
bash run_on_5090.sh skill-collect collect --real --cond A --role exploration \
  --selection outputs/skill_next01/selection.json \
  --reference-session outputs/e1e2_A_formal01 --out outputs/skill_A02
```
再train新版本、B开发、evaluate、feedback。selection保留问题/证据/原因，直接转为下一批，参考重复计入预算；
a入口拒绝夹带l或力度变化。固定采集和简单反馈可通过 feedback 的 --mode fixed/simple_feedback/agent_evidence 选择；
预算以计划尝试和实际尝试分别记录，不能把跳过/失败或开发B开销删除。

标签菜单额外记录首次接触需要恢复/已经正常/暂不适用/未知，及交接前是否实际达到目标接触。
不确定按回车保留未知，不能用计划偏移或模型输出代填真值。
A保存首次轻触；B保存每次动作/动作后观测/确认窗/交接及下游抓取结果，开发转移可作为后续探索证据。
未完成、作废与失败保留，旧记录不会被补造新标签。
B预先校验模型与力/触觉/坐标协议，并固定恢复原训练T0，不会悄悄重定义参考位。

## 4. 冻结与独立评测

模型、运行规则、力协议及观测范围确定后，先冻结，再开全新独立B目录；
其结果只用于评测/证书，feedback拒绝 independent 报告。
若需调整规则或补点，开启下一版本并重新保留独立评测。

```bash
bash run_on_5090.sh skill-collect collect --real --cond B --role final_validation \
  --mode development --fit outputs/skill_model01.json --out outputs/skill_B_final01
bash run_on_5090.sh skill-feedback evaluate --session outputs/skill_B_final01 \
  --role independent --out outputs/skill_B_final01_report.json
bash run_on_5090.sh skill-feedback certificate --report outputs/skill_B_final01_report.json \
  --fit outputs/skill_model01.json --config config.skill.bottle.yaml \
  --out outputs/skill_certificate01.json
```
开发模式只允许在已支持的候选域试验；final_validation这个数据角色仍封存，不能拿来调参。
证书绑定模型、完整runtime、上下文、原始证据哈希，验证调用/动作/退出三者联合表现。
默认至少30条合格样本、Wilson95%下界>=0.8；不足输出空observed_cells及原因。
需要可用的连续观测单元时，在开发阶段依据训练/开发证据预定义
config.skill.bottle.yaml 的 certification.observation_cells：
cell_id、status、context、centroid_bounds_px（两指×u/v×上下界）。
这些单元在独立采集前随session冻结，禁止事后扩大或重叠。
不配置时仅精确观测点可被验证；不能用少量旧样本任意画宽适用域。

拿到非空合格证书后，以同一模型/runtime导出新bundle或collect传 --certificate 并用deployment。
原先已导出的无证据bundle是不可覆盖的起始版本，不能通过原地改JSON绕过完整性校验。

## 5. 独立物理因素：局部 l 接触位置，再检查力度和少量组合

配置 config.skill.extension.yaml；l与a正交，必须依据实测T0核对它与世界高度的关系，
不能直接把局部l称作绝对瓶身高度。目标/单位/执行接口已配置化。
先固定a、姿态和轻触力度，采用独立重接触重复；再力控制实验，最后少量a+l组合。
开爪后退让、平移、轻触；不提起。原高度/力/开度/速度限制保持。

已生成：
outputs/skill_extension_height_plan_20261106.json（9次pilot）；
outputs/skill_extension_height_training_plan_20261106.json（24次，8个重接触block）；
outputs/skill_extension_force_plan_20261106.json（9次力度）和 outputs/skill_extension_combinations_plan_20261106.json（18次组合）。
9次pilot不足既有20样本训练门槛，不放宽门槛。

```bash
bash run_on_5090.sh skill-extension collect --real \
  --design-file outputs/skill_extension_height_training_plan_20261106.json \
  --out outputs/skill_height_A
bash run_on_5090.sh skill-extension train --session outputs/skill_height_A \
  --out outputs/skill_height_model.json
bash run_on_5090.sh skill-extension collect --real \
  --design-file outputs/skill_extension_height_training_plan_20261106.json \
  --fit outputs/skill_height_model.json --mode development --role development \
  --out outputs/skill_height_B_dev
```
冻结后另开B目录用 --role independent。l模型复用同一学习器，监督为实测TCP投影l；
必须固定a/c/姿态和力，通过触觉可识别性/整重复留出门禁。没有真实数据时不可用，
禁止把a模型改轴名当l模型。l无接触/边缘方向回退未获证据，保持拒绝。
l B具有实测l动作回执、新观测和相同success/incomplete交接；当天只有假硬件测试，没有真实l结果。
力度/组合改变后必须重新检查适用条件，不能自动沿用固定力度的l模型。

## 6. 方法记忆、逐轮证据与成本

最终v2目录：
- rounds/：9轮“问题→实验选择及原因→结果→skill更新”，原始证据与哈希。
- method_memory/、method_memory_links.json：原 MethodStore API，验证/晋升规则不变；当前均pending，不因离线代理指标晋升。
- cost_events/、cost_summary.json、source_costs.jsonl：实际历史310尝试/75失败；新增机器人0。
  A、B开发、B保留、失败、适用性检查、机器人阶段、人工复位、源任务经验及离线计算分列。
  历史未知复位/纯运动成本为null；旧试跑的离线计算单列superseded，不丢掉失败尝试。
- learning_curve.json、sample_efficiency.json：24/48/72各轮及共同门槛首次观测预算；不插值为保证、不把回放计为新交互。
- 每次实采meta.json：skill_observations、probe_confirmations、skill_outputs、skill_ground_truth、
  skill_cost、operator_waits及原HDF5/关键帧。机器人阶段时长含阶段等待，不能冒称纯机械运动时长；
  复位+确认等待单列，无法区分的纯复位时长仍为null。

feedback通过旧 MethodStore 检索/记录方法。旧库部分文件属于原容器root，推荐上面的run_on_5090.sh入口，
没有改变旧库权限；history_enabled=false或feedback --history-off保留无历史对照接口。
当前只有药瓶内证据，跨任务变量/方法适用性检查接口已留出，未声称跨任务迁移已验证。


## 7. 继续使用原 LLM Agent

config.rsi.skill.yaml 让原 RSI 使用 opt-in 的完整 skill 执行器、原 ChatGPT provider、
方法DSL、比较预算及晋升流程。旧配置不变。创建新 campaign 前可将
skill_feedback_report 设置为独立于最终评测的已生成开发报告路径，例如 outputs/skill_B_dev01_report.json。
Agent 的 references.skill_development 包含调用/恢复/退出/适用性缺口和原始证据；
桥接层会重算报告拒绝篡改或独立评测泄漏。null 表示尚无新skill开发反馈。
例如现场运行：

```bash
bash run_on_5090.sh rsi run --real --config config.rsi.skill.yaml \
  --seed-session outputs/e1e2_A_formal01 --out outputs/rsi_skill01
```

本次未调用远程LLM服务，也未启动该实机campaign。离线对照使用可审计本地规则；
不能把它的结果称为LLM自主探索的实验结果。新反馈报告冻结到新campaign配置，
更新反馈后开下一campaign，避免改动正在运行轮次的快照。

l轴统一三输出评测也可使用skill-feedback evaluate；未来证书使用config.skill.height.yaml，
该配置与l执行器完全一致，不能使用a配置替代。

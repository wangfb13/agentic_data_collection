# 通用符号状态图 V3

仓库位于 `6f5090:/home/dm/DM_das/bottle_grasp_experiment`。本次完成 V3 方案的离线步骤 1–7；没有启动机器人、没有采集新的 A/B，也没有把新模型部署到主策略。

## 入口与代码

```bash
cd /home/dm/DM_das/bottle_grasp_experiment
bash run_on_5090.sh state-graph analyze \
  --config config.state_graph.symbolic_v3_20261112.json
```

这是离线命令。默认读取 V2 的只读特征缓存，在 `outputs/state_graph/20261112_symbolic_v3/` 写新节点、图、模型、事实和 V1–V11 诊断。V9 默认只回放历史前缀并生成导演上下文；没有调用外部 LLM，也不证明采集效率。`--allow-agent` 才会请求真实导演建议，仍不执行采集。

| 位置 | 内容 |
|---|---|
| `bottle_scan/state_graph/skill_interface.py` | 规格、动态加载、观测和动作接口 |
| `bottle_scan/state_graph/{extract_nodes,noise,graph,sink,labels}.py` | 通用节点、距离、四类边、多后置条件及路径标签 |
| `bottle_scan/state_graph/{dynamics,value,policy,skill}.py` | f、V、π 与观测—执行—再观测运行时 |
| `bottle_scan/state_graph/{clusters,facts,operators,director_bridge}.py` | 聚类、符号事实、规则操作、LLM 协议 |
| `bottle_scan/state_graph/{acceptance,report,plots}.py` | 明确标注的 B 验收、按试验组诊断、图表 |
| `skills/contact_position_a/` | 白瓶接触位置实例与 V4 记录适配 |
| `skills/contact_position_a/legacy_v2/` | 旧数据解析及兼容实现；旧方法不再混入通用框架 |
| `tests/fixtures/skill_toy/` | 二维动作、不同观测和两个后置条件的合成实例 |

新 skill 需要规格、观测/动作提取、后置条件判定和执行器适配；无需改通用图或模型。换真实技能仍需相应数据验证，V11 只验证软件通用性。

## 实现要点

- 唯一门控是有效观测与 `task_id`。当前白瓶轻重两类合并为一个任务；绿瓶是另一个任务。协议、时期、盖板/传感器版本、开度、重量只记录，不进入门控或模型输入。
- 当前观测为触觉接触几何 6 维、两指归零 SDK 六维力共 12 维、冻结 DINOv2 ViT-S/14 的 384 维视觉特征。视觉权重 0.5。触觉位置用像素；不使用旧 MAE 特征作为主距离。
- 重复噪声在训练折内拟合：每组残差先乘 `sqrt(n/(n−1))`，修正去均值的小样本偏差，再做 Ledoit–Wolf、差分协方差 `2Σ` 和 Bonferroni。高维视觉 PCA 也只用训练折。
- 删除运动学边。只保留实测转移、观测动力学推断、互近邻相似和后置条件汇点。
- 每个后置条件独立计算 Beta 平滑成功率、`−log p`、R 和调用/退出标签。动作标签是路径上的动作向量之和。
- f 默认 5 个小 MLP；V 和 π 为小样本 Ridge 集成，另有 I/β 对照头。所有设计矩阵仅从 `obs` 构造，f 额外接收动作。隐藏属性、计划坐标和标签不进入输入。
- f 只预测声明的触觉几何及力分量，因此推断边的终点匹配限于这些分量。它没有预测视觉；可选规划对未预测分量保持最新值。该近似在模型和报告中显式记录。
- `p_off` 标签来自分组交叉拟合残差超过阈值，表示模型无法解释的变化；不能直接认定为物体被推动。集成标准差也不是已校准置信区间。
- `unsupported` 与有实测失败证据的 `unreachable` 分开。隐藏属性显著影响结果时可以分块采集；当前观测能力不足时可以提出 `request_capability`。

## 后置条件与历史证据

当前保留四个独立后置条件：

1. `handoff_forcepolicy_floor0330`：当前训练/部署目标，尚无有效 L2 记录。
2. `handoff_e1e2_squeeze_floor0336`：旧夹紧控制器的实测结果。
3. `handoff_e1e2_squeeze_floor0330`：保留原 0.330 批的旧夹紧结果。
4. `handoff_forcepolicy_floor0336`：为未来同盖板、同消费者的成对比较准备，目前无实测结果。

不能把 SQUEEZE 成功当成 ForcePolicy 成功，也不能把不同消费者之间的差异解释成闭合下限的因果效果。现有 ForcePolicy 的目标力来自模型动态预测；规格和记录标明 `predicted_by_consumer`，没有伪造一个固定 5 SDK 目标。

当前消费者版本暂未冻结；采集时记录实际 checkpoint 身份，冻结时固定版本。身份用于证据追溯，不阻止用户更换模型。

## B 验收与导演

在分析配置的 `B_assessments` 中列出 JSON 数组或 JSONL 文件。每个已评估试验至少包含：

```json
{
  "task_id": "white_bottle|tcp_translation_a",
  "group_id": "campaign/trial_001",
  "role": "development",
  "post_condition": "handoff_forcepolicy_floor0330",
  "evaluated": true,
  "initial_truth": "NeedAction",
  "invoked": true,
  "target_truth": true,
  "exit_declared": true,
  "success": true,
  "rejected": false,
  "task_success": null
}
```

真值必须来自预先约定的复核/消费者结果，不能抄模型输出。每个试验组一条；独立 B 还要求 `post_freeze: true` 和 `frozen_model_id`。门槛沿用现有开发 B/独立 B 配置。通过后才产生 `dev_B_pass` / `indep_B_pass`；当前没有合格的新 B，不会产生这些事实。

每张图会输出 `director_current_context.json`，其中接入实际 B 验收；历史前缀回放不读取未来 B。`director_state.frozen_independent` 标识独立阶段，只有该阶段的 `indep_B_pass` 才允许 `complete`。导演提议通过 `bottle_scan.skill_director.advance_symbolic()` 接入现有 provider，采集方案经实例的 `translate_plan()` 转换。此次没有自动执行新 campaign。

## V4 调度与记录

新增独立入口：

```bash
bash /home/dm/DM_das/contact_scheduler_20261104_experiment_b_v4_single_open_20261110/run_state_graph_v3_0330.sh skill-check
```

`skill-check` 只做离线预检。新的配置关闭 `single_correction_open_handoff`，复用 V4 已有的修正—重新轻触—判断退出闭环；旧单次纠正入口的配置保留。当前入口仍加载原 V4 恢复模型，用于后续收集新消费者证据，**不是一个已经训练完成的 V3 ForcePolicy skill**。

开启 V3 记录后，`events.jsonl` 增加修正前后观测窗口、动作回执、交接事件和消费者结果。原始 DD/力在已有 `streams.h5`；保存空载参考和从主程序现有腕部相机接收器读取的图像，不另建相机 PULL 连接。离线抽取沿用五帧触觉几何中位数和同窗口归零力。

- L1：读取实际力就绪事件。明确的 ForcePolicy 就绪超时记录为失败；一般传感器异常保持未评估。
- L2：提起保持无掉落，需真实结果或复核记录。
- L3：主任务完成，需真实结果或复核记录。
- 主策略恢复执行只证明 L1，不会自动变成 L2/L3 成功。

真实实验后可以追加复核结果，例如：

```bash
.venv/bin/python -m skills.contact_position_a.record_outcome \
  --events /实际新episode目录/events.jsonl \
  --l2 success --l3 success --evidence '实际视频路径及观察时间段'
```

这条命令只追加记录，不控制机器人。失败时把对应参数改为 `failure`；没有走主任务时不填 `--l3`。将新 `events.jsonl` 加入分析配置 `sources` 后可抽成图节点。

新 campaign 运行前设置 `CONTACT_GRAPH_MASS_CLASS`、`CONTACT_GRAPH_APPROACH_POSE`、`CONTACT_GRAPH_SENSOR_VERSION`、`CONTACT_GRAPH_TIME_PERIOD` 作为记录属性；这些值不参与模型推理。

## 还需要的真机数据

按方案第 12 节，在操作员授权后采集：同一盖板下轻/重白瓶平衡分块；A 轻触直接交给 ForcePolicy；B 保存每次修正前后轻触和实际动作；记录 L1/L2，走主任务时登记 L3；逐步扩展偏移到出现失败；至少两种明确命名的接近姿态；0.330/0.336 在同一消费者下成对比较。随后训练、冻结，再采独立 B 和主任务对照。

旧数据里没有登记两种姿态，所以真实 V10 暂不可算；不能从日期或 T0 变化猜造姿态标签。历史分组验证不能当作新 campaign 的独立验收。

## 文件与可追溯性

V3 结果在 `outputs/state_graph/20261112_symbolic_v3/`，报告为 `REPORT_zh.md`，交付总结为 `IMPLEMENTATION_zh.md`，图在 `figures/`，各任务目录见 `task_index.json`。`code_before/`、`scheduler_code_before/` 保留本次修改前代码。原 session、旧标签和 V1/V2 分析产物未改写；完整性检查范围和哈希记录见 `integrity_audit.json`，不宣称逐字节重验全部原始 H5。

## 可选触觉编码器（2026-11-12）

已增加独立 `tac_embed` 模态，支持正式 MAE 与 MAE＋力 λ=0.3 两组权重。启用命令：`bash scripts/analyze_state_graph_tactile.sh mae` 或 `force`；每指共享编码器输出 768 维，默认双指拼接 1536 维。默认物理特征配置保留。详见 [触觉编码器接入说明](docs/STATE_GRAPH_TACTILE_ENCODER_zh.md)。

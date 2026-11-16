# 白瓶与绿瓶 a 轴离线实验 · 2026-11-16

本目录保存固定接触参考回放、白瓶 fixed5 的历史对照，以及白瓶和绿瓶分别训练、整批留出的离线结果。白瓶 fixed5 的历史闭合下限为 0.330，绿瓶历史 E1/E2 SQUEEZE 的下限为 0.336。固定接触部署的 ForcePolicy 交接目标单独评估。

| 问题 | 报告与结果 |
|---|---|
| 冻结接触参考与 a 轴局部执行采用什么契约？ | [部署说明与离线测试](../../FIXED_CONTACT_A_zh.md) · [偏移模型验证](fixed_contact_replay/offline_report.json) · [部署窗口回放](fixed_contact_replay/recorded_window_replay.json) |
| 轻白瓶 −5 mm 的纠正何时由计划指定、何时由图策略自主选择？ | [Batch 3–6 记录复核](白瓶轻瓶负5毫米_fixed5对照复核.md) |
| Batch 6 的 0003 和 0005 在首次接触和夹紧时有何不同？去掉腕图后 Ready 误判是否减少？ | [接触与力曲线](batch6_0003_0005_fixed5.png) · [整批留出报告](fixed5_batch6_holdout/对照报告.md) · [逐窗口预测](fixed5_batch6_holdout/results.json) · [复算脚本](fixed5_batch6_holdout/replay.py) |
| 触觉偏移能否跟踪实际移动？图与直接学习、无图邻居各自表现如何？动作能否预测下次接触？ | [白瓶与绿瓶对照报告](offline_245/离线验证报告.md) · [位移对照图](offline_245/offset_pairs.png) · [逐折汇总](offline_245/summary.json) · [实验脚本和逐条 JSONL](offline_245/) |

两种瓶子的局部位移估计在通过检查的窗口对上均跟随 TCP 移动。白瓶 6/22 对通过，绿瓶 94/114 对通过。整批留出时，完整图相对无图方法的放行收益随瓶子和数据划分变化。动作后触觉可以预测，绿瓶上简单线性模型的综合误差最低。结论限于已记录的 a 轴运动和相应下游。

已有预测文件可直接复算汇总。在仓库根目录运行以下命令。

```bash
python3 docs/experiments/2026-11-16/offline_245/summarize.py
```

从头训练和回放需要 6f5090 上的原始采集目录、冻结模型及对应 Python 环境。具体数据来源和参数见两份报告。

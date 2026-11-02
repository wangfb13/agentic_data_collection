# RSI 交付验证记录（2026-11-02）

本次增加独立 `rsi` 入口：版本化采集方法、持久经验检索、受限方法提案、ContactPolicy 训练、同起点等预算对照、学习曲线、证据判定和断点恢复。旧 `e1e2`、`agentic run/resume`、setup/run 入口保留。

## 已执行

- 在 5090 的隔离源码副本中、使用现有 `daimon_ros1_das` 容器及 `.venv_robot` 运行 372 项离线测试，全部通过。覆盖新 RSI/ContactPolicy，以及旧 Agentic 计划、执行、学习、评估、认证适配和 E1/E2 回归。
- 检查 `rsi --help`、`agentic rsi --help`、启动脚本语法和差异格式。
- 从已有真实数据 `outputs/e1e2_A_formal01` 离线训练模型 `aa7343bd5fb879d82c9e`。按完整试验分组留出 20 条样本，平均绝对误差 0.8335 mm，90 分位绝对误差 1.8136 mm，`usable_for_B=true`。
- 固定验证组的回归检查通过：后续新增数据只加入训练，不因会话数变化重新划分验证集；独立 B 评估及最终验证数据不进入训练。

完整回归命令（在源码目录执行）：

```bash
python -m unittest \
  tests.test_contact_policy tests.test_rsi tests.test_rsi_review \
  tests.test_rsi_methods tests.test_rsi_proposer \
  tests.test_agentic_campaign tests.test_agentic_autoplan \
  tests.test_agentic_planning tests.test_agentic_learning \
  tests.test_agentic_executor tests.test_agentic_provider \
  tests.test_agentic_evaluation tests.test_agentic_design tests.test_e1e2 -q
```

离线模型保存在 5090：

```text
outputs/rsi_source_review_20261102/offline_check/contact_policy.json
```

`rsi run` 会为新任务重新训练并记录自己的模型；这个检查产物不会覆盖任何在用策略。

## 验证范围

验证期间没有执行任何机械臂或夹爪运动命令。没有执行真实 RSI 方法对照采集，也没有验证新提示词的在线模型回复；Agent 网络和授权沿用现有接入，提案协议使用离线测试验证。

上述误差衡量的是由当前触觉描述子预测几何纠正量的能力，监督值来自已测 PROBE 相对固定 T0 的位移。它不证明纠正动作最优，不证明抓取成功率提升，也不证明跨问题的递归改进已经成立。后续需要由独立实机 B 评估、等预算方法对照和不同真实问题的复验提供证据。

日常命令和暂停恢复见 [RSI 操作指南](RSI_GUIDE_zh.md)，实现边界见 [RSI 设计说明](RSI_DESIGN.md)。

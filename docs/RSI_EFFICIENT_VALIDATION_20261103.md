# RSI 精简采集验证记录（2026-11-03）

新增精简配置、旧 RSI 完整 A 批次导入、跨批次单点/点间距离限制、覆盖统计与评分、筛查和独立确认。旧会话继续使用原冻结协议，B 和最终评价不导入策略训练。

验证在 5090 的隔离源码目录及 `daimon_ros1_das` 容器内进行，没有运行机械臂或夹爪运动命令：

- 离线回归 379 项通过；最终的中断预算检查和单点例外又运行了 43 项相关回归，通过。
- `--seed-campaign` 只读检查识别原 `e1e2_A_formal01` 和 12 个已完成的 A 批次，共 13 个源会话。B 评价批次没有加入训练数据。
- 使用上述真实 A 数据离线训练，按完整 session 平衡划分得到 92 条训练和 30 条留出，平均绝对误差约 0.517 mm、90 分位误差约 1.476 mm，当前门槛 `usable_for_B=true`。这是几何预测的留出误差，不是实机恢复成功率，也不证明跨日泛化。
- 使用上述数据和已有 ChatGPT 授权，在隔离输出目录真实请求 Agent 生成规则，并只生成第一批计划文件；没有执行计划或连接机器人。Agent 的新分支首批选点为 −2、+8 mm，旧分支为 −10、−5 mm，均各 2 次；后续真实提案可以不同。
- 离线闭环覆盖仅筛查不晋升，以及触发确认后使用全新 B 批次作正式对照；确认数据和筛查数据均不进入策略训练。冻结规则或等预算条件被修改时拒绝继续。

完整版回归：

```bash
python -m unittest tests.test_rsi_efficient tests.test_contact_policy \
  tests.test_rsi tests.test_rsi_review tests.test_rsi_methods \
  tests.test_rsi_proposer tests.test_agentic_campaign tests.test_agentic_autoplan \
  tests.test_agentic_planning tests.test_agentic_learning tests.test_agentic_executor \
  tests.test_agentic_provider tests.test_agentic_evaluation tests.test_agentic_design \
  tests.test_e1e2 -q
```

该验证没有实施精简方案的真实采集、正式 B 确认或最终验证。小样本筛查可能漏掉改进；确定新方案是否有效需要新的独立实机证据。

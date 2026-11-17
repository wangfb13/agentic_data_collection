# P3 白瓶 a/h 与 ForcePolicy 提起保持实验

在 6f5090 的 ROS 容器中执行：

```bash
bash /home/dm/DM_das/bottle_grasp_experiment/scripts/start_p3_forcepolicy_loop.sh --allow-agent --real
```

按屏幕提示放轻瓶或重瓶。瓶底放稳，右爪全开空载，对准本次瓶身后停止遥操。预留单轴修正、向上 40 mm 和返回的路径。每次从当前接触开始，无须回原 T0。a 是动作开始时的 TCP 正 z，h 是基座正 z，每条最多纠正一次。a 可取 −10 至 +10 mm，h 可取 −3 至 +45 mm，优先向上。

A 由导演选择直接交接或指定单轴纠正。有效的最后一次轻触保持原抓位，把控制交给现有 ForcePolicy。力目标由 checkpoint 推理，闭合下限 0.330。程序检查双指接触、力控就绪、TCP 反馈以及 2 秒保持。下降时沿用已有卸载检测。瓶底由桌面支撑后输入 TABLE；用 y、n 或 u 登记刚才瓶底确实离桌并保持、没有做到或看不清。u 保留未知。位置模型只读取触觉与双指六维力，瓶重仅用于实验分组。

B 计划中的空动作列表表示由候选模型决定。模型可直接交接、选有实测转移支撑的单轴动作，或拒判。纠正后重新轻触并检查退出条件。起点类别是同瓶重、至少两组来自该版拟合快照的独立 A 无动作邻近观测一致给出的对照标签。混合或不足的证据保留未知。这个比较评估相似接触条件下的恢复，不能证明同一次抓取的反事实。开发 B 未过会回到 A，重新训练后再做开发 B。通过后冻结原候选，用新试次做独立 B。独立失败的结果单独封存；若还够新一轮 A 和两套 B 的预算，才会进入新的尝试。

数据目录为 `outputs/state_graph/campaign_white_ah_here_P3_forcepolicy`。`continuation.json` 索引旧试次，`seed_nodes.json` 为原有观测快照。旧组保留原来的路径与证据。新增预算为 60 次，包含 A 和两套 B；A/开发 B 每批最多 4 次，独立 B 预留 12 次。重启同一命令会续跑同一队列。技术故障停机后，根据屏幕提示在瓶已放稳、机械臂停止且夹爪全开空载时选 SKIP 可暂缓本条。用同一入口的 `--list-deferred` 查看，`--retry-deferred all` 将暂缓的试次放回队列。

看 `trials/<试次>/result.json` 的决策结果、实际位移回执及消费者证据。`consumer.json` 分别记录 L1 力控就绪、L2 提起保持和现场见证；`force_trace.jsonl` 保存逐次参考与控制反馈。模型和训练状态在 `revisions`，验收在 `development_acceptance.json` 和 `independent_acceptance.json`。完整 StarVLA 任务不在本实验的评估范围内。

离线复核可在仓库根目录运行 `.venv/bin/python -m pytest -q tests/test_p3_forcepolicy_loop.py tests/test_p3_force_consumer.py`。在容器中运行 `bash run_on_5090.sh python -m tests.p3_force_cpu_smoke` 可用内存模拟的夹爪和 TCP 测试部署权重与原力控器。两项演练都不连接机器人。


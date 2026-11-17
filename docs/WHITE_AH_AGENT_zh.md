# 白瓶 P2 的 a/h 探索采集

在 6f5090 的 starvla 容器里运行。沿用 V4 的符号导演、计划校验、批次恢复、双指 PROBE 和 move-tcp 实测回执。白瓶放在已记录的 P2 标记，夹爪全开，姿态与 P2 center 一致，结束遥操。每个试次终端会显示应放轻瓶还是重瓶。按现场提示确认移动、轻触和瓶位。

```bash
cd /home/dm/DM_das/bottle_grasp_experiment
bash scripts/start_state_graph_ah.sh --allow-agent --real
```

仅生成导演计划可省略 `--real`。两个参数都省略时只做离线建图和证据整理。三种方式复用 `outputs/state_graph/campaign_white_ah_P2`，重启会读取原进度。慢闭合反馈超差停机、已识别的传感器故障或 TCP 到位超时时，本次退出，把原始窗口和移动回执归档到 `incomplete_attempts/<trial_id>/<开始时间>/`，本次不计试次、失败数或训练样本。下一次启动会在现场确认后从同一编号、同一瓶重、同一计划的首触重做。结局不明且找不到明确故障回执时，`awaiting_result` 保留现场复核入口。

## 导演能选什么

`translation_available(h)` 声明上下平移能力，`direction_sample_missing(h,negative)` 表示缺少符合条件的实测向下转移。两者都不预测恢复收益。导演可以选单个方向、混合安排一批对照、重复同一设置测量噪声，或暂停。实际动作 `translation_tcp_a_base_h` 的两个分量按 `[a,h]` 排列，单位 mm。每次移位前开爪，移动后复触。每个试次最多一次探索移位。相对 P2，a 起点和终点在 −10 到 +10 mm，h 在 −3 到 +45 mm；每步 a 最多 10 mm、h 最多 45 mm。从高处可以向下返回。导演可比较不同高度的轻触，也可做向上动作前后的对照。一批最多 4 次，本轮预算 24 次。

`[+3,0]` 沿动作开始时的 TCP +a，`[0,+3]` 沿 base_link +Z，`[0,-3]` 沿 base_link −Z。l 保持工具轴原定义。准备位置使用 P2 的实测中心作为实验起点基准，保留原标准接触参考。实际基座高度也检查 −3 到 +45 mm；a 轴倾斜时，基座高度变化 = h + a × a 轴在基座 Z 上的分量。跨越整个范围的准备移动按直线分段，每段至多 45 mm，速度 2 mm/s。若实测起点与计划有少量误差，边界处可在 2 mm 反馈容差内缩短探索指令；计划与移动前计算的向量写入 action_command.json，最终指令目标和反馈位姿在 move-tcp 回执内。现场核对开爪时夹爪到桌面的净空；move-tcp 本身不规划避障。

切换范围时，先前有实测或需要技术重试的试次保留。旧批中尚未执行的计划可由导演按新范围重排，已完成的数据继续使用。

## 看哪些证据

- `director_context.json` 的 facts 与 applicable_operators 显示导演看到的能力、方向覆盖与可选操作。实际计划与理由保存在 `director/`。
- `seed_nodes.json` 复用 2026-11-17 的 P1、P2 有效轻触。动作前后窗口和相邻移动回执匹配才产生实测边。P2 两次向上命令之间没有轻触，只构成一条净转移。
- `trials/<trial_id>/request.json` 含计划。`result.json` 含有效窗口、实测动作和排除原因。每次实际动作及原始双指流保存在同一试次下。动作记录用动作时 TCP a 和基座 Z 组成的两个方向分解实测位移，同时记录平面外残差。学习输入为双指接触质心／面积及双指六维力。P2 相对位移仅用于实验测量。
- 本入口采集动作和触觉转移，策略效果及下游成功率留给有明确消费者的验证。单次双侧轻触、上移或旧 a 模型的 OOD 均不标为下游成功或失败。旧 a/l 和 fixed5 入口继续独立使用。

## 中断后先采同批剩余试次

用原来的 P3 启动命令。单条试次中断时，确认机械臂已停止、瓶子放稳、夹爪张开空载后，可输入 `SKIP` 将该条暂缓。回车则暂停处理。下一条仍显示原本的瓶型、动作和运动确认。Ctrl+C 停止程序。重启会先查找已保存的结果。

批内其他条目结算后，程序停在 `deferred_review`，打印未完成试次、原因及原始记录目录。暂缓保留原试次名额和已发生的移动记录，不作为有效评估或实验失败。已正常结束的无效双侧轻触等实验结果照常结算。

查看清单、修好后恢复待补试次、再继续采集分别执行以下命令。也可以将 `all` 换成清单中的一个或多个试次 ID，只补指定条目。

```bash
bash scripts/start_state_graph_ah.sh --from-current P3 --list-deferred
bash scripts/start_state_graph_ah.sh --from-current P3 --retry-deferred all
bash scripts/start_state_graph_ah.sh --from-current P3 --allow-agent --real
```

前两条只查看或整理队列；第三条沿用原真机确认流程。

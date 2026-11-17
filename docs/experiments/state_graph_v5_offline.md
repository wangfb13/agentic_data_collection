# 状态图 v5 离线复现

在 6f5090 的 `/home/dm/DM_das/bottle_grasp_experiment` 工作区操作。以下命令均为离线入口。

## 比较历史白瓶与绿瓶

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/compare_state_graph_versions.py
```

报告写入 `docs/experiments/state_graph_v5_comparison.md`。完整预测、分组、冻结图、噪声和模型分别写入 `outputs/state_graph/v5_history_comparison/{white_bottle,green_bottle}/{v4,v5}/`。报告按同一个历史 SQUEEZE 消费者比较，ForcePolicy 的现场提起保持需要对应试次的实测结局。

## 看 P3 新一轮会怎么采

在 6f5090 的原推理容器内执行下列命令，可从旧 P3 读入已完成的试次，使用确定性导演生成一批待执行请求。输出目录可以换成新的名称。报告文件来自冻结的 13 个 P3 试次组，程序校验所引节点属于种子证据。

```bash
bash scripts/start_p3_forcepolicy_loop.sh --graph-version 5 \
  --director-mode fixed_priority \
  --ranking-report outputs/state_graph/v5_representation_p3/ranking.json \
  --out outputs/state_graph/v5_P3_offline_release
```

查看 `state.json` 中的 `round_stage`、`spent` 和 `pending`。在 `rounds/round_0001/plan.json` 中看本轮的操作、前提事实、角色分配、预先声明的开发 B 起点和预测。开发 B 和独立 B 分别预声明中心、r 轴 ±3 mm，以及 h 轴 +3、+9、+15 mm，轻重各一次，共各 12 个独立试次。这些是验证分布，采集导演仍可选择包络内的其他幅度。独立 B 在初始化时预留。dispatch_attempts 在派发前计入潜在物理尝试，技术重试不会退款；spent.trials 单列已占用的可评估试次。首次采集前的 `ready_to_collect` 表示计划已经写盘，真实试次计数为零。

`--director-mode llm --allow-agent` 使用仓库现有的导演提供者。每个目录保存初始化时选择的模式。运行旧目录时指定 `--graph-version 4`。单纯采 r/h 转移的入口与包含训练、开发 B、双下游和独立 B 的 P3 入口分别见 `state_graph_v5_placement_consumers.md`。

## 核对一轮的证据

每轮产物按 `plan → execute → train → dev_validate → summarize → update_rule` 保存。计划中的 `dev_plan_digest`、训练收据中的 `model_id` 和试次请求可以交叉核对。开发 B 从下一轮起才具备训练资格。独立 B 始终单独封存。已开始且没有结果的请求通过现有恢复入口读取证据。

`round_report.json` 包含分下游图状态、首触能力边界、分质量的开发成功率与 Wilson 区间、失败原因、预测判定及相较上一轮的变化。`round_summary.md` 的每条陈述引用报告中的 JSON Pointer。配对比较只使用通过终端相似度和位姿检查的试次，按实测首触坐标分箱。信赖区间使用两类不一致对概率的 Bonferroni 联合界，McNemar 使用双侧精确二项检验；分箱结果为各箱的区间。缺少配对时保留空统计。

## 表征与输入

通用开关及默认数值在 `bottle_scan/state_graph/profiles/`。P3 的指面毫米几何需要左右指的真实标定；按瓶重判断切向力还需记录轻重瓶重量、SDK 到竖直支撑力的标定。现场资料和输入字段见 `state_graph_v5_placement_consumers.md`。可用 `--overrides <标定和配置的 JSON 文件> --out <新目录>` 在初始化时固定这些值。自动下游判定会保留缺少证据的结果。

编码器比较报告在 `outputs/state_graph/v5_representation_p3/ranking.json`，对应的真实帧索引在 `feature_sources.json`。只有在规定场景的分组留出指标满足条件后，导演才获得 `select_representation`。

## 合成回归

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

`pytest.ini` 将默认发现目录限定为仓库的 `tests/`；历史备份目录内有重名的旧测试。轮次、角色隔离、专家收敛、配对判定和表征测试都用合成数据。现场力阈值及闭环恢复成功率需要新的实测证据。


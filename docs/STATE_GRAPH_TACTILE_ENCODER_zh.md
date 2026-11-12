# V3 可选触觉编码器模态

位置：`6f5090:/home/dm/DM_das/bottle_grasp_experiment`。

新增 `tac_embed`，与原有 `tac` 接触几何、`force` 六维力、`vis` 视觉并列。原物理特征配置保留；指定新配置才启用学习特征。没有重新训练触觉编码器、没有改原 session，也没有启动真机。

## 使用

在 6f5090 的仓库目录执行一条离线命令：

```bash
cd /home/dm/DM_das/bottle_grasp_experiment
bash scripts/analyze_state_graph_tactile.sh mae
```

切换到 MAE＋力监督（λ=0.3）：

```bash
bash scripts/analyze_state_graph_tactile.sh force
```

脚本使用现有容器内的 PyTorch，对保存的轻触窗口编码并缓存，再在宿主机运行状态图分析；不启动任何硬件服务。默认 GPU，可设置 `TACTILE_ENCODER_DEVICE=cpu`。已经缓存的窗口不会重复做编码器前向。

只跑图分析、使用已完成的缓存：

```bash
bash run_on_5090.sh state-graph analyze \
  --config config.state_graph.symbolic_v3_tac_mae_20261112.json
```

缓存缺失时提示先预计算，不会静默退回物理特征。

## 编码方式

- 权重来自已经完成的正式实验 `final_mae_s0` / `final_force_s0`，均为 100 epoch、seed 0。力模型 λ=0.3。使用 seed 0 是固定接入实例，不是按这批图数据挑选最佳种子。
- 单指共享编码器，输入原始编码 DD 的三个通道：depth、deformation-x、deformation-y。不是可视化 RGB，也不是减去当前空载参考后的 DD。
- 读取 checkpoint 保存的训练集统计，先转 float32，再执行 `(encoded − offset − mean) / std`。形变偏置是 30000，编码比例为 1000；不使用旧模型的 25000 偏置和裁剪。
- 与训练完全一致：96×128 → 192×256，PyTorch 双线性插值，`align_corners=False`；不翻转、不做逐帧归一化、不改变长宽比。
- `eval()`、冻结参数、mask ratio=0，全 patch（不含 CLS）平均池化。每指 768 维，取力投影层之前的特征；不使用预测力输出作为 embedding。
- 每个轻触窗口：两指各 5 个独立保存帧，逐帧编码后分别取特征中位数。
- 默认双指拼接 `[z_left,z_right]`，共 **1536 维**。左右差值是可选项，不增加新的信息。

在实例规格 `tactile_encoder.representation` 中可选：

| 设置 | 输出维度 |
|---|---:|
| `single_left` | 768 |
| `single_right` | 768 |
| `concat` | 1536 |
| `concat_difference` | 2304，顺序为左、右、左减右 |

改变表示后需要重新预计算和拟合图，不能把旧图当成新表示的图。

## 哪些模块使用它

新配置中，`tac_embed` 参与距离、聚类和 V／π 输入；PCA 与重复噪声仍只在训练折拟合，默认最多 32 个主成分。Bonferroni 自动按启用的模态数计算，加入后为 4 个模态。物理触觉、力和视觉仍保留。

f 默认继续使用低维触觉几何和力，输出的物理意义保持一致；本次没有将 1536 维特征直接塞入只有少量转移数据的动力学模型。框架允许另行把 `tac_embed` 加进 `dynamics.inputs` 做对照，但需要重新训练和评估，不能沿用本次 f 的结果。

`obs` 只保存数值特征。编码器版本、归一化、窗口帧哈希和组合方式放在 `feature_evidence` 中，不进入模型输入。原有轻/重属性、偏移坐标、后置条件真值仍不作为特征。

## 历史数据、部署窗口与缓存

历史窗口使用已有 `frames` 中的原始 `dd_encoded_left/right` 引用。V4 新 episode 也能从保存的轻触时间戳定位相应编码帧，再走同一个提取器。没有保存编码 DD 的窗口不能伪造编码器特征；默认关闭该模态时不受影响。

缓存标识包括模型与源代码版本、训练归一化、池化、拼接方式，以及两指原始帧内容哈希。改变权重、帧内容或表示会分开缓存。CPU 图分析可只读缓存，不需要安装 PyTorch；真正前向在现有 PyTorch 环境完成。

## 文件位置

- MAE 规格：`skills/contact_position_a/spec.tac_mae.json`
- MAE＋力规格：`skills/contact_position_a/spec.tac_force.json`
- 通用提取器：`bottle_scan/state_graph/modalities/tactile_embedding.py`
- 预计算入口：`bottle_scan/state_graph/prepare_tactile.py`
- 训练模型副本及缓存：`outputs/tactile_encoder_v3_20261112/`
- 两组图输出：`outputs/state_graph/20261112_symbolic_v3_tac_mae/`、`outputs/state_graph/20261112_symbolic_v3_tac_force/`
- 原 V3 输出：`outputs/state_graph/20261112_symbolic_v3/`，保留作对照。

这次接入解决“编码器特征进入状态图”的链路；没有补出缺失的白瓶转移或 ForcePolicy L2 结果，也不把一次固定种子的离线检查当作触觉编码器提高真机效果的证明。

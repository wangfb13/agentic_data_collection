# 白瓶换位置后的 a 轴估计验证

使用同一只白瓶、同一夹爪和同一接近角度，在当前机器人桌面上选两个能安全侧抓的位置 P1、P2。建议沿桌面相隔 30–50 mm，标好瓶底轮廓和瓶身朝向。只用遥操粗对齐同一段瓶身。整个实验冻结 11 月 10 日的标准接触参考和 171 窗白瓶模型。center 是首次采样的名字。手眼或绝对 T0 均不参与推理。

## 容器与命令

若在 6f5090 宿主机，从一个终端进入容器。

```bash
docker exec -it daimon_ros1_das bash
```

在该容器同一个 shell 内执行下列准备命令。ROS、右手 X5 和双腕流沿用现有服务。腕图是原始采集器要记录的关键帧，模型只使用双指轻触描述子。停止遥操指令发布，以免其它进程持有控制权。全开空载，检查实际左／右指序列号与 check 打印一致。检查瓶子、指尖、掌根、桌面在两次小位移的整段路径具有足够净空。

```bash
cd /home/dm/DM_das/bottle_grasp_experiment
RUN="outputs/white_placement_$(date +%Y%m%d_%H%M%S)"
echo "$RUN"
tool() { bash run_on_5090.sh python -m skills.contact_position_a.placement_probe "$@"; }
probe() { tool probe --run "$RUN" --placement "$1" --trial "$2" --real; }
jog() { bash run_on_5090.sh move-tcp --arm right --frame tcp --direction "$1" --distance-mm "$2" --speed-mm-s 2 --pose-tolerance-mm 2 --max-distance-mm 6.1 --log-dir "$RUN/moves" --real; }
tool check
```

11 月 17 日这台机器人右手实际发布 X2620410318 和 X2620410347。使用本机时，在容器当前实验终端设置下列变量，核对左指 0318、右指 0347。此变量只供白瓶 placement_probe 使用。返回原先 0505／0305 设备时执行 `unset WHITE_PLACEMENT_X5`。check 显示选定的配置，传感器流由 probe 连接时检查。

```bash
export WHITE_PLACEMENT_X5=0318_0347
tool check
```

预期模型 ID 为 `19c20374214fddd4392c`，白瓶参考为 `white_bottle_a_contact_target_20261110_v1`。夹爪当前 TCP 的 +z 为 +a，包括它在世界竖直方向上的分量。每条 jog 输入 MOVE 之前读实际起点和目标高度，检查整只工具的直线净空。move-tcp 不检查桌面碰撞。

## 两个摆放

先放 P1。用全开的右爪遥操到瓶身中段，两指分列两侧、与瓶身各留间隙。停止遥操。下列每行独立执行。probe 提示时输入 PROBE，执行空载扣零、原地轻触、松爪。轻触为双指均值 1.5 ± 0.25 SDK，稳定 0.2 秒，闭合下限 0.330。观察瓶子整次未滑动、转动或倾倒，才输入 YES。jog 在全开空载时执行，核对目标后输入 MOVE。移动结束判据为到目标 2 mm、1° 以内。3 mm 小步的实际位移可能比请求小很多。若移动超时，先看 moves 日志的 end_measured_mm，核对夹爪全开、运动已停止和瓶位未变，再决定是否在实测位置轻触。保留有效 center；从实际位置计算下一步。瓶位已变时另起摆放组。

```bash
probe P1 center
jog z+ 3
probe P1 plus
jog z- 6
probe P1 minus
```

在确认已松爪后，用遥操把手退到与瓶子有充足间距的位置，再把同一瓶移至 P2。保持同一高度、瓶身朝向和 TCP 朝向，粗接近后停止遥操。在 P2 颠倒扰动顺序。

```bash
probe P2 center
jog z- 3
probe P2 minus
jog z+ 6
probe P2 plus
tool score --run "$RUN"
```

上述 6 mm 是正负两端都按名义位移估算的跨度。如果本组中心到首端的实测位移明显偏小，按实测位置重新算去另一端的动作，同时核对轴外漂移。实际位移由轻触窗口实测 TCP 在夹爪 a 轴上的投影得到，看到“到位”只表示落入运动容差。两次夹持之间物体须保持静止。评分要求时间配对误差在 100 ms 内，轻触前后 TCP 漂移不超过 1 mm / 1°，相对起始观测的非 a 轴位移不超过 1 mm、角度变化不超过 1°。报告还给出两次 center 的朝向差；若超过 1°，本轮同时改变了位置和方向。

## 独立测量绝对偏移（可选，需在移动该组瓶子之前完成）

若有与历史标准夹法独立对齐过的量具／定位夹具，在每组结束、瓶子仍原位时，用量具复现该标准夹法的手部位置与朝向，机械臂停稳。量具应约束瓶身部位、沿 a 的插入深度及夹爪姿态。量具必须在新机器人上对应当前同一 TCP 和手指几何。只读保存评分用的基准。在 P1 用以下命令，在 P2 将 P 改成 P2。写下量具编号、如何对齐到历史标准、测量误差或重复性，以及可追溯证据的位置。datum 文件只由 score 使用。无法确认标准对应关系时，跳过此项。

```bash
P=P1
bash run_on_5090.sh record-pose --arm right --label "${P}_standard_witness" --output "$RUN/$P/datum.json"
read -r -p '输入实际量具、与历史标准的对齐依据和测量误差：' note
printf '%s\n' "$note" > "$RUN/$P/datum_source.txt"
```

## 看结果

详细报告在本轮的 report.json，每个摆放与试次有自己的 observation.json、空载 DD 和原始记录。先查每组接受数／总尝试数和拒绝原因，再比较正负方向、各点实测变化与预测变化。plus_minus_span_mm 可在中心点拒绝、两端接受时提供端到端的变化对照。无独立标准时 absolute_MAE_mm 为 null。相对变化正确仍可能存在共同的零点偏差。独立标准有效时比较每组 absolute_MAE_mm 和逐点绝对偏移误差（表中保留符号以观察系统偏差），解释时同时考虑量具误差。3 次／位置只作首轮诊断，不足以认证泛化可靠性。此次不测恢复、夹紧提起或任务成功率。

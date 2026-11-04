# ContactPolicy：模型、调用条件和成功退出条件

同一个模型 JSON 现在包含三部分：原有触觉修正模型、`skill_conditions.start` 调用条件、`skill_conditions.exit` 成功退出条件。无需另训判断模型。条件由受校验的 Python 逻辑和执行配置生成，Agent 不提供表达式或 Python 代码。

## 产物与判断含义

模型族仍为 `numpy_tanh_mlp_ensemble`，根据两指当前 PROBE 描述子预测沿 a 轴的一维修正。`predict()` 的特征检查、分布外和不确定性拒绝保持有效；`decide()` 的容差与既有无接触、停滞和贴边后备规则保持原行为。

| 接口 | 返回 `result: true` 的条件 |
| --- | --- |
| `should_start(probe, *, status, bilateral, context)` | 尚未执行技能；适用范围与当前观测检查通过；双侧接触；学习策略未拒绝，且 `abs(e_mm) > tol_mm`。 |
| `should_exit(probe, *, status, bilateral, context)` | 正在执行技能；适用范围与当前观测检查通过；原 PROBE 已完成稳定双接触检查，此刻仍双侧接触；学习策略未拒绝，且 `abs(e_mm) <= tol_mm`。 |

两个接口只计算布尔结果和原因，不发布命令、不移动、不开合夹爪、不更新模型。它们供外部调度器直接调用，本次不替换现有 E1/E2 的动作状态机。

`should_exit: false` 可能表示仍需修正，也可能表示失败。必须查看 `reason`，由调度器沿用原异常处理；拒绝、无效观测、接触丢失、超时或耗尽不能作为成功退出。无接触和贴边仍由原 `decide()` 后备规则处理，新接口不把这些规则冒充为学习模型的调用或成功条件。

新产物的条件结构如下（节选，其余模型字段照常保存）：

```json
{
  "schema": "contact_policy_v1",
  "family": "numpy_tanh_mlp_ensemble",
  "skill_conditions": {
    "version": "contact_policy_conditions_v1",
    "thresholds": {"tol_mm": 2.0, "probe_stable_s": 0.2, "probe_frames": 5},
    "start": {
      "evaluator": "valid_probe_learned_decision_outside_tolerance_v1",
      "active": false,
      "status": "contact",
      "bilateral": true,
      "decision_controller": "contact_policy_v1",
      "rejected": false,
      "accepted": false,
      "error_comparison": "abs(e_mm) > tol_mm"
    },
    "exit": {
      "evaluator": "stable_probe_learned_decision_within_tolerance_v1",
      "active": true,
      "status": "contact",
      "bilateral": true,
      "decision_controller": "contact_policy_v1",
      "rejected": false,
      "accepted": true,
      "error_comparison": "abs(e_mm) <= tol_mm",
      "stability_evidence": "completed_PROBE.valid/stable_start_ns/timestamp_ns/window"
    }
  }
}
```

完整文件还保存 `scope.required_context` 和两段 `description_zh`。比较字符串只作机器可读定义；运行时不 `eval` 它们，而是严格校验版本、固定字段和判据，再执行对应的受校验函数。实际阈值从执行配置的 `servo.tol_mm`、`force.probe_stable_s`、`tactile.probe_frames` 写入；上面是现有默认值。

## 输入从哪里来

`probe` 直接使用本次成功 PROBE 的返回值，即 `Runner.regulate('PROBE', ..., with_dd=True)` 产生的记录。接口读取已有字段：

- `valid`、两指 `descriptors`：观测及学习特征有效性。
- `stable_start_ns`、`timestamp_ns`、两指 `window[].frame_ns/frame_fids`：成功退出时核对已有稳定阶段和独立 DD 窗口的记录是否齐全。
- `bilateral` 参数：调用当下的双侧接触检查结果，例如既有 `Runner.bilateral(snapshot)`；不能用几秒前的成功 PROBE 代替此刻的接触判断。

稳定性由原 PROBE 对两指力、力带和持续时间的检查建立。这两个接口不会根据主机时间差重新测量力稳定性，也不要求新的传感器字段。`probe_window()` 晚于力快照运行，故 DD 时间允许晚于该快照的 `timestamp_ns`。

`context` 是调度器根据现场状态和原检查生成的上下文。每个字段都必须明确提供；缺失或类型不符返回假，不能为通过判断而固定填真。

| 字段 | 要求与来源 |
| --- | --- |
| `object_supported` | `True`；物体由桌面等现有支撑面支撑。模型不能从描述子证明此事，调用方须确认。 |
| `airborne` | `False`；来自现有提起/恢复状态。重抓会松爪，不能在空中调用。 |
| `unsafe_recovery` | `False`；现有异常恢复状态已处理。 |
| `frame_compatible` | `True`；原硬件/协议签名、实机或模拟属性、TCP/轴定义与固定 T0 的检查通过。 |
| `motion_bounds_ok` | `True`；原 MotionBounds 对修正目标及两段退让目标的检查通过；其坐标、旋转、高度和位移限制继续有效。 |
| `observation_current` | `True`；原数据新鲜度检查通过，且 probe 属于当前尝试；不能重用旧的居中记录来退出。 |
| `timed_out` | `False`；原执行时间限制没有触发。 |
| `servo_exhausted` | `False`；原最大重抓次数尚未耗尽。 |
| `active` | 调用判断时为 `False`；执行中的成功退出判断时为 `True`。 |

这些上下文只用于门禁，不进入神经网络；模型仍不读取计划偏移 δ、绝对 TCP 或瓶子真值。`motion_bounds_ok` 可以先对 `decide()` 给出的候选动作做原边界检查再提供；判断通过也不代替执行前的再次检查。

## 外部调度器调用示例

以下示例中的 `probe`、`status`、`bilateral_now` 和 `checked_context` 均由调用方提供，来源见上表：

```python
import json
from bottle_scan.contact_policy import ContactPolicy
from bottle_scan.e1e2_core import load_settings

cfg = load_settings('config.e1e2.yaml')  # 只读配置，不连接硬件
model = json.loads(open('outputs/contactpolicy_new.json').read())
policy = ContactPolicy(model, cfg['servo'])

start = policy.should_start(
    probe, status=status, bilateral=bilateral_now,
    context={**checked_context, 'active': False},
)
print(start['result'], start['reason'])

# 执行器完成下一次 PROBE 后，传新的观测和实时上下文。
exit_result = policy.should_exit(
    next_probe, status=next_status, bilateral=next_bilateral_now,
    context={**next_checked_context, 'active': True},
)
print(exit_result['result'], exit_result['reason'])
# exit_result['result'] 为 True 才代表修正成功、可交还主策略。
```

常见 `reason`：`correction_required`、`already_within_tolerance`、`stable_within_tolerance`、`correction_still_required`、`invalid_probe`、`stable_probe_evidence_missing`、`bilateral_contact_missing`、`explicit_fallback_required`、`policy_out_of_distribution`、`policy_uncertain`。上下文失败会指出具体字段，例如 `context_not_satisfied:servo_exhausted`。

## 训练、冻结和旧模型

`rsi train --session ... --out 新路径.json` 自动保存条件并在终端输出。它只读配置与已有数据；无需配置开度下限，不会启动向导、连接机器人或要求运动确认。已有输出路径内容不同时拒绝覆盖；请指定新文件。RSI 常规训练使用会话已解析的执行配置，最终 `REPORT.md` 直接展示模型、何时调用、何时成功退出及实际阈值。

新增字段被整个模型的 `integrity_sha256` 和 `model_id` 覆盖。新模型的运行时 `servo.tol_mm` 必须与模型保存值相同，否则加载拒绝，防止成功容差被静默放宽。内容哈希用于检测变更，并不是可信作者的签名。

旧模型继续按原哈希加载，原 `decide()` 兼容。新接口为旧模型在内存中生成条件，容差用加载时的 `servo.tol_mm`，稳定时间与帧数取既有 E1/E2 默认值；返回值标注 `conditions_source: legacy_runtime_defaults`。这不代表旧文件原本含有这些阈值。若实际旧实验使用过不同稳定配置，应导出带实际配置的新副本再交给外部调度器。

纯离线导出可以使用 `with_skill_conditions(old_model, cfg)`；它校验原模型，复制权重与来源，记录 `conditions_parent_model_id`，生成新哈希，返回新字典，不写历史文件。RSI 冻结遇到旧产物时自动执行同样的复制，保存为新的 `models/<model_id>.json`。已经带条件的模型按原 ID 冻结；冻结记录同时保存条件和完整模型指纹。旧的已完成会话不会被自动回写升级。已经开始最终独立验证的旧会话也继续使用原冻结产物，避免在验证中途改变版本；可以另外离线导出供未来调度器使用。

本改动不改变采集预算、选点、B 筛查/确认、机器人动作、异常处理或评估成功标准。训练仍排除独立评价数据。

## 离线验证

```bash
python -m unittest tests.test_contact_policy tests.test_contact_policy_conditions \
  tests.test_rsi tests.test_rsi_conditions tests.test_rsi_efficient
```

测试使用合成描述子、已有字段格式的 PROBE 记录和离线执行器替身。覆盖调用/退出、容差边界、无效/拒绝/失去接触、上下文限制、稳定证据、后备规则不变、保存/加载、条件篡改拒绝、旧模型兼容和冻结复制。测试不连接机器人、不采集新数据。

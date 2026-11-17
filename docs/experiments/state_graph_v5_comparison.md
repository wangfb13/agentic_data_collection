# 状态图 v4 与 v5 的历史对照

数据来自 `outputs/state_graph/v4_implementation_20261113/historical_rebuild/nodes.jsonl`。按试次组做 5 折交叉验证，同一试次的所有窗口只进入一折。每一折用训练组重新估计噪声、尺度和 Ready 邻域。

主表下游为 `handoff_e1e2_squeeze_floor0336`。正确放行指留出终端实际成功且被 Ready 门放行，错误放行指留出终端实际失败却被放行。表中分母含无近邻和证据不足的终端。

| 物体 | 版本 | Ready | Recoverable | Unreachable | Unsupported | 正确放行 | 错误放行 | 留出成功 / 可评估 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| white_bottle | v4 | 3 | 0 | 0 | 48 | 2 | 0 | 11 / 23 |
| white_bottle | v5 | 0 | 0 | 0 | 51 | 0 | 0 | 11 / 23 |
| green_bottle | v4 | 358 | 28 | 0 | 3 | 234 | 3 | 235 / 252 |
| green_bottle | v5 | 0 | 0 | 0 | 389 | 42 | 2 | 235 / 252 |

### white_bottle · v4

共 51 个窗口、50 个试次组。簇符号分布为 `{'Stochastic': 2, 'Unknown': 1, 'Unsupported': 11}`。可恢复标签来源为 `{}`。

有效半径为 `v4，统一 D ≤ 1`。该次实际解析的全部开关和数值保存在 `outputs/state_graph/v5_history_comparison/white_bottle/v4/spec.json`，节点、标签及逐样本留出预测位于同目录。

### white_bottle · v5

共 51 个窗口、50 个试次组。簇符号分布为 `{'Stochastic': 2, 'Unsupported': 10}`。可恢复标签来源为 `{}`。

有效半径为 `{'similar_radius': 1.0, 'ready_radius': 1.0, 'radius': 1.0, 'support_distance': 1.0, 'endpoint_radius': 1.0}`。该次实际解析的全部开关和数值保存在 `outputs/state_graph/v5_history_comparison/white_bottle/v5/spec.json`，节点、标签及逐样本留出预测位于同目录。

### green_bottle · v4

共 389 个窗口、275 个试次组。簇符号分布为 `{'NeedAction': 12, 'Ready': 29, 'Unknown': 1}`。可恢复标签来源为 `{'observed': 15, 'uses_inferred': 13}`。

有效半径为 `v4，统一 D ≤ 1`。该次实际解析的全部开关和数值保存在 `outputs/state_graph/v5_history_comparison/green_bottle/v4/spec.json`，节点、标签及逐样本留出预测位于同目录。

### green_bottle · v5

共 389 个窗口、275 个试次组。簇符号分布为 `{'Stochastic': 1, 'Unsupported': 242}`。可恢复标签来源为 `{}`。

有效半径为 `{'similar_radius': 0.32805376136662534, 'ready_radius': 0.32805376136662534, 'radius': 0.3936645136399504, 'support_distance': 0.3936645136399504, 'endpoint_radius': 0.32805376136662534}`。该次实际解析的全部开关和数值保存在 `outputs/state_graph/v5_history_comparison/green_bottle/v5/spec.json`，节点、标签及逐样本留出预测位于同目录。

## 证据范围

两版输入均沿用历史窗口实际保存的触觉几何、力与腕图特征。历史表没有指面毫米标定、逐图块读出、同一试次配对的双下游结果或新导演的事前预测，因此这些功能只做合成验证，历史 McNemar 和预测准确率均不可估计。v4 的来源从标签的实际路径回查。uses_inferred 表示路径包含模型推断，observed 表示整条路径仅含实测转移。

## 复现

从仓库根目录运行 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/compare_state_graph_versions.py`。

历史输入内容摘要为 `33e99721e507c1fc846c5cc36d3fa2b0a9116f848e21967e4717b84315cc3fcf`。报告数值只适用于上述下游与已有对象。

## Ready 收缩发生在哪里

- white_bottle 的全量图中，单个 Ready 邻域最多含 17 个独立终端组。最高后验下界为 0.741，放行门槛为 0.8。
- green_bottle 的全量图中，单个 Ready 邻域最多含 9 个独立终端组。最高后验下界为 0.741，放行门槛为 0.8。

## 参数

历史特征适配覆盖值为 `{"observation": {"tac_version": "legacy", "tac_metric": false, "auto_select": false, "tac_embed_readout": "mean"}}`。

**facts_director**

```json
{
  "facts": {
    "measured_placement_coordinates": true,
    "boundary_before_stochastic": true,
    "per_side_no_failure": true,
    "stratified_boundary": true,
    "boundary_cross_tolerance_mm": null,
    "noise_based_resolution": true,
    "boundary_resolution_mm": 0.5,
    "boundary_noise_multiplier": 2,
    "capability_predicates": true,
    "alpha": 0.05,
    "min_axis_trials": 3,
    "confidence": 0.95,
    "bootstrap_reps": 200,
    "seed": 0,
    "observability_step_mm": 2.5,
    "stochastic_probability_band": [
      0.2,
      0.8
    ],
    "stochastic_min_trials": 4,
    "tactile_modalities": [
      "tac_metric",
      "tac",
      "geometry"
    ]
  },
  "operators": {
    "handle_drift_and_unsupported": true,
    "decouple_dynamics": true
  },
  "dynamics": {
    "min_action_gain": 0.05,
    "min_amplitude_levels": 2
  },
  "director": {
    "mode": "llm",
    "use_priors": true,
    "operator_scoring": true,
    "checkable_predictions": true,
    "frontier_weight": 1,
    "plan_reps": 1,
    "densify_points": 3,
    "initial_frontier_step_mm": 2.5
  },
  "campaign": {
    "integrated_frontier": true
  },
  "loop": {
    "rules_via_operators": true
  }
}
```

**geometry_consumer**

```json
{
  "geometry": {
    "horizontal_basis": true,
    "minimum_horizontal_projection": 0.1,
    "coordinate_residual_mm": 1.0,
    "orientation_tolerance_deg": 1.0,
    "receipt_tolerance_mm": 1.0,
    "placement_envelopes_mm": {
      "r_offset_mm": [
        -10,
        10
      ],
      "h_offset_mm": [
        -3,
        45
      ]
    },
    "max_step_mm": [
      10,
      45
    ],
    "reference_probe_repetitions": 1,
    "recalibration_probe_repetitions": 2
  },
  "runtime": {
    "max_steps": 3
  },
  "consumer": {
    "auto_l2": true,
    "dual_paired": true,
    "target_from_rule": true,
    "fixed_force_failure_causes": true,
    "fixed_force_sdk": 5.0,
    "force_tolerance_sdk": 0.5,
    "stable_force_s": 0.5,
    "fixed_closure_floor": 0.33,
    "floor_tolerance": 0.001,
    "floor_saturation_s": 0.5,
    "lift_mm": 40.0,
    "hold_s": 2.0,
    "grip_timeout_s": 60.0,
    "load_tolerance": 0.25,
    "load_tolerance_unit": "N",
    "slip_threshold": 2.0,
    "slip_threshold_unit": "px",
    "load_calibration": null,
    "weight_newtons": {},
    "audit_rate": 0.1,
    "pair_position_tolerance_mm": 1.0,
    "pair_orientation_tolerance_deg": 1.0
  }
}
```

**graph_rep**

```json
{
  "graph": {
    "resolution_scaled_radius": true,
    "resolution_min_edges": 3,
    "min_action_bin_relative_tolerance": 0.2,
    "similar_radius": 0.5,
    "success_value": true,
    "step_penalty": 0.08,
    "recover_p_min": 0.5,
    "borrow_similar_actions": true,
    "similar_penalty": 0.08
  },
  "sinks": {
    "ready_radius": 0.5,
    "lower_bound_ready": true,
    "ready_quantile": 0.05
  },
  "clusters": {
    "radius": 0.6
  },
  "runtime": {
    "support_distance": 0.6
  },
  "dynamics": {
    "endpoint_radius": 0.5
  },
  "labels": {
    "include_inferred": false
  },
  "noise": {
    "combine": "joint",
    "repeat_dimension_multiplier": 2.0,
    "max_neighbor_fraction_change": 0.2
  },
  "observation": {
    "tac_version": "calibrated_rh",
    "tac_embed_readout": "grid",
    "tac_grid": [
      2,
      3
    ],
    "tac_temperature": 1.0,
    "tac_calibration": null,
    "tac_metric": true,
    "auto_select": true,
    "metric": {
      "input_modality": "tac_embed",
      "displacement_weight": 1.0,
      "repeat_weight": 0.5,
      "ridge": 0.01,
      "min_groups": 3,
      "folds": 4,
      "max_error_mm": 1.5,
      "a_axis_tcp": [
        0.0,
        0.0,
        1.0
      ]
    },
    "representations": {
      "current": "legacy_geometry",
      "folds": 4,
      "confidence": 0.95,
      "bootstrap_reps": 300,
      "seed": 71,
      "required_scenes": [
        "a_axis",
        "h_body",
        "h_shoulder",
        "different_angles",
        "different_objects"
      ],
      "candidates": [
        "legacy_geometry",
        "calibrated_rh",
        "force",
        "mean",
        "attention",
        "grid",
        "soft_argmax",
        "tac_metric",
        "calibrated_rh+force",
        "tac_metric+force"
      ],
      "min_comparison_groups": 4,
      "min_valid_bootstrap_fraction": 0.9,
      "matched_position_tolerance_mm": 1.0
    }
  }
}
```

**rounds**

```json
{
  "loop": {
    "rounds": true,
    "dev_to_train": true,
    "expert_to_train": true,
    "structured_summary": true,
    "rules_via_operators": true,
    "termination_rules": true,
    "summary_style": "evidence_fields",
    "confidence": 0.95,
    "seed": 17,
    "default_exploration_allocation": {
      "direct": 1,
      "policy": 0,
      "expert": 0
    }
  },
  "consumer": {
    "target_from_rule": true
  },
  "report": {
    "paired_consumers": true,
    "coordinate_bin_mm": 2.5,
    "confidence": 0.95
  }
}
```

## 留出与全量图的区别

留出预测使用折内估计的噪声和动作尺度，图节点计数来自全量重建。两者的邻域可以不同。错误放行减少时，应同时查看正确放行数和拒绝覆盖，不能据此单独判定部署性能提升。

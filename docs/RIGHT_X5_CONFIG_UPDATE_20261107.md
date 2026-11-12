# 右手传感器配置与主策略兼容更新（2026-11-07）

药瓶项目和 V4 调度器的活动配置已改为右手传感器 `left: X26040505`、`right: X26040305`。左右对应来自已有右手推理脚本，并非本次单指按压实测。系统会检查主策略和恢复技能使用的序列号是否一致。

新调度器配置使用新的硬件配置文件和对应的文件哈希；旧硬件配置、旧训练会话、旧恢复模型和旧验证凭据保留。新配置中的 `hardware_rebind` 明确记录：物理左右对应、受压 Fz 正负和旧恢复模型在新传感器上的效果未重新验证。恢复技能仍为 development/candidate，不宣称通过新硬件验收。旧开度下限、T0 和力参数没有被伪造为新实测值。

## 检查用户选择的主策略

在 5090 或其 ROS 容器中：

```bash
cd /home/dm/DM_das/contact_scheduler_20261104_experiment_b_v4
CONTACT_EXPERIMENT_B_OFFSET_MM=5 \
CONTACT_SKILL_RECOVERY=enabled \
CONTACT_SKILL_AFTER=handoff \
CONTACT_MAIN_POLICY_PATH=/home/dm/DM_das/starvla-feat-sparky_x5_starvla/outputs/261026_starvla_qwenpi_greenbottle_far_normal_smalltag/50000/pretrained_model \
bash run_contact_261010.sh skill-check
```

`skill-check` 不接入相机和机器人、不发运动命令。它现在会实际解析选定主策略配置，而不只是检查配置文件存在及部分字段。当前检查点启用原来的两通道力触觉输入，但 `use_dd_contact=false`；运行期间程序只为这个禁用的扩展创建临时兼容视图。原始 `config.json` 和 4 个权重分片均不修改或复制。若 DD 扩展启用、含 DD 模块权重、出现未知配置字段或分片缺失，程序拒绝自动兼容。

本次离线对照两套主策略权重索引：835 个参数名及形状一致。CPU 离线加载用户指定检查点成功，原有六维力触觉输入保持启用。一个缺失名称是已加载词嵌入共享权重的别名，已确认与 `lm_head.weight` 指向同一个参数；没有其他缺失或多余权重。

本次没有执行 `skill-run`，没有验证真实场景中的模型表现、力方向、传感器左右安装或动作安全。配置字节哈希更新只证明文件完整，不证明新硬件完成物理标定。检查点和硬件变更备份见各项目的 `backups/right_x5_20261107_1794045308112474342/`。

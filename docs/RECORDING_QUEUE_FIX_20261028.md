# setup 开度标定采集队列满修复（2026-10-28）

现场错误：`outputs/wizard_errors/1793167155584906133.json`，发生于 `WidthBootstrap.measure(body)`。约 9.5 秒后 128 MiB 异步写入缓冲耗尽。磁盘当时仍有 134 GB；尺寸输入不是报错原因。旧记录为 incomplete，未得到有效开度点，且 release_complete=false，不认为夹爪已回开。

## 修复

后台异步 HDF5 的数值载荷从逐帧 gzip 改为 h5py 自带的无损 LZF。形状、dtype、每个来源帧、序号和先写入提交再回执的语义保留；直接使用 H5Writer 的默认 gzip 兼容。文件属性记载实际压缩方式。

腕部接收端已经解码每帧供图像检查，新增对原始 JPEG 的 SHA-256 绑定；后台保存前核验同一字节串，避免再次完整解码两路 1920×1080、约 59 fps 的视频。没有此接收端校验标记的通用 JPEG 输入仍执行解码校验。JPEG 原样保存，不重编码、不缩放。

未扩大 128 MiB 缓冲、未减帧、未丢样、未修改力环或运动门禁。只更新 recording.py、hardware_sensors.py 及相应回归测试。用户实机配置和 setup 断点字节不变，后续站点/验收绑定使用当前源码。

## 5090 验证

相关 73 项单元检查通过。真实来源只读连续采集 45.001 秒：双指 DD（编码版和解码版）、六维力、胶面原图以及两路腕部 JPEG。启用胶面全程记录以覆盖开度标定 PROBE 的连续负载，不执行真实轻触。

- 队列峰值 1.524 MiB，45 秒内无增长性积压、无溢出，关闭后未完成队列为零。
- 左/右各 1132/1235 帧 DD、胶面原图、六维力，接收计数与保存计数逐流一致。
- 左/右腕图像 2656/2655 张，共 5311 张全部逐文件验证 SHA-256 与接收端记录一致。
- recording_status.complete=true；实际 HDF5 压缩方式核验为 LZF。
- 只创建只读 ROS 观察器与传感器连接，没有创建控制发布者，没有执行机械臂或夹爪动作。此次结果证明当前负载下的 45 秒吞吐，不替代运动/力控验收。

原始验证目录：`outputs/wizard_queue_fix_20261028/readonly_capture_20261028_143440/`。汇总：`outputs/wizard_queue_fix_20261028/readonly_latest.json`。源码备份：`outputs/recording_queue_fix_backup_20261028_143328/`。

## 继续 setup

先现场确认瓶子有桌面支撑，使用现有控制方式把夹爪松开。然后重新执行：

```bash
cd /home/dm/DM_das/bottle_grasp_experiment
bash run_on_5090.sh setup --real
```

已保存的尺寸和完成步骤继续使用，未完成的瓶身开度标定重新测量；旧失败记录由程序自动归档，不用删除数据或手写 JSON。若现场设备/控制源码另外改变，则仍走原有身份变化与重新验收规则。

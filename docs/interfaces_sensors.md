# 传感器接口、数据语义与待实测项

本文件来自 2026-10-26 对 5090 源文件的只读审计。未 import SDK，未打开任何相机或触觉设备，未启动发布器。下列仓库相对路径均位于 `/home/dm/DM_das/starvla-feat-sparky_x5_starvla`。没有第三视角相机接口或保存项。

## 已有接口的证据

| 项目 | 静态证据 | 第一版处理 |
|---|---|---|
| 两路腕部图像 | `shfile/infer_261022_starvla_dd_recovery.sh:71` 起，left X5 `.10`，right X5 `.11`；`:615` 配置两个不同 X5 的 wrist，30 fps | 视作 wrist_1/wrist_2 两个独立源；明天确认安装位置、实际视野及两个视野都能看到瓶身。不能把 left/right 名字解释为同一夹爪双摄 |
| 腕相机协议 | `src/lerobot/cameras/wrist_fisheye_x5/camera_wrist_fisheye_x5.py:140` 起 ZMQ PULL，RCVHWM=1、CONFLATE=1；`:305` JPEG 解码；config `:41` 默认 5552 | 单客户端会消费 PUSH 分流消息，不能与现有消费者争抢；使用原始端点必须先确认独占消费或设计代理 |
| 腕图时间 | 上述 camera `:355` 起，在解码读取之后用 `perf_counter()`、本地计数 | 不是硬件曝光时间。保存 host monotonic、源时间为空、timestamp_kind；帧率配置 30 不等于已测 30 |
| 右手两指映射 | launcher `:90` 起：left finger `X2620410318`，right finger `X2620410347`，注释要求逐指按压确认 | 明天确认 serial 与物理手指对应关系；禁止自动推定轴向 |
| 六维力顺序 | launcher `:95` 起；`scripts/single_gripper_force_control.py:1436`，`Fx,Fy,Fz,Mx,My,Mz` | 原样保存六个 SDK 数值，不擅自改单位 |
| 力单位/法向 | `single_gripper_force_control.py:24` 起使用有符号 Fz，`:31` 起明确 SDK raw 不是已确认 N/gf；`:151` 拒绝未知 N/gf；`:1455` 起声明安装法向未确认 | force_unit=`sdk_raw`；法向轴、压缩正号、物理标定为待确认配置。已有旧实验的 Fz/-1 不能当新实验已校准 |
| 触觉原图 | `shfile/261022_starvla_dd_recovery/x5/publish_tactile_dd.py:37` 起复制 uint8 raw；`:97` 起 raw metadata；默认 raw 39010 | 这是触觉 RGB/灰度图，不是原始 DD |
| 已有 DD 39030 | publisher `:109` 起 uint16 `[96,128,3]`，D/Dx/Dy；`runtime/dd_codec.py:54` 起先 ×1000/偏置30000截断，再 INTER_AREA 降采样 | 不是 raw SDK DD。不得从这里解码后冒充原始 DD；真实采集需直接保存 SDK 原始浮点 depth 和 flow 或等价无损接口 |
| SDK 直接结果 | publisher `:53` 起 `getDeformation2D()`/`getDepth()` 返回 `(fid,array)`，flow/depth fid 对齐；`:65` 力可能不带 fid | DD 可知接口形状 H×W 和 H×W×2，实际 H/W、dtype、单位、像素间距待实测；保留 `force_fid_verified=false`，不得伪造同步 |
| 源时间 | publisher `:56`、`:79`、`:99`，wall time 在 getter 调用后取得；明确 `sdk_read_wall_time_not_hardware_exposure_time` | 时钟漂移/网络延迟未校准，不能把两机 wall time 当精确采集时间 |
| 丢帧 | publisher `:189` 队列满丢最旧；`:243` SNDHWM=2；`:272` 丢 >200ms；camera latest-only | 必须保留 source_sequence、每流独立本地序号、缺口记录；现有链路不能声称无丢帧 |

SDK V1.2.11 本地副本位于 `/home/dm/DM_das/umi_images/umi_data_offline-dev_new/code/touch_proc/touch_proc/core_proc/Touch_SDK_Publish_V1.2.11`。核心 `dmrobotics/src/dmSDK.py` 等经 PyArmor 混淆，静态源代码不能验证内部输出物理单位。没有从 SDK 中推导或臆造像素间距。

## 第一版原始存储 schema

`recording.H5Writer(path)` 以新建模式创建 `streams.h5`，重复路径拒绝覆盖。调用：

```python
writer.append("tactile/left/dd", raw_dd, time.monotonic_ns(),
              device_timestamp=None,
              attrs={"trial_id": "trial_0001", "stage": "HOLD",
                     "source_sequence": 123, "serial": "实际序列号",
                     "timestamp_kind": "host_receive_monotonic_only",
                     "unit": "sdk_raw", "channels": ["depth", "dx", "dy"]})
```

每流存于 `/streams/{stream}`，例如 `/streams/tactile/left/dd`：

| 字段 | 类型与语义 |
|---|---|
| `data` | N×原始 shape，原始 dtype，gzip 无损；禁止量化、缩放、resize、缺帧补零；shape/dtype 改变即报错 |
| `monotonic_ns` | int64，采集主机自己的单调时间，每流非递减；不是曝光时间 |
| `device_timestamp` | float64，缺失 NaN；原始精度同时存于 metadata |
| `sequence` | int64，本记录器每流从0递增，与 SDK 源帧号不同 |
| `metadata_json` | 每样本 JSON，含 stage/trial/serial/source_sequence/units、时间来源等调用方 attrs；`device_timestamp_raw` 精确保留原始整数时间 |
| `committed_count` 属性 | 已完整写入样本数，异常恢复读取只用 `[0:committed_count]` |

允许 depth 与 deformation 分流保存，也允许 SDK 本来已返回 H×W×3 时按原样存储；不能为凑 3 通道而偷偷改原始数组。力、实际 TCP、命令 TCP、夹爪状态各自独立流。调用方必须保留原始流与描述符流的区别，不能用零 DD 填补未读取数据。

`JpegWriter(directory).append(camera, image_or_jpeg_bytes, monotonic_ns, device_timestamp=None, attrs=None)` 保存 `{camera}/{sequence:08d}.jpg` 与独立 `timestamps.csv`。传入原始 JPEG bytes 原样保留；传入 RGB uint8 数组则 quality=95 重编码，明确是有损图像。DD 永远不经 JPEG。每路相机独立序号与时间，不按机器人控制周期重复写上一帧。

## 固定参考与位移检查

`sensing.TemplateGuard` 强制恰好两个腕图参考与各自瓶身 ROI。参考数组深拷贝且只读；比较及复位不修改参考。参考以无损 PNG 加 SHA256 存档，`save` 拒绝覆盖。只有用户显式重新记录 T0，才构建新的 guard/reference 记录。

匹配使用灰度 `TM_CCOEFF_NORMED`，在参考 ROI 周边搜索；每路均报告 dx/dy/Euclidean offset(px)、分数、峰值唯一性与阈值。任一路超阈、低置信、重复纹理、尺寸改变、搜索触边、缺图或未标定，整体 `passed=false`；运行器必须暂停并反复重新拍摄比较，不能更新参考来放行。

`calibrate(moved_images, noise_images=[静止图对1,静止图对2,...], movement_mm=2, threshold_fraction=.5)` 为每路记录约2mm移动对应的像素值，默认阈值取该位移的50%。要求至少两组静止重复拍摄，阈值必须覆盖3倍静止像素偏移且至少0.5px；若有相机视角对此方向不敏感，则拒绝标定，要求改用可见方向重新测量。这只是示范方向的经验像素阈值，非完整相机标定；不能声称保证任意空间方向的2mm位移均可见。

`dd_descriptors` 基于正的 reference-subtracted depth，返回接触像素面积、深度加权质心、几何质心、二阶矩。`edge_fraction` 为边缘区域正变形权重/总正变形权重；另存 `edge_area_fraction` 为边缘接触像素数占比。`cop_px` 是深度加权代理，未经压力校准不能声称是真实力学 CoP。像素间距缺失时 mm/mm² 字段为 null，不以假设值伪装实测。像素 x=列、y=行；传入的物理 pitch 顺序为 `(x,y)`。

#!/usr/bin/env python3
"""Preview or explicitly execute a measured, single-arm TCP target via public ROS.

Only the command-topic ownership section of the config is used. No gripper
controller, camera or tactile source is loaded.
The default is read-only. --real still requires typing MOVE for each shown target.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import uuid

import numpy as np

from bottle_scan.geometry import pose_error, validate_transform
from bottle_scan.hardware_arm import RosArmController, RosArmObserver, _matrix_quaternion_xyzw
from bottle_scan.config import load_config
from bottle_scan.command_ownership import (guard_all_commands, inspect_all_commands,
    current_ros_node_names, format_ownership_report)


class MoveCancelled(RuntimeError):
    pass


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("必须是有限正数")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("right", "left"), default="right")
    p.add_argument("--config", default="config.ros.yaml", help="读取命令话题完整清单；无需先填完几何标定")
    p.add_argument("--real", action="store_true", help="允许现场确认后发送 TCP；默认只读预览")
    targets = p.add_mutually_exclusive_group()
    targets.add_argument("--xyz-mm", nargs=3, type=float, metavar=("X", "Y", "Z"))
    targets.add_argument("--delta-mm", nargs=3, type=float, metavar=("DX", "DY", "DZ"))
    targets.add_argument("--pose-file", type=Path)
    targets.add_argument("--direction", choices=("x+", "x-", "y+", "y-", "z+", "z-"))
    p.add_argument("--distance-mm", type=positive, default=5.)
    p.add_argument("--frame", choices=("base", "tcp"), default="base")
    p.add_argument("--quaternion-xyzw", nargs=4, type=float, metavar=("QX", "QY", "QZ", "QW"))
    p.add_argument("--speed-mm-s", type=positive, default=5.)
    p.add_argument("--angular-speed-deg-s", type=positive, default=5.)
    p.add_argument("--max-distance-mm", type=positive, default=100.)
    p.add_argument("--max-angle-deg", type=positive, default=15.)
    p.add_argument("--timeout-s", type=positive, default=60.)
    p.add_argument("--pose-tolerance-mm", type=positive, default=2., help="实测 TCP 到位位置容差，默认 2 mm")
    p.add_argument("--pose-tolerance-deg", type=positive, default=1., help="实测 TCP 到位角度容差，默认 1 度")
    p.add_argument("--log-dir", type=Path, default=Path("outputs/tcp_moves"))
    return p


def validate_args(args):
    if args.quaternion_xyzw is not None and args.xyz_mm is None:
        raise ValueError("--quaternion-xyzw 只能与 --xyz-mm 一起使用")
    for name in ("xyz_mm", "delta_mm", "quaternion_xyzw"):
        value = getattr(args, name)
        if value is not None and not np.all(np.isfinite(value)):
            raise ValueError(name + " 必须全部是有限数值")
    if args.frame != "base" and (args.xyz_mm is not None or args.pose_file is not None):
        raise ValueError("--frame tcp 仅适用于 --delta-mm 或 --direction；绝对位姿总是 base_link 坐标")
    if args.quaternion_xyzw is not None:
        quaternion_rotation(args.quaternion_xyzw)
    return args


def quaternion_rotation(values):
    q = np.asarray(values, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("四元数必须是四个有限数值，顺序 xyzw")
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.) > .001:
        raise ValueError("四元数必须为单位四元数（范数误差 <= 0.001）")
    x, y, z, w = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def measured_pose(state, side):
    if state.get("selected_arm") != side or state.get("translation_unit") != "mm":
        raise ValueError("反馈机械臂或平移单位与所选机械臂/mm 不一致")
    arm = state["arms"][side]
    if state.get("base_frame") != "base_link" or arm.get("base_frame") != "base_link":
        raise ValueError("需要 base_link 基座坐标反馈")
    if arm.get("tcp_frame") != side + "_gripper_base_link":
        raise ValueError("TCP 必须使用所选臂的 gripper_base_link 实测 TF")
    return validate_transform(arm["T_base_tcp_mm"]).copy()


def load_pose(path, side, state):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("arm") != side:
        raise ValueError("位姿文件 arm 与 --arm 不一致")
    if "pose_source" in data:  # record_pose.py's measurement format.
        source = data["pose_source"]
        original = measured_pose(source, side)
        base = source["base_frame"]
        tcp = source["arms"][side]["tcp_frame"]
        if not np.allclose(original, data["T_base_tcp_mm"], rtol=0, atol=1e-7):
            raise ValueError("文件 T_base_tcp_mm 与 pose_source 实测位姿不一致")
    else:  # Explicit minimal format, never infer a missing frame or unit.
        base, tcp = data.get("base_frame"), data.get("tcp_frame")
        if data.get("translation_unit") != "mm":
            raise ValueError("简式位姿文件必须明确 translation_unit: mm")
    arm = state["arms"][side]
    if base != state["base_frame"] or tcp != arm["tcp_frame"]:
        raise ValueError("位姿文件的 base_frame/tcp_frame 与当前实测坐标系不一致")
    return validate_transform(data["T_base_tcp_mm"]).copy()


def build_target(args, state):
    current = measured_pose(state, args.arm)
    target = current.copy()
    if args.pose_file is not None:
        target = load_pose(args.pose_file, args.arm, state)
    elif args.xyz_mm is not None:
        target[:3, 3] = args.xyz_mm
        if args.quaternion_xyzw is not None:
            target[:3, :3] = quaternion_rotation(args.quaternion_xyzw)
    else:
        delta = args.delta_mm
        if args.direction:
            delta = np.zeros(3)
            delta["xyz".index(args.direction[0])] = args.distance_mm * (1 if args.direction[1] == "+" else -1)
        if delta is None:
            raise ValueError("没有指定目标")
        delta = np.asarray(delta, dtype=float)
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            raise ValueError("位移必须是三个有限数值，单位 mm")
        target[:3, 3] += current[:3, :3] @ delta if args.frame == "tcp" else delta
    validate_transform(target)
    check_limits(args, current, target)
    return current, target


def check_limits(args, current, target):
    distance, angle = pose_error(current, target)
    if distance > args.max_distance_mm + 1e-9 or angle > args.max_angle_deg + 1e-9:
        raise ValueError("目标超出单次限值：%.3f mm / %.3f deg；允许 %.3f mm / %.3f deg" %
                         (distance, angle, args.max_distance_mm, args.max_angle_deg))
    return distance, angle


class SampleDrain:
    """Discard explicitly non-acquisition events continuously, including human waits."""
    def __init__(self, arm):
        self.arm = arm
        self.stop = threading.Event()
        self.error = None
        self.count = 0
        self.thread = None

    def start(self):
        if hasattr(self.arm, "drain_samples"):
            self.thread = threading.Thread(target=self._run, name="tcp_move_event_drain", daemon=True)
            self.thread.start()
        return self

    def _run(self):
        while not self.stop.is_set():
            try:
                self.count += len(self.arm.drain_samples())
            except Exception as exc:
                self.error = exc
                return
            self.stop.wait(.005)

    def check(self):
        if self.error:
            raise RuntimeError("TCP 来源事件 drain 失败：" + str(self.error)) from self.error
        fault = getattr(self.arm, "fault", None)
        if fault:
            raise RuntimeError("机械臂反馈/控制故障：" + str(fault))

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1.)
            if self.thread.is_alive():
                raise RuntimeError("TCP 来源事件 drain 线程未退出")


class MoveSession:
    def __init__(self, args, arm, *, input_fn=input, output=print, cancel=None,
                 monotonic=time.monotonic, sleep=time.sleep, ownership_check=None):
        self.args, self.arm = args, arm
        self.input_fn, self.output = input_fn, output
        self.cancel = cancel if cancel is not None else threading.Event()
        self.monotonic, self.sleep = monotonic, sleep
        self.drain = SampleDrain(arm)
        self.active = False
        self.ownership_check = ownership_check

    def check(self):
        if self.cancel.is_set():
            raise MoveCancelled("收到中断请求")
        self.drain.check()

    def ask(self, prompt):
        # input() may block for minutes; its daemon only reads text, never acts.
        answers = queue.Queue()
        def read():
            try:
                answers.put((True, self.input_fn(prompt)))
            except BaseException as exc:
                answers.put((False, exc))
        threading.Thread(target=read, name="tcp_move_operator_input", daemon=True).start()
        while True:
            self.check()
            try:
                ok, value = answers.get(timeout=.02)
            except queue.Empty:
                continue
            self.check()
            if not ok:
                raise value
            return value.strip()

    def observation(self):
        self.check()
        state = self.arm.snapshot()  # Observer enforces fresh device + host timestamps.
        measured_pose(state, self.args.arm)
        if state.get("fault"):
            raise RuntimeError(str(state["fault"]))
        return state

    def show(self, state, current, target):
        distance, angle = pose_error(current, target)
        self.output("机械臂：%s；基座：%s；TCP：%s（夹爪基座原点，不是指尖）" %
                    (self.args.arm, state["base_frame"], state["arms"][self.args.arm]["tcp_frame"]))
        for name, value in (("当前", current), ("目标", target)):
            self.output("%s xyz(mm)=%s  quaternion(xyzw)=%s" %
                        (name, np.round(value[:3, 3], 6).tolist(), np.round(_matrix_quaternion_xyzw(value), 8).tolist()))
        self.output("位移 %.3f mm；旋转 %.3f deg；软件发布速度 %.3f mm/s、%.3f deg/s" %
                    (distance, angle, self.args.speed_mm_s, self.args.angular_speed_deg_s))
        self.output("实测到位容差：%.3f mm / %.3f deg" %
                    (self.args.pose_tolerance_mm, self.args.pose_tolerance_deg))

    def wait_subscriber(self):
        publisher = getattr(self.arm, "_publisher", None)
        if publisher is None:
            raise RuntimeError("命令 publisher 未连接")
        deadline = self.monotonic() + min(5., self.args.timeout_s)
        while publisher.get_num_connections() < 1:
            self.observation()
            if self.monotonic() >= deadline:
                raise TimeoutError("公开 ROS TCP 命令话题无订阅者；未提交移动")
            self.sleep(.02)

    def attempt(self, target_args):
        state = self.observation()
        current, target = build_target(target_args, state)
        self.show(state, current, target)
        record = {"schema":"bottle_tcp_move_v1", "arm": self.args.arm,
                  "real": self.args.real, "created_unix_ns":time.time_ns(),
                  "base_frame":state["base_frame"], "tcp_frame":state["arms"][self.args.arm]["tcp_frame"],
                  "translation_unit":"mm", "quaternion_order":"xyzw",
                  "preview_observation":state, "start_measured_mm":current.tolist(),
                  "target_mm":target.tolist(), "result":"preview_created",
                  "software_speed_mm_s":self.args.speed_mm_s,
                  "software_angular_speed_deg_s":self.args.angular_speed_deg_s,
                  "max_distance_mm":self.args.max_distance_mm,
                  "max_angle_deg":self.args.max_angle_deg, "timeout_s":self.args.timeout_s,
                  "pose_tolerance_mm":self.args.pose_tolerance_mm,
                  "pose_tolerance_deg":self.args.pose_tolerance_deg,
                  "event_handling":"source events drained during operator tool; not an experiment recording"}
        self.args.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.args.log_dir / ("tcp_move_%d_%s.json" % (time.time_ns(), uuid.uuid4().hex[:10]))
        # Create and flush BEFORE allowing any command. Never overwrite another attempt.
        with path.open("x+", encoding="utf-8") as log:
            def save():
                log.seek(0)
                json.dump(record, log, ensure_ascii=False, indent=2, allow_nan=False)
                log.write("\n")
                log.truncate()
                log.flush()
            save()
            try:
                if not self.args.real:
                    record["result"] = "preview_only"
                    self.output("只读预览：没有创建 TCP 命令 publisher，也没有发送动作。加 --real 后仍需确认 MOVE。")
                    return "preview_only"
                answer = self.ask("确认现场可直线到达上述目标且无人/障碍干涉；输入 MOVE 执行，其他输入取消：")
                if answer != "MOVE":
                    record["result"] = "operator_cancelled"
                    self.output("已取消；未发送 TCP。")
                    return "operator_cancelled"
                self.wait_subscriber()
                fresh = self.observation()
                fresh_pose = measured_pose(fresh, self.args.arm)
                drift_mm, drift_deg = pose_error(current, fresh_pose)
                record["confirmed_observation"] = fresh
                if drift_mm > 1. or drift_deg > 1.:
                    raise ValueError("确认期间起点变化超过 1 mm / 1 deg；旧目标已拒绝，请刷新并重新选择")
                check_limits(self.args, fresh_pose, target)
                # Re-read the master after the human wait. A second controller
                # may have started while this target was being reviewed.
                if self.ownership_check is None:
                    raise RuntimeError("未提供命令发布者检查；拒绝发送 TCP")
                record["command_ownership_before_move"] = self.ownership_check()
                record["execution_start_mm"] = fresh_pose.tolist()
                record["result"] = "submitted"
                save()
                self.check()
                self.arm.move_linear(target, self.args.speed_mm_s, self.args.angular_speed_deg_s)
                self.active = True
                deadline = self.monotonic() + self.args.timeout_s
                while True:
                    final = self.observation()
                    final_pose = measured_pose(final, self.args.arm)
                    if self.arm.motion_done():
                        distance, angle = pose_error(target, final_pose)
                        if distance > self.args.pose_tolerance_mm or angle > self.args.pose_tolerance_deg:
                            raise RuntimeError("控制器报告结束，但实测 TCP 尚未达到目标的 %.3f mm / %.3f deg 容差；实测误差 %.3f mm / %.3f deg" %
                                               (self.args.pose_tolerance_mm, self.args.pose_tolerance_deg, distance, angle))
                        self.active = False
                        record["result"] = "reached_by_measured_feedback"
                        record["final_observation"] = final
                        self.output("实测 TCP 已到达目标（%.3f mm / %.3f deg 容差，误差 %.3f mm / %.3f deg）；夹爪未操作。" %
                                    (self.args.pose_tolerance_mm, self.args.pose_tolerance_deg, distance, angle))
                        return record["result"]
                    if self.monotonic() >= deadline:
                        raise TimeoutError("TCP 移动等待实测到位超时")
                    self.sleep(.02)
            except BaseException as exc:
                record["result"] = "interrupted" if isinstance(exc, (MoveCancelled, KeyboardInterrupt)) else "failed"
                record["error"] = str(exc) or type(exc).__name__
                if self.active:
                    # Only a successfully submitted, unfinished motion may cause a hold.
                    try:
                        from bottle_scan.stop_validation import software_stop
                        software_stop(self.arm)
                        record["software_hold"] = "fresh_measured_pose_published"
                    except Exception as hold_error:
                        record["software_hold"] = "unavailable: " + str(hold_error)
                        self.output("软件保持不可用：%s；请使用现场底层停止/急停。" % hold_error)
                    self.active = False
                raise
            finally:
                try:
                    final = self.arm.snapshot()
                    record["end_measured_mm"] = measured_pose(final, self.args.arm).tolist()
                    record["end_observation"] = final
                except Exception as exc:
                    record["end_measurement_error"] = str(exc)
                record["finished_unix_ns"] = time.time_ns()
                record["drained_source_events_total"] = self.drain.count
                save()
                self.output("移动记录：" + str(path.resolve()))

    def interactive(self):
        self.output("菜单：基座 b:x+ b:x- b:y+ b:y- b:z+ b:z-；TCP t:x+ t:x- t:y+ t:y- t:z+ t:z-；xyz 输入绝对位置；file 载入位姿；r 刷新；q 退出")
        while True:
            state = self.observation()
            current = measured_pose(state, self.args.arm)
            self.output("当前基座 xyz(mm)=" + str(np.round(current[:3, 3], 3).tolist()))
            choice = self.ask("选择方向/操作：").lower()
            if choice in ("q", "quit", "exit"):
                return 0
            if choice in ("r", "refresh", ""):
                continue
            target_args = copy.copy(self.args)
            target_args.xyz_mm = target_args.delta_mm = target_args.pose_file = target_args.direction = None
            target_args.quaternion_xyzw = None
            target_args.frame = "base"
            try:
                if choice.startswith(("b:", "t:")) and choice[2:] in ("x+", "x-", "y+", "y-", "z+", "z-"):
                    target_args.frame = "base" if choice[0] == "b" else "tcp"
                    target_args.direction = choice[2:]
                    entered = self.ask("移动距离 mm [默认 5]：")
                    target_args.distance_mm = positive(entered or "5")
                elif choice == "xyz":
                    target_args.xyz_mm = [float(v) for v in self.ask("输入 base_link 的 X Y Z（mm，用空格分隔，保持当前姿态）：").split()]
                    if len(target_args.xyz_mm) != 3:
                        raise ValueError("请输入恰好三个数值")
                elif choice == "file":
                    target_args.pose_file = Path(self.ask("输入 record-pose JSON 路径：")).expanduser()
                else:
                    raise ValueError("未知操作；例如 b:z+ 或 t:x-，或 xyz/file/r/q")
                validate_args(target_args)
                self.attempt(target_args)
            except (ValueError, argparse.ArgumentTypeError, FileNotFoundError, json.JSONDecodeError) as exc:
                self.output("本次操作未完成：" + str(exc))


def main(argv=None, *, observer_factory=None, controller_factory=None, input_fn=input, output=print,
         ownership_guard=None, ownership_inspector=None, config_loader=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        p.error(str(exc))
    guard = ownership_guard or guard_all_commands
    inspector = ownership_inspector or inspect_all_commands
    cfg = None
    def check_ownership():
        report = guard(cfg, own_nodes=current_ros_node_names())
        output(format_ownership_report(report))
        return report
    factory = (controller_factory or RosArmController) if args.real else (observer_factory or RosArmObserver)
    control_options = {"pose_tolerance_mm":args.pose_tolerance_mm,
                       "pose_tolerance_deg":args.pose_tolerance_deg} if args.real else {}
    arm = factory(arm_side=args.arm, **control_options)
    cancel = threading.Event()
    session = MoveSession(args, arm, input_fn=input_fn, output=output, cancel=cancel,
                          ownership_check=check_ownership)
    old_signals = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM) + ((signal.SIGHUP,) if hasattr(signal, "SIGHUP") else ()):
            old_signals[signum] = signal.signal(signum, lambda *_: cancel.set())
    try:
        output("执行移动前请结束向同一机械臂发指令的其它推理/遥操进程，避免多个命令源。")
        output("基座 XYZ 是 base_link 坐标，不自动等于人的左右/前后；TCP 轴来自当前 gripper_base_link。")
        output("此工具不做碰撞检测或绕障；软件速度/保持不等于底层硬停止。夹爪状态不改变。")
        try:
            cfg = (config_loader or load_config)(args.config)
            if args.real:
                check_ownership()  # Before controller.connect creates its publisher.
            else:
                output(format_ownership_report(inspector(cfg, own_nodes=current_ros_node_names())))
        except Exception as exc:
            if args.real:
                raise
            output("只读预览可继续；命令发布者核对失败：" + str(exc))
        arm.connect()  # Connection alone never commands either arm.
        session.drain.start()
        if any(getattr(args, name) is not None for name in ("xyz_mm", "delta_mm", "pose_file", "direction")):
            session.attempt(args)
            return 0
        return session.interactive()
    except (MoveCancelled, KeyboardInterrupt, EOFError) as exc:
        output("已结束：" + (str(exc) or type(exc).__name__))
        return 130
    except Exception as exc:
        output("TCP 工具失败：" + str(exc))
        return 2
    finally:
        try:
            session.drain.close()
        finally:
            try:
                arm.close()  # No implicit command on normal/idle exit.
            finally:
                for signum, handler in old_signals.items():
                    signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())

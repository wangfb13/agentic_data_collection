"""A4 software-stop and publisher-process-loss measurements.

The parent observes only. Real motion lives in an explicitly started child. A
SIGKILL trial never sends a fallback hold or return from the parent. The last
target is accepted only when the parent's actual topic observation matches the
child's final publication intent, which is written before EVERY publish. A
child killed during that write, or an intent not seen on the topic, invalidates
the trial instead of inventing a final command from the requested endpoint.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import threading
import time

import numpy as np

from .geometry import check_pose, check_segment, pose_error, validate_transform
from .session import atomic_json, fingerprint, plain


def stop_test_defaults(*, demo=False):
    return {"distance_mm":50., "repeats":3,
            "direction_base":[1.,0.,0.] if demo else "TODO_unit_direction_base",
            "trigger_s":.15 if demo else "TODO_measured_trigger_s",
            "settle_window_s":.12 if demo else "TODO_measured_settle_window_s",
            "settle_translation_mm":.025 if demo else "TODO_measured_position_noise_mm",
            "settle_rotation_deg":.1 if demo else "TODO_measured_rotation_noise_deg",
            "timeout_s":3. if demo else "TODO_stop_observation_timeout_s",
            "max_extra_mm":4. if demo else "TODO_accepted_extra_displacement_mm",
            "sample_hz":100., "command_quiet_s":.25,
            "report_path":"outputs/validation/stop_report.json"}


def validate_stop_settings(cfg):
    options=cfg.get("stop_test",{})
    errors=[]
    for key in ("distance_mm","trigger_s","settle_window_s","settle_translation_mm",
                "settle_rotation_deg","timeout_s","max_extra_mm","sample_hz","command_quiet_s"):
        value=options.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            errors.append("stop_test."+key+" requires a positive finite measured value")
    if type(options.get("repeats")) is not int or options["repeats"]<3:
        errors.append("stop_test.repeats must be at least 3 per test mode")
    try:
        direction=np.asarray(options["direction_base"],dtype=float)
        if direction.shape!=(3,) or not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-1)>1e-6:
            raise ValueError()
    except (ValueError,TypeError,KeyError):errors.append("stop_test.direction_base requires a measured base-frame unit vector")
    speed=cfg.get("motion",{}).get("free_speed_mm_s")
    if isinstance(speed,bool) or not isinstance(speed,(int,float)) or not math.isfinite(speed) or speed<=0:
        errors.append("motion.free_speed_mm_s requires measured positive experiment speed")
    if not errors:
        duration=options["distance_mm"]/cfg["motion"]["free_speed_mm_s"]
        if options["trigger_s"]>=duration:
            errors.append("stop trigger must precede the planned target arrival")
        if options["timeout_s"]<=options["settle_window_s"]+options["command_quiet_s"]:
            errors.append("stop observation timeout must exceed settling plus command quiet windows")
    return errors


def stop_report_binding(cfg):
    relevant={key:copy.deepcopy(cfg.get(key)) for key in
              ("hardware","motion","tool","geometry","safety","gripper","axes","preparation","stop_test")}
    relevant["stop_test"].pop("report_path",None)
    # This threshold gates observer freshness and measured frame-gap validity.
    # Other recording settings (especially synthetic stress load) are not used
    # by stop-test's independent read-only observation path.
    relevant["recording"]={"source_max_age_s":cfg.get("recording",{}).get("source_max_age_s")}
    root=Path(__file__).resolve().parents[1]
    files=("bottle_scan/hardware_arm.py","bottle_scan/geometry.py","bottle_scan/stop_validation.py",
           "bottle_scan/command_ownership.py","stop_test.py","move_tcp.py")
    sources={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
    return {"config_sha256":fingerprint(relevant),"source_sha256":fingerprint(sources),"source_files":sources}


def software_stop(arm):
    """Shared Ctrl-C/cancel path: cancel interpolation and hold measured pose."""
    stamp=time.monotonic_ns()
    arm.stop_arm()
    return {"request_monotonic_ns":stamp,"method":"cancel_interpolation_then_fresh_measured_pose_hold"}


def stop_path(cfg, measured, width_mm):
    errors=validate_stop_settings(cfg)
    if errors:raise ValueError("; ".join(errors))
    start=validate_transform(measured)
    nominal=validate_transform(cfg["preparation"]["axis_test_pose"])
    translation,rotation=pose_error(start,nominal)
    if translation>cfg["motion"]["pose_tolerance_mm"] or rotation>cfg["motion"]["pose_tolerance_deg"]:
        raise ValueError("Stop test starts only at axis_test_pose; use move-tcp to reset, then retry")
    if not np.isfinite(width_mm) or abs(width_mm-cfg["gripper"]["open_mm"])>cfg["gripper"]["open_tolerance_mm"]:
        raise ValueError("Stop test requires fresh measured w_open; it never opens the gripper automatically")
    target=start.copy()
    target[:3,3]+=np.asarray(cfg["stop_test"]["direction_base"])*cfg["stop_test"]["distance_mm"]
    T0=np.asarray(cfg["T0"],float)
    reasons=(check_pose(start,cfg,T0,width_mm=width_mm,allow_contact=False)
             +check_pose(target,cfg,T0,width_mm=width_mm,allow_contact=False)
             +check_segment(start,target,cfg,T0,width_mm=width_mm,allow_contact=False))
    if reasons:raise ValueError("Stop-test full free-space path rejected: "+str(reasons))
    return {"start_tcp_mm":start.tolist(),"target_tcp_mm":target.tolist(),"measured_width_mm":float(width_mm),
            "path_check":"passed", "speed_mm_s":cfg["motion"]["free_speed_mm_s"],
            "angular_speed_deg_s":cfg["motion"]["angular_speed_deg_s"]}


def recheck_confirmed_path(cfg,path,measured,width_mm):
    # The pose target remains the reviewed target. Recheck the complete segment
    # from the new measured start, allowing only the existing arrival tolerance.
    stop_path(cfg,measured,width_mm)
    distance,angle=pose_error(np.asarray(measured),np.asarray(path["start_tcp_mm"]))
    if distance>cfg["motion"]["pose_tolerance_mm"] or angle>cfg["motion"]["pose_tolerance_deg"]:
        raise ValueError("Start changed after MOVE; refresh stop-test preview")
    target=np.asarray(path["target_tcp_mm"])
    reasons=check_segment(np.asarray(measured),target,cfg,np.asarray(cfg["T0"]),width_mm=width_mm,allow_contact=False)
    if reasons:raise ValueError("Confirmed stop-test path rejected: "+str(reasons))
    return {**path,"start_tcp_mm":plain(measured),"measured_width_mm":float(width_mm)}


def stable_window(samples, request_ns, options):
    if not samples:return None
    end=samples[-1]["monotonic_ns"]
    window_ns=int(options["settle_window_s"]*1e9)
    if end-request_ns<window_ns:return None
    selected=[row for row in samples if row["monotonic_ns"]>=request_ns and row["monotonic_ns"]>=end-window_ns]
    if len(selected)<3 or selected[-1]["monotonic_ns"]-selected[0]["monotonic_ns"]<window_ns*.9:return None
    origin=np.asarray(selected[0]["tcp_mm"])
    errors=[pose_error(origin,np.asarray(row["tcp_mm"])) for row in selected]
    if max(x[0] for x in errors)>options["settle_translation_mm"] or max(x[1] for x in errors)>options["settle_rotation_deg"]:
        return None
    return {"settled_monotonic_ns":selected[0]["monotonic_ns"],"settle_detected_monotonic_ns":end,
            "settle_window_samples":len(selected),"window_translation_span_mm":max(x[0] for x in errors),
            "window_rotation_span_deg":max(x[1] for x in errors)}


def _write_intent(shared, values):
    version=int(shared[0])+1
    shared[0]=version # odd: an interrupted write is never trustworthy
    shared[1:1+len(values)]=values
    shared[0]=version+1 # even BEFORE calling publish, so a missing final frame is detected


def read_intent(shared):
    before=int(shared[0]);values=list(shared[1:11]);after=int(shared[0])
    if before<=0 or before!=after or before%2:
        return None
    return {"write_version":before,"wire_values":values}


def verify_last_target(intent, commands):
    if not intent or not commands:return False
    # Compare actual source sec/nsec + arm mask + seven source pose numbers,
    # without receiver-side rounding or conversion to a nominal final target.
    if intent["wire_values"]!=commands[-1]["wire_values"]:return False
    timestamps=[int(row["wire_values"][0])*1_000_000_000+int(row["wire_values"][1]) for row in commands]
    return all(b>a for a,b in zip(timestamps,timestamps[1:]))


def summarize_trial(cfg, trial):
    samples=trial["measurements"]
    request=trial.get("stop_request_monotonic_ns")
    if not request or not samples:
        return {**trial,"valid":False,"passed":False,"invalid_reason":"missing stop request or measured feedback"}
    before=[row for row in samples if row["monotonic_ns"]<=request]
    after=[row for row in samples if row["monotonic_ns"]>=request]
    if not before or not after:
        return {**trial,"valid":False,"passed":False,"invalid_reason":"no measured samples bracketing stop request"}
    reference=np.asarray(before[-1]["tcp_mm"])
    final=np.asarray(samples[-1]["tcp_mm"])
    extra=max(pose_error(reference,np.asarray(row["tcp_mm"]))[0] for row in after)
    stable=stable_window(samples,request,cfg["stop_test"])
    verified=verify_last_target(trial.get("last_publish_intent"),trial["commands"])
    # Initial spawn/connection wait is not part of motion evidence. Evaluate
    # receive gaps only once movement has started.
    moving=[row["monotonic_ns"] for row in samples if row["monotonic_ns"]>=trial.get("motion_started_monotonic_ns",samples[0]["monotonic_ns"])]
    max_gap=max((b-a for a,b in zip(moving,moving[1:])),default=0)/1e9
    acknowledged=trial.get("software_stop_acknowledged") is True if trial["mode"]=="software" else trial.get("child_exitcode")==-signal.SIGKILL
    valid=bool(stable and verified and acknowledged and not trial.get("error") and max_gap<=cfg["recording"]["source_max_age_s"])
    result={**trial,**(stable or {}),"valid":valid,"passed":valid and extra<=cfg["stop_test"]["max_extra_mm"],
            "max_extra_displacement_mm":extra,"accepted_extra_limit_mm":cfg["stop_test"]["max_extra_mm"],
            "stop_request_pose_mm":reference.tolist(),"stop_request_pose_source_ns":before[-1]["monotonic_ns"],
            "stop_request_pose_age_s":(request-before[-1]["monotonic_ns"])/1e9,"final_tcp_mm":final.tolist(),
            "last_published_target_verified":verified,
            "maximum_moving_feedback_gap_s":max_gap,
            "last_target_evidence":"final complete pre-publish intent equals parent's received source-stamped command"}
    if trial["commands"]:
        last=trial["commands"][-1]
        result.update(last_observed_target_tcp_mm=last["tcp_mm"],last_observed_target_source_ros_ns=last["source_ros_ns"],
                      final_to_last_target_error_mm_deg=list(pose_error(final,np.asarray(last["tcp_mm"]))))
    if not valid:
        result["invalid_reason"]=(trial.get("error") or ("stop request not acknowledged/child death not observed" if not acknowledged else
                                  "last published frame unverified" if not verified else
                                  "stale/gapped feedback" if max_gap>cfg["recording"]["source_max_age_s"] else "no measured settled window"))
    return result


class _IntentPublisher:
    def __init__(self,publisher,shared,side):self.publisher,self.shared,self.side=publisher,shared,side
    def get_num_connections(self):return self.publisher.get_num_connections()
    def unregister(self):return self.publisher.unregister()
    def publish(self,message):
        field=getattr(message,self.side+"_arm_goal_pose")
        p,q=field.pose.position,field.pose.orientation
        values=[float(field.header.stamp.secs),float(field.header.stamp.nsecs),float(message.pose_selection),
                p.x,p.y,p.z,q.x,q.y,q.z,q.w]
        _write_intent(self.shared,values)
        self.publisher.publish(message)


def _child_real(pipe,shared,cfg):
    # This function is entered only by an operator-authorized --real subprocess.
    from .hardware_arm import RosArmController
    from .hardware_gripper import RosNormalizedGripperObserver,WidthCalibration
    from .command_ownership import guard_all_commands,current_ros_node_names
    from move_tcp import SampleDrain
    side=cfg["hardware"]["arm_side"]
    options=cfg["hardware"]["public_ros"]
    arm=RosArmController(arm_side=side,base_frame=cfg["hardware"]["base_frame"],
        max_age_s=cfg["recording"]["source_max_age_s"],publish_hz=options["arm_publish_hz"],
        max_step_mm=options["arm_command_step_mm"],max_step_deg=options["arm_command_step_deg"],
        pose_tolerance_mm=cfg["motion"]["pose_tolerance_mm"],pose_tolerance_deg=cfg["motion"]["pose_tolerance_deg"])
    grip=RosNormalizedGripperObserver(side,calibration=WidthCalibration(cfg["hardware"]["gripper_width_calibration"]))
    drain=SampleDrain(arm)
    cancel=threading.Event();active=False
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:cancel.set())
    try:
        guard_all_commands(cfg)
        arm.connect();grip.connect();drain.start()
        grip_deadline=time.monotonic()+5.
        while not grip.snapshot()["fresh"]:
            if time.monotonic()>=grip_deadline:raise RuntimeError("No fresh measured gripper state before stop test")
            if cancel.is_set():raise RuntimeError("Cancelled while awaiting gripper state")
            time.sleep(.01)
        arm._publisher=_IntentPublisher(arm._publisher,shared,side)
        pipe.send({"event":"ready","node":current_ros_node_names()[0]})
        while not cancel.is_set():
            drain.check()
            if not pipe.poll(.01):continue
            message=pipe.recv();kind=message["kind"]
            if kind=="start":
                state=arm.snapshot();opening=grip.snapshot()
                if not opening["fresh"] or not opening["valid"]:raise RuntimeError("Stale gripper opening before stop test")
                path=recheck_confirmed_path(cfg,message["path"],state["arms"][side]["T_base_tcp_mm"],opening["width_mm"])
                guard_all_commands(cfg,own_nodes=current_ros_node_names())
                arm.move_linear(path["target_tcp_mm"],path["speed_mm_s"],path["angular_speed_deg_s"])
                active=True
                pipe.send({"event":"started","monotonic_ns":time.monotonic_ns()})
            elif kind=="stop":
                info=software_stop(arm);active=False
                pipe.send({"event":"stopped",**info})
            elif kind=="close":break
        if cancel.is_set() and active:
            pipe.send({"event":"cancelled",**software_stop(arm)});active=False
    except BaseException as exc:
        if active:
            try:software_stop(arm)
            except Exception:pass
        try:pipe.send({"event":"error","error":type(exc).__name__+": "+str(exc)})
        except Exception:pass
    finally:
        drain.close();grip.close();arm.close();pipe.close()


def _child_fake(pipe,shared,cfg,actual):
    """Wall-clock simulator child; killed by an actual SIGKILL, never imports ROS."""
    side=cfg["hardware"]["arm_side"]
    start=np.asarray(cfg["preparation"]["axis_test_pose"],float)
    target=start.copy();origin=None;last_ns=0
    pipe.send({"event":"ready","node":"/synthetic_stop_child_"+str(os.getpid())})
    def publish(pose):
        nonlocal last_ns
        stamp=max(time.monotonic_ns(),last_ns+1);last_ns=stamp
        values=[float(stamp//1_000_000_000),float(stamp%1_000_000_000),float(1 if side=="left" else 2),
                *list(pose[:3,3]/1000.),0.,0.,0.,1.]
        _write_intent(shared,values)
        pipe.send({"event":"command","wire_values":values,"tcp_mm":pose.tolist(),
                   "source_ros_ns":stamp,"received_monotonic_ns":time.monotonic_ns(),"simulated":True})
    class Arm:
        def stop_arm(self):
            nonlocal origin
            origin=None
            pose=start.copy();pose[:3,3]=list(actual)
            publish(pose)
    arm=Arm()
    while True:
        if pipe.poll(.005):
            message=pipe.recv()
            if message["kind"]=="start":
                target=np.asarray(message["path"]["target_tcp_mm"])
                origin=time.monotonic_ns();pipe.send({"event":"started","monotonic_ns":origin})
            elif message["kind"]=="stop":pipe.send({"event":"stopped",**software_stop(arm)})
            elif message["kind"]=="close":break
        if origin is not None:
            distance=np.linalg.norm(target[:3,3]-start[:3,3])
            fraction=min(1.,(time.monotonic_ns()-origin)/1e9*cfg["motion"]["free_speed_mm_s"]/distance)
            pose=start.copy();pose[:3,3]+=fraction*(target[:3,3]-start[:3,3]);publish(pose)


class RealObserver:
    """Only state/gripper/command subscribers; no parent actuator publisher."""
    def __init__(self,cfg):
        from .hardware_arm import RosArmObserver
        from .hardware_gripper import RosNormalizedGripperObserver,WidthCalibration
        self.cfg=cfg;self.side=cfg["hardware"]["arm_side"]
        self.arm=RosArmObserver(arm_side=self.side,max_age_s=cfg["recording"]["source_max_age_s"],state_queue_size=100)
        self.grip=RosNormalizedGripperObserver(self.side,calibration=WidthCalibration(cfg["hardware"]["gripper_width_calibration"]))
        self.commands=[];self.error=None;self.node=None;self.lock=threading.Lock();self.sub=None

    def connect(self):
        from .hardware_arm import _command_message_type
        self.arm.connect();self.grip.connect()
        self.sub=self.arm._rospy.Subscriber("/wbc_goal_pose",_command_message_type(),self._command,queue_size=1000)
        deadline=time.monotonic()+5.
        while True:
            try:self.read();break
            except RuntimeError:
                if time.monotonic()>=deadline:raise
                time.sleep(.01)
        return self

    def _command(self,message):
        if not self.node:return
        try:
            caller=getattr(message,"_connection_header",{}).get("callerid")
            if caller!=self.node:raise RuntimeError("Unexpected command publisher during stop observation: "+str(caller))
            from .hardware_arm import _pose_matrix_mm
            field=getattr(message,self.side+"_arm_goal_pose");p,q=field.pose.position,field.pose.orientation
            if message.pose_selection!=(1 if self.side=="left" else 2):raise RuntimeError("Unexpected arm mask")
            state=self.arm.snapshot();tool=np.asarray(state["arms"][self.side]["T_flange_tcp_mm"])
            tcp=_pose_matrix_mm([p.x,p.y,p.z],[q.x,q.y,q.z,q.w])@tool
            if field.header.frame_id.lstrip("/")!=self.cfg["hardware"]["base_frame"].lstrip("/"):
                raise RuntimeError("Command base frame mismatch")
            values=[float(field.header.stamp.secs),float(field.header.stamp.nsecs),float(message.pose_selection),
                    p.x,p.y,p.z,q.x,q.y,q.z,q.w]
            row={"tcp_mm":tcp.tolist(),"wire_values":values,"source_ros_ns":int(field.header.stamp.to_nsec()),
                 "received_monotonic_ns":time.monotonic_ns(),"publisher":caller,
                 "tool_tf_source":"parent live TF at feedback time","tool_tf":tool.tolist()}
            with self.lock:self.commands.append(row)
        except Exception as exc:self.error=str(exc)

    def begin(self,node):
        with self.lock:self.commands=[];self.error=None;self.node=node

    def drain_commands(self):
        if self.error:raise RuntimeError(self.error)
        with self.lock:rows,self.commands=self.commands,[]
        return rows

    def read(self):
        state=self.arm.snapshot();grip=self.grip.snapshot()
        if not grip["fresh"] or not grip["valid"] or grip["width_mm"] is None:
            raise RuntimeError("Fresh calibrated gripper width required")
        value=state["arms"][self.side]
        return {"monotonic_ns":value["received_monotonic_ns"],"tcp_mm":value["T_base_tcp_mm"],
                "source_ros_timestamp_s":value["source_ros_timestamp_s"],"width_mm":grip["width_mm"],
                "width_source_monotonic_ns":grip["monotonic_ns"]}

    def close(self):
        if self.sub:self.sub.unregister()
        self.grip.close();self.arm.close()


def execute_trial(cfg,mode,index,*,simulate,observer=None,confirm=input,output=print):
    if mode not in ("software","sigkill"):raise ValueError("Unknown stop test mode")
    ctx=mp.get_context("spawn")
    shared=ctx.Array("d",16,lock=False)
    actual=ctx.Array("d",list(np.asarray(cfg["preparation"]["axis_test_pose"])[:3,3]),lock=False)
    target=np.asarray(cfg["preparation"]["axis_test_pose"],float)
    if simulate:
        initial={"monotonic_ns":time.monotonic_ns(),"tcp_mm":target.tolist(),"width_mm":cfg["gripper"]["open_mm"]}
    else:
        initial=observer.read()
    path=stop_path(cfg,initial["tcp_mm"],initial["width_mm"])
    output(json.dumps({"mode":mode,"repeat":index,**path},ensure_ascii=False))
    if not simulate and confirm("核对自由空间路径。输入 MOVE 开始本次停止测试；其它输入取消：").strip()!="MOVE":
        raise RuntimeError("Stop test cancelled; no motion submitted")
    if not simulate:
        from .command_ownership import guard_all_commands,current_ros_node_names
        guard_all_commands(cfg,own_nodes=current_ros_node_names())
        fresh=observer.read()
        recheck_confirmed_path(cfg,path,fresh["tcp_mm"],fresh["width_mm"])
    parent,child=ctx.Pipe()
    proc=ctx.Process(target=_child_fake if simulate else _child_real,
                    args=(child,shared,cfg,actual) if simulate else (child,shared,cfg),name="bottle_stop_test_child")
    trial={"mode":mode,"repeat":index,"simulated":simulate,"path":path,"measurements":[initial],"commands":[]}
    killed=False;started=None;requested=False;last_tick=time.monotonic();last_command_ns=0
    proc.start();child.close()
    try:
        if not parent.poll(15):raise TimeoutError("Stop-test child connection timed out")
        ready=parent.recv()
        if ready.get("event")!="ready":raise RuntimeError(str(ready))
        if not simulate:
            observer.begin(ready["node"])
            guard_all_commands(cfg,own_nodes=(*current_ros_node_names(),ready["node"]))
        parent.send({"kind":"start","path":path})
        deadline=time.monotonic()+cfg["stop_test"]["timeout_s"]+cfg["stop_test"]["trigger_s"]+5
        while time.monotonic()<deadline:
            while parent.poll():
                try:message=parent.recv()
                except EOFError:break
                event=message["event"]
                if event=="started":
                    started=message["monotonic_ns"];trial["motion_started_monotonic_ns"]=started
                elif event=="stopped":
                    trial["software_stop_acknowledged"]=True
                    trial["controller_stop_call_monotonic_ns"]=message["request_monotonic_ns"]
                    trial["stop_dispatch_latency_s"]=(message["request_monotonic_ns"]-trial["stop_request_monotonic_ns"])/1e9
                elif event=="command":trial["commands"].append(message);target=np.asarray(message["tcp_mm"]);last_command_ns=message["received_monotonic_ns"]
                elif event=="error":raise RuntimeError(message["error"])
            now=time.monotonic_ns()
            if simulate:
                tick=time.monotonic();dt=tick-last_tick;last_tick=tick
                # Passive plant model keeps tracking the last received goal
                # after publisher death, making process-loss behaviour visible.
                position=np.asarray(list(actual));position+=(target[:3,3]-position)*(1-math.exp(-dt/.08))
                actual[:]=position.tolist();pose=target.copy();pose[:3,3]=position
                measurement={"monotonic_ns":now,"tcp_mm":pose.tolist(),"width_mm":cfg["gripper"]["open_mm"]}
            else:
                commands=observer.drain_commands();trial["commands"].extend(commands)
                if commands:last_command_ns=commands[-1]["received_monotonic_ns"]
                measurement=observer.read()
                if abs(measurement["width_mm"]-cfg["gripper"]["open_mm"])>cfg["gripper"]["open_tolerance_mm"]:
                    raise RuntimeError("Measured opening changed during stop test")
            if measurement["monotonic_ns"]>trial["measurements"][-1]["monotonic_ns"]:
                trial["measurements"].append(measurement)
            if started and not requested and (now-started)/1e9>=cfg["stop_test"]["trigger_s"]:
                requested=True
                if mode=="software":
                    # Include IPC/dispatch latency in extra travel. Recording
                    # only the later child stop_arm() call would hide it.
                    trial["stop_request_monotonic_ns"]=time.monotonic_ns()
                    parent.send({"kind":"stop"})
                else:
                    trial["stop_request_monotonic_ns"]=time.monotonic_ns()
                    os.kill(proc.pid,signal.SIGKILL);killed=True
                    proc.join(timeout=1.)
                    if proc.is_alive():raise RuntimeError("SIGKILL did not terminate test child")
                    trial["child_exitcode"]=proc.exitcode
            request=trial.get("stop_request_monotonic_ns")
            if request:
                if (now-request)/1e9>cfg["stop_test"]["timeout_s"]:break
                if ((mode=="sigkill" or trial.get("software_stop_acknowledged"))
                        and stable_window(trial["measurements"],request,cfg["stop_test"])
                        and last_command_ns and (now-last_command_ns)/1e9>=cfg["stop_test"]["command_quiet_s"]):break
            if not proc.is_alive() and not killed:raise RuntimeError("Stop-test child exited before test completion")
            time.sleep(1/cfg["stop_test"]["sample_hz"])
        trial["last_publish_intent"]=read_intent(shared)
    except BaseException as exc:
        trial["error"]=type(exc).__name__+": "+str(exc)
    finally:
        if proc.is_alive():
            # Before/after a software trial, interruption uses the same child
            # cancel path. A killed child is never replaced by a parent hold.
            if trial.get("error") or not trial.get("stop_request_monotonic_ns"):
                try:parent.send({"kind":"stop"})
                except (BrokenPipeError,EOFError):pass
            try:parent.send({"kind":"close"})
            except (BrokenPipeError,EOFError):pass
            proc.join(timeout=2.)
            if proc.is_alive():
                os.kill(proc.pid,signal.SIGTERM);proc.join(timeout=2.)
                trial["error"]="Child failed to finish cleanup; report invalid"
        if not simulate:observer.node=None
        parent.close()
    return summarize_trial(cfg,trial)


def gate_stop_report(cfg,path=None):
    path=Path(path or cfg["stop_test"]["report_path"])
    report=json.loads(path.read_text())
    if (report.get("schema")!="bottle_stop_validation_v2" or report.get("simulated") is not False
            or report.get("operator_confirmed") is not True or report.get("passed") is not True
            or report.get("binding")!=stop_report_binding(cfg)):
        raise ValueError("A current real PASS-confirmed stop-test report is required")
    trials=report.get("trials",[])
    for mode in ("software","sigkill"):
        selected=[x for x in trials if x.get("mode")==mode]
        if len(selected)<cfg["stop_test"]["repeats"] or any(
                x.get("simulated") is not False or not x.get("valid") or not x.get("passed")
                or not x.get("last_published_target_verified")
                or not isinstance(x.get("max_extra_displacement_mm"),(int,float))
                or not 0<=x["max_extra_displacement_mm"]<=cfg["stop_test"]["max_extra_mm"] for x in selected):
            raise ValueError("Stop report lacks passing repeats for "+mode)
    return report


def run_stop_test(cfg,*,simulate,output_path,confirm=input,output=print):
    errors=validate_stop_settings(cfg)
    if errors:raise ValueError("; ".join(errors))
    path=Path(output_path)
    if path.exists():raise FileExistsError("Refusing to overwrite stop-test report: "+str(path))
    report={"schema":"bottle_stop_validation_v2","simulated":bool(simulate),"binding":stop_report_binding(cfg),
            "created_unix_ns":time.time_ns(),"trials":[],"passed":False,"operator_confirmed":False,
            "parent_control":"read_only_observer; no automatic return/hold after SIGKILL",
            "scope":{"axis_test_pose":cfg["preparation"]["axis_test_pose"],
                     "direction_base":cfg["stop_test"]["direction_base"],
                     "speed_mm_s":cfg["motion"]["free_speed_mm_s"],
                     "claim":"measured only at this free-space pose/direction/speed; not a workspace-wide hardware-stop guarantee"}}
    atomic_json(path,report)
    observer=None
    try:
        if not simulate:
            from .command_ownership import guard_all_commands
            guard_all_commands(cfg)
            observer=RealObserver(cfg)
            observer.connect()
        for mode in ("software","sigkill"):
            for index in range(1,cfg["stop_test"]["repeats"]+1):
                if not simulate:
                    confirm("用 move-tcp 将实验臂复位到 axis_test_pose，保持夹爪张开，退出定位工具；回车只读核对起点：")
                trial=execute_trial(cfg,mode,index,simulate=simulate,observer=observer,confirm=confirm,output=output)
                report["trials"].append(trial);atomic_json(path,report)
                output(json.dumps({key:trial.get(key) for key in
                    ("mode","repeat","valid","passed","max_extra_displacement_mm","last_published_target_verified","invalid_reason")},ensure_ascii=False))
                if not trial["valid"]:raise RuntimeError("Invalid stop-test evidence; see saved report")
        report["passed"]=all(row["passed"] for row in report["trials"])
        if report["passed"]:
            report["operator_confirmed"]=False if simulate else confirm("核对本报告各重复结果。输入 PASS 确认验收：").strip()=="PASS"
        report["finished_unix_ns"]=time.time_ns()
        atomic_json(path,report)
        return report
    except BaseException as exc:
        report["error"]=type(exc).__name__+": "+str(exc);report["passed"]=False
        atomic_json(path,report);raise
    finally:
        if observer:observer.close()

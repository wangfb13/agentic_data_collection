"""Immutable non-experiment-arm parking evidence and image/pose association."""
from __future__ import annotations
import copy
import math
import time
import numpy as np
from .geometry import pose_error,validate_transform
from .session import plain


def park_guard_defaults():
    return {"translation_mm":1.,"rotation_deg":.5,"joint_topic":"/joint_states",
            "joint_names":{side:[f"arm_{side}_{i}_joint" for i in range(1,8)] for side in ("left","right")},
            "mapping_evidence":"Existing Spark adapter schema plus read-only /joint_states observation on 2026-10-27"}


def validate_park_settings(cfg):
    errors=[];options=cfg.get("park_guard",{})
    for key in ("translation_mm","rotation_deg"):
        value=options.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            errors.append("park_guard."+key+" requires positive finite tolerance")
    try:
        from .hardware_observer import RosJointObserver
        RosJointObserver(cfg)
    except (KeyError,TypeError,ValueError) as exc:errors.append("park_guard joint source: "+str(exc))
    return errors


class ParkGuardError(RuntimeError):
    def __init__(self,message,report=None):super().__init__(message);self.report=report


def _fresh(stamp,now,max_age,description):
    if not isinstance(stamp,(int,np.integer)) or not 0<=(now-int(stamp))/1e9<=max_age:
        raise ParkGuardError("Stale/future/missing "+description+"; stop and refresh observation")


def _park_observation(session,observation,now):
    side="left" if session.cfg["hardware"]["arm_side"]=="right" else "right"
    age=session.cfg["recording"]["source_max_age_s"]
    try:
        arm=observation["arms"][side];joints=observation["joints"]
        _fresh(arm["received_monotonic_ns"],now,age,side+" TCP")
        _fresh(joints["received_monotonic_ns"],now,age,side+" joints")
        pose=validate_transform(arm["T_base_tcp_mm"])
        values=joints["arms"][side]
        expected=session.cfg["park_guard"]["joint_names"][side]
        if values["names"]!=expected or len(values["position_rad"])!=7 or not np.isfinite(values["position_rad"]).all():
            raise ParkGuardError("Parked-arm joint feedback does not match configured measured names")
        return {"side":side,"T_base_tcp_mm":pose.tolist(),"tcp_source":copy.deepcopy(arm),
                "joints":copy.deepcopy(values),"joint_source":{k:copy.deepcopy(v) for k,v in joints.items() if k!="arms"},
                "recorded_monotonic_ns":int(now)}
    except (KeyError,TypeError,ValueError) as exc:raise ParkGuardError("Incomplete parked-arm observation: "+str(exc)) from exc


class ParkGuard:
    def __init__(self,session):self.session=session

    @classmethod
    def capture(cls,session,observation,*,now_ns=None):
        if "nonexperiment_arm_park" in session.meta:
            raise ParkGuardError("Park reference is immutable; create a new session after deliberate camera-arm relocation")
        errors=validate_park_settings(session.cfg)
        if errors:raise ParkGuardError("; ".join(errors))
        record=_park_observation(session,observation,time.monotonic_ns() if now_ns is None else now_ns)
        record["reference_policy"]="immutable_for_session; no automatic return or rebaseline"
        record["camera_mount_confirmed_by_operator"]=True
        session.meta["nonexperiment_arm_park"]=record
        selected=session.cfg["hardware"]["arm_side"]
        session.meta["camera_roles"]={side+"_wrist":("eye_in_hand" if side==selected else "fixed_external_view_on_parked_arm") for side in ("left","right")}
        session.save_meta()
        return cls(session)

    def check(self,observation,*,context="CHECK",now_ns=None):
        reference=self.session.meta.get("nonexperiment_arm_park")
        if reference is None:raise ParkGuardError("Session lacks parked-arm TCP and joint reference; complete prepare")
        current=_park_observation(self.session,observation,time.monotonic_ns() if now_ns is None else now_ns)
        if current["side"]!=reference["side"]:raise ParkGuardError("Experiment arm changed within session")
        translation,rotation=pose_error(np.asarray(reference["T_base_tcp_mm"]),np.asarray(current["T_base_tcp_mm"]))
        options=self.session.cfg["park_guard"]
        report={"context":context,"side":current["side"],"translation_mm":translation,"rotation_deg":rotation,
                "passed":translation<=options["translation_mm"] and rotation<=options["rotation_deg"],
                "threshold_translation_mm":options["translation_mm"],"threshold_rotation_deg":options["rotation_deg"],
                "observation":current}
        if not report["passed"]:
            raise ParkGuardError("Parked "+current["side"]+" arm moved; restore the original parking pose with move-tcp before retrying. Reference was not updated.",report)
        return report


def image_pose_metadata(snapshot,session,*,camera_stamp_ns=None):
    """Associate both measured TCPs with B; never claim camera-frame ground truth."""
    stamp=int(snapshot["timestamp_ns"] if camera_stamp_ns is None else camera_stamp_ns)
    arms=snapshot["arms"]
    result={"B_base_mm":plain(session.meta["B"]),"B_definition":"T0 @ grasp_center_in_tcp",
            "image_host_monotonic_ns":stamp,"camera_pose_sync":"host-time association; not hardware exposure synchronization",
            "camera_roles":copy.deepcopy(session.meta.get("camera_roles",{})),"pose_sources":{}}
    age=session.cfg["recording"]["source_max_age_s"]
    for side in ("left","right"):
        source=arms[side];pose=validate_transform(source["T_base_tcp_mm"])
        receive=int(source["received_monotonic_ns"])
        skew=(receive-stamp)/1e9
        if abs(skew)>age:raise ParkGuardError("Image/TCP association too old for "+side)
        result["T_base_"+side+"TCP"]=pose.tolist()
        result["pose_sources"][side]={"received_monotonic_ns":receive,
            "source_ros_timestamp_s":source.get("source_ros_timestamp_s"),"image_to_pose_skew_s":skew,
            "tcp_frame":source.get("tcp_frame"),"base_frame":source.get("base_frame")}
    return result

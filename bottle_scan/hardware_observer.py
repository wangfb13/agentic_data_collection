"""Additional public ROS observations. No command publishers or service calls."""
from __future__ import annotations
import math
import threading
import time


class RosJointObserver:
    """Read named joint positions, preserving source/receipt times and radians."""
    def __init__(self,cfg,*,rospy_module=None,message_type=None,clock=time.monotonic_ns):
        options=cfg["park_guard"]
        self.topic=options["joint_topic"]
        self.names=options["joint_names"]
        self.max_age_s=cfg["recording"]["source_max_age_s"]
        self._ros=rospy_module;self._type=message_type;self.clock=clock
        self._subscriber=None;self._latest=None;self._lock=threading.Lock()
        if (not isinstance(self.topic,str) or not self.topic.startswith("/") or set(self.names)!={"left","right"}
                or any(not isinstance(v,list) or len(v)!=7 or len(set(v))!=7 or not all(isinstance(x,str) and x for x in v)
                       for v in self.names.values())):
            raise ValueError("Explicit joint-state topic and seven distinct names per arm required")

    def connect(self):
        if self._subscriber is not None:return self
        if self._ros is None:
            import rospy
            self._ros=rospy
        if self._type is None:
            from sensor_msgs.msg import JointState
            self._type=JointState
        if not self._ros.core.is_initialized():self._ros.init_node("bottle_passive_joint_observer",anonymous=True,disable_signals=True)
        self._subscriber=self._ros.Subscriber(self.topic,self._type,self._receive,queue_size=10)
        return self

    def _receive(self,message):
        stamp=self.clock()
        with self._lock:self._latest=(message,stamp)

    def snapshot(self):
        with self._lock:latest=self._latest
        if latest is None:raise RuntimeError("No passive joint-state feedback")
        message,received=latest
        if not 0<=(self.clock()-received)/1e9<=self.max_age_s:raise RuntimeError("Stale passive joint-state receipt")
        source=float(message.header.stamp.to_sec())
        now=float(self._ros.Time.now().to_sec())
        if not math.isfinite(source) or source<=0 or not -self.max_age_s<=now-source<=self.max_age_s:
            raise RuntimeError("Stale/invalid passive joint-state source timestamp")
        if len(message.name)!=len(message.position) or len(set(message.name))!=len(message.name):
            raise RuntimeError("Malformed/duplicate joint-state names")
        mapping=dict(zip(message.name,message.position))
        arms={}
        for side,names in self.names.items():
            if any(name not in mapping for name in names):raise RuntimeError("Joint-state missing "+side+" arm names")
            values=[float(mapping[name]) for name in names]
            if not all(math.isfinite(x) for x in values):raise RuntimeError("Nonfinite passive joint positions")
            arms[side]={"names":list(names),"position_rad":values}
        return {"arms":arms,"topic":self.topic,"received_monotonic_ns":received,
                "source_ros_timestamp_s":source,"source_sequence":int(message.header.seq),
                "unit":"rad","read_only":True,"timestamp_kind":"joint_state_header_and_host_receive"}

    def close(self):
        if self._subscriber is not None:self._subscriber.unregister();self._subscriber=None


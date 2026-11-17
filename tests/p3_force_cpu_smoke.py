"""Container-only CPU smoke with actual consumer weights and in-memory finger/TCP I/O."""
import tempfile
import threading
import time
from pathlib import Path
import numpy as np
from bottle_scan.state_graph import campaign
from skills.contact_position_a import p3_force


class SimulatedHardware:
    def __init__(self):
        self.grip=self
        self.control_lock=threading.RLock()
        self.mode='hold'
        self.armed=True
        self.overforce=False
        self.hold_drop_seen=False
        self.opening=.367
        self.T=np.eye(4)
        self.stage='probe'
        self.moves=[]
        self.cfg={'force':{'normal_axis':2},'lower':dict(unload_abs_sdk=.5,unload_ratio=.2,frames=3,min_descent_mm=1.,mapping_min_cosine=.7),
                  'motion':dict(near_speed_mm_s=20.,free_speed_mm_s=20.,max_lift_mm=60.,pose_tolerance_mm=2.)}

    def force_sample(self):
        stamp=time.monotonic_ns()//20_000_000*20_000_000
        force=max(.12,.7+(.367-self.opening)*120)
        return dict(sequence=[stamp,stamp],stamps=[stamp,stamp],wrench_zeroed=[[.3,0.,force,0.,0.,0.]]*2,
                    normal=[force,force],contact_thresholds=[.1,.1])

    def fresh(self):return {'normalized':self.opening}
    def _publish(self,command,kind):self.opening=command
    def check(self):pass
    def hold(self):pass
    def set_context(self,s):self.stage=s
    def ownership(self):pass
    def begin_record(self,p):pass
    def end_record(self):pass
    def wait(self,t):time.sleep(t)
    def pose(self):return self.T.copy()
    def move(self,target,speed):
        self.T=target.copy()
        self.moves.append(self.T[2,3])
    def motion_done(self):return True
    def stop_arm(self):pass


def main():
    h=SimulatedHardware()
    path=campaign.read('config.state_graph.campaign_v4.json')['execution']['environment']['RIGHT_FORCE_POLICY_PATH']
    expected=p3_force.identity(path)
    predictor,controller=p3_force.load(expected)
    worker=p3_force.sampling_worker(h,expected,predictor)
    with tempfile.TemporaryDirectory(prefix='offline_p3_force_') as d:
        try:
            worker.start()
            responses=iter(['TABLE','y'])
            o=p3_force.consume(h,expected,Path(d),controller,worker,ask=lambda _:next(responses))
            assert o['L1'] and o['L2'] and h.moves==[40.,0.], (o,h.moves)
            print('SIMULATED I/O + physical deployment checkpoint',expected['version'],
                  'L1',o['L1'],'L2',o['L2'],'TCP',h.moves,'hold_s',o['measured_hold_s'])
        finally:
            worker.stop()


if __name__=='__main__':
    main()


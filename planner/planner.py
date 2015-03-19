import numpy as np
from com_planner import COMPlanner
from foot_planner import FootPlanner
from init_pose_solver import InitPoseSolver


def to3D(v, x):
    return np.concatenate((v, [x]))


class Planner(object):
    def __init__(self, skel, ref):
        self.skel = skel
        self.ref = ref

    def solve(self):
        self.com = COMPlanner(self.skel, self.ref)
        self.com.solve()

        self.foot = FootPlanner(self.skel, self.ref)

        q0 = self.ref.pose_at(0, self.skel.id)
        x0 = np.concatenate((q0, np.zeros(self.skel.ndofs)))
        C0 = to3D(self.com.solution['C'][0], 0.0)
        dC0 = to3D(self.com.solution['dC'][0], 0.0)
        LF0 = self.foot.toe_at_frame(0)
        self.init_pose = InitPoseSolver(self.skel, x0, C0, dC0, LF0)
        self.init_pose.solve()

    def render(self):
        self.com.render()
        self.foot.render()

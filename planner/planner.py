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
        # 1. Build a COM trajectory
        self.com = COMPlanner(self.skel, self.ref)
        self.com.solve()

        # 2. Build a swing foot trajectory
        self.foot = FootPlanner(self.skel, self.ref)
        self.foot.solve()

        # 3. Shift the trajectory
        delta_x = -0.05
        self.com.shift(x=delta_x)
        self.foot.shift(x=delta_x)

        # 4. Solve the initial pose
        q0 = self.ref.pose_at(0, self.skel.id)
        x0 = np.concatenate((q0, np.zeros(self.skel.ndofs)))
        C0 = to3D(self.com.solution['C'][0], 0.0)
        dC0 = to3D(self.com.solution['dC'][0], 0.0)
        LF0 = self.foot.toe_at_frame(0)
        RF0 = [0.267 + delta_x, -0.741, 0.090]
        self.init_pose = InitPoseSolver(self.skel, x0, C0, dC0, LF0, RF0)
        self.init_pose.solve()

    def render(self):
        self.com.render()
        self.foot.render()

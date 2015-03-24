import numpy as np
from com_planner import COMPlanner
from foot_planner import FootPlanner
from init_pose_solver import InitPoseSolver
from motion_solver import MotionSolver


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
        self.foot.set_params([0.0, 0.10, 0.0, 0.10])
        self.foot.solve()

        print '# ref frames:', self.ref.num_frames
        print '# com frames:', self.com.num_frames()
        print '# foot frames:', self.foot.num_frames()
        assert (self.com.num_frames() == self.foot.num_frames())

        # 3. Shift the trajectory
        delta_x = -0.06
        self.com.shift(x=delta_x)
        self.foot.shift(x=delta_x)

        # 4. Solve the initial pose
        q0 = self.ref.pose_at(0, self.skel.id)
        x0 = np.concatenate((q0, np.zeros(self.skel.ndofs)))
        C0 = self.com.solution['C3D'][0]
        dC0 = self.com.solution['dC3D'][0]
        LF0 = self.foot.toe_at_frame(0)
        RF0 = [[0.27 + delta_x, -0.74, 0.09],
               [0.51 + delta_x, -0.74, 0.09]]
        self.init_pose = InitPoseSolver(self.skel, x0, C0, dC0, LF0, RF0[0])
        self.init_pose.solve()

        # 5. Solve the entire pose
        self.motion = MotionSolver(self.skel, self.ref,
                                   self.com, self.foot, RF0)
        # self.motion.solve(self.init_pose.solution_q0)

    def render(self):
        self.com.render()
        self.foot.render()
        self.motion.render()

    def init_state(self):
        return self.init_pose.solution

    def pose_at(self, index):
        return self.motion.pose_at(index)

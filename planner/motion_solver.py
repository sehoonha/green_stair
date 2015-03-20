import logging
import gltools
import numpy as np
from numpy.linalg import norm
import scipy.optimize


class MotionSolver(object):
    def __init__(self, skel, ref, com_planner, foot_planner, rfoot):
        self.logger = logging.getLogger(__name__)
        self.skel = skel
        self.ref = ref
        self.com_planner = com_planner
        self.foot_planner = foot_planner
        self.rfoot = rfoot

        self.render_frame = 0

    def nframes(self):
        return self.ref.num_frames

    def com_at(self, frame):
        return self.com_planner.solution['C3D'][frame]

    def lfoot_at(self, frame):
        return self.foot_planner.toe_at_frame(frame)

    def cost(self, x):
        self.skel.q = x
        C = self.skel.C
        lf = self.skel.body('h_toe_left').to_world([0.13, -0.05, 0.0])
        rf = self.skel.body('h_heel_right').to_world([-0.03, -0.05, 0])
        rf2 = self.skel.body('h_toe_right').to_world([0.13, -0.05, 0])

        Chat = self.com_at(self.solver_frame)
        LFhat = self.lfoot_at(self.solver_frame)
        RFhat = self.rfoot

        v = []
        v.append(norm(C - Chat) ** 2)
        v.append(norm(lf - LFhat) ** 2)
        v.append(norm(rf - RFhat[0]) ** 2)
        v.append(norm(rf2 - RFhat[1]) ** 2)
        value = np.array(v).dot(np.array([2.0, 1.0, 1.0, 1.0]))
        return value

    def solve_frame(self, index):
        self.solver_frame = index
        logger = self.logger
        x0 = self.ref.pose_at(index, self.skel.id)
        options = {'maxiter': 100000, 'ftol': 10e-10}
        res = scipy.optimize.minimize(self.cost, x0,
                                      method='SLSQP',
                                      options=options)
        logger.info('Frame %d: %.6f' % (index, self.cost(res.x)))
        return res.x

    def solve(self):
        self.motions = []
        for i in range(0, self.nframes(), 10):
            q_i = self.solve_frame(i)
            self.motions.append(q_i)

    def pose_at(self, frame_index):
        index = min(frame_index / 10, len(self.motions) - 1)
        return self.motions[index]

    def render(self):
        # self.render_frame = (self.render_frame + 10) % self.nframes()
        C = self.com_at(self.render_frame)
        LF = self.lfoot_at(self.render_frame)
        RF = self.rfoot
        gltools.render_trajectory([LF, C] + RF, [1.0, 0.0, 1.0])

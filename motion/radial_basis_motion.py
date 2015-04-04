import numpy as np
import math
import logging
from pydart import SkelVector
from parameterized_motion import ParameterizedMotion


class RadialBasisDof(object):
    def __init__(self, skel, dofs, w0=0.0, s0=0.1, t0=0.0):
        self.skel = skel
        self.dofs = dofs

        self.w0 = w0
        self.s0 = s0
        self.t0 = t0
        self.set_params(np.zeros(self.num_params()))
        self.step_index = -1

    def num_params(self):
        return 3

    def params(self):
        return np.array([self.w, self.s, self.t])

    def set_params(self, params):
        self.w = params[0]
        self.s = params[1]
        self.t = params[2]

    def eval(self, x):
        vec = SkelVector(skel=self.skel)
        vec[self.dofs] = 1.0

        w = self.w0 + self.w
        s = max(self.s0 + self.s, 0.01)
        t = self.t0 + self.t
        value = w * math.exp(-(1.0) / (2.0 * s * s) * (x - t) * (x - t))
        return value * vec


class RadialBasisMotion(ParameterizedMotion):
    def __init__(self, skel, ref, stair):
        super(RadialBasisMotion, self).__init__(skel, ref)
        self.logger = logging.getLogger(__name__)
        self.set_stair_info(stair)

        self.basis = []
        T = self.step_duration
        self.num_steps = 2
        num_steps = self.num_steps
        for i, H in enumerate(np.arange(0.0, T * (num_steps - 1) + 1e-5, T)):
            self.logger.info('add a set of basis for %d, %f' % (i, H))
            swing = 'left' if i % 2 == 0 else 'right'
            stance = 'right' if i % 2 == 0 else 'left'

            self.current_step = i

            # Swing leg
            self.add('j_shin_%s' % swing, -0.2, 0.1, H + 0.1, i)
            self.add('j_heel_%s_1' % swing, -0.2, 0.1, H + 0.1, i)
            self.add('j_thigh_%s_z' % swing, 0.0, 0.1, H + 0.56, i)
            self.add('j_shin_%s' % swing, 0.0, 0.1, H + 0.56, i)
            self.add('j_heel_%s_1' % swing, 0.0, 0.1, H + 0.56, i)

            # Stance leg
            self.add('j_shin_%s' % stance, 0.0, 0.1, H + 0.4, i)
            self.add('j_heel_%s_1' % stance, 0.0, 0.1, H + 0.6, i)

            # Balance
            self.add(('j_thigh_left_z', 'j_thigh_right_z'),
                     0.0, 0.2, H + 0.7, i)
            self.add(('j_thigh_%s_x' % stance), 0.0, 0.2, H + 0.4, i)

        self.logger.info('num params = %d' % self.num_params())

        # The first step
        x0 = [-0.0672191468377, -0.122859391369, -1.1031810238, -0.21210558256,
              -0.539546331848, 1.16622926629, -0.0535456005807, -0.693885409161,
              0.2144731381, -0.107783384896, -0.422426054874, 0.173536449941,
              0.342376540197, 0.553799216004, 1.36186028791, -0.294479149823,
              1.12904136918, -0.278965313751, -0.917548460234, -0.355557197286,
              -0.260391964555, 0.837186910621, -0.69438000397, 0.3698845051,
              0.0, 0.0, 0.0]
        x1 = [-0.00864127397088, -1.09886253149, 0.100563283461, 0.388348151271,
              -0.565839959958, 0.295652379077, 0.146206930372, -0.889133698506,
              0.0796791625689, -0.356224078557, -0.396099606848, -0.759045493803,
              -0.533283136842, -0.700774364026, -0.423634517524, -0.572995237876,
              0.462607824585, 1.1621945558, -0.454856880038, -0.326646601424,
              -0.282473630795, 0.0690317261902, 0.447993573256, -0.00913258004654,
              0.161057576834, 0.42793261044, 0.286690331098]
        # Normal stair
        x = [-0.04428199, -0.39795869, -1.16566691, 0.18960578, -1.0356194,
             1.24110003, 0.18127525, -0.36398466, 0.29674578, -0.58550015,
             -0.2357539, 0.55711293, 0.22790107, 0.51672396, 0.84291321,
             -0.35048593, 0.69452953, -0.82191132, -1.05813989, -0.47442107,
             -0.31192093, 0.43742374, -0.57616262, 0.63872899, -0.34321954,
             -0.16181328, -0.1158621, 0.00260821, -0.85506207, -0.05502631,
             0.40266842, -0.40559741, 0.6228111, 0.35385019, -1.00988068,
             -0.16541781, -0.44459508, -0.40761518, -0.69360236, -0.74002868,
             -0.68004935, -0.22928733, -0.78569774, 0.43305855, 1.08351675,
             -0.37316346, -0.31019323, -0.37988893, 0.08943427, 0.51211106,
             -0.10002729, -0.07475386, 0.35289616, 0.23554987]

        # self.params = np.zeros(self.num_params())
        # self.set_params(x0, 0)
        # self.set_params(x1, 1)
        self.set_params(x)

    def add(self, dof, w0, s0, t0, step=-1):
        b = RadialBasisDof(self.skel, dof, w0=w0, s0=s0, t0=t0)
        b.step_index = step
        self.basis.append(b)

    def set_stair_info(self, stair):
        self.step_duration = stair.step_duration

    def basis_at_step(self, step_index):
        return [b for b in self.basis if b.step_index == step_index]

    def num_params(self, step_index=-1):
        if step_index == -1:
            return sum([self.num_params(s) for s in range(self.num_steps)])
        m = self.basis[0].num_params()
        basis = self.basis_at_step(step_index)
        return len(basis) * m

    def accumulated_num_params(self, step_index=-1):
        return sum([self.num_params(s) for s in range(step_index)])

    def params_at_step(self, step_index):
        lo = self.accumulated_num_params(step_index)
        hi = self.accumulated_num_params(step_index + 1)
        return self.params[lo:hi]

    def set_params(self, params, step_index=None):
        if step_index is not None:
            lo = self.accumulated_num_params(step_index)
            hi = self.accumulated_num_params(step_index + 1)
            self.params[lo:hi] = params
            basis = self.basis_at_step(step_index)
        else:
            basis = self.basis
            self.params = params
        m = basis[0].num_params()
        for i, b in enumerate(basis):
            lo = m * i
            hi = m * (i + 1)
            b.set_params(params[lo:hi])

    def parameterized_pose_at_frame(self, frame_index):
        q = self.pose_at_frame(frame_index, isRef=True)
        t = float(frame_index) * self.h

        for b in self.basis:
            q += b.eval(t)
        return q

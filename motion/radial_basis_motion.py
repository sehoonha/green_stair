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
        self.fixed_basis = []
        T = self.step_duration
        for i, H in enumerate(np.arange(0.0, T * 2 + 0.0001, T)):
            self.logger.info('add a set of basis for %d, %f' % (i, H))
            swing = 'left' if i % 2 == 0 else 'right'
            stance = 'right' if i % 2 == 0 else 'left'

            # Swing leg
            self.add('j_shin_%s' % swing, -0.2, 0.1, H + 0.1)
            self.add('j_heel_%s_1' % swing, -0.2, 0.1, H + 0.1)
            self.add('j_thigh_%s_z' % swing, 0.0, 0.1, H + 0.56)
            self.add('j_shin_%s' % swing, 0.0, 0.1, H + 0.56)
            self.add('j_heel_%s_1' % swing, 0.0, 0.1, H + 0.56)

            # Stance leg
            self.add('j_shin_%s' % stance, 0.0, 0.1, H + 0.4)
            self.add('j_heel_%s_1' % stance, 0.0, 0.1, H + 0.6)

            # Balance
            self.add(('j_thigh_left_z', 'j_thigh_right_z'), 0.0, 0.2, H + 0.7)
            if i == 1:
                self.fixed_basis = self.basis
                self.basis = []

        self.logger.info('num params = %d' % self.num_params())

        # The first step
        x0 = [-0.0672191468377, -0.122859391369, -1.1031810238, -0.21210558256,
              -0.539546331848, 1.16622926629, -0.0535456005807, -0.693885409161,
              0.2144731381, -0.107783384896, -0.422426054874, 0.173536449941,
              0.342376540197, 0.553799216004, 1.36186028791, -0.294479149823,
              1.12904136918, -0.278965313751, -0.917548460234, -0.355557197286,
              -0.260391964555, 0.837186910621, -0.69438000397, 0.3698845051]
        x1 = [1.07608044082, -0.910098092922, 0.590878536888, -0.724897980105,
             -0.531144514471, 0.820173746843, 1.22861780446, -0.936173898349,
             -0.045251974521, 0.593708760878, -0.876838384345, 0.471510712002,
             0.679889799726, -0.500891848342, -0.781161678601, -0.159312563193,
             -0.133887341204, 0.53567436901, 0.238689335532, -0.2568289899,
             1.47963881916, -0.0430946286444, -0.1936184422, 0.945678493312]
        self.set_fixed_params(x0 + x1)

        # Default parameter
        x = np.zeros(self.num_params())

        # x[:5] = [-0.2, -0.2, 0.28, 0.0, 0.1]
        # x[5:8] = [0.0, 0.0, -0.4]
        # x[8:16] = x[0:8]
        # x[16:24] = x[0:8]
        self.set_params(x)

    def add(self, dof, w0, s0, t0):
        b = RadialBasisDof(self.skel, dof, w0=w0, s0=s0, t0=t0)
        self.basis.append(b)

    def set_stair_info(self, stair):
        self.step_duration = stair.step_duration

    def num_params(self):
        m = self.basis[0].num_params()
        return len(self.basis) * m

    def set_params(self, params):
        self.params = params
        m = self.basis[0].num_params()
        for i, b in enumerate(self.basis):
            lo = m * i
            hi = m * (i + 1)
            b.set_params(params[lo:hi])

    def set_fixed_params(self, params):
        self.fixed_params = params
        m = self.fixed_basis[0].num_params()
        for i, b in enumerate(self.fixed_basis):
            lo = m * i
            hi = m * (i + 1)
            b.set_params(params[lo:hi])

    def parameterized_pose_at_frame(self, frame_index):
        q = self.pose_at_frame(frame_index, isRef=True)
        t = float(frame_index) * self.h

        for b in self.basis + self.fixed_basis:
            q += b.eval(t)
        return q

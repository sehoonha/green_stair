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
        self.num_steps = 3
        num_steps = self.num_steps
        for i, H in enumerate(np.arange(0.0, T * (num_steps - 1) + 1e-5, T)):
            # H += 0.2
            self.logger.info('add a set of basis for %d, %f' % (i, H))
            swing = 'left' if i % 2 == 0 else 'right'
            stance = 'right' if i % 2 == 0 else 'left'

            self.current_step = i

            # Swing leg
            self.add('j_thigh_%s_z' % swing, 0.0, 0.1, H + 0.0, i)
            self.add('j_shin_%s' % swing, -0.2, 0.1, H + 0.1, i)
            self.add('j_heel_%s_1' % swing, -0.2, 0.1, H + 0.1, i)
            self.add('j_thigh_%s_z' % swing, 0.0, 0.1, H + 0.56, i)
            self.add('j_shin_%s' % swing, 0.0, 0.1, H + 0.56, i)
            self.add('j_heel_%s_1' % swing, 0.0, 0.1, H + 0.56, i)

            # Stance leg
            # self.add('j_shin_%s' % stance, 0.0, 0.1, H + 0.0, i)
            self.add('j_shin_%s' % stance, 0.0, 0.1, H + 0.4, i)
            self.add('j_heel_%s_1' % stance, 0.0, 0.1, H + 0.6, i)

            # Balance
            self.add(('j_thigh_left_z', 'j_thigh_right_z'),
                     0.0, 0.2, H + 0.7, i)
            self.add(('j_thigh_%s_x' % stance), 0.0, 0.2, H + 0.4, i)
            # self.add(('j_thigh_%s_x' % stance), 0.0, 0.2, H + 0.0, i)

        self.logger.info('num params = %d' % self.num_params())

        # Normal stair
        # x0 = [-0.13575228, -1.23037064, 0.3248233, 0.05172304, -0.20124602,
        #       0.163591, 0.08896968, -0.57222897, 0.07211617, -0.084136,
        #       -0.16761146, 0.45039613, -0.01256396, -0.16700529, -0.06607622,
        #       0.22568788, 0.1357137, -0.14131936, -0.34955953, -0.42422587,
        #       0.73792703, 0.87185791, -0.4021209, 0.0179604, 0.18064052,
        #       -0.31485322, 0.4256871, -0.51300866, 0.50080178, -0.72331661,
        #       0.24086195, -1.12743003, -1.1460612, -0.36523056, -1.5857851,
        #       -0.78694066, 0.31047676, -0.90466361, 0.29099397, 0.13049505,
        #       0.29426282, 0.49431881, -0.89230081, -0.67962224, 0.56997248,
        #       0.02782986, 0.12547225, 0.45138546, -0.48114401, -0.49110618,
        #       -0.25763609, 0.20675982, 0.32819063, 0.5467555, -0.16351248,
        #       -0.04469256, 0.61370465, -0.08121555, -0.55214494, -0.47528739]
        x0 = np.zeros(self.num_params())
        self.params = x0
        self.set_params(x0)

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

    def params_at_step(self, step_index=-1):
        if step_index == -1:
            return self.params
        lo = self.accumulated_num_params(step_index)
        hi = self.accumulated_num_params(step_index + 1)
        return self.params[lo:hi]

    def set_params(self, params, step_index=-1):
        if step_index != -1:
            lo = self.accumulated_num_params(step_index)
            hi = self.accumulated_num_params(step_index + 1)
            self.params[lo:hi] = params
            basis = self.basis_at_step(step_index)
        else:
            basis = self.basis
            n = len(self.params)
            m = len(params)
            self.params[:m] = params
            if m < n:
                self.params[m:n] = params[n - m:]
        m = basis[0].num_params()
        for i, b in enumerate(basis):
            lo = m * i
            hi = m * (i + 1)
            if lo >= len(params):
                continue
            b.set_params(params[lo:hi])

    def parameterized_pose_at_frame(self, frame_index):
        q = self.pose_at_frame(frame_index, isRef=True)
        t = float(frame_index) * self.h

        for b in self.basis:
            q += b.eval(t)
        return q

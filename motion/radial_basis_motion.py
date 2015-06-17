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
        self.time_duration = (0, 100.0)

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
        (lo, hi) = self.time_duration
        # if x < lo or hi < x:
        if x < lo:
            return 0.0

        w = self.w0 + self.w
        s = max(self.s0 + self.s, 0.1)
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
            if i == 0:
                self.time_duration = (0.0, H + T)
            else:
                self.time_duration = (H, H + T)
            self.logger.info('time duration = (%f, %f)' % self.time_duration)

            # Swing leg
            self.add('j_thigh_%s_z' % swing, 0.0, 0.1, H + 0.0, i)
            self.add('j_shin_%s' % swing, -0.2, 0.1, H + 0.1, i)
            self.add('j_heel_%s_1' % swing, -0.2, 0.1, H + 0.1, i)
            self.add('j_thigh_%s_z' % swing, 0.0, 0.1, H + 0.56, i)
            self.add('j_shin_%s' % swing, 0.0, 0.1, H + 0.56, i)
            self.add('j_heel_%s_1' % swing, 0.0, 0.1, H + 0.56, i)

            # Stance leg
            self.add('j_shin_%s' % stance, 0.0, 0.1, H + 0.0, i)
            self.add('j_shin_%s' % stance, 0.0, 0.1, H + 0.4, i)
            self.add('j_heel_%s_1' % stance, 0.0, 0.1, H + 0.6, i)

            # Balance
            self.add(('j_thigh_left_z', 'j_thigh_right_z'),
                     0.0, 0.2, H + 0.7, i)
            self.add(('j_thigh_%s_x' % stance), 0.0, 0.2, H + 0.4, i)
            self.add(('j_thigh_%s_x' % stance), 0.0, 0.2, H + 0.0, i)

        self.logger.info('num params = %d' % self.num_params())

        self.logger.info('dim = %d' % self.num_params())
        x0 = np.zeros(self.num_params())
        # x0 = np.random.rand(self.num_params())
        self.params = x0
        self.set_params(x0)

    def add(self, dof, w0, s0, t0, step=-1):
        b = RadialBasisDof(self.skel, dof, w0=w0, s0=s0, t0=t0)
        b.step_index = step
        b.time_duration = self.time_duration
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
        stay = 0
        if frame_index < stay:
            t = float(frame_index) * self.h
            q = self.pose_at_frame(0, isRef=True)
            q = SkelVector(q, skel=self.skel)

            b0 = RadialBasisDof(self.skel, 'j_heel_right_1',
                                w0=-0.4, s0=0.1, t0=0.0)
            for b in [b0]:
                q += b.eval(t)

            return q
        else:
            frame_index -= stay
            q = self.pose_at_frame(frame_index, isRef=True)
            t = float(frame_index) * self.h
            for b in self.basis:
                q += b.eval(t)
            return q

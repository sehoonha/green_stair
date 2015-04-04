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
        num_steps = 3
        for i, H in enumerate(np.arange(0.0, T * (num_steps - 1) + 1e-5, T)):
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
            self.add(('j_thigh_%s_x' % stance), 0.0, 0.2, H + 0.4)
            if i == num_steps - 2:
                self.fixed_basis = self.basis
                self.basis = []

        self.logger.info('num params = %d' % self.num_params())

        # The first step
        x0 = [-0.0672191468377, -0.122859391369, -1.1031810238, -0.21210558256,
              -0.539546331848, 1.16622926629, -0.0535456005807, -0.693885409161,
              0.2144731381, -0.107783384896, -0.422426054874, 0.173536449941,
              0.342376540197, 0.553799216004, 1.36186028791, -0.294479149823,
              1.12904136918, -0.278965313751, -0.917548460234, -0.355557197286,
              -0.260391964555, 0.837186910621, -0.69438000397, 0.3698845051,
              0.0, 0.0, 0.0]
        # x1 = [0.1107635, -0.58207559, 0.10293539, 0.22393839, -0.08530127,
        #       0.30843779, 0.13334581, -0.3077393, 0.32339919, -0.36242648,
        #       -0.46411873, -0.24853438, -0.51053262, -0.45206713, -0.15476215,
        #       -0.39470619, 0.4301999, 0.73164228, -0.14601511, -0.13841301,
        #       0.43436786, 0.0969217, 0.3206579, -0.26200865, 0.19463631,
        #       0.21383287, 0.00741732]
        x1 = [0.17340690245776783, -0.44854044303117019, 0.35329865291636675, 0.22150244518813275, -0.51043011160061491, 0.21057739724230126, -0.026018931088830473, -0.7274952577466528, 0.15619083777437776, -0.61133528083292354, -0.30006738661674992, -0.3921390899507235, -0.53382263164326993, -0.51710308594222121, -0.41820166567603489, -0.41702037341909237, 0.45585812842758799, 0.85807429092532794, -0.22532920141503221, -0.12051417667420183, 0.42297036618733852, 0.087071908725300379, 0.26363861585199511, 0.055665928878806045, 0.13029185983854008, 0.077369174767526014, 0.10187778568827305]
        # X1 = [1.07608044082, -0.910098092922, 0.590878536888, -0.724897980105,
        #      -0.531144514471, 0.820173746843, 1.22861780446, -0.936173898349,
        #      -0.045251974521, 0.593708760878, -0.876838384345, 0.471510712002,
        #      0.679889799726, -0.500891848342, -0.781161678601, -0.159312563193,
        #      -0.133887341204, 0.53567436901, 0.238689335532, -0.2568289899,
        #      1.47963881916, -0.0430946286444, -0.1936184422, 0.945678493312]
        x1 = [-0.00864127397088, -1.09886253149, 0.100563283461, 0.388348151271,
              -0.565839959958, 0.295652379077, 0.146206930372, -0.889133698506,
              0.0796791625689, -0.356224078557, -0.396099606848, -0.759045493803,
              -0.533283136842, -0.700774364026, -0.423634517524, -0.572995237876,
              0.462607824585, 1.1621945558, -0.454856880038, -0.326646601424,
              -0.282473630795, 0.0690317261902, 0.447993573256, -0.00913258004654,
              0.161057576834, 0.42793261044, 0.286690331098]
        self.set_fixed_params(x0 + x1)

        # x0 = [-0.24440991, 0.65962406, -0.43324682, -0.00706849, -0.72066726,
        #       -0.41449409, 0.24986759, 0.08870128, 0.21190674, 0.36893778,
        #       -0.2687791, -0.52962706, -0.54285017, -0.89235126, -0.24816659,
        #       -0.39064749, 0.64092972, -1.04927789, 0.46531869, -0.81279681,
        #       0.14037567, 0.46877291, 0.15319342, 1.00340705, -0.4746195,
        #       -0.95575141, 0.1660687 ]
        # self.set_fixed_params(x0)

        # Default parameter
        x = np.zeros(self.num_params())

        # x[:5] = [-0.2, -0.2, 0.28, 0.0, 0.1]
        # x[5:8] = [0.0, 0.0, -0.4]
        # x[8:16] = x[0:8]
        # x[16:24] = x[0:8]
        self.set_params(x1)

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

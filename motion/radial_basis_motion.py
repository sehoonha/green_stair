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

        # Normal stair
        # x = [-0.04428199, -0.39795869, -1.16566691, 0.18960578, -1.0356194,
        #      1.24110003, 0.18127525, -0.36398466, 0.29674578, -0.58550015,
        #      -0.2357539, 0.55711293, 0.22790107, 0.51672396, 0.84291321,
        #      -0.35048593, 0.69452953, -0.82191132, -1.05813989, -0.47442107,
        #      -0.31192093, 0.43742374, -0.57616262, 0.63872899, -0.34321954,
        #      -0.16181328, -0.1158621, 0.00260821, -0.85506207, -0.05502631,
        #      0.40266842, -0.40559741, 0.6228111, 0.35385019, -1.00988068,
        #      -0.16541781, -0.44459508, -0.40761518, -0.69360236, -0.74002868,
        #      -0.68004935, -0.22928733, -0.78569774, 0.43305855, 1.08351675,
        #      -0.37316346, -0.31019323, -0.37988893, 0.08943427, 0.51211106,
        #      -0.10002729, -0.07475386, 0.35289616, 0.23554987]
        x = [-0.11429011780171319, -0.44721378361028086, -1.2676412062421372,
             0.16798102501789883, -1.1403948562708339, 1.2136311408949532,
             0.21961422095335636, -0.38461287652525322, 0.28512931139290554,
             -0.44010592770549584, -0.38946642968849232, 0.62535917761220861,
             0.065695027049031032, 0.38311730531990729, 0.83186798461906164,
             -0.24097296452890565, 0.64496431185863567, -0.5192216897009172,
             -1.0825104385925108, -0.52938678795912752, -0.41973940871248272,
             0.27105623272088225, -0.49595545881937586, 0.74175682504238938,
             -0.58436470929911366, -0.46065044854551213, 0.30383421149766898,
             0.078468013887232901, -1.1538281208337431, 0.013970350035493087,
             0.58619766595747347, -0.34528084831197092, 0.67041210672958484,
             0.25672394822575295, -0.9265674273504888, -0.16247053679558587,
             -0.61538881140944657, -0.34614862440782407, -0.85982218422084067,
             -1.1128341623009994, -0.87675672450898157, -0.46174524034442399,
             -0.3929034836675751, 0.38711144118850149, 1.1770297193596462,
             -0.43510372834714339, -0.12684448379629354, -0.36924301115183056,
             0.092615222258651195, 0.34200827547013612, -0.084586016228997266,
             0.10528094041233542, 0.29257705890936425, 0.3504977972086083]

        # Spring stair: 0.2s
        x = [-0.29036564636201589, -0.54382305925350904, -0.97673111171896188,
             0.42283960833265011, -1.2319339854281444, 0.91237743664550397,
             0.2621262263252489, -0.32370080583705191, 0.46639507082294018,
             -0.76444889504552116, -0.39234608422631234, 0.32456104375299244,
             0.13563980787303762, 0.71339637099797981, 0.56623588895047294,
             -0.17274013544509129, 0.81859541922773937, -0.5627659319097964,
             -1.0148510512115907, -0.55971518138168785, -0.46599235315612703,
             0.21745476858459678, -0.65074335040906217, 0.87454936363837177,
             -0.79525445615965928, -0.35767941791881452, -0.18565431175465122,
             -0.21441470473595156, -0.75521609161857883, -0.27940692624368851,
             0.61245517602069399, -0.52361045349348856, 0.602316650016126,
             0.50069732794705502, -1.0815380256738318, -0.11153070601685844,
             -0.48611465784111557, -0.80594609790317917, -0.66465506172396938,
             -0.99041064448737481, -0.37351240273642639, -0.10764436922953118,
             -0.78709403550373924, 0.30713300290701828, 0.76756824153842673,
             -0.37460629424924835, -0.15862427515751351, -0.30348348271393172,
             0.086624732013351025, 0.4432386666173439, 0.1346224833730871,
             0.26988397354808635, -0.11596319453419862, 0.42434456411315075]

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

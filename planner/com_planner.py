import math
import logging
import numpy as np
from numpy.linalg import norm
import scipy.optimize
from telescopic_inverted_pendulum import TelescopicInvertedPendulum as TIP
import gltools


class COMPlanner(object):
    def __init__(self, skel, ref):
        self.logger = logging.getLogger(__name__)
        self.skel = skel
        self.ref = ref

        self.m = self.skel.m
        self.h = 0.001
        self.T = 0.734
        self.F = np.array([0.38 - 0.05, -0.73])  # 2D projected stance foot

        # Extract from the first frame
        self.skel.q = ref.pose_at(0, self.skel.id)
        self.C0 = self.skel.C[:2] - self.F  # 2D projected COM

        # Extract from the last frame
        self.skel.q = ref.pose_at(-1, self.skel.id)
        self.C1 = self.skel.C[:2] - self.F  # 2D projected COM

        d = self.C0  # Initial displacement vector
        r0 = norm(d)
        th0 = math.atan2(d[0], d[1])  # CW from y axis
        self.params0 = np.array([th0, r0, 0.0, 0.0, r0, r0, r0])
        # self.cost(self.params0, True)

    def num_params(self):
        return 7

    def set_params(self, params):
        self.x0 = TIP.State(params[:4])
        self.r_t = np.concatenate((params[1:2], params[4:7]))

    def params(self):
        return np.concatenate((self.x0, self.r_t[1:]))

    def cost(self, params, verbose=False):
        self.set_params(params)
        tip = TIP(self.m)
        X = tip.simulate(self.x0, self.h, self.T,
                         rhat_func=self.length_at_time)
        C = [np.array([x.x, x.y]) for x in X]
        dC = [np.array([x.dx, x.dy]) for x in X]
        term0 = norm(C[0] - self.C0) ** 2
        term1 = norm(C[-1] - self.C1) ** 2
        w_p = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 1.0])
        term2 = norm((params - self.params0) * w_p) ** 2
        term3 = norm(dC[0]) ** 2
        cost = term0 + term1 + 0.01 * term2 + 0.01 * term3
        if verbose:
            logger = self.logger
            logger.info('---- COM planner ----')
            logger.info('cost: %.6f' % cost)
            logger.info('term0= %.6f: %s %s' % (term0, C[0], self.C0))
            logger.info('term1= %.6f: %s %s' % (term1, C[-1], self.C1))
            logger.info('term2= %.6f: %s %s' % (term2, params, self.params0))
            logger.info('term3= %.6f: %s' % (term3, dC[0]))
            logger.info('---------------------')
            # for x in X:
            #     logger.info('%.4f %.4f %s' % (x.x, x.y, str(x)))
            self.solution = dict()
            self.solution['X'] = X
            self.solution['C'] = [c + self.F for c in C]
            C3D = [np.concatenate((c, [0])) for c in self.solution['C']]
            self.solution['C3D'] = C3D
            self.solution['dC'] = dC
            dC3D = [np.concatenate((d, [0])) for d in self.solution['dC']]
            self.solution['dC3D'] = dC3D

        # print cost
        return cost

    def length_at_time(self, t):
        w = t / self.T
        w0 = (1 - w) * (1 - w) * (1 - w)
        w1 = 3.0 * (1 - w) * (1 - w) * w
        w2 = 3.0 * (1 - w) * w * w
        w3 = w * w * w
        return np.array([w0, w1, w2, w3]).dot(self.r_t)

    def num_frames(self):
        return len(self.solution['C'])

    def solve(self):
        self.counter = 0
        logger = self.logger
        # x0 = [10.0, 10.0]
        logger.info('x0 = %s' % self.params0)
        options = {'maxiter': 100000, 'ftol': 10e-10}
        logger.info('options = %s' % options)
        res = scipy.optimize.minimize(self.cost, self.params0,
                                      method='SLSQP',
                                      options=options)
        logger.info('result = %s' % res)
        logger.info('cost = %.6f' % (self.cost(res.x, True)))
        logger.info('finished to solve optimization')
        logger.info('OK')

    def shift(self, x=0.0, y=0.0):
        offset = np.array([x, y])
        for c in self.solution['C']:
            c += offset

    def render(self):
        C3D = self.solution['C3D']
        gltools.render_trajectory(C3D, [1.0, 0.0, 1.0])
        F3D = np.concatenate((self.F, [0]))
        gltools.render_point(F3D, [1.0, 0.0, 1.0])

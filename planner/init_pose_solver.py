import logging
import numpy as np
from numpy.linalg import norm
import scipy.optimize


class InitPoseSolver(object):
    def __init__(self, skel, x0, C0, dC0, LF0, RF0):
        self.logger = logging.getLogger(__name__)
        self.skel = skel
        self.x0 = x0
        self.q0 = x0[:self.skel.ndofs]
        self.C0 = C0
        self.dC0 = dC0
        self.LF0 = LF0
        self.RF0 = RF0

    def cost(self, x, verbose=False):
        self.skel.x = x
        q = self.skel.q
        C = self.skel.C
        dC = self.skel.Cdot
        lf = self.skel.body('h_toe_left').to_world([0.13, -0.05, 0.0])
        rf = self.skel.body('h_heel_right').to_world([-0.03, -0.05, 0])
        v = []
        v.append(norm(q - self.q0) ** 2)
        v.append(norm(C - self.C0) ** 2)
        v.append(norm(dC - self.dC0) ** 2)
        v.append(norm(lf - self.LF0) ** 2)
        v.append(norm(rf - self.RF0) ** 2)
        value = np.array(v).dot(np.array([0.1, 2.0, 1.0, 1.0, 1.0]))
        if verbose:
            logger = self.logger
            logger.info('---- InitPoseSolver ----')
            logger.info('cost: %.6f' % value)
            logger.info('q = %.6f: %s %s' % (v[0], q, self.q0))
            logger.info('C = %.6f: %s %s' % (v[1], C, self.C0))
            logger.info('dC = %.6f: %s %s' % (v[2], dC, self.dC0))
            logger.info('lf = %.6f: %s %s' % (v[3], lf, self.LF0))
            logger.info('rf = %.6f: %s %s' % (v[4], rf, self.RF0))
            logger.info('------------------------')
        return value

    def solve(self):
        logger = self.logger
        logger.info('x0 = %s' % self.x0)
        options = {'maxiter': 100000, 'ftol': 10e-10}
        logger.info('options = %s' % options)
        res = scipy.optimize.minimize(self.cost, self.x0,
                                      method='SLSQP',
                                      options=options)
        self.solution = res.x
        logger.info('result = %s' % res)
        logger.info('cost = %.6f' % (self.cost(res.x, True)))
        logger.info('finished to solve optimization')
        logger.info('OK')

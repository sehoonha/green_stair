import logging
import scipy.optimize


class MotionOptimizer(object):
    def __init__(self, motion, evaluator):
        self.logger = logging.getLogger(__name__)
        self.motion = motion
        self.evaluator = evaluator

    def obj(self, x):
        self.motion.set_params(x)
        cost = self.evaluator.cost()
        # self.logger.debug('params = %s' % x)
        self.logger.debug('cost = %.10f' % cost)
        return cost
        # return x[0] ** 2 + (x[1] - 1.3) ** 2

    def solve(self):
        logging.info('start to solve optimization')
        logger = self.logger
        x0 = self.motion.params()
        # x0 = [10.0, 10.0]
        logger.info('x0 = %s' % x0)
        options = {'maxiter': 100000, 'maxfev': 100000,
                   'xtol': 10e-8, 'ftol': 10e-8}
        logger.info('options = %s' % options)
        res = scipy.optimize.minimize(self.obj, x0,
                                      method='SLSQP',
                                      options=options)
        logger.info('result = %s' % res)
        logger.info('finished to solve optimization')
        logger.info('OK')

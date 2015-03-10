import logging
import scipy.optimize


class MotionOptimizer(object):
    def __init__(self, motion, evaluator):
        self.logger = logging.getLogger(__name__)
        self.motion = motion
        self.evaluator = evaluator

    def obj(self, x):
        self.counter += 1
        self.motion.set_params(x)
        cost = self.evaluator.cost()
        # self.logger.debug('params = %s' % x)
        if self.counter % 100 == 1:
            self.logger.debug('%d: cost = %.10f' % (self.counter, cost))
        return cost
        # return x[0] ** 2 + (x[1] - 1.3) ** 2

    def solve(self):
        self.counter = 0
        logging.info('start to solve optimization')
        logger = self.logger
        x0 = self.motion.params()
        # x0 = [10.0, 10.0]
        logger.info('x0 = %s' % x0)
        options = {'maxiter': 100000, 'maxfev': 100000,
                   'xtol': 10e-10, 'ftol': 10e-10}
        logger.info('options = %s' % options)
        res = scipy.optimize.minimize(self.obj, x0,
                                      method='SLSQP',
                                      options=options)
        logger.info('result = %s' % res)
        logger.info('finished to solve optimization')
        logger.info('OK')

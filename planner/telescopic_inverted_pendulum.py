import numpy as np
from math import sin, cos


class TelescopicInvertedPendulum(object):
    class State(np.ndarray):
        def __new__(cls, data):
            obj = np.asarray(data[:4]).view(cls)
            return obj

        def __init__(self, data):
            # Create properties
            for i, name in enumerate(['th', 'r', 'dth', 'dr']):
                def fget(self, index=i):
                    return self[index]

                def fset(self, value, index=i):
                    self[index] = value
                setattr(self.__class__, name, property(fget, fset))

        @property
        def x(self):
            return self.r * sin(self.th)

        @property
        def y(self):
            return self.r * cos(self.th)

        @property
        def dx(self):
            th, r, dth, dr = self
            return dr * sin(th) + r * cos(th) * dth

        @property
        def dy(self):
            th, r, dth, dr = self
            return dr * cos(th) - r * sin(th) * dth

        def __array_finalize__(self, obj):
            if obj is None:
                return

    def __init__(self, m):
        self.m = m
        self.g = 9.81

    def deriv(self, x, f):
        (th, r, dth, dr) = (x.th, x.r, x.dth, x.dr)
        (m, g) = (self.m, self.g)
        f_r = f[1]
        ddth = -(2 * dr * dth - g * sin(th)) / r
        ddr = f_r / m - (-r * dth * dth + g * cos(th))
        return np.array([dth, dr, ddth, ddr])

    def simulate(self, x0, dt, T, control=None, rhat_func=None):
        self.rhat_func = rhat_func
        X = list()
        x = x0
        X.append(x)
        for t in np.arange(0.0, T, dt):
            f = np.zeros(2)
            if control is not None:
                f = control(x, t)
            elif rhat_func is not None:
                f = self.control_length(x, t)
            dx = self.deriv(x, f)
            x = x + dt * dx
            X.append(x)
        return X

    def control_length(self, x, t):
        th, r, dth, dr = x.th, x.r, x.dth, x.dr
        m, h, g = 1.0, 0.01, 9.81
        r_hat = self.rhat_func(t)
        dr_hat = (r_hat - r) / h
        f_r = m / h * (dr_hat - dr) + m * (-r * dth * dth + g * cos(th))
        # f_r = m / h * (dr_hat - dr)
        # print 'force', f_r, 'r', r, r_hat, 'dr', dr, dr_hat
        return np.array([0.0, f_r])

if __name__ == '__main__':
    tip = TelescopicInvertedPendulum(1.0)
    x0 = TelescopicInvertedPendulum.State([0.5, 1.0, 0.0, 0.0])
    print 'State:', x0, x0.x, x0.y

    X = tip.simulate(x0, 0.01, 0.5, rhat_func=lambda t: 1.0 - 0.2 * t)

    for x in X:
        print x, x.x, x.y

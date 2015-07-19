import numpy as np
from scipy.optimize import minimize
from pydart import SkelVector
from numpy.linalg import norm
import sys


def mirror_pose(skel, q):
    ret = np.array(q)
    for i, dof in enumerate(skel.dofs):
        if 'left'in dof.name:
            j = skel.dof_index(dof.name.replace('left', 'right'))
        elif 'right'in dof.name:
            j = skel.dof_index(dof.name.replace('right', 'left'))
        else:
            j = i
        scale = 1.0
        if i > 6 and ('_x' in dof.name or '_y' in dof.name):
            scale = -1.0
        # print j, '<--', i, '(', scale, ')'
        ret[i] = scale * q[j]
    # exit(0)
    return ret


def ik_adjust(skel, q):
    saved_state = skel.x
    skel.q = q
    saved_q = skel.q
    Chat = skel.C + np.array([0.05, 0.0, 0.0])
    LFhat = skel.body('h_toe_left').C + np.array([0.0, 0.0, -0.03])
    RFhat = skel.body('h_toe_right').C + np.array([0.0, 0.0, 0.10])

    def x_to_q(x):
        q = np.array(saved_q)
        q[:12] += x[:12]
        q[14:16] += x[12:14]
        q[17:21] += x[14:18]
        return q

    def mycost(x):
        q = x_to_q(x)
        skel.q = q
        C = skel.C
        LF = skel.body('h_toe_left').C
        RF = skel.body('h_toe_right').C

        ret = 0.0
        ret += 0.1 * (norm(x) ** 2)
        ret += 0.5 * (norm(C - Chat) ** 2)
        ret += 0.5 * (norm(LF - LFhat) ** 2)
        ret += 0.5 * (norm(RF - RFhat) ** 2)
        return ret

    x0 = np.zeros(18)
    res = minimize(mycost, x0, method='SLSQP')
    new_q = x_to_q(res.x)
    print '.',
    sys.stdout.flush()
    skel.x = saved_state
    return np.array(new_q)

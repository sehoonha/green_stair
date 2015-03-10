from gltools import render_point, render_line
import math
import numpy as np
from numpy.linalg import norm


class ObjPt(object):
    def __init__(self, body, local, target):
        self.body = body
        self.local = local
        self.target = target
        self.time = 0.0

    def pos(self):
        return self.body.to_world(self.local)

    def cost(self):
        d = np.array(self.pos()) - np.array(self.target)
        return norm(d) ** 2

    def render(self):
        render_point(self.pos(), [0.0, 1.0, 0.0], 0.03)
        render_point(self.target, [0.0, 0.0, 1.0], 0.03)
        render_line(self.pos(), self.target, [0.0, 0.0, 0.0])

    def __str__(self):
        return 'ObjPt[%s, %s at %f] %s %s' % (self.body.name, self.local,
                                              self.time,
                                              self.pos(), self.target)


class MotionEvaluator(object):
    def __init__(self, skel, motion):
        self.skel = skel
        self.motion = motion

        self.objs = []
        lh = 'h_heel_left'
        lt = 'h_toe_left'
        rh = 'h_heel_right'
        rt = 'h_toe_right'

        self.add(0.0, lh, [-0.03, -0.05, 0], [-0.03, -0.95, -0.11])
        self.add(0.0, lt, [0.13, -0.05, 0.0], [0.22, -0.95, -0.11])
        # self.add(0.0, rh, [-0.03, -0.05, 0], [0.27, -0.74, 0.09])
        # self.add(0.0, rt, [0.13, -0.05, 0.0], [0.51, -0.74, 0.09])
        self.add(0.365, lh, [-0.03, -0.05, 0], [0.55, -0.53, -0.11])
        self.add(0.365, lt, [0.13, -0.05, 0], [0.80, -0.53, -0.11])
        for t in np.linspace(0.0, 0.365, 0.1):
            self.add(t, rh, [-0.03, -0.05, 0], [0.27, -0.74, 0.09])
            self.add(t, rt, [0.13, -0.05, 0.0], [0.51, -0.74, 0.09])

    def add(self, time, name, local, target):
        body = self.skel.body(name)
        o = ObjPt(body, local, target)
        o.time = time
        self.objs.append(o)

    def cost(self):
        t = -1.0
        ret = 0.0
        for obj in self.objs:
            if math.fabs(obj.time - t) > 1e-5:
                t = obj.time
                q_t = self.motion.pose_at(t)
                self.skel.q = q_t
                # print 'set_pose at', t
            ret += obj.cost()
            # print 'cost: ', obj.cost()
        return ret

    def render(self):
        for obj in self.objs:
            obj.render()

    def __str__(self):
        return '\n'.join([str(o) for o in self.objs])

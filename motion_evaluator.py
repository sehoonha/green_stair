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
        lhs = np.array(self.pos())
        rhs = np.array(self.target)
        d = lhs - rhs

        penalty = 0.0
        if lhs[1] < rhs[1]:
            penalty = (rhs[1] - lhs[1]) ** 2

        return norm(d) ** 2 + penalty

    def render(self):
        render_point(self.pos(), [0.0, 1.0, 0.0], 0.03)
        render_point(self.target, [0.0, 0.0, 1.0], 0.03)
        render_line(self.pos(), self.target, [0.0, 0.0, 0.0])

    def __str__(self):
        return 'ObjPt[%s, %s at %f] %s %s' % (self.body.name, self.local,
                                              self.time,
                                              self.pos(), self.target)


class ObjCOM(object):
    def __init__(self, skel, target):
        self.skel = skel
        self.target = target
        self.time = 0.0

    def pos(self):
        return self.skel.C

    def cost(self):
        d = np.array(self.pos()) - np.array(self.target)
        return norm(d) ** 2

    def render(self):
        render_point(self.pos(), [1.0, 1.0, 0.0], 0.03)
        render_point(self.target, [0.0, 0.0, 1.0], 0.03)
        render_line(self.pos(), self.target, [0.0, 0.0, 0.0])

    def __str__(self):
        return 'ObjCOM[at %f] %s %s' % (self.time, self.pos(), self.target)


class MotionEvaluator(object):
    def __init__(self, skel, motion):
        self.skel = skel
        self.motion = motion

        self.objs = []
        lh = 'h_heel_left'
        lt = 'h_toe_left'
        rh = 'h_heel_right'
        rt = 'h_toe_right'

        # Swing foot locations
        self.add(0.0, lh, [-0.03, -0.05, 0], [-0.03, -0.95, -0.11])
        self.add(0.0, lt, [0.13, -0.05, 0.0], [0.22, -0.95, -0.11])
        self.add(0.365, lh, [-0.03, -0.05, 0], [0.55, -0.53, -0.11])
        self.add(0.365, lt, [0.13, -0.05, 0], [0.80, -0.53, -0.11])

        # Stance foot locations
        for t in np.arange(0.0, 0.365, 0.05):
            self.add(t, rh, [-0.03, -0.05, 0], [0.27, -0.74, 0.09])
            self.add(t, rt, [0.13, -0.05, 0.0], [0.51, -0.74, 0.09])

        # COM locations
        self.add_com(0.0, [0.079 + 0.20, 0.080, 0.001])
        self.add_com(0.365, [0.386, 0.299, 0.001])

        # When the swing foot is about to leave
        t = 0.12
        self.add_com(t, [0.079 + 0.25, 0.080, 0.001])
        self.add(t, lh, [-0.03, -0.05, 0], [-0.01, -0.76, -0.11])
        self.add(t, lt, [0.13, -0.05, 0.0], [0.22, -0.95, -0.11])

        # Do not collide toes with a stair
        t = 0.15
        self.add(t, lt, [0.13, -0.05, 0.0], [0.24, -0.79, -0.11])
        t = 0.282
        self.add(t, lt, [0.13, -0.05, 0.0], [0.52, -0.45, -0.11])

        # Sort
        self.objs.sort(key=lambda x: x.time)

        # Shit targets
        for obj in self.objs:
            obj.target[0] -= 0.07

    def add(self, time, name, local, target):
        body = self.skel.body(name)
        o = ObjPt(body, local, target)
        o.time = time
        self.objs.append(o)

    def add_com(self, time, target):
        o = ObjCOM(self.skel, target)
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
        return ret / float(len(self.objs))

    def render(self):
        for obj in self.objs:
            obj.render()

    def __str__(self):
        header = 'Motion Evaluator (%d objs)' % len(self.objs)
        body = '\n'.join([str(o) for o in self.objs])
        footer = '\n\n'
        return header + body + footer

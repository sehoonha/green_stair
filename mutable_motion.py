import numpy as np
import math


class RadialBasisDof(object):
    def __init__(self, skel, dof, w=0.0, s=0.1, x=0.0):
        self.skel = skel
        if isinstance(dof, str):
            self.index = self.skel.dof_index(dof)
        else:
            self.index = dof

        # Fetch the kernel parameters
        self.w = w  # weight
        self.s = s  # standard deviation
        self.x = x  # center

        # Store the pose vector
        self.vec = np.zeros(skel.ndofs)
        self.vec[self.index] = 1.0

    def num_params(self):
        return 3

    def params(self):
        return np.array([self.w, self.s, self.x])

    def set_params(self, params):
        (self.w, self.s, self.x) = tuple(params)

    def eval(self, t):
        (w, s, x) = (self.w, self.s, self.x)
        value = w * math.exp(-(1.0) / (2.0 * s * s) * (t - x) * (t - x))
        return value * self.vec


class MutableMotion(object):
    def __init__(self, skel, ref):
        self.skel = skel
        self.ref = ref
        # Init internal variables
        self.basis = []
        self.h = self.skel.world.dt
        # Initialize basis
        for i, dof in enumerate(self.skel.dofs):
            print i, dof.name
        self.init_basis()

    def init_basis(self):
        joints = ['j_pelvis_rot_z', 'j_pelvis_pos_x', 'j_pelvis_pos_y',
                  'j_thigh_left_z', 'j_shin_left', 'j_heel_left_1',
                  'j_thigh_right_z', 'j_shin_right', 'j_heel_right_1']
        for j in joints:
            self.add_basis(j, 0.0, 0.1, 0.05)
            self.add_basis(j, 0.0, 0.1, 0.95)

    def add_basis(self, dof, w0=0.0, s0=0.1, x0=0.0):
        b = RadialBasisDof(self.skel, dof, w0, s0, x0)
        self.basis.append(b)

    def num_params(self):
        return len(self.basis) * 3

    def params(self):
        params = np.zeros(self.num_params())
        for i, b in enumerate(self.basis):
            params[3 * i:3 * i + 3] = b.params()
        return params

    def set_params(self, params):
        for i, b in enumerate(self.basis):
            b.set_params(params[3 * i:3 * i + 3])

    def pose_at(self, t):
        frame_index = int(t / self.h)
        frame_index = min(frame_index, self.ref.num_frames - 1)
        pose = self.ref.pose_at(frame_index, self.skel.id)
        for b in self.basis:
            pose += b.eval(t)
        return pose

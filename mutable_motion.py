import numpy as np
import math
import jsonpickle


jsonpickle_default_skel = None


class RadialBasisDof(object):
    def __init__(self, skel, dof, w=0.0, s=0.1, x=0.0):
        self.skel = skel
        if isinstance(dof, str):
            self.index = self.skel.dof_index(dof)
        else:
            self.index = dof

        # Fetch the kernel parameters
        # self.w = w  # weight
        # self.s = s  # standard deviation
        # self.x = x  # center
        self.set_params([w, s, x])

    def num_params(self):
        return 3

    def params(self):
        return np.array([self.w, self.s, self.x])

    def set_params(self, params):
        (self.w, self.s, self.x) = tuple(np.array(params))

    def eval(self, t):
        vec = np.zeros(self.skel.ndofs)
        vec[self.index] = 1.0

        (w, s, x) = (self.w, self.s, self.x)
        value = w * math.exp(-(1.0) / (2.0 * s * s) * (t - x) * (t - x))
        return value * vec

    def __getstate__(self):
        data = dict(self.__dict__)
        del data['skel']
        return data

    def __setstate__(self, data):
        self.__dict__.update(data)
        self.skel = jsonpickle_default_skel


class MutableMotion(object):
    def __init__(self, skel, ref):
        self.skel = skel
        self.ref = ref
        # Init internal variables
        self.basis = []
        self.h = self.skel.world.dt
        # Initialize basis
        self.init_basis()

    def init_basis(self):
        joints = ['j_pelvis_rot_z', 'j_pelvis_pos_x', 'j_pelvis_pos_y',
                  'j_thigh_left_z', 'j_shin_left', 'j_heel_left_1',
                  'j_thigh_right_z', 'j_shin_right', 'j_heel_right_1']
        for j in joints:
            self.add_basis(j, 0.0, 0.1, 0.05)
            self.add_basis(j, 0.0, 0.1, 0.2)
            self.add_basis(j, 0.0, 0.5, 0.3)
            self.add_basis(j, 0.0, 0.1, 0.5)
            self.add_basis(j, 0.0, 0.1, 0.7)
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

    def ref_pose_at_frame(self, frame_index):
        frame_index = min(frame_index, self.ref.num_frames - 1)
        pose = self.ref.pose_at(frame_index, self.skel.id)
        return pose

    def ref_velocity_at_frame(self, frame_index):
        if frame_index == 0:
            return self.velocity_at_first()
        elif frame_index == self.ref.num_frames - 1:
            return self.velocity_at_last()
        elif frame_index > self.ref.num_frames - 1:
            return np.zeros(self.skel.ndofs)
        h = self.h
        q0 = self.ref.pose_at(frame_index - 1, self.skel.id)
        q2 = self.ref.pose_at(frame_index + 1, self.skel.id)
        vel = (-0.5 * q0 + 0.5 * q2) / h
        return vel

    def velocity_at_first(self):
        """ Backward finite difference with 2 accuracy """
        h = self.h
        q0 = self.ref.pose_at(0, self.skel.id)
        q1 = self.ref.pose_at(1, self.skel.id)
        q2 = self.ref.pose_at(2, self.skel.id)
        vel = (-1.5 * q0 + 2.0 * q1 - 0.5 * q2) / h
        return vel

    def velocity_at_last(self):
        """ Backward finite difference with 2 accuracy """
        h = self.h
        q0 = self.ref.pose_at(-3, self.skel.id)
        q1 = self.ref.pose_at(-2, self.skel.id)
        q2 = self.ref.pose_at(-1, self.skel.id)
        vel = (0.5 * q0 - 2.0 * q1 + 1.5 * q2) / h
        return vel

    def __getstate__(self):
        data = dict(self.__dict__)
        del data['skel']
        del data['ref']
        return data

    def __setstate__(self, data):
        self.__dict__.update(data)

    def save(self, filename):
        with open(filename, 'w+') as fout:
            frozen = jsonpickle.encode(self)
            fout.write(frozen)

    def load(self, filename):
        global jsonpickle_default_skel
        jsonpickle_default_skel = self.skel
        with open(filename, 'r') as fin:
            frozen = fin.read()
            dup = jsonpickle.decode(frozen)
            self.__dict__.update(dup.__dict__)

import numpy as np
from pydart import SkelVector


class ParameterizedMotion(object):
    def __init__(self, skel, ref):
        self.skel = skel
        self.ref = ref
        # Init internal variables
        self.h = self.skel.world.dt

        # Bake the reference COM
        self.bake_ref_com()

    def bake_ref_com(self):
        self.ref_com = []
        self.ref_com_dot = []
        self.ref_head = []
        x = self.skel.x
        for i in range(self.ref.num_frames):
            q = self.pose_at_frame(i, isRef=True)
            qdot = self.velocity_at_frame(i, isRef=True)
            self.skel.q = q
            self.skel.qdot = qdot
            self.ref_com.append(self.skel.C)
            self.ref_com_dot.append(self.skel.Cdot)
            self.ref_head.append(self.skel.body('h_head').C)
        self.skel.x = x

    def num_params(self):
        pass

    def set_params(self, params):
        self.params = params

    def parameterized_pose_at_frame(self, frame_index):
        """Should be inherited"""
        pass

    def pose_at_frame(self, frame_index, isRef=False):
        if isRef:
            return self.ref_pose_at_frame(frame_index)
        else:
            return self.parameterized_pose_at_frame(frame_index)

    def ref_pose_at_frame(self, frame_index):
        frame_index = min(frame_index, self.ref.num_frames - 1)

        q = self.ref.pose_at(frame_index, self.skel.id)
        return SkelVector(q, self.skel)

    def ref_com_at_frame(self, frame_index):
        frame_index = min(frame_index, self.ref.num_frames - 1)
        return self.ref_com[frame_index]

    def ref_com_dot_at_frame(self, frame_index):
        frame_index = min(frame_index, self.ref.num_frames - 1)
        return self.ref_com_dot[frame_index]

    def ref_head_at_frame(self, frame_index):
        frame_index = min(frame_index, self.ref.num_frames - 1)
        return self.ref_head[frame_index]

    def velocity_at_frame(self, frame_index, isRef=False):
        if frame_index == 0:
            return self.velocity_at_first_frame(isRef)
        elif frame_index == self.ref.num_frames - 1:
            return self.velocity_at_last_frame(isRef)
        elif frame_index > self.ref.num_frames - 1:
            return np.zeros(self.skel.ndofs)
        h = self.h
        q0 = self.pose_at_frame(frame_index - 1, isRef)
        q2 = self.pose_at_frame(frame_index + 1, isRef)
        vel = (-0.5 * q0 + 0.5 * q2) / h
        return vel

    def velocity_at_first_frame(self, isRef):
        """ Forward finite difference with 2 accuracy """
        h = self.h
        q0 = self.pose_at_frame(0, isRef)
        q1 = self.pose_at_frame(1, isRef)
        q2 = self.pose_at_frame(2, isRef)
        vel = (-1.5 * q0 + 2.0 * q1 - 0.5 * q2) / h
        return vel

    def velocity_at_last_frame(self, isRef):
        """ Backward finite difference with 2 accuracy """
        h = self.h
        q0 = self.pose_at_frame(-3, isRef)
        q1 = self.pose_at_frame(-2, isRef)
        q2 = self.pose_at_frame(-1, isRef)
        vel = (0.5 * q0 - 2.0 * q1 + 1.5 * q2) / h
        return vel

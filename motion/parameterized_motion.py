import numpy as np
from pydart import SkelVector


class ParameterizedMotion(object):
    def __init__(self, skel, ref):
        self.skel = skel
        self.ref = ref
        # Init internal variables
        self.h = self.skel.world.dt

    def num_params(self):
        pass

    def set_params(self, params):
        self.params = params

    def params(self):
        return self.params

    def parameterized_pose_at_frame(self, frame_index):
        """Should be inherited"""
        pass

    def pose_at_frame(self, frame_index, isRef=False):
        if isRef:
            return self.ref_pose_at_frame(frame_index)
        else:
            return self.parameterized_pose_at_frame(frame_index)

    def ref_pose_at_frame(self, frame_index):
        q = self.ref.pose_at(frame_index, self.skel.id)
        return SkelVector(q, self.skel)

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

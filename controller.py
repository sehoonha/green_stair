import numpy as np
from numpy.linalg import inv
from jacobian_transpose import JTController


class Controller:
    """ Add damping force to the skeleton """
    def __init__(self, skel, h, ref):
        self.h = h
        self.skel = skel
        self.ref = ref

        # Spring-damper
        ndofs = self.skel.ndofs
        self.qhat = self.skel.q
        self.Kp = np.diagflat([0.0] * 6 + [800.0] * (ndofs - 6))
        self.Kd = np.diagflat([0.0] * 6 + [40.0] * (ndofs - 6))

        # Jacobian transpose
        self.jt = JTController(self.skel)

    # def update_target_by_frame(self, frame_idx):
    #     if frame_idx < self.ref.num_frames:
    #         self.qhat = self.ref.pose_at(frame_idx, skel_id=0)
    #         I = self.skel.dof_indices(['j_heel_left_1', 'j_heel_right_1'])
    #         self.qhat[I] += 0.2

    def compute(self):
        skel = self.skel

        invM = inv(skel.M + self.Kd * self.h)
        p = -self.Kp.dot(skel.q + skel.qdot * self.h - self.qhat)
        d = -self.Kd.dot(skel.qdot)
        qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        tau = p + d - self.Kd.dot(qddot) * self.h

        # frame = self.skel.world.frame
        # if frame < 100:
        #     tau += self.jt.apply('h_toe_right', [-1000, 1000, 0])

        # Make sure the first six are zero
        tau[:6] = 0
        return tau
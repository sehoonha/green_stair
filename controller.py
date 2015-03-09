import numpy as np
from numpy.linalg import inv


class Controller:
    """ Add damping force to the skeleton """
    def __init__(self, skel, h, ref):
        self.h = h
        self.skel = skel
        self.ref = ref

        # Spring-damper
        ndofs = self.skel.ndofs
        self.qhat = self.skel.q
        self.Kp = np.diagflat([0.0] * 6 + [400.0] * (ndofs - 6))
        self.Kd = np.diagflat([0.0] * 6 + [40.0] * (ndofs - 6))

        # Jacobian transpose
        self.jt = JTController(self.skel)
        for i, b in enumerate(self.skel.dofs):
            print i, b.name

    def update_target_by_frame(self, frame_idx):
        if frame_idx < self.ref.num_frames:
            self.qhat = self.ref.pose_at(frame_idx, skel_id=0)
            # I = self.skel.dof_indices(['j_heel_left_1'])
            # self.qhat[I] += 1.0

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


class JTController:
    """
    # Usage
    self.jt = JTController(self.skel)
    tau += self.jt.apply( ["l_hand", "r_hand"], f )
    """
    def __init__(self, _skel):
        self.skel = _skel

    def apply(self, bodynames, f):
        if not isinstance(bodynames, list):
            bodynames = [bodynames]
        f = np.array(f)

        tau = np.zeros(self.skel.ndofs)
        for bodyname in bodynames:
            # J = self.skel.getBodyNodeWorldLinearJacobian(bodyname)
            J = self.skel.body(bodyname).world_linear_jacobian()
            JT = np.transpose(J)
            tau += JT.dot(f)
        return tau

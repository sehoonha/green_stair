import numpy as np
import logging
import posetools


class FileInfoWorld(object):
    def __init__(self):
        self.num_frames = 0
        self.skels = []
        self.pose = []
        self.logger = logging.getLogger(__name__)

    def load(self, filename):
        logger = self.logger
        num_skels = 0
        skel_id = 0
        data = []
        with open(filename) as fin:
            self.logger.info('open %s OK' % filename)
            for lineno, line in enumerate(fin.readlines()):
                tokens = line.split()
                if lineno == 0:
                    self.num_frames = int(tokens[1])
                    logger.info('numFrames = %d' % self.num_frames)
                elif lineno == 1:
                    num_skels = int(tokens[1])
                    self.logger.info('numSkeletons = %d' % num_skels)
                elif lineno == 2:
                    if num_skels * 2 != len(tokens):
                        self.logger.warning('invalid number of skeletons')
                    self.skels = [(tokens[2 * i], int(tokens[2 * i + 1]))
                                  for i in range(num_skels)]
                    logger.info('skels = %s' % self.skels)
                elif len(tokens) == 0:
                    pass
                elif tokens[0] == 'Contacts':
                    pass
                else:
                    if len(tokens) != self.skels[skel_id][1]:
                        logger.warning('invalid data: skel_id= %d' % skel_id)
                    pose = np.array([float(x) for x in tokens])
                    data.append(pose)
                    skel_id = (skel_id + 1) % len(self.skels)
            logger.info('parsing %s OK' % filename)
        n = len(self.skels)
        self.pose = [data[i * n:(i + 1) * n] for i in range(self.num_frames)]

    def modify_pose(self, skel):
        new_pose = []
        for i in range(self.num_frames):
            pose_frame = []
            for j, q_j in enumerate(self.pose[i]):
                if j == 0:
                    q_j_new = posetools.ik_adjust(skel, q_j)
                    pose_frame.append(q_j_new)
                else:
                    pose_frame.append(q_j)
            # Collect to the new_pose
            new_pose.append(pose_frame)
        # Update the original pose
        self.pose = new_pose
        self.num_frames = len(self.pose)

    def append_mirrored_motion(self, skel):
        for i in range(self.num_frames):
            pose_frame = []
            for j, q_j in enumerate(self.pose[i]):
                if j == 0:
                    q_j_mirrored = posetools.mirror_pose(skel, q_j)
                    q_j_mirrored[3] += 0.29
                    q_j_mirrored[4] += 0.20
                    if i == 0:
                        # self.append_interpolated_motion(q_j_mirrored, 215)
                        self.append_interpolated_motion(q_j_mirrored, 65)
                    pose_frame.append(q_j_mirrored)
                else:
                    pose_frame.append(q_j)
            self.pose.append(pose_frame)
        self.num_frames = len(self.pose)

    def append_interpolated_motion(self, rhs, num_frames):
        weights = np.linspace(0.0, 1.0, num_frames + 2)
        weights = weights[1:-1]
        ref_frame = self.pose[-1]
        for w in weights:
            pose_frame = []
            for j, q_j in enumerate(ref_frame):
                if j == 0:
                    q_j_interp = (1 - w) * q_j + w * rhs
                    pose_frame.append(q_j_interp)
                else:
                    pose_frame.append(q_j)
            self.pose.append(pose_frame)
        self.num_frames = len(self.pose)

    def append_shifted_motion(self, skel):
        for i in range(self.num_frames):
            pose_frame = []
            for j, q_j in enumerate(self.pose[i]):
                if j == 0:
                    q_j_shifted = np.array(q_j)
                    q_j_shifted[3] += 0.29 * 2
                    q_j_shifted[4] += 0.2 * 2
                    if i == 0:
                        self.append_interpolated_motion(q_j_shifted, 65)
                    pose_frame.append(q_j_shifted)
                else:
                    pose_frame.append(q_j)
            self.pose.append(pose_frame)
        self.num_frames = len(self.pose)

    def add_offset(self):
        for i in range(self.num_frames):
            for j, q_j in enumerate(self.pose[i]):
                if j != 0:
                    continue
                q_j[2] -= 5.0 * (3.1415 / 180.0)
                # q_j[3] += 0.06
                q_j[3] += 0.07
                q_j[4] += 0.03

    def pose_at(self, frame_index, skel_id):
        return np.array(self.pose[frame_index][skel_id])

    def poses(self, skel_id):
        return [np.array(frame[skel_id]) for frame in self.pose]

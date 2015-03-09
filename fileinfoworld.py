import numpy as np
import logging


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

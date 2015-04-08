import logging
import matplotlib.pyplot as plt
from math import fabs


class PlotterTorque(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def plot(self, con, title):
        plt.ioff()
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        colors = ['r', 'b']
        pp = []
        legends = []
        for j, jname in enumerate(['j_shin_left', 'j_shin_right']):
            self.logger.info('plot %s' % jname)
            j_index = con.skel.dof_index(jname)
            legends.append(jname + '_torque')
            legends.append(jname + '_power')
            velocities = con.history['qdot']
            torques = con.history['tau']
            x = [float(i) / 1000.0 for i in range(len(torques))]
            y = [(tau[j_index]) for tau in torques]
            y2 = [(qdot[j_index] * tau[j_index])
                  for qdot, tau in zip(velocities, torques)]
            color = colors[j]
            p = plt.plot(x, y, color=color, linewidth=2)
            pp.append(p[0])
            p2 = plt.plot(x, y2, ls='--', color=color, linewidth=2)
            pp.append(p2[0])
        font = {'size': 28}
        # plt.title('Compare %d Trials on %s' % (num_trials, prob_name),
        t = plt.title(title,
                      fontdict={'size': 32})
        t.set_y(0.92)
        font = {'size': 28}
        plt.xlabel('# Samples', fontdict=font)
        plt.ylabel('Cost', fontdict=font)
        # plt.legend(pp, self.data.keys(), numpoints=1, fontsize=20)
        # plt.legend(pp, legends, numpoints=1, fontsize=26,
        plt.axes().set_xlim(0.0, 1.6)  # Walking
        plt.axes().set_ylim(-300.0, 600.0)  # Walking
        plt.legend(pp, legends, fontsize=26,
                   # bbox_to_anchor=(0.15, 0.15))
                   # loc='lower left')
                   loc='upper right')
        plt.savefig('torque.png', bbox_inches='tight')
        plt.close(fig)

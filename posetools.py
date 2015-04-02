import numpy as np


def mirror_pose(skel, q):
    ret = np.array(q)
    for i, dof in enumerate(skel.dofs):
        if 'left'in dof.name:
            j = skel.dof_index(dof.name.replace('left', 'right'))
        elif 'right'in dof.name:
            j = skel.dof_index(dof.name.replace('right', 'left'))
        else:
            j = i
        scale = 1.0
        if i > 6 and ('_x' in dof.name or '_y' in dof.name):
            scale = -1.0
        # print j, '<--', i, '(', scale, ')'
        ret[i] = scale * q[j]
    # exit(0)
    return ret

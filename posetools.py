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
        ret[i] = q[j]
    return ret

from com_planner import COMPlanner
from foot_planner import FootPlanner


class Planner(object):
    def __init__(self, skel, ref):
        self.skel = skel
        self.ref = ref

    def solve(self):
        self.com = COMPlanner(self.skel, self.ref)
        self.com.solve()

        self.foot = FootPlanner(self.skel, self.ref)

    def render(self):
        self.com.render()
        self.foot.render()

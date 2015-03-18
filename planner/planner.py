from com_planner import COMPlanner


class Planner(object):
    def __init__(self, skel, ref):
        self.skel = skel
        self.ref = ref

    def solve(self):
        self.com = COMPlanner(self.skel, self.ref)
        self.com.solve()

    def render(self):
        self.com.render()

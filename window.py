import logging
from PyQt4 import QtGui
from pydart.qtgui import PyDartQtWindow
from pydart.qtgui.trackball import Trackball


class Window(PyDartQtWindow):
    def __init__(self, title, simulation):
        super(Window, self).__init__(title, simulation)
        self.logger = logging.getLogger(__name__)

    def initActions(self):
        PyDartQtWindow.initActions(self)
        self.loadAction = self.createAction('&Load', self.loadEvent)
        self.saveAction = self.createAction('&Save', self.saveEvent)
        self.printAction = self.createAction('Print', self.printEvent)

        self.optAction = self.createAction('Opt', self.optEvent)
        self.killAction = self.createAction('Kill', self.killEvent)

    def initToolbar(self):
        self.timeText = QtGui.QLabel('Time: 0.0000', self)

        # Update self.toolbar_actions
        # "list[x:x] += list2" is Python idiom for add list to the another list

        my_toolbar_actions = [self.printAction, self.timeText, None,
                              self.optAction, self.killAction, None]
        self.toolbar_actions[4:4] += my_toolbar_actions

        # Call the parent function to initialize the toolbar
        PyDartQtWindow.initToolbar(self)

    def initMenu(self):
        PyDartQtWindow.initMenu(self)
        self.fileMenu.addAction(self.loadAction)
        self.fileMenu.addAction(self.saveAction)

    def idleTimerEvent(self):
        PyDartQtWindow.idleTimerEvent(self)
        self.timeText.setText('T: %.4f' % self.sim.world.t)

    def loadEvent(self):
        filename = 'output.json'
        self.sim.motion.load(filename)
        self.logger.info('load file: ' + filename)
        self.sim.reset()

    def saveEvent(self):
        filename = 'output.json'
        self.sim.motion.save(filename)
        self.logger.info('save file: ' + filename)

    def printEvent(self):
        print('print event')

    def optEvent(self):
        self.sim.optimize()

    def killEvent(self):
        self.sim.kill_optimizer()

    def cam0Event(self):
        """ Change the default camera """
        self.glwidget.tb = Trackball(phi=-2.1, theta=-6.6, zoom=1.0,
                                     rot=[-0.06, 0.01, -0.02, 1.00],
                                     trans=[-0.73, -0.19, -3.47])

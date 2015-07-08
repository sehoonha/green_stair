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

        self.refreshAction = self.createAction('Refresh', self.refreshEvent)

        self.optAction = self.createAction('Opt', self.optEvent)
        self.killAction = self.createAction('Kill', self.killEvent)
        self.plotAction = self.createAction('Plot', self.plotEvent)

    def initToolbar(self):
        self.timeText = QtGui.QLabel('Time: 0.0000', self)
        self.actSpin = QtGui.QDoubleSpinBox(self)
        self.actSpin.setValue(self.sim.stair._activation)
        self.actSpin.setSingleStep(0.1)
        self.actSpin.valueChanged.connect(self.actSpinChangedEvent)
        self.prefix = QtGui.QLineEdit('', self)
        self.postfix = QtGui.QLineEdit('', self)
        # Update self.toolbar_actions
        # "list[x:x] += list2" is Python idiom for add list to the another list
        self.solXSpin = QtGui.QSpinBox(self)
        self.solXSpin.setMaximum(0)
        self.solXSpin.valueChanged.connect(self.solSpinChangedEvent)
        self.solYSpin = QtGui.QSpinBox(self)
        self.solYSpin.setMaximum(0)
        self.solYSpin.valueChanged.connect(self.solSpinChangedEvent)

        my_toolbar_actions = [self.printAction, self.timeText, None,
                              self.actSpin, None,
                              self.refreshAction,
                              self.solXSpin, self.solYSpin, None,
                              self.optAction, self.killAction,
                              self.plotAction, None]
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
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Open file',
                                                     '.', '*.json')
        if len(filename) == 0:
            self.logger.warning('cancel the load')
            return
        self.sim.motion.load(filename)
        self.logger.info('load file: ' + filename)
        self.sim.reset()

    def saveEvent(self):
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save file',
                                                     '.', '*.json')
        if len(filename) == 0:
            self.logger.warning('cancel the save')
            return
        if '.json' not in filename[-5:]:
            filename += '.json'

        self.sim.motion.save(filename)
        self.logger.info('save file: ' + filename)

    def actSpinChangedEvent(self):
        v = self.actSpin.value()
        self.logger.info('actSpinChangedEvent: %f' % v)
        self.sim.stair.set_activation(v)

    def refreshEvent(self):
        max_x, max_y = self.sim.refresh_solutions()
        self.solXSpin.setMaximum(max_x)
        self.solYSpin.setMaximum(max_y)

    def solSpinChangedEvent(self):
        x = self.solXSpin.value()
        y = self.solYSpin.value()
        self.logger.info('solSpinChangedEvent: %d %d' % (x, y))
        self.sim.change_solution(x, y)

    def printEvent(self):
        print('print event')

    def optEvent(self):
        self.sim.optimize()
        self.refreshEvent()

    def killEvent(self):
        self.sim.kill_optimizer()

    def plotEvent(self):
        self.sim.plot_torques()

    def cam0Event(self):
        """ Change the default camera """
        self.glwidget.tb = Trackball(phi=-2.1, theta=-6.6, zoom=1.0,
                                     rot=[-0.06, 0.01, -0.02, 1.00],
                                     trans=[-0.73, -0.19, -3.47])

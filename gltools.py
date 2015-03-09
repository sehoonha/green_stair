from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


def render_COM(skel):
    glPushMatrix()
    glColor3d(1.0, 0.0, 0.0)
    glTranslated(*skel.C)
    glutSolidSphere(0.05, 4, 2)
    glPopMatrix()

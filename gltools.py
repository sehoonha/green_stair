from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


def render_COM(skel):
    glPushMatrix()
    glColor3d(1.0, 0.0, 0.0)
    glTranslated(*skel.C)
    glutSolidSphere(0.05, 4, 2)
    glPopMatrix()


def render_point(x, color, size=0.05):
    glPushMatrix()
    glColor3d(*color)
    glTranslated(*x)
    glutSolidSphere(size, 4, 2)
    glPopMatrix()


def render_line(x, y, color):
    glPushMatrix()
    glColor3d(*color)
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glVertex3d(*x)
    glVertex3d(*y)
    glEnd()
    glPopMatrix()


def render_trajectory(pts, color):
    glPushMatrix()
    glColor3d(*color)
    glLineWidth(2.0)
    # glBegin(GL_LINES)
    glBegin(GL_LINE_STRIP)
    for pt in pts:
        glVertex3d(*pt)
    glEnd()
    glPopMatrix()

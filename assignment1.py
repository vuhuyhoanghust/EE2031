import numpy as np
from numpy import array, dot, pi
from numpy import linalg as LA
from sympy import symbols
from sympy import solve
from scipy import integrate
import math
from numpy.linalg import inv

def sg(number):
    if number >= 0:
        return "+"
    return "-"

def dispVec(vec, name):
    print('%s = %.2fax %s%.2fay %s%.2faz'
    % (name, vec[0][0], sg(vec[0][1]), abs(vec[0][1]),
    sg(vec[0][2]), abs(vec[0][2])))

def dispVectru(vec, name):
    print('%s = %.2fa_rho %s%.2fa_phi %s%.2faz'
    % (name, vec[0][0], sg(vec[0][1]), abs(vec[0][1]),
    sg(vec[0][2]), abs(vec[0][2])))

def dispVeccau(vec, name):
    print('%s = %.2fa_r %s%.2fa_theta %s%.2fa_phi'
    % (name, vec[0][0], sg(vec[0][1]), abs(vec[0][1]),
    sg(vec[0][2]), abs(vec[0][2])))

def calcE(q, pos1, pos2):
    pos = pos1 - pos2
    return -q/(4*pi*ep*pow(LA.norm(pos,2),3))*pos
ep = 1e-9/36/pi
# Exercise 1

print('Exercise 1: \n')
q1 = 25*1e-9
q2 = 60*1e-9
q1Vec = array([[4,-2,7]])
q2Vec = array([[-3,4,-2]])

# a
q3Vec = array([[1,2,3]])
e1 = calcE(q1, q1Vec, q3Vec)
e2 = calcE(q2, q2Vec, q3Vec)
dispVec(e1+e2, "P3")

"""
# b
y = symbols('y')
func1 = q1*q1Vec[0][0]/(q1Vec[0][0]**2+q1Vec[0][2]**2+(q1Vec[0][1]-y)**2)**(3/2)
func2 = q2*q2Vec[0][0]/(q2Vec[0][0]**2+q2Vec[0][2]**2+(q2Vec[0][1]-y)**2)**(3/2)
func = func1+ func2
sol = solve(func,y)
print(sol)
"""



# Exercise 2
print('Exercise 2: \n')
q = 120e-9
aPos = np.array([[0,0,1]])
bPos = np.array([[0,0,-1]])

#a
pPos = np.array([[0.5,0,0]])
e1 = calcE(q, aPos, pPos)
e2 = calcE(q, bPos, pPos)
e_p = e1+e2
dispVec(e_p, "Ep")

#b
x = symbols('x')
func = x/(4*pi*ep*(0.5**2)) - e_p[0][0] 
sol = solve(func,x)
posf = sol[0]*1e9
print('Q0 = %.2lf nC' %(posf))




# Exercise 3
print('Exercise 3: \n')
q = 2e-6
aPos = np.array([[4,3,5]])
pPos = np.array([[8,12,2]])
Ep = calcE(q, aPos, pPos)
phi = math.atan(pPos[0][1]/pPos[0][0])
Erho = Ep[0][0]*math.cos(phi) + Ep[0][1]*math.sin(phi)
Ephi = -Ep[0][0]*math.sin(phi) + Ep[0][1]*math.cos(phi)
Ez = Ep[0][2]
Etru = np.array([[Erho, Ephi, Ez]])
dispVectru(Etru, "Etru")



# Exercise 4
print('Exercise 4: \n')
#a
Ez = 1e3
qPos = np.zeros((1,3))
pPos = np.array([[-2,1,-1]])
q = symbols('q')
func = q*pPos[0][2]/(4*pi*ep*pow(LA.norm(pPos,2),3))-Ez
sol = solve(func,q)
q0 = sol[0]
print('Q0 = %.2f uC' %(q0*1e6))
#b
mPos = np.array([[1,6,5]])
Em = calcE(q0, qPos, mPos)
dispVec(Em, "ED")
# tru
phi = math.atan(mPos[0][1]/mPos[0][0])
Erho = Em[0][0]*math.cos(phi) + Em[0][1]*math.sin(phi)
Ephi = -Em[0][0]*math.sin(phi) + Em[0][1]*math.cos(phi)
Ez = Em[0][2]
Etru = np.array([[Erho, Ephi, Ez]])
dispVectru(Etru, "Etru")

#cau
r = LA.norm(mPos)
the = math.acos(mPos[0][2]/r)
phi = math.atan(mPos[0][1]/mPos[0][0])
mat = np.array([[math.sin(the)*math.cos(phi),math.sin(the)*math.sin(phi),math.cos(the)],
                [math.cos(the)*math.cos(phi),math.cos(the)*math.sin(phi),-math.sin(the)],
                [-math.sin(phi), math.cos(phi), 0]])
Ecau = np.dot(mat, np.transpose(Em))
dispVeccau(np.transpose(Ecau), "Ecau")




# Exercise 5
print('Exercise 5: \n')
r1 = 3e-2
r2 = 5e-2
muV = 0.2e-6
#a
Q = 4/3*pi*(r2**3-r1**3)*muV
print('Q = %.2f pC' %(Q*1e12))
#b
r = symbols('r')
func = r**3-r1**3 - 0.5*(r2**3 - r1**3)
sol = solve(func, r)
print('r2 = %.2f cm' %(abs(sol[0])*100))




# Exercise 6
print('Exercise 6: \n')
muL = 2e-6
pPos = np.array([[1,2,3]])
#a
phi = math.atan(pPos[0][1]/pPos[0][0])
rho = math.sqrt(pPos[0][0]**2+pPos[0][1]**2)
const = muL/(2*pi*ep*rho)
Ep = np.array([[const*math.cos(phi), const*math.sin(phi), 0]])
dispVec(Ep, "Ep")
#b
const = muL/(4*pi*ep)
Ex = integrate.quad(lambda z: 
        pPos[0][0]/(pPos[0][0]**2+pPos[0][1]**2
        +(pPos[0][2]-z)**2)**1.5, -4, 4)
Ey = integrate.quad(lambda z: 
        pPos[0][1]/(pPos[0][0]**2+pPos[0][1]**2
        +(pPos[0][2]-z)**2)**1.5, -4, 4) 
Ez = integrate.quad(lambda z: 
        (pPos[0][2]-z)/(pPos[0][0]**2+pPos[0][1]**2
        +(pPos[0][2]-z)**2)**1.5, -4, 4)          
ED = const*np.array([[Ex, Ey, Ez]])
dispVec(np.transpose(ED), "ED")




# Exercise 7
print('Exercise 7: \n')
muS = 2e-6
rhoLim= 0.2
aPos=np.array([[0,0,0.5]])
f = lambda phi, rho: muS*rho*aPos[0][2]/(4*pi*ep*(rho**2+aPos[0][2]**2)**1.5)
sol = integrate.dblquad(f, 0, 0.2, 0, 2*pi)
print('EA = %.2faz' %(sol[0]))
print('EB = -%.2faz' %(sol[0]))




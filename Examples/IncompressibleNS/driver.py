"""
Incompressible Navier-Stokes using discrete exterior
calclus.

An attempt at following the methods laid out in [1].

1. Discrete exterior calculus discretization of
incompressible Navier-Stokes equations over surface
simplicial meshes
"""
from pydec import simplicial_complex, d, delta, whitney_innerproduct, \
     simplex_quivers
from numpy import loadtxt
from scipy import real, zeros
from scipy.linalg import inv
from scipy.linalg import eig
from matplotlib.pylab import quiver, figure, triplot, show
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

vertices = 2*np.pi * np.random.rand(6000, 2)
vertices -= np.pi
triangles = sp.spatial.Delaunay(vertices).simplices
print("Finish triangulation")

sc = simplicial_complex((vertices, triangles))
print("Generated simplicial complex")

# simulation parameters
# time step
dt = .1
# viscosity
mu =  .01

###################################################
# hodge stars. 
star0 = sc[0].star.toarray()
star1 = sc[1].star.toarray()
inv_star0 = np.diag(np.reciprocal(np.diag(star0)))
inv_star1 = np.diag(np.reciprocal(np.diag(star1)))

# exterior derivatives.
d_0 = sc[0].d
d_1 = sc[1].d
d_b = .5 * np.abs(d_0.T) @ np.diag(d_1.T @ np.ones(d_1.shape[0]))
dual_ext = (-d_0.T) @ star1
d_b_primal = inv_star0 @ d_b 

# Page 13, middle of page.
U_op = star1 @ d_0 
V_op = inv_star1 @ U_op

def UV(psi):
    return U_op @ psi, V_op @ psi

# Middle of page 12
def W(V): return .5 * np.diag(V) @ np.abs(d_0)

def F(U, V):
    '''Unlabeled, but implements F from the paper, right under 17'''
    pre_mult = d_b_primal @ V
    return 1/dt * (-d_0.T) @ U + dual_ext @ (mu*d_0 @ pre_mult- W(V) @ pre_mult)

A = (1./(dt)) * (-d_0.T) @ U_op
def first_half_step(psi):
    '''Implements (18)'''
    print("Start first half step")
    # Want to solve equation Ax =b.
    U, V = UV(psi)

    pre_mult = inv_star0 @ (-d_0.T) @ U
    b = F(U, V) + dual_ext @ (-mu*d_0 @ pre_mult + W(V) @ pre_mult)

    print("Solving first half step")
    return np.linalg.solve(A, b)

def second_half_step(psi):
    '''Implements (19)'''
    print("Start second half step")
    U, V = UV(psi)
    R = dual_ext @ d_0
    A = 1/dt * R + (mu * R  + dual_ext @ W(V)) @ inv_star0 @ R

    print("Solving second half step")
    return np.linalg.solve(A, F(U, V))

def time_step(psi):
    psi_n1 = first_half_step(psi)
#    print("Finished first half step")
#    psi_n2 = second_half_step(psi_n1) 
#    print("Finished second half step")
    return psi_n1

def coord_stream(coord):
    x = coord[0]
    y = coord[1]
    return -np.sin(x)*np.sin(y)

#plt.triplot(sc.vertices[:,0], sc.vertices[:,1], sc.simplices.copy())

psi = np.ones(vertices.shape[0])
for i, x in enumerate(vertices):
    psi[i] = coord_stream(x)

for i in range(20):
    print(f"----------------Begin iteration {i}---------------------")
    psi = time_step(psi)    
    plt.tricontourf(vertices[:,0], vertices[:,1], triangles, psi)
    plt.savefig(f'iteration-{i}')




from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

start_time = 0
end_time = 0

def graphe(x,u, ti):
    plt.plot(x,u, '-b', label=ti)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Courbe')
    plt.legend()
    plt.show()

nx = 200
try:
    assert nx >= SIZE
except AssertionError:
    print("Nombre de processus superieur au nombre de points.")


CFL = 1
c = 1
L = 1000
x = np.linspace(0,L,nx)
dx = L/(nx-1)
dt = CFL*dx/abs(c)
u = np.zeros(nx)

for i in range(nx):
    if 300 < x[i] < 400:
        u[i] = 10
    else:
        u[i] = 0

if RANK==0:
    graphe(x,u,"Solution initiale")

def solve_1d_linearconv(u, un, nt, nx, dt, dx, c):

    for n in range(nt):  
        un = np.copy(u)
        
        for i in range(1, nx): 
            u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])
            
        # Envoi des bords au processus voisin
        if RANK < SIZE-1:
            COMM.send(u[-1], dest=RANK+1, tag=2)
        if RANK != 0:
            u[0] = COMM.recv(source=RANK - 1, tag=2)
            
    return 0

#---------------------------
# Calcul ||
#---------------------------

start_time = time.time()

if RANK == 0:
    # Le partage du domaine aux process
    u_loc = np.array_split(u,SIZE)
    
    for i in range(SIZE-1):
        u_loc[i+1] = np.insert(u_loc[i+1], 0, u_loc[i][-1])
        
    # L'envoi des domaines à chaque process
    for process in range(SIZE):
        COMM.send(u_loc[process], dest=process, tag=0)

# Reception des domaines et calcul
u = COMM.recv(source = 0, tag = 0)
nx = len(u)
un = np.zeros(nx)
nt = 100

solve_1d_linearconv(u, un, nt, nx, dt, dx, c)
COMM.send(u[1:], dest=0, tag=1)

u_final = np.array([0])
if RANK == 0:
    for i in range(SIZE):
        un_loc = COMM.recv(source=i, tag=1)
        u_final = np.concatenate((u_final, un_loc))
        
    graphe(x,un, "Solution finale")
    
end_time = time.time()
total_time = end_time - start_time

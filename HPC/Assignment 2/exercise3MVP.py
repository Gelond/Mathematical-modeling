
import numpy as np
from scipy.sparse import lil_matrix
from numpy.random import rand, seed
from numba import njit
from mpi4py import MPI

''' This program compute parallel csc matrix vector multiplication using mpi '''

COMM = MPI.COMM_WORLD
nbOfproc = COMM.Get_size()
RANK = COMM.Get_rank()

seed(42)

def matrixVectorMult(A, b, x):
    
    row, col = A.shape
    for i in range(row):
        a = A[i]
        for j in range(col):
            x[i] += a[j] * b[j]

    return 0

########################initialize matrix A and vector b ######################
#matrix sizes
SIZE = 1000
Local_size = SIZE//nbOfproc
reste = SIZE % nbOfproc

if RANK < reste:
    Local_size += 1

# counts = block of each proc
counts = COMM.allgather(Local_size*SIZE)
counts=np.array(counts)

if RANK == 0:
    """A = np.array([
    [1, 2, 0, 0],
    [0, 3, 4, 0],
    [5, 0, 0, 6],
    [0, 0, 7, 8]], dtype=np.double)"""
    A = lil_matrix((SIZE, SIZE))
    A[0, :100] = rand(100)
    A[1, 100:200] = A[0, :100]
    A.setdiag(rand(SIZE))
    A = A.toarray()
    b = rand(SIZE)
else :
    A = None
    b = None
    
b = COMM.bcast(b, root=0)

#########Send b to all procs and scatter A (each proc has its own local matrix#####
LocalMatrix = np.zeros((Local_size, SIZE), dtype=np.double)
# Scatter the matrix A
COMM.Scatterv([A, counts, MPI.DOUBLE], LocalMatrix, root=0)
print(f"{RANK}, got this\n {LocalMatrix} \n")

#####################Compute A*b locally#######################################
LocalX = np.zeros(Local_size)
#print(f"{RANK}, got this {LocalX} \n")


start = MPI.Wtime()
matrixVectorMult(LocalMatrix, b, LocalX)
stop = MPI.Wtime()

maxTime = COMM.reduce((stop - start)*1000, op=MPI.MAX, root=0)

if RANK==0:
    with open('CPU_time.txt', 'a') as fichier:
        fichier.write(str(maxTime)+" ")

if RANK == 0:
    print("CPU time of parallel multiplication is ", (stop - start)*1000)

##################Gather te results ###########################################
# sendcouns = local size of result
sendcounts = COMM.gather(Local_size, root=0)
if RANK == 0:
    X = np.zeros(SIZE)
else :
    X = None

COMM.Gatherv(LocalX, [X,sendcounts], root=0)

# Gather the result into X
##################Print the results ###########################################
if RANK == 0 :
    #print(sendcounts)
    X_ = A.dot(b)
    print("The result of A*b using dot is :", np.max(X_ - X))
    #print("The result of A*b using dot is :", round(np.max(X_ - np.array(X).reshape(-1))))
    #print("The result of A*b using parallel version is :", X)


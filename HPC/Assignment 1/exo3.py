
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

itrs = 10
a = 0

for i in range(itrs):
    if RANK == 0:
        COMM.send(a, dest=1)
        a = COMM.recv(source=1)
        print(f"Iteration {i}: I, process {RANK}, I received {a} from the process 1.")
        
    if RANK == 1:
        a = COMM.recv(source=0)
        print(f"Iteration {i}: I, process {RANK}, I received {a} from the process 0.")
        a += 1
        COMM.send(a, dest=0)

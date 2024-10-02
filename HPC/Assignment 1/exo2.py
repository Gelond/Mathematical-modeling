
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

a = 0
while True:
    try:
        if RANK == 0:
            a = int(input("Entrez un entier: "))
            for p in range(1,SIZE):
                COMM.send(a, dest = p)
                
        else:
            a = COMM.recv(source = 0)

        if a < 0:
            break

        print(f"Process {RANK} got {a}")
        
    except ValueError:
        print("Entree invalide.")

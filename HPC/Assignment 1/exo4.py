
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
data = 0
if RANK == 0:
    try:
        data = int(input("Entrez un entier: "))
    except ValueError:
        print("Entree invalide.")
        exit()
        
    print(f"I, process {RANK}, I received {data} from the process {RANK}.")
        
    COMM.send(data, dest=1)
        
else:
    data = COMM.recv(source=(RANK - 1))
    print(f"I, process {RANK}, I received {data} from the process {(RANK - 1)}.")

    if RANK != SIZE - 1:
        data += RANK
        COMM.send(data, dest=(RANK + 1))    
    
        
    

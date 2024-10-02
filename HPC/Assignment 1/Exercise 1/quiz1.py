from mpi4py import MPI

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()

# Print "Hello World" from each process
print(f"Hello World" )

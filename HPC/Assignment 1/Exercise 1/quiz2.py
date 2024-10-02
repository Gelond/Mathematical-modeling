from mpi4py import MPI

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()

print(f"Hello World from process {RANK} of {SIZE}".format ( RANK = RANK , SIZE = SIZE ) )

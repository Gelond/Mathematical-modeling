from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

if RANK == 0:
    print(f"Hello World from process {RANK} of {SIZE}".format ( RANK = RANK , SIZE = SIZE ) )

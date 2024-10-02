
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

def compute_gradient(data, labels, weight):
    predictions = data * weight
    errors = predictions - labels
    gradient = 2 * np.dot(data, errors) / len(data)
    return gradient

def generate_data(n_samples, with_noise = True):
    x = np.random.rand(n_samples)
    y = 2*x + np.random.normal(loc=0, scale=1, size=len(x)) if with_noise else 0
    return x,y

np.random.seed(42)
itrs = 100
eps = 0.001

if RANK==0:
    n_samples = 100000
    x,y = generate_data(n_samples, True)
    x_loc = np.array_split(x,SIZE)
    y_loc = np.array_split(y,SIZE)
    
    weight = np.random.rand()

else:
    x_loc = None
    y_loc = None
    weight = None
    
x_loc = COMM.scatter(x_loc, root=0)
y_loc = COMM.scatter(y_loc, root=0)
weight = COMM.bcast(weight, root=0)

for i in range(itrs):
    gradient = compute_gradient(x_loc, y_loc, weight)
    sum_gradient = COMM.allreduce(gradient, op=MPI.SUM)
    COMM.Barrier()
    weight -= eps*sum_gradient
    
COMM.Barrier()

if RANK==0:
    print(f"Final model parameters (weight) is {weight}")
    

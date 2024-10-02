
from mpi4py import MPI
import random

def compute_points(INTERVAL):
    
    random.seed(42)  
    
    circle_points= 0
    total_points=0

    # Total Random numbers generated= possible x 
    # values* possible y values 
    for i in range(INTERVAL**2): 
      
        # Randomly generated x and y values from a 
        # uniform distribution 
        # Range of x and y values is -1 to 1 
        
        rand_x= random.uniform(-1, 1) 
        rand_y= random.uniform(-1, 1) 
      
        # Distance between (x, y) from the origin 
        origin_dist= rand_x**2 + rand_y**2
      
        # Checking if (x, y) lies inside the circle 
        if origin_dist<= 1: 
            circle_points+= 1
        
        total_points+=1
    
    return circle_points, total_points

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

if RANK==0:
    INTERVAL = 1000
    inter = INTERVAL // SIZE
    reste = INTERVAL % SIZE
    intervals = [inter for i in range(SIZE)]

    for i in range(reste):
        intervals[i] += 1
        
else:
    intervals = None

interval_loc = COMM.scatter(intervals, root=0)
    
print(f"I, process {RANK}, I have {interval_loc}")
    
numbers = compute_points(interval_loc)
circle_points = COMM.reduce(numbers[0], op=MPI.SUM, root=0)
total_points = COMM.reduce(numbers[1], op=MPI.SUM, root=0)

if RANK==0:
    pi = 4*(circle_points/total_points)
    print(f"I, process {RANK}, pi is {pi}")

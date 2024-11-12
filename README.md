# voronoi-diagram
## Important
Tested on:
- Arch Linux 6.8.9 (as for 02/06/24)
- GCC: 14.1.1
- CUDA: 12.4
## Instructions
### 2D Voronoi (default)
Compilation:
```
make
```
Execution:
```
Run as ./test N S MODE ITER DEVICE MEM_GPU BS DIST REDUX Select-MU MU SAMPLE MOLECULES
N - length/width of grid
S - number of seeds
MODE - (0) JFA, (1) dJFA, (2) rJFA, (3) drJFA
ITER - number of iterations
DEVICE - Wich device to use, fill with ID of GPU
MEM_GPU - Memory management, 1: manual, 0: auto, anything if DEVICE = 0
BS - block size of GPU (32 max), anything if DEVICE = 0
DIST - distance method, 1: manhattan, 0: euclidean
REDUX - if redux method is used, 0: no, 1: yes
Select MU - election of MU, 1: arbitrary, 0:estimated
MU - value of MU if arbitrary
SAMPLE - 0: rand uniform, 1: lennard jones sample, 2: nbody
MOLECULES - for lennard jones sample {25,50,75,100,125,150,175,200}
DEBUG SIM - if compare against naive, 1: yes, 0: no
```

Example:
```
./test $((2**14)) $((2**10))    0 10   0    0 32 0    0 1 4    0 50 0
N: 16384, N_p(if used): 4096
MU: 4
Sample: 0
Comparison: 0
PBC: 0
Blocksize: 32
Method: 0
Distance used(1: Manhattan, 0:Euclidean): 0
k: 16384, k_m: 2048, k_r: 4096, k_rm: 512
TOTAL TIME REFERENCE: 0.000012, TOTAL TIME PROPOSAL: 3.043226
```



### 3D Voronoi
To compile the 3D version of the program, type the following
```
make cmp3d
```
To execute the program, is the same as before, run as ```test``` and provide the same arguments, be careful with memory consumption, for ```N``` the grid is going to be NxNxN

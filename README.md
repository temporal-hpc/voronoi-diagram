# voronoi-diagram
## Important
Tested on Arch Linux 6.8.9 (as for 02/06/24)
GCC: 14.1.1
CUDA: 12.4
Currently only supported on Linux and tested with the tools & versions mentioned before
## Instructions
### 2D Voronoi (default)
To compile the default version just type the following
```
make
```
If you want to save results use
```
make save
```
To execute, you can run as ```./test``` where you have to specify the following arguments.
- N: size of the grid (NxN), ex: 2000
- S: number of seeds, ex: 500 (be sure that S<=N)
- Mode: (0)JFA, (1)dJFA, (2)rJFA, (3)drJFA *Not useful actually, default ```0```, check ```Makefile```
- Iters/Steps: Total steps to do on simulation, ex: 100
- Device: GPU to use if multiple GPU's available *Not used, just pick the first GPU
- MEM GPU: Management of memory, (1) Manual and (0) Auto *Just use manual (default is ```1```), auto is actually not implemented
- Block size/BS: Size of the block for the GPU (BSxBS for 2D and BSxBSxBS in 3D), default is ```32``` in 2D
- Distance function: (0) Euclidean, (1) Manhattan
- Redux: If redux mode is used (i.e if you're using rJFA or drJFA), ```0``` for ```True``` and ```1``` for ```False```
- Mu: If redux is used, specify factor of reduction, ex: ```4``` (reduction means for subgrid N/Mu)
- Sample: Sample to use, (0) Random movement, (1) Lennard-Jones, (2) N-body, recommended 0 and 2
- Molecules: Only for Lennard Jones, specify amount of particles to use
- Debug simulation: If you want to compare against the precise result (brute force solution)

You can also check the ```test``` commands on the ```Makefile``` if you don't want to write every argument.
The command ```make pat test``` was intended to use in the server Patagon (patagon.uach.cl), but can be adapted to any server that uses ```SLURM```
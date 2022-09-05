#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include "utils.h"
#include "voronoi.h"

#ifdef DLSS
#include "../DLSS/include/nvsdk_ngx.h"
#include "../DLSS/include/nvsdk_ngx_defs.h"
#include "../DLSS/include/nvsdk_ngx_params.h"
#include "../DLSS/include/nvsdk_ngx_helpers.h"
#endif

#define ll long long

struct Setup{

    int N;
    int S;
    int mode;
    int iters;
    int device;
    int mem_gpu;
    int block_size;
    int distance_function;
    int redux;
    int seed;
    int mu;

    int *v_diagram;
    int *seeds;
    int *areas;

    int N_p;
    int *redux_v_diagram;

    int *gpu_v_diagram;
    int *gpu_seeds;
    int *gpu_redux_seeds;
    int *gpu_delta;
    int *gpu_delta_max;
    int *gpu_areas;

    int *gpu_redux_vd;

};

Setup initialize_variables(Setup setup, int N, int S, int mode, int iters, int device, int mem_gpu, int block_size, int distance_function, int redux,int mu){
    setup.N = N;
    setup.S = S;
    setup.mode = mode;
    setup.iters = iters;
    setup.device = device;
    setup.mem_gpu = mem_gpu;
    setup.block_size = block_size;
    setup.distance_function = distance_function;
    setup.redux = redux;
    setup.seed = 0;
    setup.mu = mu;
    setup.N_p = N/mu;
    return setup;
}

Setup allocate_arrays(Setup setup){
    setup.v_diagram = (int*)malloc(setup.N*setup.N*sizeof(int));
    setup.seeds = (int*)malloc(setup.S*sizeof(int));
    setup.areas = (int*)malloc(setup.S*sizeof(int));
    setup.redux_v_diagram = (int*)malloc(setup.N_p*setup.N_p*sizeof(int));
    return setup;
}
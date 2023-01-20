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
#include "nbody.cuh"

#define EPSILON 0.001
#define METHOD "dJFA"

const double G = 6.6743e-11;
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
    int sample;
    int molecules;
    int comparison;
    int pbc;
    int k;
    int k_m;
    int k_r;
    int k_rm;
    float L_avg;
    float rL_avg;

    int *v_diagram;
    int *seeds;
    int *areas;

    int N_p;
    int *redux_v_diagram;
    int *backup_v_diagram;

    int *gpu_v_diagram;
    int *gpu_seeds;
    int *gpu_redux_seeds;
    int *gpu_delta;
    int *gpu_delta_max;
    int *gpu_areas;

    int *gpu_redux_vd;
    int *gpu_backup_vd;

    dim3 normal_block;
    dim3 normal_grid;
    dim3 seeds_block;
    dim3 seeds_grid;
    dim3 redux_block;
    dim3 redux_grid;
    curandState *r_device;

    //Nbody params
    //double G = 6.6743e-11;  // gravitational constant
    double DT = 1.0;  // time step
    double M = 1.0; // particle mass
    double *seeds_vel;
    double *gpu_seeds_vel;

};

void initialize_variables(Setup *setup, int N, int S, int mode, int iters, int device, int mem_gpu, int block_size, int distance_function, int redux,int mu, int sample, int molecules, int comparison){
    setup->N = N;
    setup->S = S;
    setup->mode = mode;
    setup->iters = iters;
    setup->device = device;
    setup->mem_gpu = mem_gpu;
    setup->block_size = block_size;
    setup->distance_function = distance_function;
    setup->redux = redux;
    setup->seed = 0;
    setup->sample = sample;
    setup->molecules = molecules;
    setup->comparison = comparison;
    setup->pbc = (sample>=1)? 1 : 0;

    setup->mu = mu;
    setup->N_p = N/mu;
    setup->L_avg = sqrt(setup->N * setup->N / setup->S);
    setup->rL_avg = sqrt(setup->N_p * setup->N_p / setup->S);

    setup->k = pow(2,int(log2(setup->N)));
    setup->k_m = pow(2,int(log2(2 * setup->L_avg)) + 1);
    setup->k_r = pow(2,int(log2(setup->N_p)));
    setup->k_rm = pow(2, int(log2(2 * setup->rL_avg)) + 1);

    setup->normal_block = dim3 (setup->block_size,setup->block_size,1);
    setup->normal_grid = dim3 ((setup->N + setup->block_size + 1)/setup->block_size, (setup->N + setup->block_size + 1)/setup->block_size,1);
    setup->seeds_block = dim3 (setup->block_size,1,1);
    setup->seeds_grid = dim3 ((setup->S + setup->block_size + 1)/setup->block_size,1,1);
    setup->redux_block = dim3 (setup->block_size,setup->block_size,1);
    setup->redux_grid = dim3 ((setup->N_p + setup->block_size + 1)/setup->block_size, (setup->N_p + setup->block_size + 1)/setup->block_size,1);
}

void allocate_arrays(Setup *setup){
    setup->v_diagram = (int*)malloc(setup->N*setup->N*sizeof(int));
    setup->backup_v_diagram = (int*)malloc(setup->N*setup->N*sizeof(int));
    setup->seeds = (int*)malloc(setup->S*sizeof(int));
    setup->areas = (int*)malloc(setup->S*sizeof(int));
    setup->redux_v_diagram = (int*)malloc(setup->N_p*setup->N_p*sizeof(int));
    setup->seeds_vel = (double*)malloc(2*setup->S*sizeof(double));
    
    cudaMalloc((void**)&setup->r_device, setup->S * sizeof(curandState));
    cudaMalloc(&setup->gpu_v_diagram, setup->N*setup->N*sizeof(int));
    cudaMalloc(&setup->gpu_backup_vd, setup->N*setup->N*sizeof(int));
    cudaMalloc(&setup->gpu_redux_vd,setup->N_p*setup->N_p*sizeof(int));
    cudaMalloc(&setup->gpu_seeds, setup->S*sizeof(int));
    cudaMalloc(&setup->gpu_delta, setup->S*sizeof(int));
    cudaMalloc(&setup->gpu_seeds_vel, setup->S*2*sizeof(double));
}

void setDeviceInfo(Setup *setup){
    if(setup->sample==2){
        for (int it=0; it<setup->S; it++) {
            setup->seeds_vel[it*2] = 0.0;//double(rand()%10000)/10000.0;
            setup->seeds_vel[it*2+1] = 0.0;//double(rand()%10000)/2500.0;
        }
    }
    
    cudaMemcpy(setup->gpu_seeds_vel, setup->seeds_vel, 2*setup->S*sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaMemcpy(setup->gpu_seeds, setup->seeds, setup->S*sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void printRunInfo(Setup *setup){
    printf("MU: %i\n", setup->mu);
    printf("Sample: %i\n", setup->sample);
    printf("Comparison: %i\n", setup->comparison);
    printf("PBC: %i\n", setup->pbc);
    printf("Distance used(1: Manhattan, 0:Euclidean): %i\n", setup->distance_function);
    printf("k: %i, k_m: %i, k_r: %i, k_rm: %i\n", setup->k, setup->k_m, setup->k_r, setup->k_rm);
    if(setup->sample == 2){
        printf("G: %.12f\n", G);
        printf("DT: %f\n", setup->DT);
        printf("M: %f\n", setup->M);
    }
}

void seetSeeds(Setup *setup){
    if(setup->sample == 0)initSeeds(setup->seeds, setup->N, setup->S);
    else read_coords(setup->seeds, setup->N, setup->S, 0, setup->molecules);
}

void getDeviceArrays(Setup *setup){
    cudaMemcpy(setup->v_diagram, setup->gpu_v_diagram, setup->N*setup->N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemcpy(setup->backup_v_diagram, setup->gpu_backup_vd, setup->N*setup->N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemcpy(setup->redux_v_diagram, setup->gpu_redux_vd, setup->N_p*setup->N_p*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void freeSpace(Setup *setup){
    cudaFree(setup->gpu_v_diagram);
    cudaFree(setup->gpu_backup_vd);
    cudaFree(setup->gpu_redux_vd);
    cudaFree(setup->gpu_seeds);
    cudaFree(setup->gpu_seeds_vel);
    cudaFree(setup->r_device);

    free(setup->v_diagram);
    free(setup->backup_v_diagram);
    free(setup->redux_v_diagram);
    free(setup->seeds);
    free(setup->seeds_vel);
}
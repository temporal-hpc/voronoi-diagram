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
    int mode; // 0 - JFA, 1 - dJFA, 2 - rJFA, 3 - drJFA
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
    int k_m; // Write as d (delta as stated in paper)
    int k_r; // Write as d_r (delta as stated in paper)
    int k_rm; // Write as d_dr (delta as stated in paper)
    float L_avg;
    float rL_avg;

    float4 *v_diagram;
    int *seeds;
    int *areas;

    int N_p;
    float4 *redux_v_diagram;
    float4 *backup_v_diagram;

    float4 *gpu_v_diagram;
    int *gpu_seeds;
    int *gpu_redux_seeds;
    int *gpu_delta;
    int *gpu_delta_max;
    int *gpu_areas;

    float4 *gpu_redux_vd;
    float4 *gpu_backup_vd;

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
    setup->L_avg = sqrt(setup->N * setup->N / setup->S) * 1.414213;
    setup->rL_avg = sqrt(setup->N_p * setup->N_p / setup->S) * 1.414213;

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

void initializeVariables3D(Setup *setup, int N, int S, int mode, int iters, int device, int mem_gpu, int block_size, int distance_function, int redux,int mu, int sample, int molecules, int comparison){
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
    setup->L_avg =int(pow(setup->N * setup->N * setup->N / setup->S, 1.0/3.0) * 1.73205);
    setup->rL_avg = int(pow(setup->N_p * setup->N_p *setup->N_p / setup->S, 1.0/3.0) * 1.73205);

    setup->k = pow(2,int(log2(setup->N)));
    setup->k_m = pow(2,int(log2(2 * setup->L_avg)) + 1);
    setup->k_r = pow(2,int(log2(setup->N_p)));
    setup->k_rm = pow(2, int(log2(2 * setup->rL_avg)) + 1);

    // Try different size of blocksize for xyz
    setup->normal_block = dim3 (setup->block_size,setup->block_size/2,setup->block_size/2);
    setup->normal_grid = dim3 ((setup->N + setup->block_size + 1)/setup->block_size, (setup->N + (setup->block_size/2) + 1)/(setup->block_size/2),(setup->N + (setup->block_size/2) + 1)/(setup->block_size/2));
    setup->seeds_block = dim3 (setup->block_size,1,1);
    setup->seeds_grid = dim3 ((setup->S + setup->block_size + 1)/setup->block_size,1,1);
    setup->redux_block = dim3 (setup->block_size,setup->block_size/2,setup->block_size/2);
    setup->redux_grid = dim3 ((setup->N_p + setup->block_size + 1)/setup->block_size, (setup->N_p + setup->block_size + 1)/setup->block_size,(setup->N_p + setup->block_size + 1)/setup->block_size);
}

void allocate_arrays(Setup *setup){
    setup->v_diagram = (float4*)malloc(setup->N * setup->N * sizeof(float4));
    setup->backup_v_diagram = (float4*)malloc(setup->N * setup->N * sizeof(float4));
    setup->seeds = (int*)malloc(setup->S*sizeof(int));
    setup->areas = (int*)malloc(setup->S*sizeof(int));
    setup->redux_v_diagram = (float4*)malloc(setup->N_p * setup->N_p * sizeof(float4));
    setup->seeds_vel = (double*)malloc(2*setup->S*sizeof(double));
    
    cudaMalloc((void**)&setup->r_device, setup->S * sizeof(curandState));
    cudaMalloc(&setup->gpu_v_diagram, setup->N * setup->N * sizeof(float4));
    cudaMalloc(&setup->gpu_backup_vd, setup->N*setup->N*sizeof(float4));
    cudaMalloc(&setup->gpu_redux_vd,setup->N_p*setup->N_p*sizeof(float4));
    cudaMalloc(&setup->gpu_seeds, setup->S*sizeof(int));
    cudaMalloc(&setup->gpu_delta, setup->S*sizeof(int));
    cudaMalloc(&setup->gpu_seeds_vel, setup->S*2*sizeof(double));
}

void allocate_arrays3D(Setup *setup){
    setup->v_diagram = (float4*)malloc(setup->N * setup->N * setup->N * sizeof(float4));
    setup->backup_v_diagram = (float4*)malloc(setup->N * setup->N * setup->N * sizeof(float4));
    setup->seeds = (int*)malloc(setup->S*sizeof(int));
    setup->areas = (int*)malloc(setup->S*sizeof(int));
    setup->redux_v_diagram = (float4*)malloc(setup->N_p * setup->N_p * setup->N_p * sizeof(float4));
    setup->seeds_vel = (double*)malloc(3*setup->S*sizeof(double));
    
    cudaMalloc((void**)&setup->r_device, setup->S * sizeof(curandState));
    cudaMalloc(&setup->gpu_v_diagram, setup->N * setup->N * setup->N * sizeof(float4));
    cudaMalloc(&setup->gpu_backup_vd, setup->N * setup->N * setup->N * sizeof(float4));
    cudaMalloc(&setup->gpu_redux_vd, setup->N_p * setup->N_p * setup->N_p * sizeof(float4));
    cudaMalloc(&setup->gpu_seeds, setup->S*sizeof(int));
    cudaMalloc(&setup->gpu_delta, setup->S*sizeof(int));
    cudaMalloc(&setup->gpu_seeds_vel, setup->S*3*sizeof(double));
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

void setDeviceInfo3D(Setup *setup){
    if(setup->sample==2){
        for (int it=0; it<setup->S; it++) {
            setup->seeds_vel[it*3] = 0.0;//double(rand()%10000)/10000.0;
            setup->seeds_vel[it*3+1] = 0.0;//double(rand()%10000)/2500.0;
            setup->seeds_vel[it*3+2] = 0.0;//double(rand()%10000)/2500.0;
        }
    }
    
    cudaMemcpy(setup->gpu_seeds_vel, setup->seeds_vel, 3*setup->S*sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaMemcpy(setup->gpu_seeds, setup->seeds, setup->S*sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void printRunInfo(Setup *setup){
    printf("N: %i, N_p(if used): %i\n", setup->N, setup->N_p);
    printf("MU: %i\n", setup->mu);
    printf("Sample: %i\n", setup->sample);
    printf("Comparison: %i\n", setup->comparison);
    printf("PBC: %i\n", setup->pbc);
    printf("Blocksize: %i\n", setup->block_size);
    printf("Method: %i\n", setup->mode);
    printf("Distance used(1: Manhattan, 0:Euclidean): %i\n", setup->distance_function);
    printf("k: %i, k_m: %i, k_r: %i, k_rm: %i\n", setup->k, setup->k_m, setup->k_r, setup->k_rm);
    if(setup->sample == 2){
        printf("G: %.12f\n", G);
        printf("DT: %f\n", setup->DT);
        printf("M: %f\n", setup->M);
    }
}

void setSeeds(Setup *setup){
    if(setup->sample == 0)initSeeds(setup->seeds, setup->N, setup->S);
    else read_coords(setup->seeds, setup->N, setup->S, 0, setup->molecules);
}

void setSeeds3D(Setup *setup){
    if(setup->sample != 1) initSeeds3D(setup->seeds, setup->N, setup->S);
    else read_coords(setup->seeds, setup->N, setup->S, 0, setup->molecules);
}

void getDeviceArrays(Setup *setup){
    cudaMemcpy(setup->v_diagram, setup->gpu_v_diagram, setup->N*setup->N*sizeof(float4), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemcpy(setup->backup_v_diagram, setup->gpu_backup_vd, setup->N*setup->N*sizeof(float4), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemcpy(setup->redux_v_diagram, setup->gpu_redux_vd, setup->N_p*setup->N_p*sizeof(float4), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void getDeviceArrays3D(Setup *setup){
    cudaMemcpy(setup->v_diagram, setup->gpu_v_diagram, setup->N*setup->N*setup->N*sizeof(float4), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemcpy(setup->backup_v_diagram, setup->gpu_backup_vd, setup->N*setup->N*setup->N*sizeof(float4), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemcpy(setup->redux_v_diagram, setup->gpu_redux_vd, setup->N_p*setup->N_p*setup->N_p*sizeof(float4), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    if(setup->N <= 100){
        int id;
        for(int j = 0; j < setup->N; ++j){
            for(int i = 0; i < setup->N; ++i){
                id = 0 * setup->N * setup->N + j * setup->N + i;
                printf("%i ", int(setup->v_diagram[id].w));
            }
            printf("\n");
        }
        printf("\n");
        for(int j = 0; j < setup->N; ++j){
            for(int i = 0; i < setup->N; ++i){
                id = 0 * setup->N * setup->N + j * setup->N + i;
                printf("%i ", int(setup->backup_v_diagram[id].w));
            }
            printf("\n");
        }
    }
    
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

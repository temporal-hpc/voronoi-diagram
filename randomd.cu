#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <omp.h>
#include <iostream>
#include <curand.h>
#include <vector>
#include <algorithm>

__global__ void moveSeeds(int *SEEDS, int *deltax, int *deltay, int N, int S, int mod, curandState *states){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int seed = tid;
    int old_x, old_y, new_x, new_y, delta_x, delta_y;
    if(tid < S){
        curand_init(seed + mod, tid, 0, &states[tid]);

        old_x = SEEDS[tid]%N;
        old_y = SEEDS[tid]/N;

        delta_x = int(curand_uniform(&states[tid])*13.f)%13 - 6;
        delta_y = int(curand_uniform(&states[tid])*13.f)%13 - 6;

        deltax[tid] = delta_x;
        deltay[tid] = delta_y;

        new_x = old_x + delta_x;
        new_y = old_y + delta_y;
        if(new_x<0 || new_x>=N) new_x = old_x;
        if(new_y<0 || new_y>=N) new_y = old_y;

        SEEDS[tid] = new_y*N + new_x;
    }
}
/*
void initSeeds(int *SEEDS, int N, int S){
    int i;
    vector<int> POSSIBLE_SEEDS;
    srand(time(0));

    for(i = 0; i < N*N; ++i) POSSIBLE_SEEDS.push_back(i);

    random_shuffle(POSSIBLE_SEEDS.begin(), POSSIBLE_SEEDS.end());
    
    for(i = 0; i < S; ++i){
        SEEDS[i] = POSSIBLE_SEEDS[i];
        #ifdef DEBUG
            if(S <= 500 )printf("%i\n", SEEDS[i]);
        #endif
    }
}
*/
int main(int argc, char **argv){
    int N = atoi(argv[1]);
    int S = atoi(argv[2]);
    int ITER = atoi(argv[3]);
    int BS = atoi(argv[4]);

    int *seeds = (int*)malloc(S*sizeof(int));
    int *delta_x = (int*)malloc(S*sizeof(int));
    int *delta_y = (int*)malloc(S*sizeof(int));

    //initSeeds(seeds, N, S);
    for(int i = 0; i < S; ++i){
        seeds[i] = i;
    }
    int *GPU_seeds;
    int *GPU_deltax;
    int *GPU_deltay;
    cudaMalloc(&GPU_seeds, S*sizeof(int));
    cudaMalloc(&GPU_deltax, S*sizeof(int));
    cudaMalloc(&GPU_deltay, S*sizeof(int));

    dim3 block_seeds(BS*BS,1,1);
    dim3 grid_seeds((S + BS + 1)/BS,1,1);

    curandState *device;
    cudaMalloc((void**)&device, grid_seeds.x * grid_seeds.y * sizeof(curandState));

    cudaDeviceSynchronize();
    cudaMemcpy(GPU_seeds, seeds, S*sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    dim3 block(BS,1,1);
    dim3 grid((N + BS + 1)/BS,1,1);

    while(ITER > 0){
        moveSeeds<<<grid, block>>>(GPU_seeds, GPU_deltax, GPU_deltay, N, S, ITER, device);
        cudaDeviceSynchronize();
        cudaMemcpy(delta_x, GPU_deltax, S*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(delta_y, GPU_deltay, S*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(seeds, GPU_seeds, S*sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        for(int i = 0; i < S; ++i){
            printf("%i %i %i\n", delta_x[i], delta_y[i], seeds[i]);
        }
        printf("\n");
        ITER--;
    }
    cudaDeviceSynchronize();
    cudaFree(GPU_seeds);
    cudaFree(GPU_deltax);
    cudaFree(GPU_deltay);
    cudaFree(device);

}
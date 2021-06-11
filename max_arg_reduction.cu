#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>
/*
template <unsigned int blockSize>
__device__ void warp_reduce(volatile int *v_data, unsigned int tid){
    if(blockSize >= 64){
        if(v_data[tid] < v_data[tid + 32]) v_data[tid] = v_data[tid + 32];
    }
    if(blockSize >= 32){
        if(v_data[tid] < v_data[tid + 16]) v_data[tid] = v_data[tid + 16];
    }
    if(blockSize >= 16){
        if(v_data[tid] < v_data[tid + 8]) v_data[tid] = v_data[tid + 8];
    }
    if(blockSize >= 8){
        if(v_data[tid] < v_data[tid + 4]) v_data[tid] = v_data[tid + 4];
    }
    if(blockSize >= 4){
        if(v_data[tid] < v_data[tid + 2]) v_data[tid] = v_data[tid + 2];
    }
    if(blockSize >= 2){
        if(v_data[tid] < v_data[tid + 1]) v_data[tid] = v_data[tid + 1];
    }
}

template <unsigned int blockSize>
__global__ void reduce_max_arg(int *d_data, int *dmax, int n){
    extern __shared__ int sd_data[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize*2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sd_data[tid] = 0;

    while(i < n){
        if(d_data[tid] < d_data[tid + blockSize]) sd_data[tid] = d_data[tid + blockSize];
        else sd_data[tid] = d_data[tid];
        i+= gridSize;
        __syncthreads();
    }
    
    if(blockSize >= 512){
        if(tid < 256){
            if(sd_data[tid] < sd_data[tid + 256]) sd_data[tid] = sd_data[tid + 256]; 
        }
        __syncthreads();
    }

    if(blockSize >= 256){
        if(tid < 128){
            if(sd_data[tid] < sd_data[tid + 128]) sd_data[tid] = sd_data[tid + 128]; 
        }
        __syncthreads();
    }

    if(blockSize >= 128){
        if(tid < 64){
            if(sd_data[tid] < sd_data[tid + 64]) sd_data[tid] = sd_data[tid + 64]; 
        }
        __syncthreads();
    }

    if(tid < 32) warp_reduce(sd_data, tid);
    if(tid == 0) dmax[blockIdx.x] = sd_data[0];

}
*/
__device__ void reduce(volatile int *data, int tid){
    if(data[tid] < data[tid + 32]) data[tid] = data[tid + 32];
    if(data[tid] < data[tid + 16]) data[tid] = data[tid + 16];
    if(data[tid] < data[tid + 8]) data[tid] = data[tid + 8];
    if(data[tid] < data[tid + 4]) data[tid] = data[tid + 4];
    if(data[tid] < data[tid + 2]) data[tid] = data[tid + 2];
    if(data[tid] < data[tid + 1]) data[tid] = data[tid + 1];
}

/*__global__ void custom_max(int *data, int *max, int n){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int limit = n/2;
    while(limit > 0){
        if(data[tid] < data[tid + limit])
    }
}*/

__global__ void simple_max(int *data, int *max, unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    extern __shared__ int sd_data[];
    //printf("HERE\n");
    if(i < n){
        sd_data[tid] = data[i];
        if( data[i] < data[i + blockDim.x]) sd_data[tid] = data[i + blockDim.x];
    }
    __syncthreads();

    for(unsigned int s = blockDim.x/2; s>32; s>>1){
        if(tid < s){
            if(sd_data[tid] < sd_data[tid + s]) sd_data[tid] = sd_data[tid + s];
        }
        __syncthreads();
    }
    if(tid < 32) reduce(sd_data, tid);
    __syncthreads();
    if(tid == 0) max[0] = sd_data[0];
}

int main(int argc, char **argv){
    int size = atoi(argv[1]);
    int bs = atoi(argv[2]);
    int *host_data = (int*)malloc(sizeof(int) * size);
    int *dmax = (int*)malloc(sizeof(int)*5);
    double t1,t2;
    unsigned int n = 1000;

    for(int i = 0; i < size; ++i) host_data[i] = i%500;

    dim3 block(bs,1,1);
    dim3 grid((size + bs + 1)/bs,1,1);

    int *GPU_data;
    int *GPU_dmax;
    cudaMalloc(&GPU_data, size*sizeof(int));
    cudaMalloc(&GPU_dmax, sizeof(int)*5);
    cudaMemcpy(GPU_data, host_data, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    t1 = omp_get_wtime();
    //reduce_max_arg< bs><<< grid, block>>>(GPU_data, GPU_dmax, size);
    simple_max<<< grid, block>>>(GPU_data, GPU_dmax, size);
    cudaDeviceSynchronize();
    t2 = omp_get_wtime() - t1;

    cudaMemcpy(dmax, GPU_dmax, sizeof(int)*5, cudaMemcpyDeviceToHost);
    printf("MAX: %i\nTIME: %d\n", dmax[0], t2);

    cudaFree(GPU_data);
    cudaFree(GPU_dmax);

}

__device__ float euclideanDistance(int x1, int x2, int y1, int y2){
    return sqrtf(float((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)));
}

__device__ void check_neighbor(int *VD, int *S, int N, int local_value,int local_x, int local_y, int pixel_x, int pixel_y){
    
    //Values of the target pixel
    int pixel = pixel_y*N + pixel_x;

    //Values of rival that is already on the target
    int rival_value = VD[pixel];
    int rival_seed = S[rival_value];
    int rival_x;
    int rival_y;

    //Distances to compare
    float dist_local;
    float dist_rival; 

    if(local_value != rival_value && pixel!=rival_seed){

        rival_x = rival_seed%N;
        rival_y = rival_seed/N;

        dist_local = euclideanDistance(local_x, pixel_x, local_y, pixel_y);
        dist_rival = euclideanDistance(rival_x, pixel_x, rival_y, pixel_y);

        if(dist_local <= dist_rival) VD[pixel] = local_value;
    }
}

__global__ void initVD(int *VD, int N){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = tidy*N + tidx;
    if(tidx < N && tidy<N){
        VD[tid] = -1;
    }
}

__global__ void init_GPUSeeds(int *VD, int *SEEDS, int S){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int local_value;
    if(tid < S){
       local_value = SEEDS[tid];
       VD[local_value] = tid;
    }

}

//For classic approach of JFA
__global__ void voronoiJFA_8Ng(int *VD, int *S, int k, int N, int s){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = tidy*N + tidx;
    int local_k = k;
    //GRID[i] -> INDICE EN SEEDS
    //Values of the local seed
    int local_value;
    int local_seed;
    int local_x;
    int local_y;

    //Values of the target pixel
    int pixel;
    int pixel_x;
    int pixel_y;
    /*if(tid < s){
        VD[S[tid]] = tid;
    }*/
    if(tidx < N && tidy < N){
    /*
    1   2   3 -> first set of neighbors
    4   P   5 -> second set of neighbors
    6   7   8 -> third set of neighbors
    */
    //while(local_k>=1){
        //printf("%i\n", VD[tid]);
        local_value = VD[tid]; //-> Extraigo indice
        if(local_value!= -1){
            local_seed = S[local_value]; //valor de real de semilla
            local_x = local_seed%N;
            local_y = local_seed/N;
            //First set
            if(tidy - local_k >= 0){
                pixel_y = tidy - local_k;
                //First neighbor
                if(tidx - local_k >= 0){
                    pixel_x = tidx - local_k;
                    pixel = pixel_y * N + pixel_x;
                    if(VD[pixel] == -1) VD[pixel] = local_value;
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y);
                }    
                //Second neighbor
                pixel_x = tidx;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y);
                //Third neighbor
                if(tidx + local_k < N){
                    pixel_x = tidx + local_k;
                    pixel = pixel_y * N + pixel_x;
                    if(VD[pixel] == -1) VD[pixel] = local_value;
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y);
                }
            }

            //Second set
            //Forth neighbor
            pixel_y = tidy;
            if(tidx - local_k >= 0){
                pixel_x = tidx - local_k;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y);
            }
            //Fifth neighbor
            if(tidx + local_k < N){
                pixel_x = tidx + local_k;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y);
            }

            //Third set
            if(tidy + local_k < N){
                pixel_y = tidy + local_k;
                //Sixth neighbor
                if(tidx - local_k >= 0){
                    pixel_x = tidx - local_k;
                    pixel = pixel_y * N + pixel_x;
                    if(VD[pixel] == -1) VD[pixel] = local_value;
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y);
                }    
                //Seventh neighbor
                pixel_x = tidx;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y);
                //Eighth neighbor
                if(tidx + local_k < N){
                    pixel_x = tidx + local_k;
                    pixel = pixel_y * N + pixel_x;
                    if(VD[pixel] == -1) VD[pixel] = local_value;
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y);
                }
            }
        //__syncthreads();
        //local_k=local_k/2;
        //__syncthreads();
        }
    }
}

//For dynamic approach of JFA*
__global__ void voronoiJFA_4Ng(int *VD, int *S, int *MAX,int k, int N, int s){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = tidy*N + tidx;
    int kmod = k;

    //Values of the local seed
    int local_value;
    int local_seed;
    int local_x;
    int local_y;
    __syncthreads();
    /* Von Neumann neighborhood
        1
    2   P   3
        4
    */
    //while(local_k>=1){
    if(tid < s){
        VD[S[tid]] = tid;
    }   
    //printf("%i\n", MAX[0]); 
    __syncthreads();
    //while(kmod>=1){
    //for(unsigned int kmod = MAX[0]; kmod>=1; kmod>>1){
        if(tidx < N && tidy < N){// && VD[tid]!=0){
            local_value = VD[tid];
            local_seed = S[local_value];
            local_x = local_seed%N;
            local_y = local_seed/N;
            //First neighbor
            if(tidy - kmod >= 0){
                check_neighbor(VD, S, N, local_value, local_x, local_y, tidx, tidy-kmod);
            }
            //Second neighbor
            if(tidx - kmod >= 0){
                check_neighbor(VD, S, N, local_value, local_x, local_y, tidx - kmod, tidy);
            }
            //Third neighbor
            if(tidx + kmod < N){
                check_neighbor(VD, S, N, local_value, local_x, local_y, tidx + kmod, tidy);
            }
            //Forth neighbor
            if(tidy + kmod < N){
                check_neighbor(VD, S, N, local_value, local_x, local_y, tidx, tidy + kmod);
            }
        }
        //__syncthreads();
        //kmod = kmod/2;
        //__syncthreads();
    //}
    
        //local_k = local_k/2;
    //}    
}

__global__ void moveSeeds(int *SEEDS, int *DELTA, int N, int S, int mod, curandState *states){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int seed = tid;
    int old_x, old_y, new_x, new_y, delta_x, delta_y;
    if(tid < S){
        curand_init(seed + mod, tid, 0, &states[tid]);

        old_x = SEEDS[tid]%N;
        old_y = SEEDS[tid]/N;

        delta_x = int(curand_uniform(&states[tid])*13.f)%13 - 6;
        delta_y = int(curand_uniform(&states[tid])*13.f)%13 - 6;

        new_x = old_x + delta_x;
        new_y = old_y + delta_y;
        
        if(new_x<0 || new_x>=N) new_x = old_x;
        if(new_y<0 || new_y>=N) new_y = old_y;
        DELTA[tid] = int(sqrtf(pow(new_x - old_x,2) + pow(new_y - old_y,2)));
        SEEDS[tid] = new_y*N + new_x;
    }
}

__global__ void lastcheck(int *VD, int *S, int s){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < s){
        VD[S[tid]] = -1;
    }
}

float euclideanDistanceCPU(int x1, int x2, int y1, int y2){
    return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}


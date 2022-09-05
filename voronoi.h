__device__ float euclideanDistance(int x1, int x2, int y1, int y2){
    return sqrtf(float((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)));
}

__device__ float manhattanDistance(int x1, int x2, int y1, int y2){
    return  fabsf(float(x1 - x2)) + fabsf(float(y1 - y2));
}

__device__ void check_neighbor(int *VD, int *S, int N, int local_value,int local_x, int local_y, int pixel_x, int pixel_y, int DIST, int mu){
    
    //Values of the target pixel
    int pixel = pixel_y*N + pixel_x;

    //Values of rival that is already on the target
    int rival_value = VD[pixel];
    int rival_seed = S[rival_value];
    int rival_x;
    int rival_y;
    //int og_pixel_x = pixel_x;
    //int og_pixel_y = pixel_x;

    //Distances to compare
    float dist_local;
    float dist_rival; 
    
    //pixel_x = pixel_x;
    //pixel_y = pixel_y;
    pixel = pixel_y*N + pixel_x;

    if(local_value != rival_value && pixel!=rival_seed){
        //pixel_x = pixel_x * mu;
        //pixel_y = pixel_y * mu;
        rival_x = rival_seed%(N*mu);
        rival_y = rival_seed/(N*mu);

	    //dist_local = euclideanDistance(local_x, pixel_x, local_y, pixel_y);
        //dist_rival = euclideanDistance(rival_x, pixel_x, rival_y, pixel_y);
        if(DIST == 0){
                dist_local = euclideanDistance(local_x, pixel_x*mu, local_y, pixel_y*mu);
                dist_rival = euclideanDistance(rival_x, pixel_x*mu, rival_y, pixel_y*mu);
        }
        else{
            dist_local = manhattanDistance(local_x, pixel_x*mu, local_y, pixel_y*mu);
            dist_rival = manhattanDistance(rival_x, pixel_x*mu, rival_y, pixel_y*mu);
        }

        if(dist_local <= dist_rival){
            //pixel_x = pixel_x;
            //pixel_y = pixel_y;
            //pixel = pixel_y*N + pixel_x;
            VD[pixel] = local_value;
        }
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
__global__ void voronoiJFA_8Ng(int *VD, int *S, int k, int N, int s, int DIST, int mu){
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
    //local_value = VD[tid];
    if(tidx < N && tidy < N ){
        /*
        1 .. 2 .. 3 -> first set of neighbors
        .
        .
        4 .. P .. 5 -> second set of neighbors
        .
        .
        6 .. 7 .. 8 -> third set of neighbors
        */
        local_value = VD[tid]; //-> Extraigo indice
        if(local_value!= -1){
            local_seed = S[local_value]; //valor de real de semilla
            local_x = local_seed%(N*mu);
            local_y = local_seed/(N*mu);
            /**/
            //First set
            if((tidx - local_k) >= 0){
                pixel_x = tidx - local_k;
                //First neighbor
                if(tidy - local_k >= 0){
                    pixel_y = tidy - local_k;
                    pixel = pixel_y * N + pixel_x;
                    if(VD[pixel] == -1) VD[pixel] = local_value;
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu);
                }    
                //Second neighbor
                pixel_y = tidy;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu);
                //Third neighbor
                if(tidy + local_k < N){
                    pixel_y = tidy + local_k;
                    pixel = pixel_y * N + pixel_x;
                    if(VD[pixel] == -1) VD[pixel] = local_value;
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu);
                }
            }
            /**/
            //Second set
            //Forth neighbor
            pixel_x = tidx;
            if(tidy - local_k >= 0){
                pixel_y = tidy - local_k;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu);
            }
            //Fifth neighbor
            if(tidy + local_k < N){
                pixel_y = tidy + local_k;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu);
            }
            /**/
            //Third set
            if(tidx + local_k < N){
                pixel_x = tidx + local_k;
                //Sixth neighbor
                if(tidy - local_k >= 0){
                    pixel_y = tidy - local_k;
                    pixel = pixel_y * N + pixel_x;
                    if(VD[pixel] == -1) VD[pixel] = local_value;
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu);
                }    
                //Seventh neighbor
                pixel_y = tidy;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu);
                //Eighth neighbor
                if(tidy + local_k < N){
                    pixel_y = tidy + local_k;
                    pixel = pixel_y * N + pixel_x;
                    if(VD[pixel] == -1) VD[pixel] = local_value;
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu);
                }
            }
        }
    }
}

__global__ void voronoiJFA_8NgV21(int *VD, int *S, int k, int N, int s, int DIST, int mu){
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
        local_value = VD[tid]; //-> Extraigo indice
        if(local_value!= -1){
            local_seed = S[local_value]; //valor de real de semilla
            //Scale to true N
            local_x = local_seed%(N*mu);
            local_y = local_seed/(N*mu);
            /**/
            //First set
            if((tidy - local_k) >= 0){
                pixel_y = tidy - local_k;
                //First neighbor
                if(tidx - local_k >= 0){
                    pixel_x = tidx - local_k;
                    pixel = pixel_y * N + pixel_x;
                    if(VD[pixel] == -1) VD[pixel] = local_value;
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu);
                }    
                //Second neighbor
                pixel_x = tidx;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu);
                //Third neighbor
                if(tidx + local_k < N){
                    pixel_x = tidx + local_k;
                    pixel = pixel_y * N + pixel_x;
                    if(VD[pixel] == -1) VD[pixel] = local_value;
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu);
                }
            }
            /**/
            /**/
            /**/
        //__syncthreads();
        //local_k=local_k/2;
        //__syncthreads();
        }
    }
}

__global__ void voronoiJFA_8NgV22(int *VD, int *S, int k, int N, int s, int DIST, int mu){
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
        local_value = VD[tid]; //-> Extraigo indice
        if(local_value!= -1){
            local_seed = S[local_value]; //valor de real de semilla
            local_x = local_seed%(N*mu);
            local_y = local_seed/(N*mu);
            /**/
            //Second set
            //Forth neighbor
            pixel_y = tidy;
            if(tidx - local_k >= 0){
                pixel_x = tidx - local_k;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu);
            }
            //Fifth neighbor
            if(tidx + local_k < N){
                pixel_x = tidx + local_k;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu);
            }
            /**/
            /**/
            /**/
            //__syncthreads();
            //local_k=local_k/2;
            //__syncthreads();
        }
    }
}

__global__ void voronoiJFA_8NgV23(int *VD, int *S, int k, int N, int s, int DIST, int mu){
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
        local_value = VD[tid]; //-> Extraigo indice
        if(local_value!= -1){
            local_seed = S[local_value]; //valor de real de semilla
            local_x = local_seed%(N*mu);
            local_y = local_seed/(N*mu);
            /**/
            //Third set
            if(tidy + local_k < N){
                pixel_y = tidy + local_k;
                //Sixth neighbor
                if(tidx - local_k >= 0){
                    pixel_x = tidx - local_k;
                    pixel = pixel_y * N + pixel_x;
                    if(VD[pixel] == -1) VD[pixel] = local_value;
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu);
                }    
                //Seventh neighbor
                pixel_x = tidx;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu);
                //Eighth neighbor
                if(tidx + local_k < N){
                    pixel_x = tidx + local_k;
                    pixel = pixel_y * N + pixel_x;
                    if(VD[pixel] == -1) VD[pixel] = local_value;
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu);
                }
            }
            /**/
            /**/
            /**/
        //__syncthreads();
        //local_k=local_k/2;
        //__syncthreads();
        }
    }
}

//For dynamic approach of JFA*
__global__ void voronoiJFA_4Ng(int *VD, int *S, int *MAX,int k, int N, int s, int DIST, int mu){
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
        local_x = local_seed%(N*mu);
        local_y = local_seed/(N*mu);
        //First neighbor
        if(tidy - kmod >= 0){
            check_neighbor(VD, S, N, local_value, local_x, local_y, tidx, (tidy-kmod), DIST, mu);
        }
        //Second neighbor
        if(tidx - kmod >= 0){
            check_neighbor(VD, S, N, local_value, local_x, local_y, (tidx - kmod), tidy, DIST, mu);
        }
        //Third neighbor
        if(tidx + kmod < N){
            check_neighbor(VD, S, N, local_value, local_x, local_y, (tidx + kmod), tidy, DIST, mu);
        }
        //Forth neighbor
        if(tidy + kmod < N){
            check_neighbor(VD, S, N, local_value, local_x, local_y, tidx, (tidy + kmod), DIST, mu);
        }
    }
    //__syncthreads();
    //kmod = kmod/2;
    //__syncthreads();
    //}
    
        //local_k = local_k/2;
    //}    
}

__global__ void init_rand(int S, int mod, curandState *states){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int seed = tid;
    if(tid < S) curand_init(1, tid, 0, &states[tid]);
}

__global__ void moveSeeds(int *SEEDS, int *DELTA, int N, int S, int mod, curandState *states){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int seed = tid;
    int old_x, old_y, new_x, new_y, delta_x, delta_y, ref_seed;
    int old_vd;
    unsigned long long int old_ref=0,old=0,result;
    __syncthreads(); 

    if(tid < S){
        DELTA[tid] = 0;
        //curand_init(seed + mod, tid, 0, &states[tid]);

        old_x = SEEDS[tid]%N;
        old_y = SEEDS[tid]/N;

        delta_x = int(curand_uniform(&states[tid])*13.f)%13 - 6;
        delta_y = int(curand_uniform(&states[tid])*13.f)%13 - 6;

        new_x = old_x + delta_x;
        new_y = old_y + delta_y;

        if(new_x<0 || new_x>=N) new_x = old_x;
        if(new_y<0 || new_y>=N) new_y = old_y;

        //

        //old = GPU_VD[new_y*N + new_x];
        //old_ref = old;
        //result = atomicCAS(&old,old_ref,tid);
        // if result == old_ref then change
        //if(){
            DELTA[tid] = int(sqrtf(pow(new_x - old_x,2) + pow(new_y - old_y,2)));
            SEEDS[tid] = new_y*N + new_x;
        //} 
    }
    
}

//Proof of concept
__global__ void reduxVoronoi(int *VD, int *redux_VD, int N, int n, int mu){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = tidy * n + tidx;
    int stride_x = mu * tidx + mu / 2;
    int stride_y = tidy * mu * N + mu/2 * N;
    if(tidx < N && tidy < N){
        //Arbitrary
        redux_VD[tid] = VD[stride_y + stride_x];
    }
}

//Proof of concept
__global__ void voronoiElection(int *VD, int *redux_VD, int N, int n){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = tidy * n + tidx;
    const int mu = 4;
    __shared__ int voronoi_space[mu*mu];
    //Revisar
    for(int j = tidy * mu; j < (tidy + 1) * mu; ++j){
        for(int i = tidx * mu; i < (tidx+1) * mu; ++i){
            voronoi_space[j%(mu*mu) + i%mu] = VD[j * N + i];
        }
    }
    //
}

//Proof of concept
__global__ void voronoiElection2(int *VD, int *redux_VD, int N, int n, int mu){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = tidy * n + tidx;
    int elect = -1;
    int votes = 0;
    int vote;
    //Revisar
    if(tidx < N && tidy < N){
        for(int j = tidy * mu; j < (tidy + 1) * mu; ++j){
            for(int i = tidx * mu; i < (tidx+1) * mu; ++i){
                vote = VD[j * N + i];
                if(vote != elect) votes -=1;
                else votes += 1;
                if(votes <= 0){
                    elect = VD[j * N + i];
                    votes = 1;
                }
                if(votes >= mu*mu/3) break;
            }
        }

        redux_VD[tid] = elect;
    }
    //
}

//If redux is used before JFA and n*n > S
__global__ void reduxVDSeeds(int *REDUX_VD, int *seeds, int N, int S, int Np, int mu){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int local_value, local_x, local_y, redux_x, redux_y, redux_coord;
    if(tid < S){
        local_value = seeds[tid];
        local_x = local_value%N;
        local_y = local_value/N;
        redux_x = local_x / mu;
        redux_y = local_y / mu;
        redux_coord = redux_y * Np + redux_x;
        REDUX_VD[redux_coord] = tid;
    }
}

__global__ void lastcheck(int *VD, int *SEEDS, int s){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < s){
        VD[SEEDS[tid]] = -1;
    }
}

__global__ void scaleVoronoi(int *VD,int *REDUX_VD, int N, int n, int mu){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = tidy * N + tidx;

    int redux_x, redux_y, redux_coord, local_value;

    if(tidx < N && tidy < N){
        redux_x = tidx / mu;
        redux_y = tidy / mu;
        redux_coord = redux_y * n + redux_x;
        VD[tid] = REDUX_VD[redux_coord];
    }
}

__global__ void clearGrid(int *VD, int *SEEDS, int N, int S){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = tidy * N + tidx;
    if(tidx < N && tidy < N){
        VD[tid] = -1;
    }
    __syncthreads();
    //if(tid < S){
    //    VD[SEEDS[tid]] = tid;
    //}
}

__global__ void scaleSeeds(int *true_seeds, int *redux_seeds, int S, int N, int N_p,int mu){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int true_seed, true_x, true_y, redux_seed, redux_y, redux_x;
    if(tid < S){
        true_seed = true_seeds[tid];
        true_x = true_seed%N;
        true_y = true_seed/N;
        redux_x = true_x/mu;
        redux_y = true_y/mu;
        redux_seed = redux_y * N_p + redux_x;
        redux_seeds[tid] = redux_seed;
    }
}


float euclideanDistanceCPU(int x1, int x2, int y1, int y2){
    return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

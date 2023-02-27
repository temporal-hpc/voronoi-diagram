#include "setup.h"

__device__ float euclideanDistance(int x1, int x2, int y1, int y2){
    return sqrtf(float((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)));
}

__device__ float manhattanDistance(int x1, int x2, int y1, int y2){
    return  fabsf(float(x1 - x2)) + fabsf(float(y1 - y2));
}

__device__ float pbcEuclideanDistance(int N, int x1, int x2, int y1, int y2){
    int tmp_x = x1 - x2;
    int tmp_y = y1 - y2;

    if(fabsf(float(tmp_x - N)) < fabsf(float(tmp_x))) tmp_x -= N;
    else if(fabsf(float(tmp_x + N)) < fabsf(float(tmp_x))) tmp_x += N;

    if(fabsf(float(tmp_y - N)) < fabsf(float(tmp_y))) tmp_y -= N;
    else if(fabsf(float(tmp_y + N)) < fabsf(float(tmp_y))) tmp_y += N;

    return sqrtf(float((tmp_x)*(tmp_x) + (tmp_y)*(tmp_y)));
}

__device__ float pbcManhattanDistance(int N, int x1, int x2, int y1, int y2){
    int tmp_x = x1 - x2;
    int tmp_y = y1 - y2;

    if(fabsf(tmp_x - N) < fabsf(tmp_x)) tmp_x -= N;
    else if(fabsf(tmp_x + N) < fabsf(tmp_x)) tmp_x += N;

    if(fabsf(tmp_y - N) < fabsf(tmp_y)) tmp_y -= N;
    else if(fabsf(tmp_y + N) < fabsf(tmp_y)) tmp_y += N;

    return  fabsf(float(tmp_x)) + fabsf(float(tmp_y));
}

__device__ void check_neighbor(int *VD, int *S, int N, int local_value,int local_x, int local_y, int pixel_x, int pixel_y, int DIST, int mu, int isPBC){
    
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
                dist_local = (isPBC == 1)? pbcEuclideanDistance(N*mu, local_x, pixel_x*mu, local_y, pixel_y*mu): euclideanDistance(local_x, pixel_x*mu, local_y, pixel_y*mu);
                dist_rival = (isPBC == 1)? pbcEuclideanDistance(N*mu, rival_x, pixel_x*mu, rival_y, pixel_y*mu): euclideanDistance(rival_x, pixel_x*mu, rival_y, pixel_y*mu);
        }
        else{
            dist_local = (isPBC == 1)? pbcManhattanDistance(N*mu, local_x, pixel_x*mu, local_y, pixel_y*mu): manhattanDistance(local_x, pixel_x*mu, local_y, pixel_y*mu);
            dist_rival = (isPBC == 1)? pbcManhattanDistance(N*mu, rival_x, pixel_x*mu, rival_y, pixel_y*mu): manhattanDistance(rival_x, pixel_x*mu, rival_y, pixel_y*mu);
        }

        if(dist_local < dist_rival){
            //pixel_x = pixel_x;
            //pixel_y = pixel_y;
            //pixel = pixel_y*N + pixel_x;
            VD[pixel] = local_value;
        }
    }
}

__global__ void bruteReference(int *VD, int *S, int N, int s, int DIST, int isPBC){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = tidy*N + tidx;
    if(tidx >= N || tidy >= N ) return;
    
    int close_index = -1,closest = -1, local_x, local_y, ref_x, ref_y;
    float d_local, d_ref;
    
    
    for(int i = 0; i < s; ++i){
        if( closest == -1 ){
            closest = S[i];
            close_index = i;
        }
        else{
            ref_x = S[i]%N;
            ref_y = S[i]/N;
            local_x = closest%N;
            local_y = closest/N;
            d_local = (isPBC == 1)? pbcEuclideanDistance(N, tidx, local_x, tidy, local_y): euclideanDistance(tidx, local_x, tidy, local_y);
            d_ref = (isPBC == 1)? pbcEuclideanDistance(N, tidx, ref_x, tidy, ref_y): euclideanDistance(tidx, ref_x, tidy, ref_y);
            
            if(d_ref < d_local){
                closest = S[i];
                close_index = i;
            }
        }
    }

    VD[tid] = close_index;
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
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 0);
                }    
                //Second neighbor
                pixel_y = tidy;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 0);
                //Third neighbor
                if(tidy + local_k < N){
                    pixel_y = tidy + local_k;
                    pixel = pixel_y * N + pixel_x;
                    if(VD[pixel] == -1) VD[pixel] = local_value;
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 0);
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
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 0);
            }
            //Fifth neighbor
            if(tidy + local_k < N){
                pixel_y = tidy + local_k;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 0);
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
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 0);
                }    
                //Seventh neighbor
                pixel_y = tidy;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 0);
                //Eighth neighbor
                if(tidy + local_k < N){
                    pixel_y = tidy + local_k;
                    pixel = pixel_y * N + pixel_x;
                    if(VD[pixel] == -1) VD[pixel] = local_value;
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 0);
                }
            }
        }
    }
}

__global__ void pbcVoronoiJFA_8Ng(int *VD, int *S, int k, int N, int s, int DIST, int mu){
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

    if(tidx < N && tidy < N ){
        local_value = VD[tid];
        if(local_value!=-1){
            local_seed = S[local_value];
            local_x = local_seed%(N*mu);
            local_y = local_seed/(N*mu);
            //First neighbor
            pixel_x = (tidx - k + N)%N;
            pixel_y = (tidy - k + N)%N;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
            //Second neighbor
            pixel_y = tidy;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
            //Third neighbor
            pixel_y = (tidy + k)%N;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
            //Fourth neighbor
            pixel_x = tidx;
            pixel_y = (tidy - k + N)%N;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
            //Fifth neighbor
            pixel_y = (tidy + k)%N;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
            //Sixth neighbor
            pixel_x = (tidx + k)%N;
            pixel_y = (tidy - k + N)%N;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
            //Seventh neighbor
            pixel_y = tidy;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
            //Eigth neighbor
            pixel_y = (tidy + k)%N;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
            
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
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 0);
                }    
                //Second neighbor
                pixel_x = tidx;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 0);
                //Third neighbor
                if(tidx + local_k < N){
                    pixel_x = tidx + local_k;
                    pixel = pixel_y * N + pixel_x;
                    if(VD[pixel] == -1) VD[pixel] = local_value;
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 0);
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
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 0);
            }
            //Fifth neighbor
            if(tidx + local_k < N){
                pixel_x = tidx + local_k;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 0);
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
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 0);
                }    
                //Seventh neighbor
                pixel_x = tidx;
                pixel = pixel_y * N + pixel_x;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 0);
                //Eighth neighbor
                if(tidx + local_k < N){
                    pixel_x = tidx + local_k;
                    pixel = pixel_y * N + pixel_x;
                    if(VD[pixel] == -1) VD[pixel] = local_value;
                    else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 0);
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

__global__ void pbcVoronoiJFA_8NgV21(int *VD, int *S,int k, int N, int s, int DIST, int mu){
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

    if(tidx < N && tidy < N ){
        local_value = VD[tid];
        if(local_value!=-1){
            local_seed = S[local_value];
            local_x = local_seed%(N*mu);
            local_y = local_seed/(N*mu);
            //First neighbor
            pixel_x = (tidx - local_k + N)%N;
            pixel_y = (tidy - local_k + N)%N;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
            //Second neighbor
            pixel_x = tidx;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
            //Third neighbor
            pixel_x = (tidx + local_k)%N;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
        }
    }
}

__global__ void pbcVoronoiJFA_8NgV22(int *VD, int *S,int k, int N, int s, int DIST, int mu){
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

    if(tidx < N && tidy < N ){
        local_value = VD[tid];
        if(local_value!=-1){
            local_seed = S[local_value];
            local_x = local_seed%(N*mu);
            local_y = local_seed/(N*mu);
            //Fourth neighbor
            pixel_y = tidy;
            pixel_x = (tidx - local_k + N)%N;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
            //Fifth neighbor
            pixel_x = (tidx + local_k)%N;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
        }
    }
}

__global__ void pbcVoronoiJFA_8NgV23(int *VD, int *S,int k, int N, int s, int DIST, int mu){
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

    if(tidx < N && tidy < N ){
        local_value = VD[tid];
        if(local_value!=-1){
            local_seed = S[local_value];
            local_x = local_seed%(N*mu);
            local_y = local_seed/(N*mu);
            //Sixth neighbor
            pixel_y = (tidy + local_k)%N;
            pixel_x = (tidx - local_k + N)%N;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
            //Seventh neighbor
            pixel_x = tidx;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
            //Eigth neighbor
            pixel_x = (tidx + local_k)%N;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
            
        }
    }
}

//Clear points above L_avg
__global__ void clearDistantPixels(int *VD, int *SEEDS,int N, int S, float L_avg, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = tidy*N + tidx;
    int local_seed, local_x, local_y;
    float dist;
    if(tidx < N && tidy < N){
        local_seed = SEEDS[VD[tid]];
        local_x = local_seed%(N*mu);
        local_y = local_seed/(N*mu);
        if(pbc == 1) dist = pbcEuclideanDistance(N*mu, tidx*mu, local_x, tidy*mu, local_y);
        else dist = euclideanDistance(tidx*mu, local_x, tidy*mu, local_y);
        if(dist >= L_avg) VD[tid] = -1;
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
    int pixel;
    __syncthreads();
    /* Von Neumann neighborhood
        1
    2   P   3
        4
    */
    //while(local_k>=1){
    if(tid < s && mu==1){
        VD[S[tid]] = tid;
    }   
    //printf("%i\n", MAX[0]); 
    __syncthreads();
    //while(kmod>=1){
    //for(unsigned int kmod = MAX[0]; kmod>=1; kmod>>1){
    if(tidx < N && tidy < N){// && VD[tid]!=0){
        local_value = VD[tid];
        if(local_value!= -1){
            local_seed = S[local_value];
            local_x = local_seed%(N*mu);
            local_y = local_seed/(N*mu);
            //First neighbor
            if(tidy - kmod >= 0){
                pixel = (tidy-kmod)*N + tidx;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                check_neighbor(VD, S, N, local_value, local_x, local_y, tidx, (tidy-kmod), DIST, mu, 0);
            }
            //Second neighbor
            if(tidx - kmod >= 0){
                pixel = tidy*N + (tidx - kmod);
                if(VD[pixel] == -1) VD[pixel] = local_value;
                check_neighbor(VD, S, N, local_value, local_x, local_y, (tidx - kmod), tidy, DIST, mu, 0);
            }
            //Third neighbor
            if(tidx + kmod < N){
                pixel = tidy*N + (tidx + kmod);
                if(VD[pixel] == -1) VD[pixel] = local_value;
                check_neighbor(VD, S, N, local_value, local_x, local_y, (tidx + kmod), tidy, DIST, mu, 0);
            }
            //Forth neighbor
            if(tidy + kmod < N){
                pixel = (tidy+kmod)*N + tidx;
                if(VD[pixel] == -1) VD[pixel] = local_value;
                check_neighbor(VD, S, N, local_value, local_x, local_y, tidx, (tidy + kmod), DIST, mu, 0);
            }
        }
        
    }
    //__syncthreads();
    //kmod = kmod/2;
    //__syncthreads();
    //}
    
        //local_k = local_k/2;
    //}    
}

__global__ void pbcVoronoiJFA_4Ng(int *VD, int *S,int k, int N, int s, int DIST, int mu){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = tidy*N + tidx;
    int local_k = k;

    //Values of the local seed
    int local_value;
    int local_seed;
    int local_x;
    int local_y;
    int pixel;
    int pixel_x;
    int pixel_y;
    __syncthreads();
    /* Von Neumann neighborhood
        1
    2   P   3
        4
    */
    //while(local_k>=1){
    if(tid < s && mu==1){
        VD[S[tid]] = tid;
    }   
    //printf("%i\n", MAX[0]); 
    __syncthreads();
    //while(kmod>=1){
    //for(unsigned int kmod = MAX[0]; kmod>=1; kmod>>1){
    if(tidx < N && tidy < N){
        local_value = VD[tid];
        if(local_value!= -1){
            local_seed = S[local_value];
            local_x = local_seed%(N*mu);
            local_y = local_seed/(N*mu);
            //First neighbor
            pixel_y = (tidy - local_k + N)%N;
            pixel_x = tidx;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
            //Second neighbor
            pixel_y = tidy;
            pixel_x = (tidx - local_k + N)%N;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
            //Third neighbor
            pixel_y = tidy;
            pixel_x = (tidx + local_k)%N;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
            //Fourth neighbor
            pixel_y = (tidy + local_k)%N;
            pixel_x = tidx;
            pixel = pixel_y * N + pixel_x;
            if(VD[pixel] == -1) VD[pixel] = local_value;
            else check_neighbor(VD, S, N, local_value, local_x, local_y, pixel_x, pixel_y, DIST, mu, 1);
        }
    }
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

void setRandDevice(Setup *setup){
    init_rand<<<setup->seeds_grid, setup->seeds_block>>>(setup->S, setup->N, setup->r_device);
    cudaDeviceSynchronize();
}

float euclideanDistanceCPU(int x1, int x2, int y1, int y2){
    return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

void dynamicSeeds(Setup setup, int iter){
    if(setup.sample == 0){
        moveSeeds<<<setup.seeds_grid, setup.seeds_block>>>(setup.gpu_seeds, setup.gpu_delta,setup.N, setup.S, 1, setup.r_device);
        cudaDeviceSynchronize();
        if(cudaGetLastError() != cudaSuccess){
                printf("Something went wrong on N-body\n");
                exit(0);
        }
    }
    else if (setup.sample == 1){
        read_coords(setup.seeds, setup.N, setup.S, 100-iter+1, setup.molecules);
        cudaMemcpy(setup.gpu_seeds, setup.seeds, sizeof(int)*setup.S, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
    else if(setup.sample == 2){
        for(int unb = 0; unb < 5; ++unb){
            updateNBodies<<< setup.seeds_grid, setup.seeds_block >>>(setup.gpu_seeds, setup.gpu_seeds_vel, setup.N, setup.S, setup.DT, G, setup.M);
            cudaDeviceSynchronize();
            if(cudaGetLastError() != cudaSuccess){
                printf("Something went wrong on N-body\n");
                exit(0);
            }
        }
        
    }
}

// This method needs to be check due to posible concurrence errors
void jfaVDIters(Setup setup,int *v_diagram, int *seeds, int N, int S, int k, int mu, dim3 grid, dim3 block, int pbc){
    int k_copy = k;
    if(pbc == 0){
        while(k_copy >= 1){
            voronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, k_copy, N, S, setup.distance_function, mu);
            cudaDeviceSynchronize();
            k_copy = k_copy/2;
        }
        voronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, 2, N, S, setup.distance_function, mu);
        voronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, 1, N, S, setup.distance_function, mu);
    }
    else{
        while(k_copy >= 1){
            pbcVoronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, k_copy, N, S, setup.distance_function, mu);
            cudaDeviceSynchronize();
            k_copy = k_copy/2;
        }
        pbcVoronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, 2, N, S, setup.distance_function, mu);
        pbcVoronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, 1, N, S, setup.distance_function, mu);
    }
    
}

void jfaVDUnique(Setup setup, int *v_diagram, int *seeds, int N, int S, int k, int mu, dim3 grid, dim3 block, int pbc){
    int k_copy = k;
    if(pbc == 0){
        while(k_copy >= 1){
            voronoiJFA_8NgV21<<< grid, block>>>(v_diagram, seeds, k_copy, N, S, setup.distance_function, mu);
            cudaDeviceSynchronize();
            voronoiJFA_8NgV22<<< grid, block>>>(v_diagram, seeds, k_copy, N, S, setup.distance_function, mu);
            cudaDeviceSynchronize();
            voronoiJFA_8NgV23<<< grid, block>>>(v_diagram, seeds, k_copy, N, S, setup.distance_function, mu);
            cudaDeviceSynchronize();
            k_copy = k_copy/2;
        }
        voronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, 2, N, S, setup.distance_function, mu);
        voronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, 1, N, S, setup.distance_function, mu);
    }
    else{
        while(k_copy >= 1){
            pbcVoronoiJFA_8NgV21<<< grid, block>>>(v_diagram, seeds, k_copy, N, S, setup.distance_function, mu);
            cudaDeviceSynchronize();
            if(cudaGetLastError() != cudaSuccess){
                printf("Something went wrong on Generation\n");
                exit(0);
            }
            pbcVoronoiJFA_8NgV22<<< grid, block>>>(v_diagram, seeds, k_copy, N, S, setup.distance_function, mu);
            cudaDeviceSynchronize();
            pbcVoronoiJFA_8NgV23<<< grid, block>>>(v_diagram, seeds, k_copy, N, S, setup.distance_function, mu);
            cudaDeviceSynchronize();
            
            k_copy = k_copy/2;
        }
        pbcVoronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, 2, N, S, setup.distance_function, mu);
        pbcVoronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, 1, N, S, setup.distance_function, mu);
    }
    
}

void mjfaVDIters(Setup setup, int *v_diagram, int *seeds, int N, int S, int k, int mu, dim3 grid, dim3 block, int pbc){
    int k_copy = k;
    if(pbc == 0){
        while(k_copy >= k/2){
            voronoiJFA_4Ng<<< grid, block>>>(v_diagram, seeds, setup.gpu_delta_max, k_copy, N, S, setup.distance_function, mu);
            cudaDeviceSynchronize();
            if(cudaGetLastError() != cudaSuccess){
                printf("Something went wrong on Von Neumman NB\n");
                exit(0);
            }
            k_copy = k_copy/2;
        }
        while(k_copy >= 1){
            voronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, k_copy, N, S, setup.distance_function, mu);
            cudaDeviceSynchronize();
            k_copy = k_copy/2;
        }
        voronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, 2, N, S, setup.distance_function, mu);
        voronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, 1, N, S, setup.distance_function, mu);
    }
    else{
        while(k_copy >= k/2){
            pbcVoronoiJFA_4Ng<<< grid, block>>>(v_diagram, seeds, k_copy, N, S, setup.distance_function, mu);
            cudaDeviceSynchronize();
            k_copy = k_copy/2;
        }
        while(k_copy >= 1){
            pbcVoronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, k_copy, N, S, setup.distance_function, mu);
            cudaDeviceSynchronize();
            k_copy = k_copy/2;
        }
        pbcVoronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, 2, N, S, setup.distance_function, mu);
        pbcVoronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, 1, N, S, setup.distance_function, mu);
    }
}

void baseJFA(Setup setup){
    clearGrid<<<setup.normal_grid, setup.normal_block>>>(setup.gpu_backup_vd, setup.gpu_seeds, setup.N, setup.S);
    cudaDeviceSynchronize();
    init_GPUSeeds<<<setup.seeds_grid, setup.seeds_block>>>(setup.gpu_backup_vd, setup.gpu_seeds, setup.S);
    cudaDeviceSynchronize();
    jfaVDUnique(setup, setup.gpu_backup_vd, setup.gpu_seeds, setup.N, setup.S, setup.k, 1, setup.normal_grid, setup.normal_block, setup.pbc);
    cudaDeviceSynchronize();
}

void dJFA(Setup setup, int iter){
    if(iter==setup.iters || iter%5==0){
        clearGrid<<<setup.normal_grid, setup.normal_block>>>(setup.gpu_backup_vd, setup.gpu_seeds, setup.N, setup.S);
        cudaDeviceSynchronize();
        init_GPUSeeds<<<setup.seeds_grid, setup.seeds_block>>>(setup.gpu_backup_vd, setup.gpu_seeds, setup.S);
        cudaDeviceSynchronize();
        jfaVDUnique(setup, setup.gpu_backup_vd, setup.gpu_seeds, setup.N, setup.S, setup.k, 1, setup.normal_grid, setup.normal_block, setup.pbc);
        cudaDeviceSynchronize();
    }
    else{
        mjfaVDIters(setup, setup.gpu_backup_vd, setup.gpu_seeds, setup.N, setup.S, setup.k_m, 1, setup.normal_grid, setup.normal_block, setup.pbc);
        cudaDeviceSynchronize();
        if(cudaGetLastError() != cudaSuccess){
            printf("Something went wrong on dJFA iter %i, k_m %i\n", iter, setup.k_m);
            exit(0);
        }
    }
}

void rJFA(Setup setup){
    clearGrid<<<setup.redux_grid, setup.redux_block>>>(setup.gpu_redux_vd, setup.gpu_seeds, setup.N_p, setup.S);
    cudaDeviceSynchronize();
    reduxVDSeeds<<<setup.seeds_grid, setup.seeds_block>>>(setup.gpu_redux_vd, setup.gpu_seeds, setup.N, setup.S, setup.N_p, setup.mu);
    cudaDeviceSynchronize();
    jfaVDUnique(setup, setup.gpu_redux_vd, setup.gpu_seeds, setup.N_p, setup.S, setup.k_r, setup.mu, setup.redux_grid, setup.redux_block, setup.pbc);
    cudaDeviceSynchronize();
    scaleVoronoi<<<setup.normal_grid, setup.normal_block>>>(setup.gpu_backup_vd, setup.gpu_redux_vd, setup.N, setup.N_p, setup.mu);
    cudaDeviceSynchronize();
    jfaVDUnique(setup, setup.gpu_backup_vd, setup.gpu_seeds, setup.N, setup.S, setup.mu/2, 1, setup.normal_grid, setup.normal_block, setup.pbc);
    cudaDeviceSynchronize();
}

void drJFA(Setup setup, int iter){
    if(iter==setup.iters || iter%5==0){
        clearGrid<<<setup.redux_grid, setup.redux_block>>>(setup.gpu_redux_vd, setup.gpu_seeds, setup.N_p, setup.S);
        cudaDeviceSynchronize();
        reduxVDSeeds<<<setup.seeds_grid, setup.seeds_block>>>(setup.gpu_redux_vd, setup.gpu_seeds, setup.N, setup.S, setup.N_p, setup.mu);
        cudaDeviceSynchronize();
        jfaVDUnique(setup, setup.gpu_redux_vd, setup.gpu_seeds, setup.N_p, setup.S, setup.k_r, setup.mu, setup.redux_grid, setup.redux_block, setup.pbc);
        cudaDeviceSynchronize();
    }
    else{
        mjfaVDIters(setup, setup.gpu_redux_vd, setup.gpu_seeds, setup.N_p, setup.S, setup.k_rm, setup.mu, setup.redux_grid, setup.redux_block, setup.pbc);
        cudaDeviceSynchronize();
    }
    scaleVoronoi<<<setup.normal_grid, setup.normal_block>>>(setup.gpu_backup_vd, setup.gpu_redux_vd, setup.N, setup.N_p, setup.mu);
    cudaDeviceSynchronize();
    jfaVDUnique(setup, setup.gpu_backup_vd, setup.gpu_seeds, setup.N, setup.S, setup.mu/2, 1, setup.normal_grid, setup.normal_block, setup.pbc);
    cudaDeviceSynchronize();
}

void compare_diagrams(Setup setup, dim3 grid_jfa, dim3 block_jfa, int pbc, double *bw_acc){

    cudaMemcpy(setup.v_diagram, setup.gpu_v_diagram, sizeof(int)*setup.N*setup.N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemcpy(setup.backup_v_diagram, setup.gpu_backup_vd, setup.N*setup.N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    int total = 0, ref_1, ref_2, x, x1, x2, y, y1, y2, place;
    double dist_1, dist_2, acc = 0.0;
    for(int i = 0; i < setup.N; ++i){
        for(int j = 0; j < setup.N; ++j){
            x = i;
            y = j;
            place = y*setup.N + x;
            ref_1 = setup.v_diagram[place];
            ref_2 = setup.backup_v_diagram[place];
            //printf("REF 1: %i. REF 2: %i\n", ref_1, ref_2);
            if(ref_1 == ref_2) total +=1;
            else{
                x1 = ref_1%setup.N;
                y1 = ref_1/setup.N;

                x2 = ref_2%setup.N;
                y2 = ref_2/setup.N;

                dist_1 = sqrt((x - x1)*(x - x1) + (y - y1)*(y - y1));
                dist_2 = sqrt((x - x2)*(x - x2) + (y - y2)*(y - y2));
                //printf("REF 1: %i, REF 2: %i, DIST 1: %f, DIST 2: %f, DIF: %f\n", ref_1, ref_2, dist_1, dist_2, abs(dist_1 - dist_2));
                if(abs(dist_1 - dist_2) < EPSILON*double(setup.N)) total +=1;
            }
        }
    }
    acc = (double)total/((double)setup.N*(double)setup.N) * 100.0;
    if(bw_acc[0] < acc) bw_acc[0] = acc;
    if(bw_acc[1] > acc) bw_acc[1] = acc;
    bw_acc[2] += acc;
    printf("%f\n",acc);

}

void itersJFA(Setup setup){
    int iter_copy = setup.iters;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms1 = 0, ms2 = 0, ms1_t = 0.0, ms2_t = 0.0;
    int k_used;
    double best_worst_acc[3];
    int count_acc = 0;
    best_worst_acc[0] = 0.0;
    best_worst_acc[1] = 100.0;
    best_worst_acc[2] = 0.0;


    while(iter_copy > 0){
        //printf("ITER %i\n", iter_copy);
        dynamicSeeds(setup, iter_copy);
        cudaEventRecord(start);
        if(setup.comparison == 1){
            bruteReference<<<setup.normal_grid, setup.normal_block>>>(setup.gpu_v_diagram, setup.gpu_seeds, setup.N, setup.S, setup.distance_function, setup.pbc);
            cudaDeviceSynchronize();
        }
        if(cudaGetLastError() != cudaSuccess){
            printf("Something went wrong on Comparison\n");
            exit(0);
        }
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms1, start, stop);
        ms1_t += ms1;
        
        cudaEventRecord(start);
        // Base JFA (JFA)
        //baseJFA(setup);

        // Dynamic JFA (dJFA)
        dJFA(setup, iter_copy);

        //Redux JFA (rJFA)
        //rJFA(setup);

        // Dynamic  - Redux JFA (dr-JFA)
        //drJFA(setup, iter_copy);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms2, start, stop);
        ms2_t += ms2;
        if(setup.comparison == 1){
            compare_diagrams(setup, setup.normal_grid, setup.normal_block, setup.pbc, best_worst_acc);
            #ifdef SSTEP
                save_step(setup.backup_v_diagram, setup.N, 100-iter_copy,1);
                save_step(setup.v_diagram, setup.N, 100-iter_copy,0);
            #endif
        }
        count_acc+=1;
        //printf("TIMES %i, M1: %f, M2: %f\n", iter_copy, ms1, ms2);
        iter_copy -= 1;

    }
    ms1_t = ms1_t * 0.001;
    ms2_t = ms2_t * 0.001;
    best_worst_acc[2] = best_worst_acc[2] / double(count_acc);
    printf("TOTAL TIME REFERENCE: %f, TOTAL TIME PROPOSAL: %f\n", ms1_t, ms2_t);
    printf("ACCURACY - AVG: %f - BEST: %f - WORST: %f\n", best_worst_acc[2], best_worst_acc[0], best_worst_acc[1]);
    //write_data(setup.N, setup.S, METHOD, setup.mu, ms1_t, ms2_t, best_worst_acc[2], best_worst_acc[0], best_worst_acc[1]);
}
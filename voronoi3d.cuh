#include "setup.h"

__device__ float euclideanDistance(int x1, int x2, int y1, int y2, int z1, int z2){
    return float( (x1 - x2) * (x1 - x2) +
           (y1 - y2) * (y1 - y2) +
           (z1 - z2) * (z1 - z2));
}

__device__ float manhattanDistance(int x1, int x2, int y1, int y2, int z1, int z2){
    return float(abs(x1 - x2) +
           abs(y1 - y2) +
           abs(z1 - z2));
}

__device__ float pbcEuclideanDistance(int N, int x1, int x2, int y1, int y2, int z1, int z2){
    int tmp_x = x1 - x2;
    int tmp_y = y1 - y2;
    int tmp_z = z1 - z2;

    if(fabsf(float(tmp_x - N)) < fabsf(float(tmp_x))) tmp_x -= N;
    else if(fabsf(float(tmp_x + N)) < fabsf(float(tmp_x))) tmp_x += N;

    if(fabsf(float(tmp_y - N)) < fabsf(float(tmp_y))) tmp_y -= N;
    else if(fabsf(float(tmp_y + N)) < fabsf(float(tmp_y))) tmp_y += N;

    if(fabsf(float(tmp_z - N)) < fabsf(float(tmp_z))) tmp_z -= N;
    else if(fabsf(float(tmp_z + N)) < fabsf(float(tmp_z))) tmp_z += N;

    return float(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z);
}

__device__ float pbcManhattanDistance(int N, int x1, int x2, int y1, int y2, int z1, int z2){
    int tmp_x = x1 - x2;
    int tmp_y = y1 - y2;
    int tmp_z = z1 - z2;

    if(fabsf(float(tmp_x - N)) < fabsf(float(tmp_x))) tmp_x -= N;
    else if(fabsf(float(tmp_x + N)) < fabsf(float(tmp_x))) tmp_x += N;

    if(fabsf(float(tmp_y - N)) < fabsf(float(tmp_y))) tmp_y -= N;
    else if(fabsf(float(tmp_y + N)) < fabsf(float(tmp_y))) tmp_y += N;

    if(fabsf(float(tmp_z - N)) < fabsf(float(tmp_z))) tmp_z -= N;
    else if(fabsf(float(tmp_z + N)) < fabsf(float(tmp_z))) tmp_z += N;

    return fabsf(tmp_x) + fabsf(tmp_y) + fabsf(tmp_z);
}

__device__ void checkNeighborDistance(
    float4 *VD,
    const int *seeds,
    const int N,
    const int local_value,
    const int local_x,
    const int local_y,
    const int local_z,
    const int pixel_x,
    const int pixel_y,
    const int pixel_z,
    const int dist,
    const int mu,
    const int isPBC
){
    int pixel = pixel_z * N * N + pixel_y * N + pixel_x;

    int rival_value = int(VD[pixel].w);
    int rival_seed = seeds[rival_value], rival_x, rival_y, rival_z;

    float dist_local;
    float dist_rival;

    if(local_value != rival_value){
        rival_x = rival_seed%(N * mu);
        rival_y = rival_seed/(N * mu) %(N * mu);
        rival_z = rival_seed/( (N * mu) * (N * mu));

        if(dist == 0){
            dist_local = (isPBC == 1)? pbcEuclideanDistance(
                N * mu,
                local_x,
                pixel_x * mu,
                local_y,
                pixel_y * mu,
                local_z,
                pixel_z * mu
            ) : euclideanDistance(
                local_x,
                pixel_x * mu,
                local_y,
                pixel_y * mu,
                local_z,
                pixel_z * mu
            );
            dist_rival = (isPBC == 1)? pbcEuclideanDistance(
                N * mu,
                rival_x,
                pixel_x * mu,
                rival_y,
                pixel_y * mu,
                rival_z,
                pixel_z * mu
            ) : euclideanDistance(
                rival_x,
                pixel_x * mu,
                rival_y,
                pixel_y * mu,
                rival_z,
                pixel_z * mu
            );
        } else{
            dist_local = (isPBC == 1)? pbcManhattanDistance(
                N * mu,
                local_x,
                pixel_x * mu,
                local_y,
                pixel_y * mu,
                local_z,
                pixel_z * mu
            ) : manhattanDistance(
                local_x,
                pixel_x * mu,
                local_y,
                pixel_y * mu,
                local_z,
                pixel_z * mu
            );
            dist_rival = (isPBC == 1)? pbcManhattanDistance(
                N * mu,
                rival_x,
                pixel_x * mu,
                rival_y,
                pixel_y * mu,
                rival_z,
                pixel_z * mu
            ) : manhattanDistance(
                rival_x,
                pixel_x * mu,
                rival_y,
                pixel_y * mu,
                rival_z,
                pixel_z * mu
            );
        }

        if(dist_local < dist_rival){
            VD[pixel].w = float(local_value);
        }
    }
}

__global__ void bruteReference(float4 *VD, int *seeds, int N, int S, int dist, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    int tid = tidz * N * N + tidy * N + tidx;

    int close_index = -1, closest = -1, local_x, local_y, local_z, ref_x, ref_y, ref_z;
    float dist_local, dist_ref;
    for(int i = 0; i < S; ++i){
        if(closest == -1){
            closest = seeds[i];
            close_index = i;
        } else{
            ref_x = seeds[i] % N;
            ref_y = seeds[i] / N % N;
            ref_z = seeds[i] / (N * N);

            local_x = closest%N;
            local_y = closest/N % N;
            local_z = closest/(N * N);

            dist_local = (pbc == 1)? pbcEuclideanDistance(
                N,
                tidx,
                local_x,
                tidy,
                local_y,
                tidz,
                local_z
            ) : euclideanDistance(
                tidx,
                local_x,
                tidy,
                local_y,
                tidz,
                local_z
            );
            dist_ref = (pbc == 1)? pbcEuclideanDistance(
                N,
                tidx,
                ref_x,
                tidy,
                ref_y,
                tidz,
                ref_z
            ) : euclideanDistance(
                tidx,
                ref_x,
                tidy,
                ref_y,
                tidz,
                ref_z
            );

            if(dist_ref < dist_local){
                closest = seeds[i];
                close_index = i;
            }
        }

        VD[tid].w = float(close_index);
    }
}

__global__ void initVD(float4 *VD, int N){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.z;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    if(tidx >= N || tidy>= N || tidz >= N) return;
    int tid = tidz * N * N + tidy * N + tidx;
    VD[tid].w = -1.0;
}

/*__global__ void initGPUSeeds(float4 *VD, int *seeds, int S){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= S) return;
    int local_value = seeds[tid];
    VD[local_value].w = float(tid);
}*/

// Split flooding in groups of 3 (Total f: 26, # of groups: 9)
//  1   2   3 [LEFT -> RIGHT, UP -> DOWN]
//  4   5   5
//  7   8   9
// 1: 1 2 3 [FRONT -> BACK]
__global__ void flood3DG1(int N, int S, float4 *VD, int *seeds, int delta, int dist, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    bool allow = true;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    int tid = tidz * N * N + tidy * N + tidx;
    int local_value = int(VD[tid].w);
    if(local_value == -1) return;
    //if(tid < 10) printf("For seed %i, value %i, init set\n", tid, seeds[tid]);
    int local_seed = seeds[local_value], local_x = local_seed%(N * mu), local_y = local_seed/(N * mu) % (N * mu), local_z = local_seed/((N * mu) * (N * mu));
    int pixel, pixel_x, pixel_y, pixel_z;
    // 1
    pixel_x = tidx - delta;
    pixel_y = tidy + delta;
    pixel_z = tidz + delta;
    if(pbc){
        pixel_x = (pixel_x + N)%N;
        pixel_y = pixel_y%N;
        pixel_z = pixel_z%N;
    }
    else{
        if(pixel_x < 0 || pixel_y >= N || pixel_z >= N) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    }
    else allow = true;
    // 2
    pixel_x = tidx - delta;
    pixel_y = tidy + delta;
    pixel_z = tidz;
    if(pbc){
        pixel_x = (pixel_x + N)%N;
        pixel_y = pixel_y%N;
    }
    else{
        if(pixel_x < 0 || pixel_y >= N) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);  
    } else allow = true;
    // 3
    pixel_x = tidx - delta;
    pixel_y = tidy + delta;
    pixel_z = tidy - delta;
    if(pbc){
        pixel_x = (pixel_x + N)%N;
        pixel_y = pixel_y%N;
        pixel_z = (pixel_z + N)%N;
    }
    else{
        if(pixel_x < 0 || pixel_y >= N || pixel_z < 0) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    } else allow = true;
}
__global__ void flood3DG2(int N, int S, float4 *VD, int *seeds, int delta, int dist, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    bool allow = true;
    int tid = tidz * N * N + tidy * N + tidx;
    int local_value;
    local_value = int(VD[tid].w);
    if(local_value == -1) return;
    int local_seed = seeds[local_value], local_x = local_seed%(N * mu), local_y = local_seed/(N * mu) % (N * mu), local_z = local_seed/((N * mu) * (N * mu));
    int pixel, pixel_x, pixel_y, pixel_z;
    // 1
    pixel_x = tidx;
    pixel_y = tidy + delta;
    pixel_z = tidz + delta;
    if(pbc){
        pixel_y = pixel_y%N;
        pixel_z = pixel_z%N;
    }
    else{
        if(pixel_y >= N) return;
        if(pixel_z >= N) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    } else allow = true;
    // 2
    pixel_z = tidz;
    if(pbc){
        pixel_y = pixel_y%N;
    }
    else{
        //if(pixel_y >= N) return;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    } else allow = true;
    // 3
    pixel_z = tidy - delta;
    if(pbc){
        pixel_z = (pixel_z + N)%N;
    }
    else{
        if(pixel_z < 0) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    }
}
__global__ void flood3DG3(int N, int S, float4 *VD, int *seeds, int delta, int dist, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    bool allow = true;
    int tid = tidz * N * N + tidy * N + tidx;
    int local_value;
    local_value = int(VD[tid].w);
    //if(local_value != -1) printf("lv: %i\n", local_value);
    if(local_value == -1) return;
    int local_seed = seeds[local_value], local_x = local_seed%(N * mu), local_y = local_seed/(N * mu) % (N * mu), local_z = local_seed/((N * mu) * (N * mu));
    int pixel, pixel_x, pixel_y, pixel_z;
    // 1
    pixel_x = tidx + delta;
    pixel_y = tidy + delta;
    pixel_z = tidz + delta;
    if(pbc){
        pixel_x = pixel_x%N;
        pixel_y = pixel_y%N;
        pixel_z = pixel_z%N;
    }
    else{
        if(pixel_y >= N) return;
        if(pixel_x >= N || pixel_z >= N) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    }
    // 2
    pixel_z = tidz;
    if(pbc){
        //pixel_y = pixel_y%N;
    }
    else{
        if(pixel_x >= N) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    } else allow = true;
    // 3
    pixel_z = tidy - delta;
    if(pbc){
        pixel_z = (pixel_z + N)%N;
    }
    else{
        if(pixel_x >= N || pixel_z < 0) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    }
    
}
__global__ void flood3DG4(int N, int S, float4 *VD, int *seeds, int delta, int dist, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    bool allow = true;
    int tid = tidz * N * N + tidy * N + tidx;
    int local_value;
    local_value = int(VD[tid].w);
    if(local_value == -1) return;
    int local_seed = seeds[local_value], local_x = local_seed%(N * mu), local_y = local_seed/(N * mu) % (N * mu), local_z = local_seed/((N * mu) * (N * mu));
    int pixel, pixel_x, pixel_y, pixel_z;
    // 1
    pixel_x = tidx - delta;
    pixel_y = tidy;
    pixel_z = tidz + delta;
    if(pbc){
        pixel_x = (pixel_x + N)%N;
        pixel_z = pixel_z%N;
    }
    else{
        if(pixel_x < 0 || pixel_z >= N) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    } else allow = true;
    // 2
    pixel_x = tidx - delta;
    pixel_z = tidz;
    if(pbc){
        pixel_x = (pixel_x + N)%N;
    }
    else{
        if(pixel_x < 0) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    } else allow = true;
    // 3
    pixel_x = tidx - delta;
    pixel_z = tidy - delta;
    if(pbc){
        pixel_x = (pixel_x + N)%N;
        pixel_z = (pixel_z + N)%N;
    }
    else{
        if(pixel_x < 0 || pixel_z < 0) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    }
}
__global__ void flood3DG5(int N, int S, float4 *VD, int *seeds, int delta, int dist, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    bool allow = true;
    int tid = tidz * N * N + tidy * N + tidx;
    int local_value;
    local_value = int(VD[tid].w);
    if(local_value == -1) return;
    int local_seed = seeds[local_value], local_x = local_seed%(N * mu), local_y = local_seed/(N * mu) % (N * mu), local_z = local_seed/((N * mu) * (N * mu));
    int pixel, pixel_x, pixel_y, pixel_z;
    // 1
    pixel_x = tidx;
    pixel_y = tidy;
    pixel_z = tidz + delta;
    if(pbc){
        pixel_z = pixel_z%N;
    }
    else{
        if(pixel_z >= N) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    } allow = true;
    // 2
    // 3
    pixel_z = tidy - delta;
    if(pbc){
        pixel_z = (pixel_z + N)%N;
    }
    else{
        if(pixel_z < 0) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    }
}
__global__ void flood3DG6(int N, int S, float4 *VD, int *seeds, int delta, int dist, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    bool allow = true;
    int tid = tidz * N * N + tidy * N + tidx;
    int local_value;
    local_value = int(VD[tid].w);
    if(local_value == -1) return;
    int local_seed = seeds[local_value], local_x = local_seed%(N * mu), local_y = local_seed/(N * mu) % (N * mu), local_z = local_seed/((N * mu) * (N * mu));
    int pixel, pixel_x, pixel_y, pixel_z;
    // 1
    pixel_x = tidx + delta;
    pixel_y = tidy;
    pixel_z = tidz + delta;
    if(pbc){
        pixel_x = pixel_x%N;
        pixel_z = pixel_z%N;
    }
    else{
        if(pixel_x >= N || pixel_z >= N) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    } else allow = true;
    // 2
    pixel_z = tidz;
    if(pbc){
        pixel_x = pixel_x%N;
    }
    else{
        if(pixel_x >= N) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    }
    // 3
    pixel_x = tidx + delta;
    pixel_z = tidy - delta;
    if(pbc){
        pixel_x = pixel_x%N;
        pixel_z = (pixel_z + N)%N;
    }
    else{
        if(pixel_x < 0 || pixel_z < 0) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    }
}
__global__ void flood3DG7(int N, int S, float4 *VD, int *seeds, int delta, int dist, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    bool allow = true;
    int tid = tidz * N * N + tidy * N + tidx;
    int local_value;
    local_value = int(VD[tid].w);
    if(local_value == -1) return;
    int local_seed = seeds[local_value], local_x = local_seed%(N * mu), local_y = local_seed/(N * mu) % (N * mu), local_z = local_seed/((N * mu) * (N * mu));
    int pixel, pixel_x, pixel_y, pixel_z;
    // 1
    pixel_x = tidx - delta;
    pixel_y = tidy - delta;
    pixel_z = tidz + delta;
    if(pbc){
        pixel_x = (pixel_x + N)%N;
        pixel_y = (pixel_y + N)%N;
        pixel_z = pixel_z%N;
    }
    else{
        if(pixel_y < 0) return;
        if(pixel_x < 0 || pixel_z >= N) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    }
    else allow = true;
    // 2
    pixel_x = tidx - delta;
    pixel_z = tidz;
    if(pbc){
        pixel_x = (pixel_x + N)%N;
    }
    else{
        if(pixel_x < 0) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    } else allow = true;
    // 3
    pixel_x = tidx - delta;
    pixel_z = tidy - delta;
    if(pbc){
        pixel_x = (pixel_x + N)%N;
        pixel_z = (pixel_z + N)%N;
    }
    else{
        if(pixel_x < 0 || pixel_z < 0) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    }
}
__global__ void flood3DG8(int N, int S, float4 *VD, int *seeds, int delta, int dist, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    bool allow = true;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    int tid = tidz * N * N + tidy * N + tidx;
    int local_value;
    local_value = int(VD[tid].w);
    if(local_value == -1) return;
    int local_seed = seeds[local_value], local_x = local_seed%(N * mu), local_y = local_seed/(N * mu) % (N * mu), local_z = local_seed/((N * mu) * (N * mu));
    int pixel, pixel_x, pixel_y, pixel_z;
    // 1
    pixel_x = tidx;
    pixel_y = tidy - delta;
    pixel_z = tidz + delta;
    if(pbc){
        pixel_x = (pixel_x + N)%N;
        pixel_y = pixel_y%N;
        pixel_z = pixel_z%N;
    }
    else{
        if(pixel_y < 0) return;
        if(pixel_z >= N) allow = false; 
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    }
    else allow = true;
    // 2
    pixel_z = tidz;
    if(pbc){
        pixel_y = pixel_y%N;
    }
    else{
        //if(pixel_x < 0) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);  
    } else allow = true;
    // 3
    pixel_z = tidy - delta;
    if(pbc){
        pixel_z = (pixel_z + N)%N;
    }
    else{
        if(pixel_x < 0 || pixel_z < 0) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    }
}
__global__ void flood3DG9(int N, int S, float4 *VD, int *seeds, int delta, int dist, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    bool allow = true;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    int tid = tidz * N * N + tidy * N + tidx;
    int local_value;
    local_value = int(VD[tid].w);
    if(local_value == -1) return;
    int local_seed = seeds[local_value], local_x = local_seed%(N * mu), local_y = local_seed/(N * mu) % (N * mu), local_z = local_seed/((N * mu) * (N * mu));
    int pixel, pixel_x, pixel_y, pixel_z;
    // 1
    pixel_x = tidx + delta;
    pixel_y = tidy - delta;
    pixel_z = tidz + delta;
    if(pbc){
        pixel_x = pixel_x%N;
        pixel_y = pixel_y%N;
        pixel_z = pixel_z%N;
    }
    else{
        if(pixel_y < 0) return;
        if(pixel_x < 0 || pixel_z >= N) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    }
    else allow = true;
    // 2
    pixel_z = tidz;
    if(pbc){
        //pixel_x = (pixel_x + N)%N;
        //pixel_y = pixel_y%N;
    }
    else{
        if(pixel_x >= N) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);  
    } else allow = true;
    // 3
    pixel_z = tidy - delta;
    if(pbc){
        pixel_z = (pixel_z + N)%N;
    }
    else{
        if(pixel_x >= N || pixel_z < 0) allow = false;
    }
    if(allow){
        pixel = pixel_z * N * N + pixel_y * N + pixel_x;
        if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
        else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
    }
}

// Special cases of neighborhood
__global__ void flood3DG1_2(int N, int S, float4 *VD, int *seeds, int delta, int dist, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    int tid = tidz * N * N + tidy * N + tidx;
    int local_value;
    local_value = int(VD[tid].w);
    if(local_value == -1) return;
    int local_seed = seeds[local_value], local_x = local_seed%(N * mu), local_y = local_seed/(N * mu) % (N * mu), local_z = local_seed/((N * mu) * (N * mu));
    int pixel, pixel_x, pixel_y, pixel_z;
    // 1
    pixel_x = tidx - delta;
    pixel_y = tidy + delta;
    pixel_z = tidz;
    if(pbc){
        pixel_x = (pixel_x + N)%N;
        pixel_y = pixel_y%N;
    }
    else{
        if(pixel_x < 0 || pixel_y >= N) return;
    }
    pixel = pixel_z * N * N + pixel_y * N + pixel_x;
    if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
    else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
}

__global__ void flood3DG3_2(int N, int S, float4 *VD, int *seeds, int delta, int dist, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    int tid = tidz * N * N + tidy * N + tidx;
    int local_value;
    local_value = int(VD[tid].w);
    if(local_value == -1) return;
    int local_seed = seeds[local_value], local_x = local_seed%(N * mu), local_y = local_seed/(N * mu) % (N * mu), local_z = local_seed/((N * mu) * (N * mu));
    int pixel, pixel_x, pixel_y, pixel_z;
    // 1
    pixel_x = tidx + delta;
    pixel_y = tidy + delta;
    pixel_z = tidz;
    if(pbc){
        pixel_x = pixel_x%N;
        pixel_y = pixel_y%N;
    }
    else{
        if(pixel_x >= N || pixel_y >= N) return;
    }
    pixel = pixel_z * N * N + pixel_y * N + pixel_x;
    if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
    else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
}

__global__ void flood3DG7_2(int N, int S, float4 *VD, int *seeds, int delta, int dist, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    int tid = tidz * N * N + tidy * N + tidx;
    int local_value;
    local_value = int(VD[tid].w);
    if(local_value == -1) return;
    int local_seed = seeds[local_value], local_x = local_seed%(N * mu), local_y = local_seed/(N * mu) % (N * mu), local_z = local_seed/((N * mu) * (N * mu));
    int pixel, pixel_x, pixel_y, pixel_z;
    // 1
    pixel_x = tidx - delta;
    pixel_y = tidy - delta;
    pixel_z = tidz;
    if(pbc){
        pixel_x = (pixel_x + N)%N;
        pixel_y = (pixel_y + N)%N;
    }
    else{
        if(pixel_x < 0 || pixel_y < 0) return;
    }
    pixel = pixel_z * N * N + pixel_y * N + pixel_x;
    if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
    else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
}

__global__ void flood3DG9_2(int N, int S, float4 *VD, int *seeds, int delta, int dist, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    int tid = tidz * N * N + tidy * N + tidx;
    int local_value;
    local_value = int(VD[tid].w);
    if(local_value == -1) return;
    int local_seed = seeds[local_value], local_x = local_seed%(N * mu), local_y = local_seed/(N * mu) % (N * mu), local_z = local_seed/((N * mu) * (N * mu));
    int pixel, pixel_x, pixel_y, pixel_z;
    // 1
    pixel_x = tidx + delta;
    pixel_y = tidy - delta;
    pixel_z = tidz;
    if(pbc){
        pixel_x = pixel_x%N;
        pixel_y = (pixel_y + N)%N;
    }
    else{
        if(pixel_x >= N || pixel_y < 0) return;
    }
    pixel = pixel_z * N * N + pixel_y * N + pixel_x;
    if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
    else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
}

__global__ void flood3DG2_2(int N, int S, float4 *VD, int *seeds, int delta, int dist, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    int tid = tidz * N * N + tidy * N + tidx;
    int local_value;
    local_value = int(VD[tid].w);
    if(local_value == -1) return;
    int local_seed = seeds[local_value], local_x = local_seed%(N * mu), local_y = local_seed/(N * mu) % (N * mu), local_z = local_seed/((N * mu) * (N * mu));
    int pixel, pixel_x, pixel_y, pixel_z;
    // 1
    pixel_x = tidx;
    pixel_y = tidy + delta;
    pixel_z = tidz;
    if(pbc){
        pixel_y = pixel_y%N;
    }
    else{
        if(pixel_y >= N) return;
    }
    pixel = pixel_z * N * N + pixel_y * N + pixel_x;
    if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
    else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
}

__global__ void flood3DG4_2(int N, int S, float4 *VD, int *seeds, int delta, int dist, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    int tid = tidz * N * N + tidy * N + tidx;
    int local_value;
    local_value = int(VD[tid].w);
    if(local_value == -1) return;
    int local_seed = seeds[local_value], local_x = local_seed%(N * mu), local_y = local_seed/(N * mu) % (N * mu), local_z = local_seed/((N * mu) * (N * mu));
    int pixel, pixel_x, pixel_y, pixel_z;
    // 1
    pixel_x = tidx - delta;
    pixel_y = tidy;
    pixel_z = tidz;
    if(pbc){
        pixel_x = (pixel_x + N)%N;
    }
    else{
        if(pixel_x < 0) return;
    }
    pixel = pixel_z * N * N + pixel_y * N + pixel_x;
    if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
    else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
}

__global__ void flood3DG6_2(int N, int S, float4 *VD, int *seeds, int delta, int dist, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    int tid = tidz * N * N + tidy * N + tidx;
    int local_value;
    local_value = int(VD[tid].w);
    if(local_value == -1) return;
    int local_seed = seeds[local_value], local_x = local_seed%(N * mu), local_y = local_seed/(N * mu) % (N * mu), local_z = local_seed/((N * mu) * (N * mu));
    int pixel, pixel_x, pixel_y, pixel_z;
    // 1
    pixel_x = tidx + delta;
    pixel_y = tidy;
    pixel_z = tidz;
    if(pbc){
        pixel_x = pixel_x%N;
    }
    else{
        if(pixel_x >= N) return;
    }
    pixel = pixel_z * N * N + pixel_y * N + pixel_x;
    if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
    else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
}

__global__ void flood3DG8_2(int N, int S, float4 *VD, int *seeds, int delta, int dist, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    if(tidx >= N || tidy >= N || tidz >= N) return;
    int tid = tidz * N * N + tidy * N + tidx;
    int local_value;
    local_value = int(VD[tid].w);
    if(local_value == -1) return;
    int local_seed = seeds[local_value], local_x = local_seed%(N * mu), local_y = local_seed/(N * mu) % (N * mu), local_z = local_seed/((N * mu) * (N * mu));
    int pixel, pixel_x, pixel_y, pixel_z;
    // 1
    pixel_x = tidx;
    pixel_y = tidy - delta;
    pixel_z = tidz;
    if(pbc){
        pixel_y = (pixel_y + N)%N;
    }
    else{
        if(pixel_y < 0) return;
    }
    pixel = pixel_z * N * N + pixel_y * N + pixel_x;
    if(int(VD[pixel].w) == -1) VD[pixel].w = float(local_value);
    else checkNeighborDistance(VD, seeds, N, local_value, local_x, local_y, local_z, pixel_x, pixel_y, pixel_z, dist, mu, pbc);
}
// Create full neighbours

// Neighbors
void voronoi26NB(Setup setup, float4 *v_diagram, int *seeds, int N, int S, int k, int mu, dim3 grid, dim3 block, int pbc){
    flood3DG1<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on G1\n");
        exit(0);
    }
    flood3DG2<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on G2\n");
        exit(0);
    }
    flood3DG3<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on G3\n");
        exit(0);
    }
    flood3DG4<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on G4\n");
        exit(0);
    }
    flood3DG5<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on G5\n");
        exit(0);
    }
    flood3DG6<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on G6\n");
        exit(0);
    }
    flood3DG7<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on G7\n");
        exit(0);
    }
    flood3DG8<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on G8\n");
        exit(0);
    }
    flood3DG9<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on G9\n");
        exit(0);
    }
}

void voronoi18NB(Setup setup, float4 *v_diagram, int *seeds, int N, int S, int k, int mu, dim3 grid, dim3 block, int pbc){
    flood3DG1_2<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    flood3DG2<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    flood3DG3_2<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    flood3DG4<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    flood3DG5<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    flood3DG6<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    flood3DG7_2<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    flood3DG8<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    flood3DG9_2<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    cudaDeviceSynchronize();
}

void voronoi6NB(Setup setup, float4 *v_diagram, int *seeds, int N, int S, int k, int mu, dim3 grid, dim3 block, int pbc){
    flood3DG2_2<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    flood3DG4_2<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    flood3DG5<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    flood3DG6_2<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    flood3DG8_2<<<grid, block>>>(N, S, v_diagram, seeds, k, setup.distance_function, mu, pbc);
    cudaDeviceSynchronize();
}

__global__ void clearGrid(float4 *VD, int *SEEDS, int N, int S){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    int tid = tidz * N * N + tidy * N + tidx;
    if(tidx < N && tidy < N && tidz < N){
        VD[tid].w = -1.0;
    }
}

__global__ void initGPUSeeds3D(float4 *VD, int *SEEDS, int S){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int local_value;
    if(tid >= S) return;
   
    local_value = SEEDS[tid];
    //if(tid < 10) printf("id %i, local value: %i\n", tid, local_value);
    VD[local_value].w = float(tid);
    
}

__global__ void reduxVDSeeds3D(float4 *RVD, int *seeds, int N, int S, int N_p, int mu){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int local_value, local_x, local_y, local_z, redux_x, redux_y, redux_z, redux_coord;
    if(tid < S){
        local_value = seeds[tid];
        local_x = local_value%N;
        local_y = local_value/N % N;
        local_z = local_value/(N * N);
        redux_x = local_x / mu;
        redux_y = local_y / mu;
        redux_z = local_z / mu;
        redux_coord = redux_z * N_p * N_p + redux_y * N_p + redux_x;
        if(redux_coord > (N_p * N_p * N_p)) printf("Alert\n");
        RVD[redux_coord].w = float(tid);
    }
}

__global__ void scaleVD3D(float4 *VD, float4 *RVD, int N, int n, int mu){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    

    if(tidx >= N || tidy >= N || tidz >= N) return;

    int tid = tidz * N * N + tidy * N + tidx;
    int redux_x, redux_y, redux_z, redux_coord;

    redux_x = tidx / mu;
    redux_y = tidy / mu;
    redux_z = tidz / mu;
    redux_coord = redux_z * n * n + redux_y * n + redux_x;
    VD[tid] = RVD[redux_coord];
    
}

__global__ void scaleVD_V2(float4 *VD, float4 *RVD, int N, int n, int mu, int pbc){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    

    if(tidx >= N || tidy >= N || tidz >= N) return;

    int tid = tidz * N * N + tidy * N + tidx;
    int redux_x, redux_y, redux_z, redux_coord, ipol_x, ipol_y, ipol_z;
    float dist_local, dist_ref;

    redux_x = tidx / mu;
    redux_y = tidy / mu;
    redux_z = tidz / mu;
    
    
    for(int i = -1; i <= 1; ++i){
        for(int j = -1; j <= 1; ++j){
            for(int k = -1; k <=1; ++k){
                if(j == 0 && i == 0 && k == 0) continue;
                dist_local = 0.f;
                dist_ref = 0.f;
                ipol_x = redux_x + i;
                ipol_y = redux_y + j;
                ipol_z = redux_z + k;
                if(pbc){
                    ipol_x = (ipol_x + N)%N;
                    ipol_y = (ipol_y + N)%N;
                    ipol_z = (ipol_z + N)%N;
                    dist_local = pbcEuclideanDistance(N, tidx, redux_x * mu, tidy, redux_y, tidz, redux_z * mu);
                    dist_ref = pbcEuclideanDistance(N, tidx, ipol_x * mu, tidy, ipol_y * mu, tidz, ipol_z * mu);
                } else{
                    if(ipol_x >= N || ipol_x < 0 ||
                       ipol_y >= N || ipol_y < 0 ||
                       ipol_z >= N || ipol_z < 0) continue;
                    dist_local = euclideanDistance(tidx, redux_x * mu, tidy, redux_y, tidz, redux_z * mu);
                    dist_ref = euclideanDistance(tidx, ipol_x * mu, tidy, ipol_y * mu, tidz, ipol_z * mu);  
                }
                if(dist_ref < dist_local){
                    redux_x = ipol_x;
                    redux_y = ipol_y;
                    redux_z = ipol_z;
                }
            }
        }
    }
    redux_coord = redux_z * n * n + redux_y * n + redux_x;
    VD[tid] = RVD[redux_coord];
}
// Methods
// Base 3DJFA

// 3D-dJFA

// 3D-rJFA

// 3D-drJFA

// Check similarity

// Simulation
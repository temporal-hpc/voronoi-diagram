//#include "setup.h"
// Util for seeds
__global__ void initRand3D(int S, int mod, curandState *states, int *seeds){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < S){
        curand_init(1, tid, 0, &states[tid]);
        //if(tid < 10) printf("For seed %i, value %i, init set\n", tid, seeds[tid]);
    }
}
void setRandDevice(Setup *setup){
    initRand3D<<<setup->seeds_grid, setup->seeds_block>>>(setup->S, setup->N, setup->r_device, setup->gpu_seeds);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on initRand3D\n");
        exit(0);
    }
}

// Distance
__device__ float4 euclideanDistanceDyn(int x1, int x2, int y1, int y2, int z1, int z2){
    float4 ret;
    ret.x = sqrtf( (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
    ret.y = abs(x1 - x2);
    ret.z = abs(y1 - y2);
    ret.w = abs(z1 - z2);
    return ret;
}

__device__ float4 pbcEuclideanDistanceDyn(int N, int x1, int x2, int y1, int y2, int z1, int z2){
    int tmp_x = x1 - x2;
    int tmp_y = y1 - y2;
    int tmp_z = z1 - z2;

    if(fabsf(float(tmp_x - N)) < fabsf(float(tmp_x))) tmp_x -= N;
    else if(fabsf(float(tmp_x + N)) < fabsf(float(tmp_x))) tmp_x += N;

    if(fabsf(float(tmp_y - N)) < fabsf(float(tmp_y))) tmp_y -= N;
    else if(fabsf(float(tmp_y + N)) < fabsf(float(tmp_y))) tmp_y += N;

    if(fabsf(float(tmp_z - N)) < fabsf(float(tmp_z))) tmp_z -= N;
    else if(fabsf(float(tmp_z + N)) < fabsf(float(tmp_z))) tmp_z += N;
    float4 ret;
    ret.x = sqrtf(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z);
    ret.y = tmp_x;
    ret.z = tmp_y;
    ret.w = tmp_z;
    return ret;
}


//  3D Nbody
__global__ void naiveNBody3D(int *particles, double *par_vel, int N, int S, double dt, double G, double M, int pbc){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= S) return;
    int par = particles[tid], par_ref,local_x, local_y, local_z, ref_x, ref_y, ref_z;
    double tmp_x, tmp_y, tmp_z;
    local_x = par%N;
    local_y = par/N % N;
    local_z = par/(N * N);
    double force_x = 0.0, force_y = 0.0, force_z = 0.0;
    double d;
    float4 tmp;
    for(int j = 0; j < S; j++){
        if(tid == j) continue;
        par_ref = particles[j];
        ref_x = par_ref%N;
        ref_y = par_ref/N % N;
        ref_z = par_ref/(N * N);
        if(pbc) tmp = pbcEuclideanDistanceDyn(N, local_x, ref_x, local_y, ref_y, local_z, ref_z);
        else tmp = euclideanDistanceDyn(local_x, ref_x, local_y, ref_y, local_z, ref_z);
        tmp_x = double(tmp.y), tmp_y = double(tmp.z), tmp_z = double(tmp.w), d = double(tmp.x);
        if(par == par_ref) d = 0.001;
        force_x = force_x + tmp_x * (G * M * M / (d * d * d));
        force_y = force_y + tmp_y * (G * M * M / (d * d * d));
        force_z = force_z + tmp_z * (G * M * M / (d * d * d));
    }
    par_vel[tid*3] = par_vel[tid*3] + force_x * dt / M;
    par_vel[tid*3 + 1] = par_vel[tid*3 + 1] + force_y * dt / M;
    par_vel[tid*3 + 2] = par_vel[tid*3 + 2] + force_z * dt / M;

    local_x += par_vel[tid*3] * dt;
    local_y += par_vel[tid*3 + 1] * dt;
    local_z += par_vel[tid*3 + 2] * dt;
    if(pbc){
        local_x = int(local_x + N)%N;
        local_y = int(local_y + N)%N;
        local_z = int(local_z + N)%N;
    }
    else{
        if(local_x < 0 || local_x >= N) local_x = par%N;
        if(local_y < 0 || local_y >= N) local_y = par/N % N;
        if(local_z < 0 || local_z >= N) local_z = par/(N * N);
    }

    particles[tid] = int(local_z) * N * N + int(local_y) * N + int(local_x);
}
//  Exact LJ (use checkpoints as references)
void readCords3D(int *seeds, int N, int S, int count, int molecules){
	string name = "sample-lj/3d-sample-";
	name.insert(name.size(), to_string(molecules));
	name.insert(name.size(), "k/coords-voro/");
	name.insert(name.size(), to_string(count));
	name.insert(name.size(),".txt");
    
	ifstream FILE(name);
	int i = 0;
    int count_seed = 0;
	char *ptr;
    int n;
    int x, y, z;
    //Box assumption
    int max_x = -1;
	int max_y = -1;
    int max_z = -1;
    
    string text;
	while (getline(FILE, text)) {
        y = -1;
        z = -1;
        n = text.length();
        char aux[n+1];
        strcpy(aux, text.c_str());
		ptr = strtok(aux," ");
        x = atoi(ptr);
        while(ptr!= NULL){
            ptr = strtok(NULL," ");
            if(y==-1) y = atoi(ptr);
            else if(z == -1) z = atoi(ptr);
            
        }
        if( max_x < x) max_x = x;
		if( max_y < y) max_y = y;
        if( max_z < z) max_z = z;
        
        seeds[count_seed++] = z*N*N + y*N + x;
		
	}
	
    int delim = max(max_x + 2, max_y + 2);
    delim = max(delim, max_z + 2);
    for(int i = 0; i < S; ++i){
        int seed = seeds[i];
        int aux_x = seed%N;
        int aux_y = seed/N % N;
        int aux_z = seed/(N*N);
		//printf("%i %i %i\n", seed, aux_x, aux_y);
        aux_x = aux_x*N/delim;
        aux_y = aux_y*N/delim;
        aux_z = aux_z*N/delim;
        seeds[i] = aux_z * N * N + aux_y * N + aux_x;
		//printf("%i %i %i\n", seeds[i], aux_x, aux_y);
        if(seeds[i] >= N*N*N){
            printf("SOMETHING WRONG %i %i %i\n", aux_x, aux_y, aux_z);
            exit(0);
        }
    }
	FILE.close();
}
void updateLJ(Setup setup, int iter){
    readCords3D(setup.seeds, setup.N, setup.S, 100 - iter + 1, setup.molecules);
    cudaMemcpy(setup.gpu_seeds, setup.seeds, sizeof(int)*setup.S, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}
//  Approximate LJ
__global__ void approxLJ(int *particles, double *par_vel, int N, int S, double dt, double G, double M, int pbc){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= S) return;
    int par = particles[tid], par_ref,local_x, local_y, local_z, ref_x, ref_y, ref_z;
    double sigma = double(N) * 0.0116;
    double max_dist = sigma * 2.5;
    double tmp_x, tmp_y, tmp_z;
    local_x = par%N;
    local_y = par/N % N;
    local_z = par/(N * N);
    double force_x = 0.0, force_y = 0.0, force_z = 0.0, force;
    double d, well = 14.0, d13, d7, sigma12, sigma6;
    float4 tmp;
    for(int j = 0; j < S; j++){
        if(tid == j) continue;
        par_ref = particles[j];
        ref_x = par_ref%N;
        ref_y = par_ref/N % N;
        ref_z = par_ref/(N * N);
        if(pbc) tmp = pbcEuclideanDistanceDyn(N, local_x, ref_x, local_y, ref_y, local_z, ref_z);
        else tmp = euclideanDistanceDyn(local_x, ref_x, local_y, ref_y, local_z, ref_z);
        tmp_x = double(tmp.y), tmp_y = double(tmp.z), tmp_z = double(tmp.w), d = double(tmp.x);
        if(d < max_dist){
            d7 = d * d * d * d * d * d * d;
            d13 = d7 * d * d * d * d * d * d;
            sigma6 = sigma * sigma * sigma * sigma * sigma * sigma;
            sigma12 = sigma * sigma;
            force = 4.0 * well * (12.0 * sigma12/d13 - 6.0 * sigma6/d7);
            force_x = force_x + tmp_x * force / d;
            force_y = force_y + tmp_y * force / d;
            force_z = force_z + tmp_z * force / d;
        }
    }
    par_vel[tid*3] = par_vel[tid*3] + force_x * dt / M;
    par_vel[tid*3 + 1] = par_vel[tid*3 + 1] + force_y * dt / M;
    par_vel[tid*3 + 2] = par_vel[tid*3 + 2] + force_z * dt / M;

    local_x += par_vel[tid*3] * dt;
    local_y += par_vel[tid*3 + 1] * dt;
    local_z += par_vel[tid*3 + 2] * dt;
    if(pbc){
        local_x = int(local_x + N)%N;
        local_y = int(local_y + N)%N;
        local_z = int(local_z + N)%N;
    }
    else{
        if(local_x < 0 || local_x >= N) local_x = par%N;
        if(local_y < 0 || local_y >= N) local_y = par/N % N;
        if(local_z < 0 || local_z >= N) local_z = par/(N * N);
    }

    particles[tid] = int(local_z) * N * N + int(local_y) * N + int(local_x);
}
//  Uniform movement
__global__ void randomMovement3D(int *seeds, int *delta, int N, int S, int mod, curandState *states, int pbc){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= S) return;
    int seed = seeds[tid];
    int old_x, old_y, old_z, new_x, new_y, new_z, delta_x, delta_y, delta_z;
    old_x = seed%N;
    old_y = seed/N % N;
    old_z = seed/(N * N);

    delta_x = int(curand_uniform(&states[tid])*13.f)%13 - 6;
    delta_y = int(curand_uniform(&states[tid])*13.f)%13 - 6;
    delta_z = int(curand_uniform(&states[tid])*13.f)%13 - 6;

    new_x = old_x + delta_x;
    new_y = old_y + delta_y;
    new_z = old_z + delta_z;
    if(pbc){
        new_x = (new_x + N)%N;
        new_y = (new_y + N)%N;
        new_z = (new_z + N)%N;
        delta[tid] = int(pbcEuclideanDistanceDyn(N, old_x, new_x, old_y, new_y, old_z, new_z).x);
    }
    else{
        if(new_x < 0 || new_x >= N) new_x = old_x;
        if(new_y < 0 || new_y >= N) new_y = old_y;
        if(new_z < 0 || new_z >= N) new_z = old_z;
        delta[tid] = int(euclideanDistanceDyn(old_x, new_x, old_y, new_y, old_z, new_z).x);
    }
    seeds[tid] = new_z * N * N + new_y * N + new_x;
    //if(tid < 10) printf("id %i, local value: %i, pbc %i\noldx %i, newx %i, oldy %i, newy%i, oldz %i, newz %i\n", tid, seeds[tid], pbc, old_x, new_x, old_y/N % N, new_y/N % N, old_z, new_z);
}
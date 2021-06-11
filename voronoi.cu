#include "global.h"

using namespace std;

int main(int argc, char **argv){
    if(argc!=8){
        printf("Run as ./prog N S MODE ITER DEVICE BS MEM_GPU\n");
        printf("N - length/width of grid\n");
        printf("S - number of seeds\n");
        printf("MODE - 0 For classic JFA, 1 For dynamic approach\n");
        printf("ITER - number of iterations\n");
        printf("DEVICE - Wich device to use, fill with ID of GPU\n");
        printf("MEM_GPU - Memory management, 1: manual, 0: auto, anything if DEVICE = 0\n");
        printf("BS - block size of GPU (32 max), anything if DEVICE = 0\n");
        return(EXIT_FAILURE);
    }
    //Initialize argv
    int N = atoi(argv[1]);
    int S = atoi(argv[2]);
    int MODE = atoi(argv[3]);
    int ITER = atoi(argv[4]);
    int DEVICE = atoi(argv[5]);
    int MEM_GPU = atoi(argv[6]);
    int BS = atoi(argv[7]);
    //printf("BS %i\n", BS);
    //ITER MODE VORONOI AND OLD SEEDS
    int *VD = (int*)malloc(N*N*sizeof(int));
    int *SEEDS = (int*)malloc(S*sizeof(int));
    int *MAX = (int*)malloc(2*sizeof(int));
    int seed = 0;
    int k, k_mod, k_copy, k_mod_or;
    k = pow(2,int(log2(N)));//2^(roof(log2(N)) - 1)
    k_mod = 16;//
    //kmod = 2^(int(log2(dmax)) + 1)
    printf("MODE: %i\n", MODE);
    k_copy = k;
    k_mod_or = k_mod;


    //TIME VARIABLES
    double T1,T2;

    //GPU SETUP
    dim3 block_jfa(BS,BS,1);
    dim3 grid_jfa((N + BS + 1)/BS, (N + BS + 1)/BS,1);

    dim3 block_seeds(BS*BS,1,1);
    dim3 grid_seeds((S + BS + 1)/BS,1,1);

    dim3 blocktest(BS,1,1);
    dim3 gridtest((S + BS + 1)/BS, 1 ,1);
    
    initSeeds(SEEDS, N, S);

    int *GPU_VD;
    int *GPU_SEEDS;
    int *GPU_DELTA;
    int *GPU_DELTAMAX;
    

    cudaMalloc(&GPU_VD, N*N*sizeof(int));
    cudaMalloc(&GPU_SEEDS, S*sizeof(int));
    cudaMalloc(&GPU_DELTA, S*sizeof(int));
    cudaMalloc(&GPU_DELTAMAX, 2*sizeof(int));
    cudaMemcpy(GPU_SEEDS, SEEDS, S*sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    initVD<<<grid_jfa, block_jfa>>>(GPU_VD, N);
    init_GPUSeeds<<<grid_seeds, block_seeds>>>(GPU_VD, GPU_SEEDS, S);

    cudaDeviceSynchronize();
    
    //printf("%i\n",k_copy);
    /**/
    while(k_copy >0){
        //printf("local_k: %i\n", k_copy);
        voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, k_copy, N, S);
        cudaDeviceSynchronize();
        k_copy= k_copy/2;
        //printf("local_k: %i\n", k_copy);
    }
    /**/
    /*
    voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, 2, N, S);*/
    voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, 1, N, S);
    //printf("%i\n",k_copy);
    /**/
    

    cudaDeviceSynchronize();

    curandState *device;
    cudaMalloc((void**)&device, grid_seeds.x * grid_seeds.y * sizeof(curandState));

    T1 = omp_get_wtime();
    //WHILE ITER
    //JFA...
    //MOVER SEMILLAS
    /*
    while(ITER >= 0){
        k_copy = k;
        k_mod = k_mod_or;
        //printf("BEFORE MOVE\n");
        moveSeeds<<< grid_seeds, block_seeds>>>(GPU_SEEDS, GPU_DELTA,N, S, ITER + N, device);
        //printf("AFTER MOVE\n");
        if(MODE == 0){
            //printf("BEFORE INIT LOOP\n");
            initVD<<<grid_jfa, block_jfa>>>(GPU_VD, N);
            //printf("AFTER INIT LOOP\n");
            //printf("BEFORE INITGPUSEEDS LOOP\n");
            //init_GPUSeeds<<<grid_seeds, block_seeds>>>(GPU_VD, GPU_SEEDS, S);
            //printf("AFTER INITGPUSEEDS LOOP\n");
            while(k_copy>=1){
                voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS,k_copy, N, S);
                k_copy = k_copy/2;
            }
            //voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, 2, N);
            //voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, 1, N);
            //printf("TRAPPED ABOVE\n");
        }
        else if(MODE == 1){
            //int change = 0;
            while(k_mod >=1){
                //simple_max<<< gridtest, blocktest>>>(GPU_DELTA, GPU_DELTAMAX, S);
                voronoiJFA_4Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, GPU_DELTAMAX, k_mod, N, S);
                //voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS,k_mod, N, S);
                //voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, 2, N, S);
                //voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, 1, N, S);
                //if(change == 0) change = 1;
                //if(change == 1) change = 0;
                k_mod = k_mod/2;
            }
            voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, 4, N, S);
            //voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, 2, N, S);
            //voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, 1, N, S);
        }
        //Rondas de correccion
        voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, 2, N, S);
        voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, 1, N, S);
        //printf("AFTER JFA\n");
        ITER--;
    }
    */
    lastcheck<<<grid_seeds, block_seeds>>>(GPU_VD, GPU_SEEDS, S);/*
    cudaDeviceSynchronize();
    T2 = omp_get_wtime() - T1;
    */
    cudaMemcpy(VD, GPU_VD, N*N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(MAX, GPU_DELTAMAX, 2*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(GPU_VD);
    cudaFree(GPU_SEEDS);
    cudaFree(GPU_DELTA);
    cudaFree(GPU_DELTAMAX);
    cudaFree(device);
    //cudaFree(device);
    save_step(VD, N, 0);
    printf("TOTAL: %f\nLAST MAX: %i\n", T2, MAX[0]);
    //printf("%i\n", VD[6758]);
    if(N <= 100)printMat(N, VD);
    free(VD);
    free(SEEDS);
    free(MAX);

}
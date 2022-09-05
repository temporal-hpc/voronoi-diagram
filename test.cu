#include "global.h"
#define EPSILON 0.0001

using namespace std;

void jfaVDUnique(Setup setup,int *v_diagram, int *seeds, int N, int S, int k, int mu, dim3 grid, dim3 block){
    int k_copy = k;
    //printf("k:%i\n", k_copy);
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

void jfaVDIters(Setup setup,int *v_diagram, int *seeds, int N, int S, int k, int mu, dim3 grid, dim3 block){
    int k_copy = k;
    while(k_copy >= 1){
        voronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, k_copy, N, S, setup.distance_function, mu);
        cudaDeviceSynchronize();
        k_copy = k_copy/2;
    }
    voronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, 2, N, S, setup.distance_function, mu);
    voronoiJFA_8Ng<<< grid, block>>>(v_diagram, seeds, 1, N, S, setup.distance_function, mu);
}

void mjfaVDIters(Setup setup,int *v_diagram, int *seeds, int N, int S, int k, int mu, dim3 grid, dim3 block){
    int k_copy = k;
    while(k_copy >= k/2){
        voronoiJFA_4Ng<<< grid, block>>>(v_diagram, seeds, setup.gpu_delta_max, k_copy, N, S, setup.distance_function, mu);
        cudaDeviceSynchronize();
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

void itersJFA(Setup setup, int k, dim3 grid, dim3 block, dim3 grid_s, dim3 block_s, dim3 grid_r, dim3 block_r, curandState *device){
    int iter_copy = setup.iters;
    double T1, T2 = 0.0;
    while(iter_copy > 0){
        moveSeeds<<< grid_s, block_s>>>(setup.gpu_seeds, setup.gpu_delta,setup.N, setup.S, 1, device);
        cudaDeviceSynchronize();
        T1 = omp_get_wtime();
        jfaVDUnique(setup, setup.gpu_v_diagram, setup.gpu_seeds, setup.N, setup.S, k, 1, grid, block);
        cudaDeviceSynchronize();
        //scaleSeeds<<<grid_s,block_s>>>(setup.gpu_seeds, setup.gpu_seeds, setup.S, setup.N, setup.N_p, setup.mu);
        //cudaDeviceSynchronize();
        clearGrid<<<grid_r, block_r>>>(setup.gpu_redux_vd, setup.gpu_seeds,setup.N_p, setup.S);
        cudaDeviceSynchronize();
        //printf("k:%i, iter:%i\n", k, iter_copy);
        reduxVDSeeds<<<grid_s, block_s>>>(setup.gpu_redux_vd, setup.gpu_seeds, setup.N, setup.S, setup.N_p, setup.mu);
        cudaDeviceSynchronize();
        jfaVDUnique(setup, setup.gpu_redux_vd, setup.gpu_seeds, setup.N_p, setup.S, 256, setup.mu, grid_r, block_r);
        cudaDeviceSynchronize();
        T2 += omp_get_wtime() - T1;
        iter_copy -= 1;
    }
    printf("Time: %f\n", T2);
}

void itersMJFA(Setup setup, int km, dim3 grid, dim3 block){
    int iter_copy = setup.iters;
    int k = sqrt(setup.N * setup.N / setup.S);
    while(iter_copy > 0){
        if(setup.redux == 1) mjfaVDIters(setup, setup.gpu_redux_vd, setup.gpu_seeds, setup.N_p, setup.S, k, setup.mu, grid, block);
        else mjfaVDIters(setup, setup.gpu_v_diagram, setup.gpu_seeds, setup.N, setup.S, k, 1, grid, block);
        iter_copy -= 1;
    }
}

int main(int argc, char **argv){
    if(argc!=10){
        printf("Run as ./prog N S MODE ITER DEVICE MEM_GPU BS DIST\n");
        printf("N - length/width of grid\n");
        printf("S - number of seeds\n");
        printf("MODE - 0 For classic JFA, 1 For dynamic approach\n");
        printf("ITER - number of iterations\n");
        printf("DEVICE - Wich device to use, fill with ID of GPU\n");
        printf("MEM_GPU - Memory management, 1: manual, 0: auto, anything if DEVICE = 0\n");
        printf("BS - block size of GPU (32 max), anything if DEVICE = 0\n");
	    printf("DIST - distance method, 1: manhattan, 0: euclidean\n");
        printf("REDUX - if redux method is used, 0: no, 1: yes\n");
        return(EXIT_FAILURE);
    }
    //test setup
    Setup setup;
    setup = initialize_variables(setup, atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), atoi(argv[7]), atoi(argv[8]), atoi(argv[9]), 4);
    setup = allocate_arrays(setup);

    int k = pow(2,int(log2(setup.N)));//
    printf("k:%i\n", k);

    //Reducir desde aca

    //GPU SETUP
    dim3 block_jfa(setup.block_size,setup.block_size,1);
    dim3 grid_jfa((setup.N + setup.block_size + 1)/setup.block_size, (setup.N + setup.block_size + 1)/setup.block_size,1);

    dim3 block_seeds(setup.block_size,1,1);
    dim3 grid_seeds((setup.S + setup.block_size + 1)/setup.block_size,1,1);

    dim3 blocktest(setup.block_size,1,1);
    dim3 gridtest((setup.S + setup.block_size + 1)/setup.block_size, 1 ,1);
    
    //dim3 redux_block;
    //dim3 redux_grid;

    curandState *device;
    cudaMalloc((void**)&device, setup.S * sizeof(curandState));
    init_rand<<< grid_seeds, block_seeds>>>(setup.S, setup.N, device);
    cudaDeviceSynchronize();

    initSeeds(setup.seeds, setup.N, setup.S);



    cudaMalloc(&setup.gpu_v_diagram, setup.N*setup.N*sizeof(int));
    cudaMalloc(&setup.gpu_seeds, setup.S*sizeof(int));
    cudaMalloc(&setup.gpu_delta, setup.S*sizeof(int));
    //cudaMalloc(&GPU_AREAS, S*sizeof(int));
    cudaMalloc(&setup.gpu_delta_max, 2*sizeof(int));
    cudaMemcpy(setup.gpu_seeds, setup.seeds, setup.S*sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    initVD<<<grid_jfa, block_jfa>>>(setup.gpu_v_diagram, setup.N);
    cudaDeviceSynchronize();
    init_GPUSeeds<<<grid_seeds, block_seeds>>>(setup.gpu_v_diagram, setup.gpu_seeds, setup.S);
    cudaDeviceSynchronize();
    jfaVDUnique(setup, setup.gpu_v_diagram, setup.gpu_seeds, setup.N, setup.S, k, 1, grid_jfa, block_jfa);

    //Probar redux aca
    dim3 redux_block(setup.block_size, setup.block_size, 1);
    dim3 redux_grid((setup.N_p + setup.block_size + 1)/setup.block_size, (setup.N_p + setup.block_size + 1)/setup.block_size, 1);
    if(setup.redux == 1){
        //dim3 redux_block(setup.block_size, setup.block_size, 1);
        //dim3 redux_grid((setup.N_p + setup.block_size + 1)/setup.block_size, (setup.N_p + setup.block_size + 1)/setup.block_size, 1);

        cudaMalloc(&setup.gpu_redux_vd,setup.N_p*setup.N_p*sizeof(int));
        cudaDeviceSynchronize();
    
        reduxVDSeeds<<<grid_seeds, block_seeds>>>(setup.gpu_redux_vd, setup.gpu_seeds, setup.N, setup.S, setup.N_p, setup.mu);
        cudaDeviceSynchronize();
        
        jfaVDUnique(setup, setup.gpu_redux_vd, setup.gpu_seeds, setup.N_p, setup.S, k, setup.mu, redux_grid, redux_block);
        cudaDeviceSynchronize();
    }
    

    int k_copy = k;
    /**/

    //printf("%i\n",setup.iters);
    /*if(setup.redux == 1) itersJFA(setup, k, redux_grid, redux_block, grid_seeds, block_seeds, device);*/
    /*else*/ itersJFA(setup, k, grid_jfa, block_jfa, grid_seeds, block_seeds, redux_grid, redux_block, device);

    /**/
    
    if(setup.redux == 1){
        //scaleVoronoi<<<grid_jfa, block_jfa>>>(setup.gpu_v_diagram, setup.gpu_redux_vd, setup.N, setup.N_p, setup.mu);
        //cudaDeviceSynchronize();
        
        //voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(setup.gpu_v_diagram, setup.gpu_seeds, 4, setup.N, setup.S, setup.distance_function, 1);
        //voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(setup.gpu_v_diagram, setup.gpu_seeds, 2, setup.N, setup.S, setup.distance_function, 1);
        //voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(setup.gpu_v_diagram, setup.gpu_seeds, 1, setup.N, setup.S, setup.distance_function, 1);
        //jfaVDUnique(setup, setup.gpu_v_diagram, setup.gpu_seeds, setup.N, setup.S, 4, 1, grid_jfa, block_jfa);
        //cudaDeviceSynchronize();
    }
    

    cudaMemcpy(setup.v_diagram, setup.gpu_v_diagram, setup.N*setup.N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    if(setup.redux==1){
        cudaMemcpy(setup.redux_v_diagram, setup.gpu_redux_vd, setup.N_p*setup.N_p*sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    
    cudaFree(setup.gpu_v_diagram);
    if(setup.redux == 1){
        cudaFree(setup.gpu_redux_vd);
        //cudaFree(setup.gpu_redux_seeds);
    }
    cudaFree(setup.gpu_seeds);
    cudaFree(setup.gpu_delta);
    cudaFree(setup.gpu_delta_max);
    cudaFree(device);
    
    save_step(setup.v_diagram, setup.N, -1,1);
    save_step(setup.redux_v_diagram, setup.N_p, 0,1);
    
    free(setup.v_diagram);
    free(setup.redux_v_diagram);
    free(setup.seeds);
    free(setup.areas);
}
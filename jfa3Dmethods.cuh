#include "voronoi3d.cuh"
#include "dynamics.cuh"

// Methods
// Unique with full neighbor
void jfa3DVDUnique(Setup setup, float4 *v_diagram, int *seeds, int N, int S, int k, int mu, dim3 grid, dim3 block, int pbc){
    int k_copy = k;
    while(k_copy >= 1){
        //printf("k: %i\n", k_copy);
        voronoi26NB(setup, v_diagram, seeds, N, S, k_copy, mu, grid, block, pbc);
        cudaDeviceSynchronize();
        if(cudaGetLastError() != cudaSuccess){
            printf("Something went wrong on 26NB\n");
            exit(0);
        }
        k_copy = k_copy/2;
        
    }
    voronoi26NB(setup, v_diagram, seeds, N, S, 2, mu, grid, block, pbc);
    voronoi26NB(setup, v_diagram, seeds, N, S, 1, mu, grid, block, pbc); 
}
// Dynamic with different neighborhoods
void djfa3DVDIters(Setup setup, float4 *v_diagram, int *seeds, int N, int S, int k, int mu, dim3 grid, dim3 block, int pbc){
    int k_copy = k;
    while(k_copy >= k/2){
        voronoi6NB(setup, v_diagram, seeds, N, S, k_copy, mu, grid, block, pbc);
        k_copy = k_copy/2;
    }
    while(k_copy >= k/8){
        voronoi18NB(setup, v_diagram, seeds, N, S, k_copy, mu, grid, block, pbc);
        k_copy = k_copy/2;
    }
    while(k_copy >= 1){
        voronoi6NB(setup, v_diagram, seeds, N, S, k_copy, mu, grid, block, pbc);
        k_copy = k_copy/2;
    }
    voronoi26NB(setup, v_diagram, seeds, N, S, 2, mu, grid, block, pbc);
    voronoi26NB(setup, v_diagram, seeds, N, S, 1, mu, grid, block, pbc); 
}

// Base 3DJFA
void baseJFA3D(Setup setup){
    clearGrid<<<setup.normal_grid, setup.normal_block>>>(setup.gpu_backup_vd, setup.gpu_seeds, setup.N, setup.S);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on clear grid\n");
        exit(0);
    }
    initGPUSeeds3D<<<setup.seeds_grid, setup.seeds_block>>>(setup.gpu_backup_vd, setup.gpu_seeds, setup.S);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on initGPUSeeds3D\n");
        exit(0);
    }
    jfa3DVDUnique(setup, setup.gpu_backup_vd, setup.gpu_seeds, setup.N, setup.S, setup.k, setup.mu, setup.normal_grid, setup.normal_block, setup.pbc);
    cudaDeviceSynchronize();
    //std::cout<<cudaGetLastError();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on Base JFA3D\n");
        exit(0);
    }
}
// 3D-dJFA
void dJFA3D(Setup setup, int iter){
    if(iter==setup.iters || iter%5 == 0){
        baseJFA3D(setup);
    } else{
        djfa3DVDIters(setup, setup.gpu_backup_vd, setup.gpu_seeds, setup.N, setup.S, setup.k_m, setup.mu, setup.normal_grid, setup.normal_block, setup.pbc);
    }
}
// 3D-rJFA
void rJFA3D(Setup setup){
    clearGrid<<<setup.redux_grid, setup.redux_block>>>(setup.gpu_redux_vd, setup.gpu_seeds, setup.N_p, setup.S);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on reduce clear grid\n");
        exit(0);
    }
    reduxVDSeeds3D<<<setup.seeds_grid, setup.seeds_block>>>(setup.gpu_redux_vd, setup.gpu_seeds, setup.N, setup.S, setup.N_p, setup.mu);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on redux VD seeds\n");
        exit(0);
    }
    jfa3DVDUnique(setup, setup.gpu_redux_vd, setup.gpu_seeds, setup.N_p, setup.S, setup.k_r, setup.mu, setup.redux_grid, setup.redux_block, setup.pbc);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on rjfa\n");
        exit(0);
    }
    scaleVD3D<<<setup.seeds_grid, setup.seeds_block>>>(setup.gpu_backup_vd, setup.gpu_redux_vd, setup.N, setup.N_p, setup.mu);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on scale vd\n");
        exit(0);
    }
    jfa3DVDUnique(setup, setup.gpu_backup_vd, setup.gpu_seeds, setup.N, setup.S, setup.mu, 1, setup.normal_grid, setup.normal_block, setup.pbc);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on restoration\n");
        exit(0);
    }
}
// 3D-drJFA
void drJFA3D(Setup setup, int iter){
    if(iter == setup.iters || iter%5 == 0){
        clearGrid<<<setup.redux_grid, setup.redux_block>>>(setup.gpu_redux_vd, setup.gpu_seeds, setup.N_p, setup.S);
        cudaDeviceSynchronize();
        if(cudaGetLastError() != cudaSuccess){
            printf("Something went wrong on clear drjfa\n");
            exit(0);
        }
        reduxVDSeeds3D<<<setup.seeds_grid, setup.seeds_block>>>(setup.gpu_redux_vd, setup.gpu_seeds, setup.N, setup.S, setup.N_p, setup.mu);
        cudaDeviceSynchronize();
        if(cudaGetLastError() != cudaSuccess){
            printf("Something went wrong on redux VD seeds drjfa\n");
            exit(0);
        }
        jfa3DVDUnique(setup, setup.gpu_redux_vd, setup.gpu_seeds, setup.N_p, setup.S, setup.k_r, setup.mu, setup.redux_grid, setup.redux_block, setup.pbc);
        cudaDeviceSynchronize();
    } else{
        djfa3DVDIters(setup, setup.gpu_redux_vd, setup.gpu_seeds, setup.N_p, setup.S, setup.k_rm, setup.mu, setup.redux_grid, setup.redux_block, setup.pbc);
    }
    scaleVD3D<<<setup.seeds_grid, setup.seeds_block>>>(setup.gpu_backup_vd, setup.gpu_redux_vd, setup.N, setup.N_p, setup.mu);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on scale vd\n");
        exit(0);
    }
    jfa3DVDUnique(setup, setup.gpu_backup_vd, setup.gpu_seeds, setup.N, setup.S, setup.mu, 1, setup.normal_grid, setup.normal_block, setup.pbc);
    cudaDeviceSynchronize();
    if(cudaGetLastError() != cudaSuccess){
        printf("Something went wrong on restoration\n");
        exit(0);
    }
}
// Check similarity
double checkDistanceSim3D(int N, int x1, int x2, int y1, int y2, int z1, int z2){
    int tmp_x = x1 - x2;
    int tmp_y = y1 - y2;
    int tmp_z = z1 - z2;

    if(fabsf(float(tmp_x - N)) < fabsf(float(tmp_x))) tmp_x -= N;
    else if(fabsf(float(tmp_x + N)) < fabsf(float(tmp_x))) tmp_x += N;

    if(fabsf(float(tmp_y - N)) < fabsf(float(tmp_y))) tmp_y -= N;
    else if(fabsf(float(tmp_y + N)) < fabsf(float(tmp_y))) tmp_y += N;

    if(fabsf(float(tmp_z - N)) < fabsf(float(tmp_z))) tmp_z -= N;
    else if(fabsf(float(tmp_z + N)) < fabsf(float(tmp_z))) tmp_z += N;

    return sqrt(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z);
}
void compare_results(Setup setup, dim3 grid_jfa, dim3 block_jfa, int pbc, double *bw_acc){
    cudaMemcpy(setup.v_diagram, setup.gpu_v_diagram, sizeof(float4) * setup.N * setup.N * setup.N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemcpy(setup.backup_v_diagram, setup.gpu_backup_vd, sizeof(float4) * setup.N * setup.N * setup.N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    int total = 0, ref_1, ref_2, x, x1, x2, y, y1, y2, z, z1, z2, place;
    double dist_1, dist_2, acc;
    for(int i = 0; i < setup.N; ++i){
        x = i;
        for(int j = 0; j < setup.N; ++j){
            y = j;
            for(int k = 0; k < setup.N; ++k){
                z = k;
                place = z * setup.N * setup.N + y * setup.N + x;
                ref_1 = int(setup.v_diagram[place].w);
                ref_2 = int(setup.backup_v_diagram[place].w);
                if(ref_1 == ref_2) total += 1;
                else{
                    x1 = ref_1%setup.N;
                    y1 = ref_1/setup.N;
                    z1 = ref_1/(setup.N * setup.N);

                    x2 = ref_2%setup.N;
                    y2 = ref_2/setup.N;
                    z2 = ref_2/(setup.N * setup.N);

                    dist_1 = checkDistanceSim3D(setup.N, x, x1, y, y1, z, z1);
                    dist_2 = checkDistanceSim3D(setup.N, x, x2, y, y2, z, z2);
                    if(abs(dist_1 - dist_2) < EPSILON * double(setup.N)) total +=1;
                }
            }
        }
    }
    acc = double(total) / double(setup.N * setup.N * setup.N) * 100.0;
    if(bw_acc[0] < acc) bw_acc[0] = acc;
    if(bw_acc[1] > acc) bw_acc[1] = acc;
    bw_acc[2] += acc;
    //printf("%f\n", acc);
}
// Simulation
void dynamicSeeds(Setup setup, int iter){
    if(setup.sample == 0){
        randomMovement3D<<<setup.seeds_grid, setup.seeds_block>>>(setup.gpu_seeds, setup.gpu_delta,setup.N, setup.S, 1, setup.r_device, setup.pbc);
        cudaDeviceSynchronize();
        if(cudaGetLastError() != cudaSuccess){
                printf("Something went wrong on Uniform\n");
                exit(0);
        }
    }
    else if (setup.sample == 1){
        // Work in progress
    }
    else if(setup.sample == 2){
        for(int unb = 0; unb < 5; ++unb){
            naiveNBody3D<<< setup.seeds_grid, setup.seeds_block >>>(setup.gpu_seeds, setup.gpu_seeds_vel, setup.N, setup.S, setup.DT, G, setup.M, setup.pbc);
            cudaDeviceSynchronize();
            if(cudaGetLastError() != cudaSuccess){
                printf("Something went wrong on N-body\n");
                exit(0);
            }
        }
        
    }
}

void itersJFA3D(Setup setup){
    int iter_copy = setup.iters;

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    float ms1 = 0.0, ms2 = 0.0, ms1t = 0.0, ms2t = 0.0;
    double best_worst_acc[3];
    int count_acc = 0;
    best_worst_acc[0] = 0.0;
    best_worst_acc[1] = 100.0;
    best_worst_acc[2] = 0.0;

    while(iter_copy > 0){
        printf("ITER %i\n", 100-iter_copy);
        dynamicSeeds(setup, iter_copy);
        cudaEventRecord(begin);
        if(setup.comparison == 1){
            bruteReference<<<setup.normal_grid, setup.normal_block>>>(setup.gpu_v_diagram, setup.gpu_seeds, setup.N, setup.S, setup.distance_function, setup.pbc);
            cudaDeviceSynchronize();
            if(cudaGetLastError() != cudaSuccess){
                printf("Something went wrong on Brute reference\n");
                exit(0);
            }
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms1, begin, end);
        ms1t += ms1;

        cudaEventRecord(begin);
        // Base JFA
        if(setup.mode == 0)baseJFA3D(setup);
        // Dynamic JFA (dJFA)
        else if(setup.mode == 1)dJFA3D(setup, iter_copy);
        // Redux JFA (rJFA)
        else if(setup.mode == 2) rJFA3D(setup);
        // Dynamic-Redux JFA (drJFA)
        else if(setup.mode == 3) drJFA3D(setup, iter_copy);
        cudaEventRecord(end);

        cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms2, begin, end);
        ms2t += ms2;
        if(setup.comparison == 1){
            compare_results(setup, setup.normal_grid, setup.normal_block, setup.pbc, best_worst_acc);
            #ifdef SSTEP
                save_step(setup.backup_v_diagram, setup.N, 100-iter_copy,1);
                save_step(setup.v_diagram, setup.N, 100-iter_copy,0);
            #endif
        }
        count_acc+=1;
        //printf("TIMES %i, M1: %f, M2: %f\n", iter_copy, ms1, ms2);
        iter_copy -= 1;
    }
    ms1t = ms1t * 0.001;
    ms2t = ms2t * 0.001;
    best_worst_acc[2] = best_worst_acc[2] / double(count_acc);
    printf("TOTAL TIME REFERENCE: %f, TOTAL TIME PROPOSAL: %f\n", ms1t, ms2t);
    printf("ACCURACY - AVG: %f - BEST: %f - WORST: %f\n", best_worst_acc[2], best_worst_acc[0], best_worst_acc[1]);
    /*
    if(setup->mode == 0) write_data(setup.N, setup.S, "JFA", setup.mu, ms1_t, ms2_t, best_worst_acc[2], best_worst_acc[0], best_worst_acc[1]);
    else if(setup->mode == 1) write_data(setup.N, setup.S, "dJFA", setup.mu, ms1_t, ms2_t, best_worst_acc[2], best_worst_acc[0], best_worst_acc[1]);
    else if(setup->mode == 2) write_data(setup.N, setup.S, "rJFA", setup.mu, ms1_t, ms2_t, best_worst_acc[2], best_worst_acc[0], best_worst_acc[1]);
    else if(setup->mode == 3) write_data(setup.N, setup.S, "drJFA", setup.mu, ms1_t, ms2_t, best_worst_acc[2], best_worst_acc[0], best_worst_acc[1]);
    /**/
}
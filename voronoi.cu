#include "global.h"
#define EPSILON 0.0001

using namespace std;

int main(int argc, char **argv){
    if(argc!=9){
        printf("Run as ./prog N S MODE ITER DEVICE MEM_GPU BS DIST\n");
        printf("N - length/width of grid\n");
        printf("S - number of seeds\n");
        printf("MODE - 0 For classic JFA, 1 For dynamic approach\n");
        printf("ITER - number of iterations\n");
        printf("DEVICE - Wich device to use, fill with ID of GPU\n");
        printf("MEM_GPU - Memory management, 1: manual, 0: auto, anything if DEVICE = 0\n");
        printf("BS - block size of GPU (32 max), anything if DEVICE = 0\n");
	printf("DIST - distance method, 1: manhattan, 0: euclidean\n");
        return(EXIT_FAILURE);
    }
    //Initialize argv
    int N = atoi(argv[1]);
    //Test purposes
    int AREA = atoi(argv[2]);
    int S = N*N/AREA;
    //int S = atoi(argv[2]);

    int MODE = atoi(argv[3]);
    int ITER = atoi(argv[4]);
    int OG_ITER = ITER;
    int DEVICE = atoi(argv[5]);
    int MEM_GPU = atoi(argv[6]);
    int BS = atoi(argv[7]);
    int DIST = atoi(argv[8]);
    //printf("%i\n", S);
    //ITER MODE VORONOI AND OLD SEEDS
    int *VD = (int*)malloc(N*N*sizeof(int));
    int *VD_REF = (int*)malloc(N*N*sizeof(int));
    int *SEEDS = (int*)malloc(S*sizeof(int));
    int *AREAS = (int*)malloc(S*sizeof(int));
    int *MAX = (int*)malloc(2*sizeof(int));
    int seed = 0;
    int k, k_mod, k_copy, k_mod_or, k_ref;
    k = pow(2,int(log2(N)));//
    if(N>=S){
        k_mod = pow(2, ceil(log2(6)) + int(log2(N)) - int(round(log2(S))) + int(log10(N)));//
    }
    else k_mod = pow(2, ceil(log2(6)) + int(log2(N)) - int(round(log2(S))) + int(log10(N)) + int(round(log10(S/N))));
    
    if(k_mod >= k/4) k_mod = k/4;
    int dmax = 6;
    int fac = int(log2(2*N/sqrt(S)));

    //TIME VARIABLES
    double T1,T2,t_create;

    //GPU SETUP
    dim3 block_jfa(BS,BS,1);
    dim3 grid_jfa((N + BS + 1)/BS, (N + BS + 1)/BS,1);

    dim3 block_seeds(BS,1,1);
    dim3 grid_seeds((S + BS + 1)/BS,1,1);

    dim3 blocktest(BS,1,1);
    dim3 gridtest((S + BS + 1)/BS, 1 ,1);
    
    initSeeds(SEEDS, N, S);

    int *GPU_VD;
    int *GPU_VD_REF;
    int *GPU_SEEDS;
    int *GPU_DELTA;
    int *GPU_DELTAMAX;
    int *GPU_AREAS;
    

    cudaMalloc(&GPU_VD, N*N*sizeof(int));
    cudaMalloc(&GPU_VD_REF, N*N*sizeof(int));
    cudaMalloc(&GPU_SEEDS, S*sizeof(int));
    cudaMalloc(&GPU_DELTA, S*sizeof(int));
    //cudaMalloc(&GPU_AREAS, S*sizeof(int));
    cudaMalloc(&GPU_DELTAMAX, 2*sizeof(int));
    cudaMemcpy(GPU_SEEDS, SEEDS, S*sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    initVD<<<grid_jfa, block_jfa>>>(GPU_VD, N);
    cudaDeviceSynchronize();
    init_GPUSeeds<<<grid_seeds, block_seeds>>>(GPU_VD, GPU_SEEDS, S);
    cudaDeviceSynchronize();
    
    //printf("%i\n",k_copy);
    
    while(k_copy >0){
        //printf("local_k: %i\n", k_copy);
        /*
        voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, k_copy, N, S);
        /**/
        voronoiJFA_8NgV21<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, k_copy, N, S, 0);
        cudaDeviceSynchronize();
        voronoiJFA_8NgV22<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, k_copy, N, S, 0);
        cudaDeviceSynchronize();
        voronoiJFA_8NgV23<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, k_copy, N, S, 0);
        /**/
        cudaDeviceSynchronize();
        k_copy= k_copy/2;
        //printf("local_k: %i\n", k_copy);
    }
    /**/
    
    voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, 2, N, S, 0);
    voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, 1, N, S, 0);
    //printf("%i\n",k_copy);
    /**/
    cudaDeviceSynchronize();
    cudaMemcpy(VD, GPU_VD, N*N*sizeof(int),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemcpy(GPU_VD_REF, VD, N*N*sizeof(int),cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    //t_create = omp_get_wtime();
    for(int i = 0; i < N*N; ++i){
        AREAS[VD[i]] +=1;
    }
    int t_areas = 0;
    double avg = 0.0;
    for(int i = 0; i < S; ++i){
        //printf("%i\n",AREAS[i]);
        t_areas += AREAS[i];
	avg += sqrt(AREAS[i]);
    }
    //printf("AREAS AVG = %f\n",double(t_areas/S));
    avg /=S;
    double sd = standard_deviation(AREAS,avg,S);
    //printf("STD: %f\n", sd);
    t_create = omp_get_wtime() - t_create;
    //printf("ELAB: %f\n", t_create);
    fac = int(log2(avg)) + 1;
    //printf("FAC: %i \n", fac);
    k_mod_or = pow(2,(max(int(log2(dmax)), fac)));
    //k_mod_or+=1;
    cudaDeviceSynchronize();
    curandState *device;
    cudaMalloc((void**)&device, S * sizeof(curandState));
    //printf("KMOD: %i\n", k_mod_or);
    T1 = omp_get_wtime();
    //WHILE ITER
    //JFA...
    //MOVER SEMILLAS
    /**/
    double acc = 0.0, total_acc = 0.0, worst=100.0, best = 0.0, total_acc_fr = 0.0, acc_fr = 0.0, best_fr = 0.0, worst_fr = 100.0;
    int  local_x, local_y, ref_x, ref_y, total_fr;
    double dist_a, dist_b;
    int times = 0;

    double total_ref = 0.0, time_ref, total_mjfa = 0.0, time_mjfa;
    init_rand<<< grid_seeds, block_seeds>>>(S, N, device);
    while(ITER >= 0){
        k_copy = k;
        k_ref = k;
        k_mod = k_mod_or;
        printf("ITER %i\n", ITER);
        moveSeeds<<< grid_seeds, block_seeds>>>(GPU_SEEDS, GPU_VD, GPU_DELTA,N, S, ITER + N, device);
        cudaDeviceSynchronize();
        
        //cudaDeviceSynchronize();
        if(MODE == 0 || MODE == 2){
            
            initVD<<<grid_jfa, block_jfa>>>(GPU_VD_REF, N);
            time_ref = omp_get_wtime();
            while(k_ref>=1){
                voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD_REF, GPU_SEEDS,k_ref, N, S, 0);
                cudaDeviceSynchronize();
                k_ref = k_ref/2;
            }
            time_ref = omp_get_wtime() - time_ref;
            total_ref += time_ref;
            //printf("TRAPPED ABOVE\n");
        }
        if(MODE == 1 || MODE == 2){
            time_mjfa = omp_get_wtime();
            while(k_mod>=k_mod_or/2){
                voronoiJFA_4Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, GPU_DELTAMAX, k_mod, N, S, DIST);
                cudaDeviceSynchronize();
                k_mod = k_mod/2;
            }
            while(k_mod >=1){
                voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS,k_mod, N, S, DIST);
                cudaDeviceSynchronize();
                k_mod = k_mod/2;
            }
            time_mjfa = omp_get_wtime() - time_mjfa;
            total_mjfa += time_mjfa;
        }
        //Rondas de correccion
        voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, 2, N, S, DIST);
        voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD, GPU_SEEDS, 1, N, S, DIST);
        voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD_REF, GPU_SEEDS, 2, N, S, 0);
        voronoiJFA_8Ng<<< grid_jfa, block_jfa>>>(GPU_VD_REF, GPU_SEEDS, 1, N, S, 0);
        //printf("AFTER JFA\n");
        ITER--;
        //printf("ITER: %i\n",ITER);
        #ifdef SAVE_ITER
            cudaDeviceSynchronize();
            if(MODE == 1 || MODE == 2)cudaMemcpy(VD, GPU_VD, N*N*sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            if(MODE == 0 || MODE == 2)cudaMemcpy(VD_REF, GPU_VD_REF, N*N*sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            cudaMemcpy(SEEDS, GPU_SEEDS, S*sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            //save_step(VD, N, ITER,1);
            //save_step(VD_REF, N, ITER,0);
        #endif
        /**/
        if(MODE == 2 && ITER%10==0){
            //acc = 0.0;
            int count = 0;
	    int count_fr = 0;
            times +=1;
            for(int y = 0; y < N; ++y){
                for(int x = 0; x < N; ++x){
                    if(VD[y*N + x] == VD_REF[y*N + x]){
			    //acc+=1.0f;
			    count += 1;
		    }
                    else{
			//printf("Voronoi en x=%i y=%i son distintos,  MJFA=%i, JFA=%i\n", x, y, VD[y*N+x], VD_REF[y*N+x]);
                        local_x = SEEDS[VD[y*N + x]]%N;
                        local_y = SEEDS[VD[y*N + x]]/N;

                        ref_x = SEEDS[VD_REF[y*N + x]]%N;
                        ref_y = SEEDS[VD_REF[y*N + x]]/N;
                        
                        dist_a = sqrt((x - local_x)*(x - local_x) + (y - local_y)*(y - local_y));
                        dist_b = sqrt((x - ref_x)*(x - ref_x) + (y - ref_y)*(y - ref_y));
			//printf("local_x=%i local_y=%i  ref_x=%i ref_y=%i\ndist_a=%f - dist_b=%f = %f\n", local_x, local_y, ref_x, ref_y, dist_a, dist_b, dist_a-dist_b);
                        
                        //if(abs(dist_a - dist_b)<= 0.01) acc+=1.0;
                        if(abs(dist_a - dist_b)<= EPSILON*N) count += 1;
			//getchar();
			//printf("%i is BORDER %i %i\n", is_border(x, y, VD_REF, N), x, y);
                    }
		    //printf("%i is BORDER\n", is_border(ref_x, ref_y, VD_REF, N));
		    if(is_border(x ,y, VD_REF, N)==1){
			//printf("CHECK BORDER\n");
			total_fr +=1;
			if(VD[y*N + x] == VD_REF[y*N + x]){
                        	    //acc+=1.0f;
                	            count_fr += 1;
        	        }
	                else{
                        	//printf("Voronoi en x=%i y=%i son distintos,  MJFA=%i, JFA=%i\n", x, y, VD[y*N+x], VD_REF[y*N+x]);
                	        local_x = SEEDS[VD[y*N + x]]%N;
        	                local_y = SEEDS[VD[y*N + x]]/N;

	                        ref_x = SEEDS[VD_REF[y*N + x]]%N;
                        	ref_y = SEEDS[VD_REF[y*N + x]]/N;

                      		dist_a = sqrt((x - local_x)*(x - local_x) + (y - local_y)*(y - local_y));
                	        dist_b = sqrt((x - ref_x)*(x - ref_x) + (y - ref_y)*(y - ref_y));
        	                //printf("local_x=%i local_y=%i  ref_x=%i ref_y=%i\ndist_a=%f - dist_b=%f = %f\n", local_x, local_y, ref_x, ref_y, dist_a, dist_b, dist_a-dist_b);
	
        	                //if(abs(dist_a - dist_b)<= 0.01) acc+=1.0;
	                        if(abs(dist_a - dist_b)<= EPSILON*N) count_fr += 1;
                        	//getchar();
                    	}
			//printf("%i %i\n", count_fr, total_fr);
		    }
		    //printf("OUT OF\n");
                }
            }
	    //printf("CHECK\n");
            acc = (double)count/((double)N*(double)N) * 100.0;
	    acc_fr = (double)count_fr/((double)total_fr) * 100.0;
	    if(acc<worst) worst = acc;
	    if(acc>best) best = acc;
	    if(acc_fr < worst_fr) worst_fr = acc_fr;
	    if(acc_fr > best_fr) best_fr = acc_fr;
            printf("accuracy = %f\n",acc);
            printf("accuracy fr = %f\n", acc_fr);
	    total_acc += acc;
	    total_acc_fr +=acc_fr;
	    total_fr = 0;
        }
        /**/
    }
    printf("%f\n",total_acc/times);
    printf("%f\n", total_acc_fr/times);
    write_acc(total_acc/times, best, worst, acc_fr, best_fr, worst_fr, N, AREA);
    lastcheck<<<grid_seeds, block_seeds>>>(GPU_VD, GPU_SEEDS, S);/**/
    cudaDeviceSynchronize();
    //printf("%f \n", total_ref);
    //printf("%f \n", total_mjfa);
    float spup = total_ref/total_mjfa;
    //printf("%f\n",spup);
    write_time(total_ref, AREA, N, 0, DIST);
    write_time(total_mjfa, AREA, N, 1, DIST);
    //write_sp(spup, AREA, N);
    T2 = omp_get_wtime() - T1;
    /**/
    cudaMemcpy(VD, GPU_VD, N*N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(MAX, GPU_DELTAMAX, 2*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(GPU_VD);
    cudaFree(GPU_VD_REF);
    cudaFree(GPU_SEEDS);
    cudaFree(GPU_DELTA);
    cudaFree(GPU_DELTAMAX);
    cudaFree(device);
    //cudaFree(device);
    save_step(VD, N, -1,1);
    /*
    for(int i = 0; i < N*N; ++i){
        AREAS[VD[i]] +=1;
    }
    int t_areas = 0;
    for(int i = 0; i < S; ++i){
        printf("AREA %i: %i\n",i,AREAS[i]);
        t_areas += AREAS[i];
    }
    printf("AREAS AVG = %f\n",double(t_areas/S));
    int avg = N*N/S;
    double sd = standard_deviation(AREAS,avg,S);
    printf("STD: %f\n", sd);
    */
    //printf("%f\n", T2);
    //printf("%i\n", VD[6758]);
    //if(N <= 100)printMat(N, VD);
    free(VD);
    free(VD_REF);
    free(SEEDS);
    free(MAX);
    free(AREAS);
}

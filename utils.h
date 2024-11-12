#include <string>
using namespace std;

__device__ void reduce(volatile int *data, int tid){
    if(data[tid] < data[tid + 32]) data[tid] = data[tid + 32];
    if(data[tid] < data[tid + 16]) data[tid] = data[tid + 16];
    if(data[tid] < data[tid + 8]) data[tid] = data[tid + 8];
    if(data[tid] < data[tid + 4]) data[tid] = data[tid + 4];
    if(data[tid] < data[tid + 2]) data[tid] = data[tid + 2];
    if(data[tid] < data[tid + 1]) data[tid] = data[tid + 1];
}

__global__ void simple_max(int *data, int *max, int n, int s){
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    extern __shared__ int sd_data[];
    //printf("HERE\n");
    if(i < n){
        sd_data[tid] = data[i];
        if( data[i] < data[i + blockDim.x]) sd_data[tid] = data[i + blockDim.x];
    }
    __syncthreads();

    for(unsigned int s = blockDim.x/2; s>32; s>>=1){
        if(tid < s){
            if(sd_data[tid] < sd_data[tid + s]) sd_data[tid] = sd_data[tid + s];
        }
        __syncthreads();
    }
    if(tid < 32) reduce(sd_data, tid);
    __syncthreads();
    if(tid == 0) max[0] = int(pow(2,int( ceilf(log2f(double(sd_data[0]) )) + log2f(n) - log2f(s)) + 2 ) );
}


__global__ void getAreas(int *VD, int *A_S, int N, int S){
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tid = tidy*N + tidx;
    extern __shared__ int sums[];

    __syncthreads();
    if(tid < S){
        A_S[tid] = 0;
        sums[tid] = 0;
    }

    __syncthreads();

    if(tid < N){
        int index = VD[tid];
        atomicAdd(&A_S[index],1);
    }
    __syncthreads();
    /*
    if(tid < N){
        sums[VD[tid]] +=1;
    }
    __syncthreads();
    if(tid < S){
        A_S[tid] = sums[tid];
    }
    */
}

void printMat(int n, int *map){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%2i ", map[i*n+j]);
            //printf("%4i ", map[i*n + j]);
        }
        printf("\n");
    }
}

void write_time(float time, int AREA, int N, int mode, int DIST){
    string name;
    if(DIST==1) name = "test_results/manhattan/time_";
    else name = "test_results/euclidean/time_";
    name.insert(name.size(),to_string(N));
    if(mode == 0) name.insert(name.size(),"/JFA");
    else if(mode == 1) name.insert(name.size(),"/MJFA");
    name.insert(name.size(),"/S");
    name.insert(name.size(),to_string(AREA));
    //ofstream FILE(name);
    ofstream myfile;
    myfile.open(name,fstream::app);
    string data = to_string(time);
    myfile<<data.insert(data.size(),"\n");
    myfile.close();
}

void write_sp(float sp, int AREA, int N){
    string name = "test_results/6th_test_";
    name.insert(name.size(),to_string(N));
    name.insert(name.size(),"/S");
    name.insert(name.size(),to_string(AREA));
    //ofstream FILE(name);
    ofstream myfile;
    myfile.open(name,fstream::app);
    string data = to_string(sp);
    myfile<<data.insert(data.size(),"\n");
    myfile.close();
}

#ifdef SSTEP
void save_step(float4 *map, int n, int step, int dec){
    string name;
    if(dec == 1) name ="example/map";
    else if(dec == 0) name="example2/map";
    name.insert(name.size(),to_string(step));
    name.insert(name.size(),".txt");
    ofstream FILE(name);
    if(FILE.is_open()){
        for(int i = 0; i < n; ++i){
            for(int j = 0; j < n; ++j){
                string data = to_string(map[i*n + j]);
                FILE<<data.insert(data.size()," ");
            }
            FILE<<"\n";
        }
        FILE.close();
    }
    else printf("Unable to write step\n");

}
#endif

void write_acc(double acc, double best, double worst, double acc_fr, double best_fr, double worst_fr, int N, int S){
	string name;
	name = "example/N";
	name.insert(name.size(),to_string(N));
	name.insert(name.size(),"/S");
	name.insert(name.size(),to_string(S));
	name.insert(name.size(),".txt");

	ofstream FILE(name);
	if(FILE.is_open()){
		string data = to_string(acc);
		data.insert(data.size()," ");
		data.insert(data.size(),to_string(best));
		data.insert(data.size()," ");
		data.insert(data.size(),to_string(worst));
		data.insert(data.size()," ");
                data.insert(data.size(),to_string(acc_fr));
		data.insert(data.size()," ");
                data.insert(data.size(),to_string(best_fr));
		data.insert(data.size()," ");
                data.insert(data.size(),to_string(worst_fr));
		FILE<<data;
		FILE<<"\n";
		FILE.close();
	}
}

void write_data(int N, int S, string method, int mu, double time_1, double time_2, double avg_acc, double best_acc, double worst_acc){
    string name = "metrics-nb/";
    name.insert(name.size(), method);
    name.insert(name.size(), "/");
    name.insert(name.size(), to_string(N));
    name.insert(name.size(), "x");
    name.insert(name.size(), to_string(N));
    name.insert(name.size(), "/");
    name.insert(name.size(), to_string(S));
    name.insert(name.size(), "_");
    name.insert(name.size(), to_string(mu));
    fstream FILE(name, fstream::app);
	if(FILE.is_open()){
        string data = to_string(time_1);
        data.insert(data.size(), " ");
        data.insert(data.size(), to_string(time_2));
        data.insert(data.size(), " ");
        data.insert(data.size(), to_string(avg_acc));
        data.insert(data.size(), " ");
        data.insert(data.size(), to_string(best_acc));
        data.insert(data.size(), " ");
        data.insert(data.size(), to_string(worst_acc));
        data.insert(data.size(), "\n");
        FILE<<data;
        FILE.close();
    }
}

void initSeeds(int *SEEDS, int N, int S){
    int i;
    vector<int> POSSIBLE_SEEDS;
    srand(time(0));

    for(i = 0; i < N*N; ++i) POSSIBLE_SEEDS.push_back(i);

    random_shuffle(POSSIBLE_SEEDS.begin(), POSSIBLE_SEEDS.end());
    
    for(i = 0; i < S; ++i){
        SEEDS[i] = POSSIBLE_SEEDS[i];
        #ifdef DEBUG
            if(S <= 500 )printf("%i\n", SEEDS[i]);
        #endif
    }
}

void initSeeds3D(int *SEEDS, int N, int S){
    int i;
    vector<int> POSSIBLE_SEEDS;
    srand(time(0));

    for(i = 0; i < N*N*N; ++i) POSSIBLE_SEEDS.push_back(i);

    random_shuffle(POSSIBLE_SEEDS.begin(), POSSIBLE_SEEDS.end());
    
    for(i = 0; i < S; ++i){
        SEEDS[i] = POSSIBLE_SEEDS[i];
        //#ifdef DEBUG
            //if(S<=500 && i <= 10 )printf("%i\n", SEEDS[i]);
        //#endif
    }
}

double standard_deviation(int *data, int avg, int S){
    double sum = 0;
    for(int i = 0; i < S; ++i){
        sum += pow(data[i] - avg,2)/S;
    }
    //sum = sum/(S);
    return sqrt(sum);
}

int is_border(int local_x, int local_y, int *VD, int N){
	
	//Third neighborhood
	int v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24;
	int v1_x = local_x - 3, v2_x = local_x - 2, v3_x = local_x - 1, v4_x = local_x, v5_x = local_x + 1, v6_x = local_x + 2, v7_x = local_x + 3;
	int v1_y = local_y - 3, v2_y = local_y - 2, v3_y = local_y - 1, v4_y = local_y, v5_y = local_y + 1, v6_y = local_y + 2, v7_y = local_y + 3;
	int VD_ref = VD[local_y * N + local_x];	
	//printf("%i %i %i\n", local_x, local_y, VD_ref);
	if(v1_y >= 0){
		if(v1_x >= 0){
			v1 = v1_y * N + v1_x;
			if(VD_ref != VD[v1]) return 1;
		}
		if(v2_x >= 0){
			v2 = v1_y * N + v2_x;
			if(VD_ref != VD[v2]) return 1;
		}
		if(v3_x >= 0){
			v3 = v1_y * N + v3_x;
			if(VD_ref != VD[v3]) return 1;
		}
		v4 = v1_y * N + v4_x;
		if(VD_ref != VD[v4]) return 1;
		if(v5_x < N){
			v5 = v1_y * N + v5_x;
			if(VD_ref != VD[v5]) return 1;
		}
		if(v6_x < N){
			v6 = v1_y * N + v6_x;
			if(VD_ref != VD[v6]) return 1;
		}
		if(v7_x < N){
			v7 = v1_y * N + v7_x;
			if(VD_ref != VD[v7]) return 1;
		}
	}
	

	if(v7_x < N){
		if(v2_y >= 0){
			v8 = v2_y * N + v7_x;
			if(VD_ref != VD[v8]) return 1;
		}
		if(v3_y >= 0){
			v9 = v3_y * N + v7_x;
			if(VD_ref != VD[v9]) return 1;
		}
		v10 = v4_y * N + v7_x;
		if(VD_ref != VD[v10]) return 1;
		if(v5_y < N){
			v11 = v5_y * N + v7_x;
			if(VD_ref != VD[v11]) return 1;
		}
		if(v6_y < N){
			v12 = v6_y * N + v7_x;
			if(VD_ref != VD[v12]) return 1;
		}
	}
	if(v7_y < N){
		if(v7_x < N){
			v13 = v7_y * N + v7_x;
			if(VD_ref != VD[v13]) return 1;
		}
		if(v6_x < N){
			v14 = v7_y * N + v6_x;
			if(VD_ref != VD[v14]) return 1;
		}
		if(v5_x < N){
			v15 = v7_y * N + v5_x;
			if(VD_ref != VD[v15]) return 1;
		}
		v16 = v7_y * N + v4_x;
		if(VD_ref != VD[v16]) return 1;
		if(v3_x >= 0){
			v17 = v7_y * N + v3_x;
			if(VD_ref != VD[v17]) return 1;
		}
		if(v2_x >= 0){
			v18 = v7_y * N + v2_x;
			if(VD_ref != VD[v18]) return 1;
		}
		if(v1_x >= 0){
			v19 = v7_y * N + v1_x;
			if(VD_ref != VD[v19]) return 1;
		}
	}
	if(v1_x >= 0){
		if(v6_y < N){
			v20 = v6_y * N + v1_x;
			if(VD_ref != VD[v20]) return 1;
		}
		if(v5_y < N){
			v21 = v5_y * N + v1_x;
			if(VD_ref != VD[v21]) return 1;
		}
		v22 = v4_y * N + v1_x;
		if(VD_ref != VD[v22]) return 1;
		if(v3_y >= 0){
			v23 = v3_y * N + v1_x;
			if(VD_ref != VD[v23]) return 1;
		}
		if(v2_y >= 0){
			v24 = v2_y * N + v1_x;
			if(VD_ref != VD[v24]) return 1;
		}
	}

	return 0;
}

void read_coords(int *seeds, int N, int S, int count, int molecules){
	string name = "sample-lj/2d-sample-";
	name.insert(name.size(), to_string(molecules));
	name.insert(name.size(), "k/coords-voro/");
	name.insert(name.size(), to_string(count));
	name.insert(name.size(),".txt");
    
	ifstream FILE(name);
	//int i = 0;
    int count_seed = 0;
	char *ptr;
    int n;
    int x,y;
    //Box assumption
    int max_x = -1;
	int max_y = -1;
    
    string text;
	while (getline(FILE, text)) {
        y = -1;
        n = text.length();
        char aux[n+1];
        strcpy(aux, text.c_str());
		ptr = strtok(aux," ");
        x = atoi(ptr);
        while(ptr!= NULL){
            ptr = strtok(NULL," ");
            if(y==-1) y = atoi(ptr);
            
        }
        if( max_x < x) max_x = x;
		if( max_y < y) max_y = y;
        
        seeds[count_seed++] = y*N + x;
		
	}
	
    int delim = max(max_x + 2, max_y + 2);
    for(int i = 0; i < S; ++i){
        int seed = seeds[i];
        int aux_x = seed%N;
        int aux_y = seed/N;
		//printf("%i %i %i\n", seed, aux_x, aux_y);
        aux_x = aux_x*N/delim;
        aux_y = aux_y*N/delim;
        seeds[i] = aux_y * N + aux_x;
		//printf("%i %i %i\n", seeds[i], aux_x, aux_y);
        if(seeds[i] >= N*N){
            printf("SOMETHING WRONG %i %i\n", aux_x, aux_y);
            exit(0);
        }
    }
	FILE.close();

}

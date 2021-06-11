using namespace std;

__device__ void reduce(volatile int *data, int tid){
    if(data[tid] < data[tid + 32]) data[tid] = data[tid + 32];
    if(data[tid] < data[tid + 16]) data[tid] = data[tid + 16];
    if(data[tid] < data[tid + 8]) data[tid] = data[tid + 8];
    if(data[tid] < data[tid + 4]) data[tid] = data[tid + 4];
    if(data[tid] < data[tid + 2]) data[tid] = data[tid + 2];
    if(data[tid] < data[tid + 1]) data[tid] = data[tid + 1];
}

__global__ void simple_max(int *data, int *max, int n){
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    extern __shared__ int sd_data[];
    //printf("HERE\n");
    if(i < n){
        sd_data[tid] = data[i];
        if( data[i] < data[i + blockDim.x]) sd_data[tid] = data[i + blockDim.x];
    }
    __syncthreads();

    for(unsigned int s = blockDim.x/2; s>32; s>>1){
        if(tid < s){
            if(sd_data[tid] < sd_data[tid + s]) sd_data[tid] = sd_data[tid + s];
        }
        __syncthreads();
    }
    if(tid < 32) reduce(sd_data, tid);
    __syncthreads();
    if(tid == 0) max[0] = int(pow(2,int(log2(double(sd_data[0])))));
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

#ifdef SSTEP
void save_step(int *map, int n, int step){
    string name = "example/map";
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
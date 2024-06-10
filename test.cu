#include "voronoi.cuh"

using namespace std;

int main(int argc, char **argv){
    if(argc!=15){
        printf("Run as ./prog N S MODE ITER DEVICE MEM_GPU BS DIST REDUX Select-MU MU SAMPLE MOLECULES\n");
        printf("N - length/width of grid\n");
        printf("S - number of seeds\n");
        printf("MODE - (0) JFA, (1) dJFA, (2) rJFA, (3) drJFA\n");
        printf("ITER - number of iterations\n");
        printf("DEVICE - Wich device to use, fill with ID of GPU\n");
        printf("MEM_GPU - Memory management, 1: manual, 0: auto, anything if DEVICE = 0\n");
        printf("BS - block size of GPU (32 max), anything if DEVICE = 0\n");
	    printf("DIST - distance method, 1: manhattan, 0: euclidean\n");
        printf("REDUX - if redux method is used, 0: no, 1: yes\n");
        printf("Select MU - election of MU, 1: arbitrary, 0:estimated\n");
        printf("MU - value of MU if arbitrary\n");
        printf("SAMPLE - 0: rand uniform, 1: lennard jones sample, 2: nbody");
        printf("MOLECULES - for lennard jones sample {25,50,75,100,125,150,175,200}k");
        printf("DEBUG SIM - if compare against naive, 1: yes, 0: no");
        return(EXIT_FAILURE);
    }
    Setup setup;
    initialize_variables(&setup,
                        atoi(argv[1]),
                        atoi(argv[2]),
                        atoi(argv[3]),
                        atoi(argv[4]),
                        atoi(argv[5]),
                        atoi(argv[6]),
                        atoi(argv[7]),
                        atoi(argv[8]),
                        atoi(argv[9]),
                        atoi(argv[11]),
                        atoi(argv[12]),
                        atoi(argv[13]),
                        atoi(argv[14])
    );
    allocate_arrays(&setup);
    printRunInfo(&setup);
    setSeeds(&setup);
    setDeviceInfo(&setup);
    setRandDevice(&setup);
    
    itersJFA(setup);
    getDeviceArrays(&setup);
    freeSpace(&setup);
}

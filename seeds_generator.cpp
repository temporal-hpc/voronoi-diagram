#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <chrono>
#include <vector>
#include <algorithm>
#include <ctime>

using namespace std;

void save_seeds(int *SEEDS, int S, int iter, int version){
    int true_iter = iter + 1;
    string name = "seeds_sets/";
    name.insert(name.size(), to_string(S));
    name.insert(name.size(), "/");
    name.insert(name.size(), to_string(version));
    name.insert(name.size(), "/");
    name.insert(name.size(), to_string(true_iter));

    ofstream FILE(name);
    if(FILE.is_open()){
        for(int i = 0; i < S; ++i){
            string data = to_string(SEEDS[i]);
            FILE<<data.insert(data.size(),"\n");
        }
        FILE.close();
    }
    else printf("Unable to write seeds\n");
}

int main(int argc, char **argv){
    if(argc != 4){
        printf("Run as ./prog S N I\n");
        printf("S - Number of seeds\n");
        printf("N _ length of grid\n");
        printf("ITER - Number of iterations\n");
        return(EXIT_FAILURE);
    }

    int S = atoi(argv[1]);
    int N = atoi(argv[2]);
    int ITER = atoi(argv[3]);
    int i,j;

    srand(time(0));
    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<int> uid(0, 12);

    int *SEEDS = (int*)malloc(S*sizeof(int));
    vector<int> POSSIBLE_SEEDS;

    for(i = 0; i < N*N; ++i) POSSIBLE_SEEDS.push_back(i);

    random_shuffle(POSSIBLE_SEEDS.begin(), POSSIBLE_SEEDS.end());
    
    for(i = 0; i < S; ++i){
        SEEDS[i] = POSSIBLE_SEEDS[i];
        if(S <= 500 )printf("%i\n", SEEDS[i]);
    }
    save_seeds(SEEDS, S, -1, 1);
    POSSIBLE_SEEDS.clear();
    printf("\n");
    //Move the seeds
    int old_x, old_y, new_x, new_y, delta_x, delta_y, occupied = 0, check = 0;
    for(int iterator = 0; iterator < ITER; ++iterator){
        for(j = 0; j < S; ++j){
            old_x = SEEDS[j]%N;
            old_y = SEEDS[j]/N;
            do{
                check = 0;
                occupied = 0;
                do{
                    delta_x = uid(rng) - 6;
                    delta_y = uid(rng) - 6;
                    new_x = old_x + delta_x;
                    new_y = old_y + delta_y;
                    if(new_x<N && new_y<N) check = 1;
                    if(new_x<0 || new_y<0) check = 0;
                    //if(check == 0)printf("Trapped here, ITER:%i\n", iterator);
                }
                while(check == 0);
                //printf("OUT\n");
                SEEDS[j] = new_y * N + new_x;
                for(i = 0; i < j; ++i){
                    if(SEEDS[i] == SEEDS[j]){
                        occupied = 1;
                        break;
                    }
                }
                if(occupied == 0 && S <= 500 )printf("OLD X: %i, OLD Y:%i, OLD SEED:%i | NEW X:%i, NEW Y:%i, NEW SEED:%i\n", old_x, old_y, old_y*N + old_x, new_x, new_y, SEEDS[j]);
            }
            while(occupied == 1);
            //if(S <= 500 )printf("%i\n", SEEDS[j]);
        }
        save_seeds(SEEDS, S, iterator, 1);
        //printf("after save\n");
    }

    free(SEEDS);

}

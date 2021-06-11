#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include "utils.h"
#include "voronoi.h"

#define ll long long
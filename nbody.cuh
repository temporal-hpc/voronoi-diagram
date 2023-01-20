__global__ void updateNBodies(int *particles, double *par_vel, int N, int S, double dt, double G, double M){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S) return;
    
    double force_x = 0.0, force_y = 0.0, pos_x = double(particles[i]%N), pos_y = double(particles[i]/N);
    
    for(int j = 0; j < S; ++j){
        if(i == j) continue;
        double body_x = double(particles[j]%N), body_y = double(particles[j]/N);
        double tmp_x = body_x - pos_x, tmp_y = body_y - pos_y;
        
        if(abs(tmp_x - N) < abs(tmp_x)) tmp_x -= N;
        else if(abs(tmp_x + N) < abs(tmp_x)) tmp_x += N;

        if(abs(tmp_y - N) < abs(tmp_y)) tmp_y -= N;
        else if(abs(tmp_y + N) < abs(tmp_y)) tmp_y += N;
        
        double d = sqrt(tmp_x*tmp_x + tmp_y*tmp_y);
        if(particles[i]==particles[j]) d = 0.001;
        force_x = force_x + tmp_x * (G * M * M / (d*d*d));
        force_y = force_y + tmp_y * (G * M * M / (d*d*d)); 
    }

    par_vel[i*2] = par_vel[i*2] + force_x * dt / M;
    par_vel[i*2 + 1] = par_vel[i*2 + 1] + force_y * dt / M;

    pos_x = pos_x + par_vel[i*2] * dt;
    pos_y = pos_y + par_vel[i*2 + 1] * dt;

    pos_x = int(pos_x + N)%N;
    pos_y = int(pos_y + N)%N;

    particles[i] = int(pos_y) * N + int(pos_x);
}
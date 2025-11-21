double mat[N][N];
double s[N][N];
int v[N];


double trig_table[256];
for (int i =0; i < 256; i++) {
    //precompute sin^2 - cos^2 for all i (only 256 possible values)
    trig_table[i] = -cos(2.0*i);
}

// fill v and s

//swap the order of the loops - why is this faster?
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        //replace modulo 256 with bitwise and 255
        mat[i][j] = s[i][j] * (trig_table[v[i] & 255]);
    }
}
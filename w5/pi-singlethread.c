#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_RAND 1

double random_number() {
    // returns a random floating point number between 0.0 and MAX_RAND
    return fmod(rand() * ((double) rand() / RAND_MAX), MAX_RAND); 
}

struct timeval t;

double get_time() {
    gettimeofday(&t, NULL);
    return t.tv_sec + (1e-6 * t.tv_usec);
}

int main(int argc, char* argv[]) {

    //read start time
    double t0 = get_time();

    // read N from the first command line argument
    int N = atoi(argv[1]);

    double total = 0;
    double in = 0;

    for (int i = 0; i < N; i++) {

        double x = random_number();
        double y = random_number();

        if ((x*x + y*y) <= 1.0) {
            in++;
        }

        total++;

        printf("current value of pi: %lf \n" , (in / total)*4.0);

    }

    double t1 = get_time(); 
    double total_time = t1 - t0;

    printf("time taken: %lfs \n", total_time);


    return 0;
}
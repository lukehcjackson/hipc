#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define MAX_RAND 1

// Modified from Numerical Recipes Page 340 https://numerical.recipes/book.html
typedef struct {
    unsigned long long u;
    unsigned long long v;
    unsigned long long w;
} random_generator;

random_generator init_random_generator(unsigned long long seed) {
    random_generator r;
    r.v = 4101842887655102017LL;
    r.w = 1;
    r.u = seed ^ r.v; 
    r.v = r.u;
    r.w = r.v;
    return r;
}

double random_double(random_generator *r) {
    //Return 64-bit random integer.
    r->u = r->u * 2862933555777941757LL + 7046029254386353087LL; 
    r->v ^= r->v >> 17; 
    r->v ^= r->v << 31; 
    r->v ^= r->v >> 8;
    r->w = 4294957665U*(r->w & 0xffffffff) + (r->w >> 32);
    unsigned long long x = r->u ^ (r->u << 21); 
    x ^= x >> 35; 
    x ^= x << 4; 
    //return 5.42101086242752217E-20 * ((x + r->v) ^ r->w);
    unsigned long long result = (x + r->v) ^ r->w;
    return result * (1.0 / 18446744073709551616.0); // 18446744073709551616.0 is 2^64
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

    //instead of initialising the random generator with the thread number in the loop,
    //precompute what the random generator will be for each thread number and use that value instead

    int max_threads = omp_get_max_threads();
    random_generator* all_generators = (random_generator*)malloc(sizeof(random_generator) * max_threads);

    for (int i = 0; i < max_threads; i++) {
        // Seed each thread differently, e.g., thread index + an initial unique seed (like time(NULL))
        all_generators[i] = init_random_generator(get_time() + i);
    }

    //use a reduction to make adding to in and total faster
    //the 'in' and 'total' variables become thread-private local copies - each thread gets its own in and total
    //during execution, each thread can modify these as it pleases (faster as no global synchronisation)
    //after the parallel region, openmp adds a synchronisation point (barrier) 
    //and applies the reduction operator (+ here) to the private copies back into the original shared variable
    #pragma omp parallel for reduction(+:in, total)
    for (int i = 0; i < N; i++) {

        //use the pre-generated random generator
        int thread_id = omp_get_thread_num();
        random_generator *my_gen = &all_generators[thread_id];

        double x = random_double(my_gen);
        double y = random_double(my_gen);

        if ((x*x + y*y) <= 1.0) {
            in++;
        }

        total++;

    }

    free(all_generators);

    //do the output outside of the loop because printf is fucking slow
    printf("value of pi: %lf \n" , (in / total)*4.0);

    double t1 = get_time(); 
    double total_time = t1 - t0;

    printf("time taken: %lfs \n", total_time);


    return 0;
}
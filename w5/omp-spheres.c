#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

#include <omp.h>

#define MAX_RAND 10000

struct timeval t;

double get_time() {
    gettimeofday(&t, NULL);
    return t.tv_sec + (1e-6 * t.tv_usec);
}

struct sphere_t {
    double x;
    double y;
    double z;
    double r;
};

double random_number() {
    // returns a random floating point number between 0.0 and MAX_RAND
    return fmod(rand() * ((double) rand() / RAND_MAX), MAX_RAND); 
}

int main(int argc, char *argv[]) {
    //read start time
    double t0 = get_time();

    // read N from the first command line argument
    int N = atoi(argv[1]);

    struct sphere_t * sphere = malloc(sizeof(struct sphere_t) * N);
    // fill with random numbers
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        sphere[i].x = random_number();
        sphere[i].y = random_number();
        sphere[i].z = random_number();
        sphere[i].r = random_number() / 4.0;
    }

    double t1 = get_time(); 
    double population_timer = t1 - t0;

    // calculate areas
    double * area = calloc(N, sizeof(double));
    double four_pi = 4.0 * M_PI;
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        area[i] = four_pi * sphere[i].r * sphere[i].r;
    }

    double t2 = get_time();
    double areas_timer = t2 - t1;

    // calculate volume
    double * volume = calloc(N, sizeof(double));
    double four_thirds_pi = four_pi / 3.0;
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        volume[i] = four_thirds_pi * sphere[i].r * sphere[i].r * sphere[i].r;
    }

    double t3 = get_time();
    double volume_timer = t3 - t2;
    
    // calculate the number of spheres each sphere intersects
    int * intersects = calloc(N, sizeof(int));
    #pragma omp parallel for 
    for (int i = 0; i < N; i++) {

        double i_x = sphere[i].x;
        double i_y = sphere[i].y;
        double i_z = sphere[i].z;
        double i_r = sphere[i].r;

        //if we know that sphere 3 intersects sphere 5, we don't need to check sphere 5 for an intersection with sphere 3
        #pragma omp parallel for
        for (int j = 0; j < i; j++) {
            if (i == j) continue; // same circle

            double x_diff = sphere[j].x - i_x;
            double y_diff = sphere[j].y - i_y;
            double z_diff = sphere[j].z - i_z;

            // calculate distance between two spheres
            //double d = sqrt(pow(sphere[j].x - sphere[i].x, 2.0) + pow(sphere[j].y - sphere[i].y, 2.0) + pow(sphere[j].z - sphere[i].z, 2.0));
            double d = sqrt((x_diff * x_diff) + (y_diff * y_diff) + (z_diff * z_diff));
            
            // if the distance is less than the sum of the radii, they intersect
            //if (d < (sphere[j].r + sphere[i].r)) intersects[i]++;
            if (d < (sphere[j].r + i_r)) {
                // if one sphere intersects another, then obviously the other sphere intersects this sphere
                intersects[i]++;
                intersects[j]++;
            } 
        }
    }

    double t4 = get_time();
    double intersect_timer = t4 - t3;

    // print out all information to the screen (consider piping this to a file)
    printf("x, y, z, r, area, volume, intersects\n");
    for (int i = 0; i < N; i++) {
        printf("%lf, %lf, %lf, %lf, %lf, %lf, %d\n", sphere[i].x, sphere[i].y, sphere[i].z, sphere[i].r, area[i], volume[i], intersects[i]);
    }

    //print timing information
    fprintf(stderr, "Time to populate spheres: %lfs \n", population_timer);
    fprintf(stderr, "Time to calculate areas: %lfs \n", areas_timer);
    fprintf(stderr, "Time to calculate volumes: %lfs \n", volume_timer);
    fprintf(stderr, "Time to calculate intersects: %lfs \n", intersect_timer);
}
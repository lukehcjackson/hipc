#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define MAX_RAND 10000

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
    clock_t start = clock();

    // read N from the first command line argument
    int N = atoi(argv[1]);

    struct sphere_t * sphere = malloc(sizeof(struct sphere_t) * N);
    // fill with random numbers
    for (int i = 0; i < N; i++) {
        sphere[i].x = random_number();
        sphere[i].y = random_number();
        sphere[i].z = random_number();
        sphere[i].r = random_number() / 4.0;
    }

    clock_t population_timer = clock() - start;

    // calculate areas
    double * area = calloc(N, sizeof(double));
    double four_pi = 4.0 * M_PI;
    for (int i = 0; i < N; i++) {
        area[i] = four_pi * sphere[i].r * sphere[i].r;
    }
    clock_t areas_timer = clock() - population_timer;

    // calculate volume
    double * volume = calloc(N, sizeof(double));
    double four_thirds_pi = four_pi / 3.0;
    for (int i = 0; i < N; i++) {
        volume[i] = four_thirds_pi * sphere[i].r * sphere[i].r * sphere[i].r;
    }
    clock_t volume_timer = clock() - areas_timer;
    
    // calculate the number of spheres each sphere intersects
    int * intersects = calloc(N, sizeof(int));
    for (int i = 0; i < N; i++) {

        double i_x = sphere[i].x;
        double i_y = sphere[i].y;
        double i_z = sphere[i].z;
        double i_r = sphere[i].r;

        //if we know that sphere 3 intersects sphere 5, we don't need to check sphere 5 for an intersection with sphere 3
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
    clock_t intersect_timer = clock() - volume_timer;
    clock_t end = clock();

    // print out all information to the screen (consider piping this to a file)
    printf("x, y, z, r, area, volume, intersects\n");
    for (int i = 0; i < N; i++) {
        printf("%lf, %lf, %lf, %lf, %lf, %lf, %d\n", sphere[i].x, sphere[i].y, sphere[i].z, sphere[i].r, area[i], volume[i], intersects[i]);
    }

    //print timing information
    fprintf(stderr, "Time to populate spheres: %lfs \n", (double)population_timer / CLOCKS_PER_SEC);
    fprintf(stderr, "Time to calculate areas: %lfs \n", (double)areas_timer / CLOCKS_PER_SEC);
    fprintf(stderr, "Time to calculate volumes: %lfs \n", (double)volume_timer / CLOCKS_PER_SEC);
    fprintf(stderr, "Time to calculate intersects: %lfs \n", (double)intersect_timer / CLOCKS_PER_SEC);
}
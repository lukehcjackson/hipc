#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ITERS 25

#define get_A(x, y) (A[(x) * n + (y)])

void jacobi_solve(double* A, double* b, double* x, int n) {
    double eps = 1e-4;

    // check that A is diagonally dominant
    int diag, sum;
    for (int row = 0; row < n; row++) {
        diag = -1;
        sum = 0;
        for (int col = 0; col < n; col++) {
            if (row == col) {
                diag = get_A(row, col);
            } else {
                sum += get_A(row, col);
            }
        }

        if (diag < sum) {
            printf("Matrix is not diagonally dominant at row %d\n", row);
            return;
        }
    }

    // iteratively improve x using the element-wise formula
    //xi^(n+1) = 1/aii * (bi - sum over j!=i(aij * xj^n))

    double* newX = (double *) calloc(n, sizeof(double));

    for (int iter = 0; iter < MAX_ITERS; iter++) {
        for (int i = 0; i < n; i++) {

            double sum = 0;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    sum += get_A(i, j) * x[j];
                }
            }
            double bracket_term = b[i] - sum;

            newX[i] = bracket_term / get_A(i, i);
        }

        for (int i = 0; i < n; i++) {
            if (fabs(x[i] - newX[i]) < eps) {
                printf("terminated early at iteration %d \n", iter);
                return;
            }
            x[i] = newX[i];
        }

    }
    free(newX);
    
}

int main(int argc, char *argv[]) {
    //get n from command line arguments
    int n = 0;
    if (argc == 2) {
        //given one argument
        n = atoi(argv[1]);
    } else {
        fprintf(stderr, "Provide n as a command line arg");
    }

    double* A = (double *) malloc(sizeof(double) * n * n);
    double* b = (double *) malloc(sizeof(double) * n);
    double* x = (double *) malloc(sizeof(double) * n);

    // A = [ 2  1 ] b = [ 11 ]
    //     [ 5  7 ]     [ 13 ]
    /*
    A[0] = 2;
    A[1] = 1;
    A[2] = 5;
    A[3] = 7; 

    b[0] = 11;
    b[1] = 13;  
    

    // initial guess is x = [ 1 ]
    //                      [ 1 ]
    x[0] = 1;
    x[1] = 1; 
    */

   //set A, b, x

    //set A
    int diag_count = 0;
    for (int i = 0; i < n*n; i++) {
        // if we are on the diagonal, A[i] = n+1 else A[i] = 1
        if (diag_count > n) {
            diag_count = 0;
        }

        if (diag_count == 0) {
            A[i] = n+1;
        } else {
            A[i] = 1;
        }

        diag_count++;
        
    }

    for (int i = 0; i < n*n; i++) {
        printf("%lf \n", A[i]);
    }

    //set b and x
    for (int i = 0; i < n; i++) {
        b[i] = 2*n;
        x[i] = 0.5f;
    }

    jacobi_solve(A, b, x, n); 

    printf("Solution is: \n");
    for (int i = 0; i < n; i++) {
        printf("[ %10lf ]\n", x[i]);
    }

}


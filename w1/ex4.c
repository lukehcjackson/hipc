#include <stdio.h>

void swap(int* a_ptr, int* b_ptr) {

    int tmp = *a_ptr;
    *a_ptr = *b_ptr;
    *b_ptr = tmp;

}

// write the code for the sort() function that sorts an integer array in ascending order
void sort(int* x, int length) {
    //have: pointer to first element of array and length of array
    //want: sorted
    int swapped;

    for (int i = 0; i < length - 1; i++) {
        swapped = 0;
        for (int j = 0; j < length - i - 1; j++) {
            if (x[j] > x[j+1]) {
                swap(&x[j], &x[j+1]);
                swapped = 1;
            }
        }

        if (!swapped) {
            break;
        }
    }

}


int main() {
    
    int length;
    printf("How many items? : ");
    scanf("%d", &length);

    int x[length];

    for (int i = 0; i < length; i++) {
        scanf("%d", &x[i]);
    }

    for (int j=0; j < length; j++){
        printf("%d ", x[j]);
    }

    sort(x, length);

    printf("The sorted array is as follows:\n");
    for (int j=0; j < length; j++){
        printf("%d ", x[j]);
    }

    return 0;
}
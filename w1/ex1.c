#include <stdio.h>

int isPrime(int n) {

    for (int i = 2; i < n; i++) {
        if (n % i == 0) {
            return 0;
        }
    }

    return 1;

}

int main() {

    int n;
    printf("Enter a positive integer:");
    scanf("%d", &n);

    for (int i = 2; i <= n; i++) {
        //if i is prime print it
        if (isPrime(i)) {
            printf("%d \n", i);
        }    
    }

    return 0;
}
#include <stdio.h> // Includes the stdio.h library 

/*
int main(int argc, char *argv[]) { // main method is called when program is run    
   
    for (int i = 0; i < 10; i++) {
        printf("Hello, world!\n");      // Use the printf function from stdio.h to print a string to the terminal     
    }

    return 0;                       // return code zero means no problems occurred 
}
*/


/*
int factorial(int m) {  // computes the factorial of a positive integer m
    int i, prod = 1;

    for (i = 2; i <= m; i++) {
        prod = prod * i;
    }
    return prod;
}

int main() {
    int n;

    printf("Enter a positive integer:");
    scanf("%d", &n); // scanf() scans the integer entered by the user

    printf("The factoral of %d is %d\n", n, factorial(n));
    return 0;
}
*/

//int a;   // An integer primitive, holds a whole number from -2,147,483,648 to 2,147,483,647 on most machines
//int *b;  // An integer pointer, holds an address of an integer variable.

/*
int main(int argc, char *argv[]) {
   int *a_pointer;
   int a_value = 5;
   int x[] = { 10, 1, 2, 5, -3 };

   a_pointer = &a_value;

   printf("The value of a_value is %d\n", a_value); // print the value of the variable a_value
   printf("The value of the pointer is %p,\n", a_pointer); // prints the address of the variable a_value
   printf("The value pointed to by the pointer is %d,\n", *a_pointer); // prints the value of the variable a_value

   *a_pointer = 10;

   printf("The value of a_value is now %d\n", a_value); // the value of a_value will be changed
   printf("The start address of the array x is %p\n", &x); // prints the base address of the array x
   printf("The address of the first element is %p\n", &x[0]); // also prints the same

   int i;
   for (i = 0; i < 5; i++) {
      printf("Value stored in address %p is %d\n", (x+i), x[i]);  // shows that array elements are stored in contiguous locations
   }

   printf("Size of each integer is %lu bytes\n", sizeof(int));  // the address of each location in array x differs by this amount from its previous location
   return 0;
}

*/


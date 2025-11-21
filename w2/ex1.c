#include <stdio.h>
#include <string.h>

int count_words(char *str) {
    // this function should return the number of words in str
    int words = 0;

    for (int i = 0; i < strlen(str); i++) {
        //printf("%c\n", str[i]);
        if (str[i] == ' ') {
            words++; //counts number of spaces
        }
    }

    return words + 1; // number of words is 1 more than number of spaces
}

int main() {
    char str[100];

    printf("Enter a string:");
    fgets(str, 100, stdin);    // this function reads a line or at most 99 bytes from stdin file stream that represents the keyboard

    printf("Number of words in the entered string is %d\n", count_words(str));

    return 0;

}
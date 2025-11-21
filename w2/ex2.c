#include <stdio.h>
#include <stdlib.h> // required for the exit() function
#include <string.h>

#define BUFSIZE 50

int main() {
    FILE *fptr1 = fopen("./testinput.txt", "r");
    FILE *fptr2 = fopen("./testoutput.txt", "w");

    if(fptr1 == NULL || fptr2 == NULL){
        fprintf(stderr, "Error opening file\n");
        exit(1);
    }

    char str[BUFSIZE];
    //printf("Content of the opened file:\n");

    while(fgets(str, BUFSIZE, fptr1) != NULL) { // read the file line by line
        fprintf(stdout, "%s", str);             // same as doing printf("%s",str);
        int flag = 0;
        flag = fwrite(&str, strlen(str)+1 , 1, fptr2);
        if (!flag) {
            fprintf(stderr, "Error writing to file\n");
            exit(1);
        }
    }

    //this almost works
    //mystery newline (?) characters at the start of lines and the end of the last line - what is causing that

    printf("\n");
    fclose(fptr1);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PARAM <PARAM>
#define SZ1 <SZ1>
#define SZ2 <SZ2>

int foo(int x) {
    int val = x * 5;
    printf("Foo says: %d\n", val);
    return val;
}

void bar(int y){
    printf("Bar sees: %d\n", y);
}

void secret() {
    puts("You’ve reached secret()!");
    system("/bin/sh");
}

int vuln(char *str){
    int canary = 0xDEADF00D;
    char buffer[SZ2];
    strcpy(buffer, str);
    if (canary != 0xDEADF00D) {
        printf("Stack smashing detected!\n");
        return 1;
    }
    return 2;
}


int main(int argc, char **argv) {
    char str[SZ1];
    FILE *badfile;
    badfile = fopen("badfile", "r");
    fread(str, sizeof(char), SZ1, badfile);
    vuln(str);

    printf("Returned Properly\n");
    return 0;
}

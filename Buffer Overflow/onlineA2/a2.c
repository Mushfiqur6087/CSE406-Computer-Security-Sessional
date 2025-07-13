#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PARAM <PARAM>
#define SZ1 <SZ1>
#define SZ2 <SZ2>

int foo(int x) {
    int hash = 0xDEADC0DE;
    int val = ((unsigned)(hash << 8) + hash) * x;
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
    char* buffer = (char*)malloc(SZ2);
    if (buffer == NULL) {
        printf("malloc failed for buffer\n");
        return 1;
    }
    int (**fp)(int) = malloc(sizeof(int (*)(int)));

    if (fp == NULL) {
        printf("malloc failed for fp\n");
        return 1;
    }
    *fp = foo;
    strcpy(buffer, str);
    (*fp)(PARAM);

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

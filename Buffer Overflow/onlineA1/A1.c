#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SZ    <SZ>
#define MAX   <MAX>

void greet(int x, int y) {
    printf("Greetings challenger, %d!\n", x);
    if(y != 0xFEEDDEAD){
        printf("Stack smashing detected!\n");
        exit(1);
    }
}

void get_shell() {
    execve("/bin/sh", NULL, NULL);
}

void vuln(char *user, char *pass) {
    char buf[SZ];
    strcpy(buf, pass);
}

int main() {
    char user[MAX], pass[MAX];
    fread(user, 1, MAX, fopen("user","r"));
    fread(pass, 1, MAX, fopen("pass","r"));
    // print address of greet
    printf("greet: %p\n", greet);
    vuln(user, pass);
    return 0;
}

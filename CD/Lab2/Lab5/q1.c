#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int curr = 0;
char str[100];

void invalid() {
    printf("-----------------ERROR!----------------\n");
    exit(0);
}
void valid() {
    printf("----------------SUCCESS!---------------\n");
    exit(0);
}

// Grammar 1
void S1();
void T1();
void S1() {
    if (str[curr] == 'a' || str[curr] == '>') {
        curr++;
    } else if (str[curr] == '(') {
        curr++;
        T1();
        if (str[curr] == ')') {
            curr++;
        } else invalid();
    } else invalid();
}
void T1() {
    S1();
    if (str[curr] == ',') {
        curr++;
        T1();
    }
}

// Grammar 2
void S2();
void U();
void V();
void W();
void S2() {
    U();
    V();
    W();
}
void U() {
    if (str[curr] == '(') {
        curr++;
        S2();
        if (str[curr] == ')') curr++;
        else invalid();
    } else if (str[curr] == 'a') {
        curr++;
        S2();
        if (str[curr] == 'b') curr++;
        else invalid();
    } else if (str[curr] == 'd') {
        curr++;
    } else invalid();
}
void V() {
    while (str[curr] == 'a') {
        curr++;
    }
}
void W() {
    while (str[curr] == 'c') {
        curr++;
    }
}

// Grammar 3
void S3();
void A();
void B();
void S3() {
    if (str[curr] == 'a') {
        curr++;
        A();
        if (str[curr] == 'c') {
            curr++;
            B();
            if (str[curr] == 'e') curr++;
            else invalid();
        } else invalid();
    } else invalid();
}
void A() {
    while (str[curr] == 'b') {
        curr++;
    }
}
void B() {
    if (str[curr] == 'd') {
        curr++;
    } else invalid();
}

// Grammar 4
void S4();
void L();
void S4() {
    if (str[curr] == '(') {
        curr++;
        L();
        if (str[curr] == ')') curr++;
        else invalid();
    } else if (str[curr] == 'a') {
        curr++;
    } else invalid();
}
void L() {
    S4();
    if (str[curr] == ',') {
        curr++;
        L();
    }
}

int main() {
    printf("Enter String: ");
    scanf("%s", str);
    strcat(str, "$'); // Ensure end marker

    curr = 0;
    S1(); // For Grammar 1
    if (str[curr] == '$') valid();
    else invalid();
    
    curr = 0;
    S2(); // For Grammar 2
    if (str[curr] == '$') valid();
    else invalid();
    
    curr = 0;
    S3(); // For Grammar 3
    if (str[curr] == '$') valid();
    else invalid();
    
    curr = 0;
    S4(); // For Grammar 4
    if (str[curr] == '$') valid();
    else invalid();
    
    return 0;
}

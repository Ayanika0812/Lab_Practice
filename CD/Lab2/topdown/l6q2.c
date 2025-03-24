/*
S -> UVW
U -> (S) | aSb | d
V -> aV | 𝜖
W -> cW | 𝜖
*/

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

void S();
void U();
void W();
void V();
void invalid();
void valid();

int curr=0;
char str[100];

void valid(){
    printf("----------------SUCCESS!!--------------");
    exit(0);
}
void invalid(){
    printf("----------------ERROR!!--------------");
    exit(0);
}

//S -> UVW
void S(){
    U();
    V();
    W();
}

//U -> (S) | aSb | d
void U(){
    if(str[curr] == '('){
        curr++;
        S();
        if(str[curr] == ')'){
            curr++;
            return;
        }else{
            invalid();   //last char not )
        }
    }
    else if(str[curr] == 'a'){
        curr++;
        S();
        if(str[curr] == 'b'){
            curr++;
            return;
        }else{
            invalid();    //last char not b
        }
    }
    else if(str[curr] == 'd'){
        curr++;
        return;
    }
    else{
        invalid();     //last char not d
    }
}

//V -> aV | 𝜖
void V(){
//first char is a    
    if(str[curr] == 'a'){
        curr++;
        V();   //Recursively 
    }
    //empty string
}

// W -> cW | 𝜖
void W(){
    if(str[curr] == 'c'){
        curr++;
        W();
    }
    // empty
}

int main(){
    printf("Enter String: ");
    scanf("%s", str);

    S();

    if(str[curr] == '\0'){
        valid();
    }else{
        invalid();
    }
    return 0;
}

/*
gcc l6q2.c -o q2


Enter String: adb
----------------SUCCESS!!--------------

Enter String: b
----------------ERROR!!--------------
*/
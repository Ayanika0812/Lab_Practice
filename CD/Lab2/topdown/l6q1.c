/*
S -> a | > | ( T )
T -> S T'
T' -> , S T' | 𝜖
*/

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

void T();
void S();
void Tprime();
void invalid();
void valid();

int curr=0;
char str[100];

//S -> a | > | ( T )
void S(){
//if first character is a || >   then curr++
    if(str[curr] == 'a'){
        curr++;
        return;
    }    
    else if(str[curr] == '>'){
        curr++;
        return;
    }
//( T )
    else if(str[curr] == '('){
        curr++;
        T();
        if(str[curr] == ')'){
            curr++;
            return;
        }else{
            invalid();  //if no ) at the end
        }
    }
    else{
        invalid();    //if no valid starting string
    }    
}

//T -> S T'
void T(){
    S();
    Tprime();      
}

//T' -> , S T' | 𝜖
void Tprime(){
//if first character is ','
    if(str[curr] == ','){
        curr++;
        S();
        Tprime();
    }
    //Epsilon  T' → 𝜖
}

void valid(){
    printf("----------------SUCCESS!!--------------");
    exit(0);
}
void invalid(){
    printf("----------------ERROR!!--------------");
    exit(0);
}

int main(){
    printf("Enter String: ");
    scanf("%s", str);

    //start symbol
    S();

    //if its the end of the string
    if(str[curr] == '\0'){
        valid(); //if parsing is successful
    }else{
        invalid(); //if extra characters are there in input
    }

    return 0;
}


/*
gcc l6q1.c -o q1

.\q1.exe
Enter String: (a,>)
----------------SUCCESS!!--------------

Enter String: a
----------------SUCCESS!!--------------

Enter String: <
----------------ERROR!!--------------
*/
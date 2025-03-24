/*
E->E+T|T 
T-> T*F|F 
F-> ( E )|id 
define 
int main
goto table and action table productions
*/
#include<stdio.h>
#include<stdlib.h>
#define ID 0
#define PLUS 1
#define STAR 2
#define LPAREN 3
#define RPAREN 4
#define DOLLAR 5
#define E 6
#define T 7
#define F 8
int action_table[12][6] = {
    {5,  0,  0, 4,  0,  0},    // State 0
    {0,  6,  0, 0,  0,  0},    // State 1
    {0, -2, 7, 0, -2, -2},     // State 2 (Reduce 2: E→T)
    {0, -4, -4, 0, -4, -4},    // State 3 (Reduce 4: T→F)
    {5,  0,  0, 4,  0,  0},    // State 4
    {0, -6, -6, 0, -6, -6},    // State 5 (Reduce 6: F→id)
    {5,  0,  0, 4,  0,  0},    // State 6
    {5,  0,  0, 4,  0,  0},    // State 7
    {0,  6,  0, 0, 11, 0},     // State 8
    {0, -1, 7, 0, -1, -1},     // State 9 (Reduce 1: E→E+T)
    {0, -3, -3, 0, -3, -3},    // State 10 (Reduce 3: T→T*F)
    {0, -5, -5, 0, -5, -5}     // State 11 (Reduce 5: F→(E))
};
int goto_table[12][3] = {
    {1, 2, 3},   // State 0
    {0, 0, 0},   // State 1
    {0, 0, 0},   // State 2
    {0, 0, 0},   // State 3
    {8, 2, 3},   // State 4
    {0, 0, 0},   // State 5
    {0, 9, 3},   // State 6
    {0, 0, 10},  // State 7
    {0, 0, 0},   // State 8
    {0, 0, 0},   // State 9
    {0, 0, 0},   // State 10
    {0, 0, 0}    // State 11
};

int productions[6][2] = {
    {E,3}, // 1: E → E + T
    {E, 1}, // 2: E → T
    {T, 3}, // 3: T → T * F
    {T, 1}, // 4: T → F
    {F, 3}, // 5: F → ( E )
    {F, 1}  // 6: F → id
};

int stack[100], top=-1;
int input[100], ip = 0;

void push(int state){
    stack[++top] = state;
}

int pop(){
    return stack[top--];
}


void parse(){
    push(0);
    int current_token = input[ip++];

    while(1){
        int state = stack[top];
        int action = action_table[state][current_token];

        if(action > 0){
            printf("Shift %d\n", action);
            push(action);                      //Pushes the new state onto the stack.
            current_token = input[ip++];       //Reads the next token.
        }
        else if(action < 0){
            int prod = -action - 1;
            printf("Reduce by production %d (" , prod+1);

            switch (prod)
            {
            case 0: printf("E gives E + T"); break;
            case 1: printf("E gives T"); break;
            case 2: printf("T gives T * F"); break;
            case 3: printf("T gives F"); break;
            case 4: printf("F gives ( E )"); break;
            case 5: printf("F gives id"); break;
            }
            printf(")\n");

            int rhs_side = productions[prod][1];
            for(int i=0;i<rhs_side;i++){
                pop();
            }
            int lhs_side = productions[prod][0];
            int new_state =  goto_table[stack[top]][lhs_side - E];
            push(new_state);
        }
        else if(action == 0){
            if(state == 1 && current_token == DOLLAR){
                printf("Accepted");
                return;
            }
            else{
                printf("Error at token %d\n",current_token);
                exit(1);
            }
        }
    }
}



int main(){
    int sample_input[] = {ID, PLUS, ID, STAR, ID, DOLLAR};
    printf("Input str:id + id * id $\n");
    for(int i =0;i<6;i++){
        input[i] = sample_input[i];
    }
    printf("Steps: \n");
    parse();
    return 0;

}
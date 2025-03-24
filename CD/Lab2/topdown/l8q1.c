/*
Program → main () { declarations statement-list }
Declarations → data-type identifier-list; declarations | ε
data-type → int | char
identifier-list → id | id, identifier-list | id[number], identifier-list | id[number]
statement_list → statement statement_list | ε
statement → assign-stat; | decision_stat
assign_stat → id = expn
expn → simple-expn eprime
eprime → relop simple-expn | ε
simple-expn → term seprime
seprime → addop term seprime | ε
term → factor tprime
tprime → mulop factor tprime | ε
factor → id | num
decision-stat → if (expn) { statement_list } dprime
dprime → else { statement_list } | ε
relop → == | != | <= | >= | > | <
addop → + | -
mulop → * | / | %
*/

/*
int main
enum
struct
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef enum{
    MAIN, INT,CHAR,ID,NUM,LPAREN,RPAREN,LBRACE,RBRACE,LBRACKET,RBRACKET,SEMICOLON,COMMA,ASSIGN,IF,ELSE,REL_OP,ADD_OP,MUL_OP,END_OF_FILE,CHAR_LITERAL
}TokenType;

typedef struct Token
{
    TokenType type;
    char* value;
    int row;
    int col;
}Token;

Token currentToken;

char* buffer;
int bufferindex =0;
int row =1;
int col =1;

//initialize
void initTokenizer(char * buf){
    buffer = buf;
    bufferindex = 0;
    row = 1;
    col = 1;
    getNextToken();         //get first token
}

//Program → main () { declarations statement-list }
void program(){
    if(currentToken.type != MAIN){
        printf("Error at line %d, column %d: Expected 'main'\n",row,col);
        exit(1);
    }

    match(MAIN);    //main
    match(LPAREN);   //  (
    match(RPAREN);    // )
    match(LBRACE);    //  {

    declarations();
    statement_list();

    if(currentToken.type != RBRACE){                          //if last char doesnt match
        printf("Error at line %d, column %d: Expected '}'\n", row, col);
        exit(1);
    }

    match(RBRACE);

    printf("Parsing Successful\n");
}


int main(int argc, char* argv[]){
    if(argc != 2){
        printf("Usage: %s filename\n",argv[0]);
        return 1;
    }

    FILE* file = fopen(argv[1], "r");
    if(!file){
        printf("Could not open file %s\n",argv[1]);
        return 1;
    }

    //read file
    fseek(file,0,SEEK_END);
    long fileSize = ftell(file);
    rewind(file);

    char* inputBuffer = malloc(fileSize + 1);
    if(!inputBuffer){
        printf("Memory allocation failed\n");
        fclose(file);
        return 1;
    }

    size_t bytesRead = fread(inputBuffer, 1 , fileSize, file);
    inputBuffer[bytesRead] = '\0';

    fclose(file);

    //Initialize tokenizer
    initTokenizer(inputBuffer);
    program();

    free(inputBuffer);

    return 0;

}
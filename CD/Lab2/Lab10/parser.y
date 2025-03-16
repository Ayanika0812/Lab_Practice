%{
    #include <stdio.h>
    #include <stdlib.h>
    #include <ctype.h>
    void yyerror(const char *msg);
    int yylex();
    int yywrap() { return 1; }
%}

%token NUMBER ID NL INT FLOAT CHAR IF ELSE WHILE FOR 
%left '+' '-'
%left '*' '/'
%right '^'
%token DECLARE
%token GT LT EQ NEQ GEQ LEQ 

%%
input: 
    | input line
    ;

line: 
    '\n'
    | declaration '\n'   { printf("Valid Declaration Statement\n"); }
    | decision '\n'      { printf("Valid Decision-Making Statement\n"); }
    | exp '\n'          { printf("Result: %d\n", $$); }
    | postfix '\n'      { printf("Valid Postfix Expression\n"); }
    ;

declaration:
    DECLARE ID ';'      
    | DECLARE ID '=' NUMBER ';'
    ;

decision:
    IF '(' condition ')' '{' '}' 
    | IF '(' condition ')' '{' '}' ELSE '{' '}'
    | WHILE '(' condition ')' '{' '}'
    | FOR '(' declaration condition ';' declaration ')' '{' '}'
    ;

condition:
    ID GT NUMBER 
    | ID LT NUMBER 
    | ID EQ NUMBER 
    | ID NEQ NUMBER 
    | ID GEQ NUMBER 
    | ID LEQ NUMBER 
    ;

exp:
    exp '+' exp   { $$ = $1 + $3; }
    | exp '-' exp { $$ = $1 - $3; }
    | exp '*' exp { $$ = $1 * $3; }
    | exp '/' exp { $$ = $1 / $3; }
    | '(' exp ')' { $$ = $2; }
    | NUMBER      { $$ = $1; }
    ;

postfix:
    NUMBER
    | postfix postfix '+' 
    | postfix postfix '-' 
    | postfix postfix '*' 
    | postfix postfix '/' 
    | postfix postfix '^' 
    | postfix 'n' 
    ;

%%

void yyerror(const char *msg) {
    printf("Error: %s\n", msg);
}

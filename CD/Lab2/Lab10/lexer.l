%{
    #include "y.tab.h"
%}

%%
"int"    { return DECLARE; }
"float"  { return DECLARE; }
"char"   { return DECLARE; }
"if"     { return IF; }
"else"   { return ELSE; }
"while"  { return WHILE; }
"for"    { return FOR; }

">"      { return GT; }
"<"      { return LT; }
"=="     { return EQ; }
"!="     { return NEQ; }
">="     { return GEQ; }
"<="     { return LEQ; }

[0-9]+   { yylval = atoi(yytext); return NUMBER; }
[a-zA-Z_][a-zA-Z0-9_]* { return ID; }
[\n]     { return NL; }
[ \t]    { /* Ignore whitespace */ }
.        { return yytext[0]; }
%%

int yywrap() { return 1; }

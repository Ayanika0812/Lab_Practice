#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>

// Function Prototypes
int isArithmeticOperator(char c);
int isRelationalOperator(char c, char next);
int isLogicalOperator(char c, char next);
int isSpecialSymbol(char c);
int isKeyword(char *str);
int isIdentifier(char *str);
int isNumeric(char *str);
int isStringLiteral(char *str);

void identifyTokens(FILE *inputFile, FILE *outputFile) {
    int row = 1, col = 1;
    char c;

    while ((c = fgetc(inputFile)) != EOF) {
        if (c == '\n') {
            row++;
            col = 1;
            continue;
        }

        if (isspace(c)) {
            col++;
            continue;
        }

        if (isArithmeticOperator(c)) {
            fprintf(outputFile, "<%c,%d,%d>\n", c, row, col);
            col++;
        } 
        else if (c == '=' || c == '<' || c == '>' || c == '!' || c == '&' || c == '|') {
            char next = fgetc(inputFile);
            if (isRelationalOperator(c, next) || isLogicalOperator(c, next)) {
                fprintf(outputFile, "<%c%c,%d,%d>\n", c, next, row, col);
                col += 2;
            } else {
                fprintf(outputFile, "<%c,%d,%d>\n", c, row, col);
                ungetc(next, inputFile);
                col++;
            }
        } 
        else if (isSpecialSymbol(c)) {
            fprintf(outputFile, "<%c,%d,%d>\n", c, row, col);
            col++;
        } 
        else if (c == '"') {  
            int start_col = col;
            char str[100];
            int i = 0;
            str[i++] = c;

            while ((c = fgetc(inputFile)) != '"' && c != EOF) {
                str[i++] = c;
                col++;
            }
            if (c == '"') {
                str[i++] = c;
                col++;
            }
            str[i] = '\0';
            fprintf(outputFile, "<%s,%d,%d>\n", str, row, start_col);
        } 
        else if (isalpha(c) || c == '_') {  
            int start_col = col;
            char str[100];
            int i = 0;

            str[i++] = c;
            while (isalnum((c = fgetc(inputFile))) || c == '_') {
                str[i++] = c;
                col++;
            }
            str[i] = '\0';
            ungetc(c, inputFile);

            if (isKeyword(str)) {
                fprintf(outputFile, "<%s,%d,%d>\n", str, row, start_col);
            } else {
                fprintf(outputFile, "<id,%d,%d>\n", row, start_col);
            }
        } 
        else if (isdigit(c)) {  
            int start_col = col;
            char num[100];
            int i = 0;
            num[i++] = c;

            while (isdigit((c = fgetc(inputFile))) || c == '.') {  
                num[i++] = c;
                col++;
            }
            num[i] = '\0';
            ungetc(c, inputFile);

            fprintf(outputFile, "<%s,%d,%d>\n", num, row, start_col);
        }

        col++;
    }
}

// Utility Functions

int isArithmeticOperator(char c) {
    return (c == '+' || c == '-' || c == '*' || c == '/' || c == '%');
}

int isRelationalOperator(char c, char next) {
    return (c == '=' && next == '=') || (c == '!' && next == '=') || 
           (c == '<' && next == '=') || (c == '>' && next == '=');
}

int isLogicalOperator(char c, char next) {
    return (c == '&' && next == '&') || (c == '|' && next == '|');
}

int isSpecialSymbol(char c) {
    return strchr("{}[]();,.:#", c) != NULL;
}

int isKeyword(char *str) {
    const char *keywords[] = {
        "int", "float", "char", "if", "else", "while", "return", "void", "for",
        "break", "continue", "switch", "case", "default"
    };
    for (int i = 0; i < sizeof(keywords) / sizeof(keywords[0]); i++) {
        if (strcmp(str, keywords[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

int main() {
    FILE *inputFile = fopen("input.txt", "r");
    FILE *outputFile = fopen("output.txt", "w");

    if (!inputFile || !outputFile) {
        printf("Error opening files.\n");
        return 1;
    }

    identifyTokens(inputFile, outputFile);

    fclose(inputFile);
    fclose(outputFile);

    printf("Output written to 'output.txt'.\n");
    return 0;
}

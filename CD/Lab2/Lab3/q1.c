#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TOKENS 100
#define MAX_LEXEME_LEN 20

typedef struct {
    char lexeme[MAX_LEXEME_LEN];
    char type[MAX_LEXEME_LEN];
    char scope; // 'G' for global, 'L' for local
    int size;
} Symbol;

Symbol globalSymbolTable[MAX_TOKENS];
int globalCount = 0;

Symbol localSymbolTable[MAX_TOKENS];
int localCount = 0;

// Token structure for getNextToken()
typedef struct {
    char lexeme[MAX_LEXEME_LEN];
    char type[MAX_LEXEME_LEN];
} Token;
//. Utility Functions

void addSymbol(Symbol *table, int *count, char *lexeme, char *type, char scope, int size) {
    strcpy(table[*count].lexeme, lexeme);
    strcpy(table[*count].type, type);
    table[*count].scope = scope;
    table[*count].size = size;
    (*count)++;
}

void displaySymbolTable(Symbol *table, int count, const char *tableName) {
    printf("\n%s:\n", tableName);
    printf("Lexeme\tType\tScope\tSize\n");
    for (int i = 0; i < count; i++) {
        printf("%s\t%s\t%c\t%d\n", table[i].lexeme, table[i].type, table[i].scope, table[i].size);
    }
}

//Lexical Analyzer and Symbol Table Population

Token getNextToken(FILE *file) {
    Token token = {"", ""};
    char ch;
    int i = 0;

    // Simulate token extraction (can be expanded for actual lexical analysis)
    while ((ch = fgetc(file)) != EOF) {
        if (ch == ' ' || ch == '\n') continue;
        if (isalpha(ch)) { // Identifier
            token.lexeme[i++] = ch;
            while (isalnum(ch = fgetc(file))) {
                token.lexeme[i++] = ch;
            }
            token.lexeme[i] = '\0';
            strcpy(token.type, "Identifier");
            ungetc(ch, file); // Return the last read character
            break;
        }
        if (isdigit(ch)) { // Number
            token.lexeme[i++] = ch;
            while (isdigit(ch = fgetc(file))) {
                token.lexeme[i++] = ch;
            }
            token.lexeme[i] = '\0';
            strcpy(token.type, "Number");
            ungetc(ch, file);
            break;
        }
    }
    return token;
}

void processFile(FILE *file) {
    char currentFunction[MAX_LEXEME_LEN] = "";
    Token token;

    while (!feof(file)) {
        token = getNextToken(file);

        if (strcmp(token.lexeme, "int") == 0 || strcmp(token.lexeme, "bool") == 0) {
            char type[MAX_LEXEME_LEN];
            strcpy(type, token.lexeme);

            token = getNextToken(file); // Get the identifier
            if (strcmp(currentFunction, "") == 0) { // Global scope
                addSymbol(globalSymbolTable, &globalCount, token.lexeme, type, 'G', 4);
            } else { // Local scope
                addSymbol(localSymbolTable, &localCount, token.lexeme, type, 'L', 4);
            }
        }

        if (strcmp(token.lexeme, "void") == 0) { // Function definition
            token = getNextToken(file); // Get function name
            strcpy(currentFunction, token.lexeme);
            addSymbol(globalSymbolTable, &globalCount, token.lexeme, "Function", 'G', 0);
        }

        if (strcmp(token.lexeme, "{") == 0) { // Start of a function
            localCount = 0; // Reset local symbol table
        }

        if (strcmp(token.lexeme, "}") == 0) { // End of a function
            displaySymbolTable(localSymbolTable, localCount, "Local Symbol Table");
            strcpy(currentFunction, ""); // Reset current function
        }
    }
}



int main() {
    FILE *file = fopen("source_code.txt", "r");
    if (!file) {
        printf("Error opening file.\n");
        return 1;
    }

    processFile(file);
    displaySymbolTable(globalSymbolTable, globalCount, "Global Symbol Table");
    fclose(file);

    return 0;
}

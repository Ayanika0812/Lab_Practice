#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LEXEME_LEN 100
#define MAX_TOKENS 1000

typedef struct {
    char lexeme[MAX_LEXEME_LEN];
    char type[20]; // Token type: KEYWORD, IDENTIFIER, DATATYPE, etc.
    int lineNo;    // Line number in the source code
} Token;

Token tokens[MAX_TOKENS];
int tokenCount = 0;
int currentLine = 1;

//Utility Func
void addToken(const char *lexeme, const char *type) {
    strcpy(tokens[tokenCount].lexeme, lexeme);
    strcpy(tokens[tokenCount].type, type);
    tokens[tokenCount].lineNo = currentLine;
    tokenCount++;
}

void displayTokens() {
    printf("Lexeme\t\tType\t\tLine\n");
    printf("-----------------------------------\n");
    for (int i = 0; i < tokenCount; i++) {
        printf("%-15s %-15s %d\n", tokens[i].lexeme, tokens[i].type, tokens[i].lineNo);
    }
}

// Lexical Analysis Logic

void lexicalAnalysis(FILE *file) {
    char ch;
    char buffer[MAX_LEXEME_LEN];
    int i = 0;

    while ((ch = fgetc(file)) != EOF) {
        if (ch == '\n') {
            currentLine++;
            continue;
        }

        if (isspace(ch)) {
            continue;
        }

        if (isalpha(ch)) { // Identifiers or Keywords
            buffer[i++] = ch;
            while (isalnum(ch = fgetc(file))) {
                buffer[i++] = ch;
            }
            buffer[i] = '\0';
            ungetc(ch, file); // Put back the last read character

            if (strcmp(buffer, "struct") == 0) {
                addToken(buffer, "KEYWORD");
            } else if (strcmp(buffer, "int") == 0 || strcmp(buffer, "float") == 0 || strcmp(buffer, "char") == 0) {
                addToken(buffer, "DATATYPE");
            } else {
                addToken(buffer, "IDENTIFIER");
            }
            i = 0;
        } else if (isdigit(ch)) { // Numbers
            buffer[i++] = ch;
            while (isdigit(ch = fgetc(file))) {
                buffer[i++] = ch;
            }
            buffer[i] = '\0';
            ungetc(ch, file);
            addToken(buffer, "NUMBER");
            i = 0;
        } else if (ch == '{' || ch == '}' || ch == ';' || ch == '=') { // Special characters
            buffer[0] = ch;
            buffer[1] = '\0';
            addToken(buffer, "SPECIAL_CHARACTER");
        }
    }
}

int main() {
    FILE *file = fopen("source_code.c", "r");
    if (!file) {
        printf("Error: Unable to open file.\n");
        return 1;
    }

    lexicalAnalysis(file);
    fclose(file);

    displayTokens();
    return 0;
}

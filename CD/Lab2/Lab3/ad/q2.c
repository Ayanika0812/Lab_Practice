#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKEN_LEN 100

// Token structure
typedef struct {
    char lexeme[MAX_TOKEN_LEN];
    char type[20]; // Token type: KEYWORD, VARIABLE, SYMBOL, etc.
    int lineNo;
} Token;

int currentLine = 1; // Keep track of line numbers
FILE *file; // Input Perl script file

// Function to classify and return the next token
Token getNextToken() {
    Token token;
    char ch;
    char buffer[MAX_TOKEN_LEN];
    int i = 0;

    while ((ch = fgetc(file)) != EOF) {
        // Ignore whitespace
        if (isspace(ch)) {
            if (ch == '\n') currentLine++;
            continue;
        }

        // Ignore #! and comments
        if (ch == '#') {
            ch = fgetc(file);
            if (ch == '!') { // Ignore #!
                while ((ch = fgetc(file)) != '\n' && ch != EOF);
                currentLine++;
            } else { // Regular comments
                while ((ch = fgetc(file)) != '\n' && ch != EOF);
                currentLine++;
            }
            continue;
        }

        // Handle variables starting with $
        if (ch == '$') {
            buffer[i++] = ch;
            while (isalnum(ch = fgetc(file)) || ch == '_') {
                buffer[i++] = ch;
            }
            buffer[i] = '\0';
            ungetc(ch, file); // Put back the last character
            strcpy(token.lexeme, buffer);
            strcpy(token.type, "VARIABLE");
            token.lineNo = currentLine;
            return token;
        }

        // Handle array reference @_ or other symbols
        if (ch == '@') {
            buffer[i++] = ch;
            if ((ch = fgetc(file)) == '_') {
                buffer[i++] = ch;
            } else {
                ungetc(ch, file);
            }
            buffer[i] = '\0';
            strcpy(token.lexeme, buffer);
            strcpy(token.type, "SYMBOL");
            token.lineNo = currentLine;
            return token;
        }

        // Handle keywords
        if (isalpha(ch)) {
            buffer[i++] = ch;
            while (isalnum(ch = fgetc(file)) || ch == '_') {
                buffer[i++] = ch;
            }
            buffer[i] = '\0';
            ungetc(ch, file);

            if (strcmp(buffer, "scalar") == 0 || strcmp(buffer, "foreach") == 0) {
                strcpy(token.lexeme, buffer);
                strcpy(token.type, "KEYWORD");
            } else {
                strcpy(token.lexeme, buffer);
                strcpy(token.type, "IDENTIFIER");
            }
            token.lineNo = currentLine;
            return token;
        }

        // Handle operators and symbols
        if (strchr("+=-*/{}();", ch)) {
            buffer[0] = ch;
            buffer[1] = '\0';

            // Handle += as a single token
            if (ch == '+' && (ch = fgetc(file)) == '=') {
                buffer[1] = '=';
                buffer[2] = '\0';
            } else {
                ungetc(ch, file);
            }
            strcpy(token.lexeme, buffer);
            strcpy(token.type, "SYMBOL");
            token.lineNo = currentLine;
            return token;
        }

        // Handle numbers
        if (isdigit(ch)) {
            buffer[i++] = ch;
            while (isdigit(ch = fgetc(file)) || ch == '.') { // Handle floats
                buffer[i++] = ch;
            }
            buffer[i] = '\0';
            ungetc(ch, file);
            strcpy(token.lexeme, buffer);
            strcpy(token.type, "NUMBER");
            token.lineNo = currentLine;
            return token;
        }
    }

    // End of file token
    strcpy(token.lexeme, "EOF");
    strcpy(token.type, "EOF");
    token.lineNo = currentLine;
    return token;
}
int main() {
    file = fopen("script.pl", "r");
    if (!file) {
        printf("Error: Unable to open the file.\n");
        return 1;
    }

    printf("Lexeme\t\tType\t\tLine\n");
    printf("------------------------------------\n");

    Token token;
    do {
        token = getNextToken();
        if (strcmp(token.type, "EOF") != 0) {
            printf("%-15s %-15s %d\n", token.lexeme, token.type, token.lineNo);
        }
    } while (strcmp(token.type, "EOF") != 0);

    fclose(file);
    return 0;
}

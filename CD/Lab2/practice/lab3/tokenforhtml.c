#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKEN_LENGTH 100

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

        // Detect HTML Tags
        if (c == '<') {
            char tag[MAX_TOKEN_LENGTH] = "<";
            int i = 1;

            while ((c = fgetc(inputFile)) != EOF && c != '>' && i < MAX_TOKEN_LENGTH - 1) {
                tag[i++] = c;
                col++;
            }
            if (c == '>') {
                tag[i++] = '>';
                col++;
            }
            tag[i] = '\0';

            fprintf(outputFile, "<TAG: %s, %d, %d>\n", tag, row, col - i + 1);
            continue;
        }

        // Detect Attributes and Attribute Values inside a tag
        if (isalpha(c)) {
            char attr[MAX_TOKEN_LENGTH] = {c};
            int i = 1;

            while ((c = fgetc(inputFile)) != EOF && (isalnum(c) || c == '-' || c == '_') && i < MAX_TOKEN_LENGTH - 1) {
                attr[i++] = c;
                col++;
            }
            attr[i] = '\0';

            fprintf(outputFile, "<ATTRIBUTE: %s, %d, %d>\n", attr, row, col - i + 1);

            if (c == '=') {
                fprintf(outputFile, "<SYMBOL: =, %d, %d>\n", row, col);
                c = fgetc(inputFile);
                col++;

                // Detect Attribute Value (inside quotes)
                if (c == '"' || c == '\'') {
                    char quote = c;
                    char value[MAX_TOKEN_LENGTH] = {c};
                    i = 1;

                    while ((c = fgetc(inputFile)) != EOF && c != quote && i < MAX_TOKEN_LENGTH - 1) {
                        value[i++] = c;
                        col++;
                    }
                    if (c == quote) {
                        value[i++] = c;
                        col++;
                    }
                    value[i] = '\0';

                    fprintf(outputFile, "<VALUE: %s, %d, %d>\n", value, row, col - i + 1);
                }
            }
            ungetc(c, inputFile);
            continue;
        }

        // Detect Text Content
        if (!ispunct(c)) {
            char text[MAX_TOKEN_LENGTH] = {c};
            int i = 1;

            while ((c = fgetc(inputFile)) != EOF && c != '<' && !ispunct(c) && i < MAX_TOKEN_LENGTH - 1) {
                text[i++] = c;
                col++;
            }
            text[i] = '\0';
            ungetc(c, inputFile);

            fprintf(outputFile, "<TEXT: %s, %d, %d>\n", text, row, col - i + 1);
            continue;
        }

        // Detect Special Characters
        if (ispunct(c)) {
            fprintf(outputFile, "<SYMBOL: %c, %d, %d>\n", c, row, col);
        }
        
        col++;
    }
}

int main() {
    FILE *inputFile = fopen("input.html", "r");
    FILE *outputFile = fopen("output.txt", "w");

    if (!inputFile || !outputFile) {
        printf("Error opening files.\n");
        return 1;
    }

    identifyTokens(inputFile, outputFile);

    fclose(inputFile);
    fclose(outputFile);

    printf("Tokenized output written to 'output.txt'.\n");
    return 0;
}

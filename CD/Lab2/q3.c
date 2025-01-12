#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define MAX_KEYWORDS 32

// List of C keywords
const char *keywords[MAX_KEYWORDS] = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if",
    "int", "long", "register", "return", "short", "signed", "sizeof",
    "static", "struct", "switch", "typedef", "union", "unsigned", "void",
    "volatile", "while"
};

// Function to check if a word is a C keyword
int isKeyword(const char *word) {
    for (int i = 0; i < MAX_KEYWORDS; i++) {
        if (strcmp(word, keywords[i]) == 0) {
            return 1; // It's a keyword
        }
    }
    return 0; // Not a keyword
}

// Function to convert a string to uppercase
void toUpperCase(char *str) {
    for (int i = 0; str[i]; i++) {
        str[i] = toupper(str[i]);
    }
}

int main() {
    FILE *inputFile, *outputFile;
    char inputFileName[100], outputFileName[100];
    char word[256];  // Buffer to store individual words
    int ch, index = 0;

    // Prompt for input file name
    printf("Enter the name of the input C file: ");
    scanf("%s", inputFileName);

    // Open the input file in read mode
    inputFile = fopen(inputFileName, "r");
    if (inputFile == NULL) {
        printf("Cannot open file %s.\n", inputFileName);
        return 1;
    }

    // Prompt for output file name
    printf("Enter the name of the output file: ");
    scanf("%s", outputFileName);

    // Open the output file in write mode
    outputFile = fopen(outputFileName, "w");
    if (outputFile == NULL) {
        printf("Cannot open file %s.\n", outputFileName);
        fclose(inputFile);
        return 1;
    }

    // Read the input file character by character
    while ((ch = getc(inputFile)) != EOF) {
        if (isalnum(ch) || ch == '_') {
            // Build a word
            word[index++] = ch;
        } else {
            if (index > 0) {
                // End of a word
                word[index] = '\0';
                if (isKeyword(word)) {
                    toUpperCase(word); // Convert keyword to uppercase
                }
                fputs(word, outputFile); // Write the word to the output file
                index = 0;
            }
            // Write the non-alphanumeric character to the output file
            putc(ch, outputFile);
        }
    }

    // Handle the last word if any
    if (index > 0) {
        word[index] = '\0';
        if (isKeyword(word)) {
            toUpperCase(word); // Convert keyword to uppercase
        }
        fputs(word, outputFile);
    }

    // Close the files
    fclose(inputFile);
    fclose(outputFile);

    printf("Processing complete. Keywords have been converted to uppercase in %s.\n", outputFileName);
    return 0;
}

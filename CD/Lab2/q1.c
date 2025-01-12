#include <stdio.h>

int main() {
    FILE *inputFile, *outputFile;
    char inputFileName[100], outputFileName[100];
    int currentChar, previousChar = 0;

    // Input file name
    printf("Enter the name of the input file: ");
    scanf("%s", inputFileName);

    // Open the input file in read mode
    inputFile = fopen(inputFileName, "r");
    if (inputFile == NULL) {
        printf("Cannot open file %s.\n", inputFileName);
        return 1;
    }

    // Output file name
    printf("Enter the name of the output file: ");
    scanf("%s", outputFileName);

    // Open the output file in write mode
    outputFile = fopen(outputFileName, "w");
    if (outputFile == NULL) {
        printf("Cannot open file %s.\n", outputFileName);
        fclose(inputFile);
        return 1;
    }

    // Process the input file
    while ((currentChar = getc(inputFile)) != EOF) {
        if (currentChar == ' ' || currentChar == '\t') {
            if (previousChar != ' ') {
                putc(' ', outputFile);  // Write a single space to the output file
                previousChar = ' ';
            }
        } else {
            putc(currentChar, outputFile);  // Write non-space characters to the output file
            previousChar = currentChar;    // Update the previous character
        }
    }

    // Close the files
    fclose(inputFile);
    fclose(outputFile);

    printf("Processing complete. Output written to %s.\n", outputFileName);
    return 0;
}

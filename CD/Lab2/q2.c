#include <stdio.h>

int main() {
    FILE *inputFile, *outputFile;
    char inputFileName[100], outputFileName[100];
    char line[1024];  // Buffer to hold each line of the file

    // Prompt for input file name
    printf("Enter the name of the input file: ");
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

    // Process the input file line by line
    while (fgets(line, sizeof(line), inputFile)) {
        if (line[0] != '#') {  // Check if the line is not a preprocessor directive
            fputs(line, outputFile);  // Write the line to the output file
        }
    }

    // Close the files
    fclose(inputFile);
    fclose(outputFile);

    printf("Processing complete. Output written to %s.\n", outputFileName);
    return 0;
}

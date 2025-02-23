/*
Grammar:
Program -> main() { declarations assign_stat }
declarations -> data-type identifier-list; declarations | ε
data-type -> int | char
identifier-list -> id | id, identifier-list
assign_stat -> id=id; | id=num;
*/

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

char input[100];
int pos = 0, row = 1, col = 1;

void error(const char *expected) {
    printf("Error: Expected %s at row %d, column %d\n", expected, row, col);
    exit(1);
}

void match(char expected) {
    if (input[pos] == expected) {
        pos++;
        col++;
    } else {
        error(&expected);
    }
}

void identifier_list();
void declarations();
void assign_stat();

void data_type() {
    if (strncmp(&input[pos], "int", 3) == 0) {
        pos += 3;
        col += 3;
    } else if (strncmp(&input[pos], "char", 4) == 0) {
        pos += 4;
        col += 4;
    } else {
        error("int or char");
    }
}

void identifier() {
    if (isalpha(input[pos])) {
        while (isalnum(input[pos])) {
            pos++;
            col++;
        }
    } else {
        error("identifier");
    }
}

void identifier_list() {
    identifier();
    if (input[pos] == ',') {
        match(',');
        identifier_list();
    }
}

void declarations() {
    if (strncmp(&input[pos], "int", 3) == 0 || strncmp(&input[pos], "char", 4) == 0) {
        data_type();
        identifier_list();
        match(';');
        declarations();
    }
}

void assign_stat() {
    identifier();
    match('=');
    if (isalpha(input[pos])) {
        identifier();
    } else if (isdigit(input[pos])) {
        while (isdigit(input[pos])) {
            pos++;
            col++;
        }
    } else {
        error("identifier or number");
    }
    match(';');
}

void program() {
    if (strncmp(&input[pos], "main", 4) == 0) {
        pos += 4;
        col += 4;
        match('(');
        match(')');
        match('{');
        declarations();
        assign_stat();
        match('}');
    } else {
        error("main");
    }
}

int main() {
    printf("Enter the input string: ");
    fgets(input, sizeof(input), stdin);
    input[strcspn(input, "\n")] = 0;
    program();
    printf("Parsing successful!\n");
    return 0;
}

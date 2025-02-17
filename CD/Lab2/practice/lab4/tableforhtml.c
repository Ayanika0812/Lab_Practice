#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define TABLE_SIZE 50

struct node {
    char lexeme[100];
    char type[20];
    struct node* next;
};

struct node* symbol_table[TABLE_SIZE];

void initialize_symbol_table() {
    for (int i = 0; i < TABLE_SIZE; i++) {
        symbol_table[i] = NULL;
    }
}

int hash(char* str) {
    int hash_value = 0;
    while (*str) {
        hash_value = (hash_value * 31 + *str) % TABLE_SIZE;
        str++;
    }
    return hash_value;
}

void display_symbol_table() {
    printf("Symbol Table:\n");
    printf("------------------------------------------------------------\n");
    printf("LexemeName\tType\n");
    printf("------------------------------------------------------------\n");

    for (int i = 0; i < TABLE_SIZE; i++) {
        struct node* entry = symbol_table[i];
        while (entry) {
            printf("%s\t\t%s\n", entry->lexeme, entry->type);
            entry = entry->next;
        }
    }
}

struct node* search_in_symbol_table(char* lexeme) {
    int index = hash(lexeme);
    struct node* entry = symbol_table[index];
    while (entry) {
        if (strcmp(entry->lexeme, lexeme) == 0) {
            return entry;
        }
        entry = entry->next;
    }
    return NULL;
}

void insert_into_symbol_table(char* lexeme, char* type) {
    if (search_in_symbol_table(lexeme)) {
        return;
    }

    struct node* new_entry = (struct node*)malloc(sizeof(struct node));
    strcpy(new_entry->lexeme, lexeme);
    strcpy(new_entry->type, type);
    new_entry->next = NULL;

    int index = hash(lexeme);
    if (!symbol_table[index]) {
        symbol_table[index] = new_entry;
    } else {
        struct node* temp = symbol_table[index];
        while (temp->next) {
            temp = temp->next;
        }
        temp->next = new_entry;
    }
}

// Remove HTML comments <!-- ... -->
void remove_html_comments(const char* input_file, const char* output_file) {
    FILE* fa = fopen(input_file, "r");
    FILE* fb = fopen(output_file, "w");

    if (!fa || !fb) {
        printf("Error opening files\n");
        exit(1);
    }

    int ch, prev_ch = -1;
    int in_comment = 0;

    while ((ch = fgetc(fa)) != EOF) {
        if (prev_ch == '<' && ch == '!') {
            char buffer[4];
            fread(buffer, 1, 3, fa); // Read next three chars
            if (strncmp(buffer, "--", 2) == 0) {
                in_comment = 1;
                while ((ch = fgetc(fa)) != EOF) {
                    if (ch == '-' && (fgetc(fa) == '-' && fgetc(fa) == '>')) {
                        in_comment = 0;
                        break;
                    }
                }
            } else {
                fputc('<', fb);
                fputc('!', fb);
                fwrite(buffer, 1, 3, fb);
            }
        } else if (!in_comment) {
            fputc(ch, fb);
        }

        prev_ch = ch;
    }

    fclose(fa);
    fclose(fb);
}

// Identify HTML tags and attributes
void identify_html_tokens(const char* input_file) {
    FILE* fp = fopen(input_file, "r");
    if (fp == NULL) {
        printf("Error opening file\n");
        exit(1);
    }

    char c, buffer[100];
    int buffer_index = 0;
    int inside_tag = 0;
    int inside_attr = 0;

    while ((c = fgetc(fp)) != EOF) {
        if (c == '<') {  // Start of a tag
            inside_tag = 1;
            buffer_index = 0;
            continue;
        }
        
        if (inside_tag) {
            if (c == ' ' || c == '>') { // End of tag name
                buffer[buffer_index] = '\0';
                insert_into_symbol_table(buffer, "HTML Tag");
                buffer_index = 0;

                if (c == '>') {
                    inside_tag = 0;
                }
                continue;
            }
            buffer[buffer_index++] = c;
        } else if (c == '=') { // Attribute assignment
            buffer[buffer_index] = '\0';
            insert_into_symbol_table(buffer, "Attribute");
            buffer_index = 0;
            inside_attr = 1;
            continue;
        } else if (c == '"' && inside_attr) { // Attribute value (skip)
            while ((c = fgetc(fp)) != EOF && c != '"');
            inside_attr = 0;
        } else if (isalnum(c) || c == '-' || c == '_') { // Collecting attributes or text
            buffer[buffer_index++] = c;
        } else {
            buffer[buffer_index] = '\0';
            if (buffer_index > 0) {
                insert_into_symbol_table(buffer, "Text Content");
            }
            buffer_index = 0;
        }
    }

    fclose(fp);
}

int main() {
    initialize_symbol_table();
    printf("Symbol Table initialized\n");

    // Remove HTML comments
    remove_html_comments("input.html", "no_comments.html");
    printf("HTML comments removed\n");

    // Identify tokens
    identify_html_tokens("no_comments.html");
    printf("HTML tokens identified\n");

    // Display symbol table
    display_symbol_table();

    return 0;
}

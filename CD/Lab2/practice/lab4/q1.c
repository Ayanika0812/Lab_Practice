#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<ctype.h>

#define TABLE_SIZE 30

struct node{
    char lexeme[100];
    char type[20];
    int size;
    struct node* next;
};

struct node* symbol_table[TABLE_SIZE];

void initialize_symbol_table(){
    for(int i=0;i<TABLE_SIZE;i++){
        symbol_table[i]=NULL;
    }
}

int hash(char* str){
    int hash_value = 0;
    while(*str){
        hash_value = (hash_value * 31 + *str) % TABLE_SIZE;
        str++;
    }
    return hash_value;
}

void display_symbol_table(){
    printf("Symbol Table:\n");
    printf("------------------------------------------------------------\n");
    printf("LexemeName\tTyoe\tSize\n");
    printf("-------------------------------------------------------------\n");

    for(int i = 0;i<TABLE_SIZE;i++){
        struct node* entry = symbol_table[i];
        while (entry)
        {
            printf("%s\t\t%s\t%d\n",entry->lexeme,entry->type,entry->size);
            entry = entry->next;
        }
        
    }
}

int is_keyword(const char* str){
    const char* keywords[]={
        "int", "float", "char", "double", "if", "else", "while", "for", "return", "void", "main", "break",
        "continue", "switch", "case", "default", "do", "sizeof", "struct", "typedef", NULL        
    };

    for(int i = 0; keywords[i] != NULL; i++){
        if(strcmp(str,keywords[i]) == 0){
            return 1;
        }
    }
    return 0;
}

struct node* search_in_symbol_table(char* lexeme){
    int index = hash(lexeme);
    struct node* entry = symbol_table[index];
    while(entry){
        if(strcmp(entry->lexeme,lexeme) == 0 ){
            return entry;
        }
        entry = entry->next;
    }
    return NULL;
}

void insert_into_symbol_table(char* lexeme , char* type , int size){

    if(search_in_symbol_table(lexeme)){
        return;
    }

    struct node* new_entry = (struct node*)malloc(sizeof(struct node));

    strcpy(new_entry->lexeme,lexeme);
    strcpy(new_entry->type,type);
    new_entry->size=size;
    new_entry->next=NULL;

    int index = hash(lexeme);
    if(!symbol_table[index]){
        symbol_table[index] = new_entry;
    }else{
        struct node* temp = symbol_table[index];
        while(temp->next){
            temp=temp->next;
        }
        temp->next = new_entry;
    }
}


//removing preprocessor
void remove_preprocessor(const char* input_file, const char* output_file){

    FILE* fa =fopen(input_file,"r");
    FILE* fb =fopen(output_file,"w");

    if(!fa || !fb){
        printf("Error opening files\n");
        exit(1);
    }
    int ch;
    while((ch=fgetc(fa))!=EOF){

        if(ch =='#'){
            while((ch=fgetc(fa))!= EOF && ch !='\n'){}
        }else{
            fputc(ch,fb);
        }
    }

    fclose(fa);
    fclose(fb);
}

//removing comments

void remove_comments(const char* input_file, const char* output_file){
    FILE* fa=fopen(input_file,"r");
    FILE* fb=fopen(output_file,"w");

    if(!fa || !fb){
        printf("Error oepning files\n");
        exit(1);
    }

    int ch, prev_ch =-1;
    int single_comment=0,multi_comment=0;

    while((ch=fgetc(fa))!=EOF){
        
        if(!multi_comment && prev_ch == '/' && ch =='/'){
            single_comment=1;
        }

        else if(!single_comment && prev_ch == '/' && ch == '*'){
            multi_comment=1;
        }

        if(single_comment && ch == '\n'){
            single_comment=0;
        }

        else if(multi_comment && prev_ch == '*' && ch == '/'){
            multi_comment=0;   //end of multline comment
            //ch = fgetc(fa);   //Move to next character
            prev_ch=-1;   //resets
            continue;  //skip writing '/'
        }
        
        else if(!single_comment && !multi_comment){
            if(prev_ch != EOF){
                fputc(prev_ch,fb);
            }
        }

        prev_ch=ch;
    }

    if(!single_comment && !multi_comment && prev_ch != EOF){
        fputc(prev_ch,fb);
    }
    fclose(fa);
    fclose(fb);

}

void remove_whitespace(const char * input_file , const char* output_file){

    FILE* fa = fopen(input_file,"r");
    FILE* fb = fopen(output_file,"w");

    if(!fa || !fb){
        printf("Error opening files\n");
        exit(1);
    }
                                                                             
    int ch, prev_ch = -1;
    int last_was_space = 0;
    int last_was_newline = 0;

    while ((ch = fgetc(fa)) != EOF) {
        if (isspace(ch)) {
            if (ch == '\n') {
                if (!last_was_newline) {
                    fputc('\n', fb);  // Preserve single newlines
                    last_was_newline = 1;
                }
            } else if (!last_was_space && prev_ch != -1) {
                fputc(' ', fb);  // Preserve single spaces
                last_was_newline = 0;
            }
            last_was_space = 1;
        } else {
            fputc(ch, fb);
            last_was_space = 0;
            last_was_newline = 0;
        }
        prev_ch = ch;
    }

    fclose(fa);
    fclose(fb);
}

void identify_tokens(const char* input_file){
    FILE* fp = fopen(input_file,"r");
    if(fp ==NULL){
        printf("Error opening file\n");
        exit(1);
    }

    int row = 1, col =1;
    char c, buffer[100];
    int buffer_index = 0;
    char type[10] = "";

    while ((c = fgetc(fp)) != EOF){

        if(c == '\n'){
            row++;
            col = 1; //next line
            continue;
        }

        if(isspace(c)){
            col++;
            continue;
        }

        if(c == '"'){  //string literal
            buffer[buffer_index++] = c;
            col++;
            while((c = fgetc(fp)) != EOF && c != '"'){
                buffer[buffer_index++] = c;
                col++;
            }

            buffer[buffer_index++] = c;  //includes last "
            col++;
            buffer[buffer_index]= '\0';

            insert_into_symbol_table(buffer, "string_literal", strlen(buffer)+1);
            buffer_index = 0;
            continue;
        }

        if(isalpha(c) || c == '_'){
            buffer[buffer_index++] = c;
            col++;
            while( isalnum((c=fgetc(fp))) || c == '_'){
                buffer[buffer_index++] = c;
                col++;
            }
            buffer[buffer_index] = '\0';

            if(strcmp(buffer,"main") == 0){
                insert_into_symbol_table(buffer,"function",0);
            }else if(strcmp(buffer,"printf") == 0){
                insert_into_symbol_table(buffer, "function",0);
            }

            else if(strcmp(buffer, "int") == 0||strcmp(buffer, "float") == 0||strcmp(buffer, "char") == 0){

                strcpy(type,buffer);
            }else {
                // Check if the token is a keyword before inserting
                if (!is_keyword(buffer)) {
                    if (strcmp(type, "int") == 0) {
                        insert_into_symbol_table(buffer, "int", 4);
                    } else if (strcmp(type, "float") == 0) {
                        insert_into_symbol_table(buffer, "float", 4);
                    } else if (strcmp(type, "char") == 0) {
                        insert_into_symbol_table(buffer, "char", 1);
                    }
                }
            }
            

            buffer_index = 0;
            ungetc(c,fp);
            continue;
        }
    }
    fclose(fp);
}

int main(){

    initialize_symbol_table();
    printf("Symbol Table initialized\n");

    //Step 1 ->  Remove preprocessor

    remove_preprocessor("q1in.c","preprocessed.c");
    printf("Preprocessor removed\n");
    //comments
    remove_comments("preprocessed.c","no_comments.c");
    printf("Comments removed\n");

    //whitespace
    remove_whitespace("no_comments.c","cleaned.c");
    printf("Whitespaces removed\n");   

    //identify tokens

    identify_tokens("cleaned.c");
    printf("Tokens identified\n");

    display_symbol_table();

    return 0;
}
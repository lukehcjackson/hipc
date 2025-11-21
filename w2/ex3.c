#include <stdio.h>
#include <stdlib.h>

struct element {
    struct element* next;
    int data;
};

struct linked_list {
    struct element* head;
};

void append_int(struct linked_list* list, int val) {
    struct element* elem = malloc(sizeof(struct element));

    elem->data = val;
    elem->next = NULL; // Really important to explicitly set this to null. Malloc does not zero memory

    if (list->head == NULL) {
        // Empty list, we need to append to head
        list->head = elem;
    } else {
        // List has some elements, find the end and append to that
        struct element* tail = list->head;
        while (tail->next != NULL) {
            tail = tail->next;
        }
        tail->next = elem;
    }

}

void insert_int(struct linked_list* list, int val) {

    //want: insert a value at the head of the linked list

    //make the element
    struct element* elem = malloc(sizeof(struct element));
    elem->data = val;

    if (list->head == NULL) {
        //empty list => make this the head
        list->head = elem;
        elem->next = NULL;
    } else {
        struct element* other = list->head;
        list->head = elem;
        elem->next = other;
    }

}

void delete_head(struct linked_list* list) {

    //want: delete head of linked list

    
    //do i need to free the memory from the initial head?

    list->head = list->head->next;

}

int main() {

    struct linked_list* list = malloc(sizeof(struct linked_list));

    struct element* element4 = malloc(sizeof(struct element));
    element4->data = 400;
    element4->next = NULL;

    struct element* element3 = malloc(sizeof(struct element));
    element3->data = 300;
    element3->next = element4;

    struct element* element2 = malloc(sizeof(struct element));
    element2->data = 200;
    element2->next = element3;

    struct element* element1 = malloc(sizeof(struct element));
    element1->data = 100;
    element1->next = element2;

    list->head = element1;


    append_int(list, 500); //works

    insert_int(list, 20); //works

    delete_head(list); //works


    struct element* tail = list->head;
    int counter = 0;
    while (tail->data != -1) {
        counter++;
        printf("Element: %d    Value: %d \n", counter, tail->data);
        if (tail->next != NULL) {
            tail = tail->next;
        } else {
            break;
        }
        
    }

    return 0;
}
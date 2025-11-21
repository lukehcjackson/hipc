#include <stdio.h>
#include <stdlib.h>

struct element {
    struct element* next;
    int data;
};

struct queue {
    struct element* head;
};

int isempty(struct queue* que) {

    if (que->head == NULL) {
        return 1;
    }

    return 0;

}

void enqueue(struct queue* que, int val) {
    struct element* elem = malloc(sizeof(struct element));

    elem->data = val;
    elem->next = NULL; // Really important to explicitly set this to null. Malloc does not zero memory

    if (que->head == NULL) {
        que->head = elem;
    } else {
        struct element* tail = que->head;
        while (tail->next != NULL) {
            tail = tail->next;
        }
        tail->next = elem;
    }

}

void dequeue(struct queue* que) {

    //do i need to free the memory from the initial head?
    que->head = que->head->next;

}

int main() {

    struct queue* my_queue = malloc(sizeof(struct queue));

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

    my_queue->head = element1;

    enqueue(my_queue, 700);

    if (isempty(my_queue)) {
        printf("queue empty\n");
    } else {
        printf("queue not empty\n");
    }

    dequeue(my_queue);


    struct element* tail = my_queue->head;
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
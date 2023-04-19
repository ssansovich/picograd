#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Value {
  float data;
  float grad;
  char op[5];
  char label[20];
  struct Value *child_a_ptr;
  struct Value *child_b_ptr;
  void (*backward)(struct Value *);
} Value;

void backward_default(Value *self_ptr) { return; }

Value* init_value(float data, const char *label) {
	Value *out = (Value *)malloc(sizeof(Value));
	out->data = data;
	out->grad = 0.0;
    out->child_a_ptr = NULL;
    out->child_b_ptr = NULL;
	strcpy(out->label, label);
	out->backward = backward_default;
	return out;
}

void backward_add(Value *self_ptr) {
    self_ptr->child_a_ptr->grad += self_ptr->grad;
    self_ptr->child_b_ptr->grad += self_ptr->grad;
    return;
};

Value* add(Value *a_ptr, Value *b_ptr, const char *label) {
	Value *out = (Value *)malloc(sizeof(Value));
    out->data = a_ptr->data + b_ptr->data,
    out->child_a_ptr = a_ptr,
    out->child_b_ptr = b_ptr,
    out->backward = backward_add,
	strcpy(out->op, "+");
	strcpy(out->label, label);
	return out;
}

void backward_mul(Value *self_ptr) {
    self_ptr->child_a_ptr->grad += self_ptr->child_b_ptr->data * self_ptr->grad;
    self_ptr->child_b_ptr->grad += self_ptr->child_a_ptr->data * self_ptr->grad;
    return;
};

Value* mul(Value *a_ptr, Value *b_ptr, const char *label) {
	Value *out = (Value *)malloc(sizeof(Value));
    out->data = a_ptr->data * b_ptr->data,
    out->child_a_ptr = a_ptr,
    out->child_b_ptr = b_ptr,
    out->backward = backward_mul,
    strcpy(out->op, "*");
    strcpy(out->label, label);
    return out;
}

Value* neg(Value *a_ptr, const char *label) {
    Value *minus_one = init_value(-1.0, "neg");
	Value *out = (Value *)malloc(sizeof(Value));
    out = mul(a_ptr, minus_one, label);
    return out;
}

Value* subtract(Value *a_ptr, Value *b_ptr, const char *label) {
    Value *neg_b = neg(b_ptr, "neg");
	Value *out = (Value *)malloc(sizeof(Value));
    out = add(a_ptr, neg_b, label);
    return out;
}

void backward_tanh(Value *self_ptr) {
  self_ptr->child_a_ptr->grad += (1 - self_ptr->data * self_ptr->data) * self_ptr->grad;
  return;
};

Value* pico_tanh(Value *a_ptr, const char *label) {
    float x = a_ptr->data;

	Value *out = (Value *)malloc(sizeof(Value));
    out->data = (float)(exp(2 * x) - 1) / (exp(2 * x) + 1);
    out->child_a_ptr = a_ptr;
    out->child_b_ptr = NULL;
    out->backward = backward_tanh;
    strcpy(out->op, "tanh");
    strcpy(out->label, label);
    return out;
}

typedef struct Node {
  Value *val_ptr;
  struct Node *next;
} Node;

bool is_in_value_linked_list(Node *curr, Value *val_ptr) {
    if (curr == NULL) {
        return false;
    } else if (curr->val_ptr == val_ptr) {
        return true;
    } else {
        return is_in_value_linked_list(curr->next, val_ptr);
    }
    return false;
}

Node* prepend(Node *head_ptr, Value *val_ptr) {
  Node *new_head_ptr = (Node *)malloc(sizeof(Node));
  new_head_ptr->val_ptr = val_ptr;
  new_head_ptr->next = head_ptr;
  return new_head_ptr;
}

Node* build_topo(Value *val_ptr, Node *visited_head_ptr, Node *topo_head_ptr) {
    if (!is_in_value_linked_list(visited_head_ptr, val_ptr)) {
        visited_head_ptr = prepend(visited_head_ptr, val_ptr);
        if (val_ptr->child_a_ptr) {
            topo_head_ptr = build_topo(val_ptr->child_a_ptr, visited_head_ptr, topo_head_ptr);
        }
        if (val_ptr->child_b_ptr) {
            topo_head_ptr = build_topo(val_ptr->child_b_ptr, visited_head_ptr, topo_head_ptr);
        }
        topo_head_ptr = prepend(topo_head_ptr, val_ptr);
    }
    return topo_head_ptr;
}

void backprop(Value *start_ptr) {
    Node *visited_head_ptr = NULL;

    Node *topo_head_ptr = NULL;
    topo_head_ptr = build_topo(start_ptr, visited_head_ptr, topo_head_ptr);

    Node *curr_node_ptr = topo_head_ptr;
    while (curr_node_ptr != NULL) {
        curr_node_ptr->val_ptr->backward(curr_node_ptr->val_ptr);
        curr_node_ptr = curr_node_ptr->next;
    }
    return;
}

void printGraph(Value *root, int depth) {
    printf("%s = data: %.4f, grad: %.4f\n", root->label, root->data, root->grad);

    if (root->child_a_ptr != NULL) {
        for (int i = 0; i <= depth; i++) {
            printf("• • ");
        }
        printGraph(root->child_a_ptr, depth + 1);
    }

    if (strcmp(root->op, "") != 0) {
        for (int i = 0; i <= depth; i++) {
            printf("• • ");
        }
        printf("%s\n", root->op);
    }

    if (root->child_b_ptr != NULL) {
        for (int i = 0; i <= depth; i++) {
            printf("• • ");
        }
        printGraph(root->child_b_ptr, depth + 1);
    }
    return;
}

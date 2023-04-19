#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct Neuron {
    Value *bias;
    int num_weights; // also the number of inputs
    Value **weights_arr;
} Neuron;

void init_random_seed() {
    srand((unsigned)time(NULL));
}

Value* gen_random_value(const char *label) {
    float random_float = (float)rand() / (RAND_MAX / 2.0) - 1.0;

    Value* out = init_value(random_float, label);
    // Value* out = init_value(0.2, label);
    return out;
}

Neuron* init_neuron(int num_inputs) {
    Neuron *out = malloc(sizeof(Neuron));
    out->bias = gen_random_value("bias");
    out->num_weights = num_inputs;
    out->weights_arr = malloc(num_inputs * sizeof(Value *));
    for (int i = 0; i < num_inputs; i++) {
        out->weights_arr[i] = gen_random_value("weight");
    }
    return out;
}

Value* pairwise_sum(Value **values, int start, int end) {
	Value *out;
    if (start == end) {
		Value *zero = init_value(0.0, "zero");
		out = add(values[end], zero, "pairwise_sum end");
    } else if (end - start == 1) {
        out = add(values[start], values[end], "pairwise_sum");
    } else {
        int mid = (start + end) / 2;
        Value *left_sum = pairwise_sum(values, start, mid);
        Value *right_sum = pairwise_sum(values, mid + 1, end);
        out = add(left_sum, right_sum, "pairwise_sum");
    }
	return out;
}

Value* run_neuron(Neuron *n_ptr, Value **inputs_arr) {
    Value **products_arr = malloc(n_ptr->num_weights * sizeof(Value *));
    // num_weights is always the same as the number of inputs
    for (int i = 0; i < n_ptr->num_weights; i++) {
        Value *prod = mul(n_ptr->weights_arr[i], inputs_arr[i], "product");
        products_arr[i] = prod;
    }

    Value *pairwise_sums_ptr = pairwise_sum(products_arr, 0, n_ptr->num_weights - 1);
	Value *biased = add(pairwise_sums_ptr, n_ptr->bias, "adding bias");
	Value *out = pico_tanh(biased, "pico_tanh");
    return out;
}

Value** get_neuron_params(Neuron *n_ptr) {
    Value **params_ptr_arr = (Value**)malloc(sizeof(Value*));
    for (int i = 0; i < n_ptr->num_weights; i++) {
        params_ptr_arr[i] = n_ptr->weights_arr[i];
    }
    return params_ptr_arr;
}

typedef struct Layer {
    int num_neurons;
    Neuron **neurons;
} Layer;

Layer* init_layer(int num_inputs, int num_neurons) {
    Layer *out = malloc(sizeof(Layer));
    out->num_neurons = num_neurons;
    out->neurons = malloc(num_neurons * sizeof(Neuron *));
    for (int i = 0; i < num_neurons; i++) {
        out->neurons[i] = init_neuron(num_inputs);
    }
    return out;
}

Value** run_layer(Layer *layer, Value **inputs_arr) {
    Value **outputs_arr = malloc(layer->num_neurons * sizeof(Value *));
    for (int i = 0; i < layer->num_neurons; i++) {
        outputs_arr[i] = run_neuron(layer->neurons[i], inputs_arr);
    }
    return outputs_arr;
}

typedef struct MLP {
    int num_inputs;
    int num_hidden_layers;
    int hidden_layer_size;
    int num_outputs;
    Layer **layers;
} MLP;

struct init_mlp_params {
    int num_inputs;
    int num_hidden_layers;
    int hidden_layer_size;
    int num_outputs;
};

MLP init_mlp(struct init_mlp_params params) {
    int num_total_layers = params.num_hidden_layers + 1; // to include output layer
    int *sizes_arr = malloc((num_total_layers + 1) * sizeof(int)); // to include initial inputs
    sizes_arr[0] = params.num_inputs;
    for (int i = 0; i < params.num_hidden_layers; i++) {
        sizes_arr[i + 1] = params.hidden_layer_size;
    }
    sizes_arr[num_total_layers] = params.num_outputs;

    MLP out = {
        .num_inputs = params.num_inputs,   
        .num_hidden_layers = params.num_hidden_layers,   
        .hidden_layer_size = params.hidden_layer_size,   
        .num_outputs = params.num_outputs,   
    };
    out.layers = malloc(num_total_layers * sizeof(Layer *));

    for (int i = 0; i < num_total_layers; i++) {
        out.layers[i] = init_layer(sizes_arr[i], sizes_arr[i + 1]);
    }

    free(sizes_arr);
    return out;
}

Value** run_mlp(MLP mlp, float *inputs_arr) {
    // Convert input floats to values
    Value **outputs_arr = malloc(sizeof(Value *) * mlp.hidden_layer_size); // max size of layer since we reuse the outputs arr, passing each result to the next layer
    for (int i = 0; i < mlp.num_inputs; i++) {
        outputs_arr[i] = init_value(inputs_arr[i], "run mlp init");
    }

    for (int i = 0; i < mlp.num_hidden_layers + 1; i++) { // run for all hidden layers plus the final output layer
        outputs_arr = run_layer(mlp.layers[i], outputs_arr);
    }
    
    return outputs_arr;
}

Value* calc_loss(float *ground_truths, Value **predictions, int num_predictions) {
    // Loss function = sum of (yout - ygt) ^ 2
    Value **products = malloc(num_predictions * sizeof(Value *));
    for (int i = 0; i < num_predictions; i++) {
        Value *gt = init_value(ground_truths[i], "ygt init");
        Value *diff = subtract(predictions[i], gt, "loss subtract");
        products[i] = mul(diff, diff, "loss prod");
    }

    Value *out = pairwise_sum(products, 0, num_predictions - 1);
    return out;
}

Value** get_mlp_params(MLP *mlp_ptr, int num_params) {
    Value **params_ptr_arr = (Value**)malloc(sizeof(Value*) * num_params);
    int param_count = 0;
    for (int i = 0; i < mlp_ptr->num_hidden_layers + 1; i++) {
        Layer *layer_ptr = mlp_ptr->layers[i];
        for (int j = 0; j < layer_ptr->num_neurons; j++) {
            Neuron *neuron_ptr = layer_ptr->neurons[j];
            for (int k = 0; k < neuron_ptr->num_weights; k++) {
                params_ptr_arr[param_count] = neuron_ptr->weights_arr[k];
                param_count++;
            }
            params_ptr_arr[param_count] = neuron_ptr->bias;
            param_count++;
        }
    }
    return params_ptr_arr;
}

void print_params(Value **params_arr, int num_params) {
    for (int i = 0; i < num_params; i++) {
        printf("%i %s (%.8f, %.8f)\n", i, params_arr[i]->label, params_arr[i]->data, params_arr[i]->grad);
    }
    return;
}

void zero_grads(Value **params_arr, int num_params) {
    for (int i = 0; i < num_params; i++) {
        params_arr[i]->grad = 0.0;
    }
    return;
}

void update_params(Value **params_arr, int num_params) {
    for (int i = 0; i < num_params; i++) {
        params_arr[i]->data += -0.01 * params_arr[i]->grad;
    }
    return;
}

Value** convert_floats_to_values(float *floats_arr, int num_floats) {
    Value **out = malloc(sizeof(Value *) * num_floats);
    for (int i = 0; i < num_floats; i++) {
        out[i] = init_value(floats_arr[i], "init value");
    }
    return out;
}

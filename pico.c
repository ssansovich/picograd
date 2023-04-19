#include "engine.c"
#include "nn.c"

int main() {
    init_random_seed();

    struct init_mlp_params mlp_params = {
        .num_inputs = 3,
        .num_hidden_layers = 2,
        .hidden_layer_size = 4,
        .num_outputs = 1,
    };
    int num_params = 41;

    int num_prediction_runs = 4;
    float input_cases[4][3] = {
        { 2.0, 3.0, -1.0 },
        { 3.0, -1.0, 0.5 },
        { 0.5, 1.0, 1.0 },
        { 1.0, 1.0, -1.0 }
    };
    float ground_truths[4] = { 1.0, -1.0, -1.0, 1.0 }; // Desired targets

    MLP mlp = init_mlp(mlp_params);
    Value **params = get_mlp_params(&mlp, num_params);

    int num_steps = 60;
    for (int i = 0; i < num_steps; i++) {
        // forward pass
        Value **predictions = malloc(sizeof(Value *) * num_prediction_runs);
        for (int j = 0; j < num_prediction_runs; j++) {
            predictions[j] = *run_mlp(mlp, input_cases[j]);
        }
        Value *loss = calc_loss(ground_truths, predictions, num_prediction_runs);

        // backward pass
        zero_grads(params, num_params);
        loss->grad = 1;
        backprop(loss);

        // update
        update_params(params, num_params);
        loss = calc_loss(ground_truths, predictions, num_prediction_runs);
        free(loss);
    }

    return 0;
}

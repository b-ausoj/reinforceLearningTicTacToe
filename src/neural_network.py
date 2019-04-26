# 2. The class NeuralNetwork
import pickle
from src import functions_nn as f


class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes1, hiddennodes2, hiddennodes3, outputnodes, learningrate, new_weights, saving_place):
        self.inodes = inputnodes
        self.hnodes1 = hiddennodes1
        self.hnodes2 = hiddennodes2
        self.hnodes3 = hiddennodes3
        self.onodes = outputnodes
        self.save = saving_place

        if new_weights == "new":
            self.wih1 = f.random_numbers_gauss(0.0, pow(self.hnodes1, -0.5), self.hnodes1, self.inodes)
            self.wh1h2 = f.random_numbers_gauss(0.0, pow(self.hnodes2, -0.5), self.hnodes2, self.hnodes1)
            self.wh2h3 = f.random_numbers_gauss(0.0, pow(self.hnodes3, -0.5), self.hnodes3, self.hnodes2)
            self.wh3o = f.random_numbers_gauss(0.0, pow(self.onodes, -0.5), self.onodes, self.hnodes3)
        else:
            read_weights = open(self.save, "rb")
            self.wh3o, self.wh2h3, self.wh1h2, self.wih1 = pickle.load(read_weights)

        self.lr = learningrate

        self.activation_function = lambda x: f.tangens_hyperbolicus(x)

        pass

    def save_weights(self):
        write_weights = open(self.save, "bw")
        weights = self.wh3o, self.wh2h3, self.wh1h2, self.wih1
        pickle.dump(weights, write_weights)
        write_weights.close()

        pass

    def train(self, inputs_list, targets_list):
        inputs = f.matrix_transposition(f.list_to_matrix(inputs_list))
        targets = f.matrix_transposition([targets_list])

        hidden_inputs_1 = f.matrix_multiplication(self.wih1, inputs)
        hidden_outputs_1 = self.activation_function(hidden_inputs_1)

        hidden_inputs_2 = f.matrix_multiplication(self.wh1h2, hidden_outputs_1)
        hidden_outputs_2 = self.activation_function(hidden_inputs_2)

        hidden_inputs_3 = f.matrix_multiplication(self.wh2h3, hidden_outputs_2)
        hidden_outputs_3 = self.activation_function(hidden_inputs_3)

        final_inputs = f.matrix_multiplication(self.wh3o, hidden_outputs_3)
        final_outputs = self.activation_function(final_inputs)

        output_errors = self.activation_function(f.subtraction(targets, final_outputs))
        hidden_errors_3 = f.matrix_multiplication(f.matrix_transposition(self.wh3o), output_errors)
        hidden_errors_2 = f.matrix_multiplication(f.matrix_transposition(self.wh2h3), hidden_errors_3)
        hidden_errors_1 = f.matrix_multiplication(f.matrix_transposition(self.wh1h2), hidden_errors_2)

        self.wh3o = f.addition(self.wh3o, f.multiplication(self.lr, f.matrix_multiplication(
            f.multiplication(output_errors,
                             f.multiplication(f.addition(1.0, final_outputs), f.subtraction(1.0, final_outputs))),
            f.matrix_transposition(hidden_outputs_3))))

        self.wh2h3 = f.addition(self.wh2h3, f.multiplication(self.lr, f.matrix_multiplication(
            f.multiplication(hidden_errors_3,
                             f.multiplication(f.addition(1.0, hidden_outputs_3), f.subtraction(1.0, hidden_outputs_3))),
            f.matrix_transposition(hidden_outputs_2))))

        self.wh1h2 = f.addition(self.wh1h2, f.multiplication(self.lr, f.matrix_multiplication(
            f.multiplication(hidden_errors_2,
                             f.multiplication(f.addition(1.0, hidden_outputs_2), f.subtraction(1.0, hidden_outputs_2))),
            f.matrix_transposition(hidden_outputs_1))))

        self.wih1 = f.addition(self.wih1, f.multiplication(self.lr, f.matrix_multiplication(
            f.multiplication(hidden_errors_1,
                             f.multiplication(f.addition(1.0, hidden_outputs_1), f.subtraction(1.0, hidden_outputs_1))),
            f.matrix_transposition(inputs))))
        pass

    def query(self, inputs_list):
        inputs = f.matrix_transposition(f.list_to_matrix(inputs_list))

        hidden_inputs_1 = f.matrix_multiplication(self.wih1, inputs)
        hidden_outputs_1 = self.activation_function(hidden_inputs_1)

        hidden_inputs_2 = f.matrix_multiplication(self.wh1h2, hidden_outputs_1)
        hidden_outputs_2 = self.activation_function(hidden_inputs_2)

        hidden_inputs_3 = f.matrix_multiplication(self.wh2h3, hidden_outputs_2)
        hidden_outputs_3 = self.activation_function(hidden_inputs_3)

        final_inputs = f.matrix_multiplication(self.wh3o, hidden_outputs_3)
        [final_outputs] = f.matrix_transposition(self.activation_function(final_inputs))

        return final_outputs

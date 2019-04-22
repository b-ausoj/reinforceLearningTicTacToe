# 2. The class NeuralNetwork
import pickle
from src import functions_nn as f


class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, new_weights, saving_place):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.save = saving_place

        if new_weights == "new":
            self.wih = f.random_numbers_gauss(0.0, pow(self.hnodes, -0.5), self.hnodes, self.inodes)
            self.who = f.random_numbers_gauss(0.0, pow(self.onodes, -0.5), self.onodes, self.hnodes)
        else:
            read_weights = open(self.save, "rb")
            (self.who, self.wih) = pickle.load(read_weights)

        self.lr = learningrate

        self.activation_function = lambda x: f.tangens_hyperbolicus(x)

        pass

    def save_weights(self):
        write_weights = open(self.save, "bw")
        weights = (self.who, self.wih)
        pickle.dump(weights, write_weights)
        write_weights.close()

        pass

    def train(self, inputs_list, targets_list):
        inputs = f.matrix_transposition([inputs_list])  # mögliches problem mit [ ]
        targets = f.matrix_transposition([targets_list])  # "

        hidden_inputs = f.matrix_multiplication(self.wih, inputs)
        hidden_outputs = f.matrix_transposition(self.activation_function(hidden_inputs))

        final_inputs = f.matrix_multiplication(self.who, hidden_outputs)
        final_outputs = f.matrix_transposition(self.activation_function(final_inputs))

        output_errors = f.matrix_transposition(self.activation_function(f.subtraction(targets, final_outputs)))
        hidden_errors = f.matrix_multiplication(f.matrix_transposition(self.who), output_errors)

        self.who = f.addition(self.who, f.multiplication(self.lr, f.matrix_multiplication(
            f.multiplication(output_errors,
                             f.multiplication(f.addition(1.0, final_outputs), f.subtraction(1.0, final_outputs))),
            f.matrix_transposition(hidden_outputs))))

        self.wih = f.addition(self.wih, f.multiplication(self.lr, f.matrix_multiplication(
            f.multiplication(hidden_errors,
                             f.multiplication(f.addition(1.0, hidden_outputs), f.subtraction(1.0, hidden_outputs))),
            f.matrix_transposition(inputs))))

        pass

    def query(self, inputs_list):
        inputs = f.matrix_transposition([inputs_list])

        hidden_inputs = f.matrix_multiplication(self.wih, inputs)
        hidden_outputs = f.matrix_transposition(self.activation_function(hidden_inputs))

        final_inputs = f.matrix_multiplication(self.who, hidden_outputs)
        [final_outputs] = self.activation_function(final_inputs)

        return final_outputs

# The functions for the NeuralNetwork
import random


# numpy.random.normal
def random_numbers_gauss(mean, standard_deviation, numbers_of_rows, numbers_of_columns):
    result_matrix = []
    for i in range(numbers_of_rows):
        empty_matrix = []
        for j in range(numbers_of_columns):
            empty_matrix += [0]
        result_matrix += [empty_matrix]
        del empty_matrix
    for k in range(numbers_of_rows):
        for l in range(numbers_of_columns):
            result_matrix[k][l] = random.gauss(mean, standard_deviation)
    return result_matrix


def tangens_hyperbolicus(argument):
    e = 2.71828
    if isinstance(argument, list):
        for i in range(len(argument)):
            for j in range(len(argument[0])):
                argument[i][j] = 1 - (2 / (1 + e ** (2 * argument[i][j])))
        argument = matrix_transposition(argument)
        return argument
    elif isinstance(argument, float):
        argument = 1 - (2 / (1 + e ** (2 * argument)))
        return argument


# numpy.array (not finished)
def list_to_matrix(input_list):
    input_list = [input_list]
    return input_list


# numpy.transpose
def matrix_transposition(matrix):
    result_matrix = []
    for i in range(len(matrix[0])):
        empty_matrix = []
        for j in range(len(matrix)):
            empty_matrix += [0]
        result_matrix += [empty_matrix]
        del empty_matrix
    for k in range(len(matrix)):
        for l in range(len(matrix[0])):
            result_matrix[l][k] = matrix[k][l]
    return result_matrix


# numpy.dot
def matrix_multiplication(first, second):
    if len(first[0]) == len(second):
        result_matrix = []
        for i in range(len(first)):
            empty_matrix = []
            for j in range(len(second[0])):
                empty_matrix += [0]
            result_matrix += [empty_matrix]
            del empty_matrix
        for k in range(len(first)):
            for l in range(len(second[0])):
                for m in range(len(second)):
                    result_matrix[k][l] += first[k][m] * second[m][l]
        return result_matrix


# * (more than necessity)
def multiplication(multiplier, multiplicand):
    if isinstance(multiplier, list) and isinstance(multiplicand, list):
        if len(multiplier) == len(multiplicand) and len(multiplier[0]) == len(multiplicand[0]):
            result_matrix = []
            for i in range(len(multiplier)):
                empty_matrix = []
                for j in range(len(multiplier[0])):
                    empty_matrix += [0]
                result_matrix += [empty_matrix]
                del empty_matrix
            for k in range(len(multiplier)):
                for l in range(len(multiplier[0])):
                    result_matrix[k][l] = multiplier[k][l] * multiplicand[k][l]
            return result_matrix
        elif len(multiplier) == len(multiplicand) and len(multiplicand[0]) == 1:
            result_matrix = []
            for i in range(len(multiplier)):
                empty_matrix = []
                for j in range(len(multiplier[0])):
                    empty_matrix += [0]
                result_matrix += [empty_matrix]
                del empty_matrix
            for k in range(len(multiplier)):
                for l in range(len(multiplier[0])):
                    result_matrix[k][l] = multiplier[k][l] * multiplicand[k][0]
            return result_matrix
        elif len(multiplier[0]) == len(multiplicand[0]) and len(multiplicand) == 1:
            result_matrix = []
            for i in range(len(multiplier)):
                empty_matrix = []
                for j in range(len(multiplier[0])):
                    empty_matrix += [0]
                result_matrix += [empty_matrix]
                del empty_matrix
            for k in range(len(multiplier)):
                for l in range(len(multiplier[0])):
                    result_matrix[k][l] = multiplier[k][l] * multiplicand[0][l]
            return result_matrix
    elif isinstance(multiplier, list) and isinstance(multiplicand, float):
        result_matrix = []
        for i in range(len(multiplier)):
            empty_matrix = []
            for j in range(len(multiplier[0])):
                empty_matrix += [0]
            result_matrix += [empty_matrix]
            del empty_matrix
        for k in range(len(multiplier)):
            for l in range(len(multiplier[0])):
                result_matrix[k][l] = multiplier[k][l] * multiplicand
        return result_matrix
    elif isinstance(multiplier, float) and isinstance(multiplicand, list):
        result_matrix = []
        for i in range(len(multiplicand)):
            empty_matrix = []
            for j in range(len(multiplicand[0])):
                empty_matrix += [0]
            result_matrix += [empty_matrix]
            del empty_matrix
        for k in range(len(multiplicand)):
            for l in range(len(multiplicand[0])):
                result_matrix[k][l] = multiplier * multiplicand[k][l]
        return result_matrix


# - (more than necessity)
def subtraction(minuend, subtrahend):
    if isinstance(minuend, list) and isinstance(subtrahend, list):
        if len(minuend) == len(subtrahend) and len(minuend[0]) == len(subtrahend[0]):
            result_matrix = []
            for i in range(len(minuend)):
                empty_matrix = []
                for j in range(len(minuend[0])):
                    empty_matrix += [0]
                result_matrix += [empty_matrix]
                del empty_matrix
            for k in range(len(minuend)):
                for l in range(len(minuend[0])):
                    result_matrix[k][l] = minuend[k][l] - subtrahend[k][l]
            return result_matrix
        elif len(minuend) == len(subtrahend) and len(subtrahend[0]) == 1:
            result_matrix = []
            for i in range(len(minuend)):
                empty_matrix = []
                for j in range(len(minuend[0])):
                    empty_matrix += [0]
                result_matrix += [empty_matrix]
                del empty_matrix
            for k in range(len(minuend)):
                for l in range(len(minuend[0])):
                    result_matrix[k][l] = minuend[k][l] - subtrahend[k][0]
            return result_matrix
        elif len(minuend[0]) == len(subtrahend[0]) and len(subtrahend) == 1:
            result_matrix = []
            for i in range(len(minuend)):
                empty_matrix = []
                for j in range(len(minuend[0])):
                    empty_matrix += [0]
                result_matrix += [empty_matrix]
                del empty_matrix
            for k in range(len(minuend)):
                for l in range(len(minuend[0])):
                    result_matrix[k][l] = minuend[k][l] - subtrahend[0][l]
            return result_matrix
    elif isinstance(minuend, list) and isinstance(subtrahend, float):
        result_matrix = []
        for i in range(len(minuend)):
            empty_matrix = []
            for j in range(len(minuend[0])):
                empty_matrix += [0]
            result_matrix += [empty_matrix]
            del empty_matrix
        for k in range(len(minuend)):
            for l in range(len(minuend[0])):
                result_matrix[k][l] = minuend[k][l] - subtrahend
        return result_matrix
    elif isinstance(minuend, float) and isinstance(subtrahend, list):
        result_matrix = []
        for i in range(len(subtrahend)):
            empty_matrix = []
            for j in range(len(subtrahend[0])):
                empty_matrix += [0]
            result_matrix += [empty_matrix]
            del empty_matrix
        for k in range(len(subtrahend)):
            for l in range(len(subtrahend[0])):
                result_matrix[k][l] = minuend - subtrahend[k][l]
        return result_matrix


# + (more than necessity)
def addition(augend, addend):
    if isinstance(augend, list) and isinstance(addend, list):
        if len(augend) == len(addend) and len(augend[0]) == len(addend[0]):
            result_matrix = []
            for i in range(len(augend)):
                empty_matrix = []
                for j in range(len(augend[0])):
                    empty_matrix += [0]
                result_matrix += [empty_matrix]
                del empty_matrix
            for k in range(len(augend)):
                for l in range(len(augend[0])):
                    result_matrix[k][l] = augend[k][l] + addend[k][l]
            return result_matrix
        elif len(augend) == len(addend) and len(addend[0]) == 1:
            result_matrix = []
            for i in range(len(augend)):
                empty_matrix = []
                for j in range(len(augend[0])):
                    empty_matrix += [0]
                result_matrix += [empty_matrix]
                del empty_matrix
            for k in range(len(augend)):
                for l in range(len(augend[0])):
                    result_matrix[k][l] = augend[k][l] + addend[k][0]
            return result_matrix
        elif len(augend[0]) == len(addend[0]) and len(addend) == 1:
            result_matrix = []
            for i in range(len(augend)):
                empty_matrix = []
                for j in range(len(augend[0])):
                    empty_matrix += [0]
                result_matrix += [empty_matrix]
                del empty_matrix
            for k in range(len(augend)):
                for l in range(len(augend[0])):
                    result_matrix[k][l] = augend[k][l] + addend[0][l]
            return result_matrix
    elif isinstance(augend, list) and isinstance(addend, float):
        result_matrix = []
        for i in range(len(augend)):
            empty_matrix = []
            for j in range(len(augend[0])):
                empty_matrix += [0]
            result_matrix += [empty_matrix]
            del empty_matrix
        for k in range(len(augend)):
            for l in range(len(augend[0])):
                result_matrix[k][l] = augend[k][l] + addend
        return result_matrix
    elif isinstance(augend, float) and isinstance(addend, list):
        result_matrix = []
        for i in range(len(addend)):
            empty_matrix = []
            for j in range(len(addend[0])):
                empty_matrix += [0]
            result_matrix += [empty_matrix]
            del empty_matrix
        for k in range(len(addend)):
            for l in range(len(addend[0])):
                result_matrix[k][l] = augend + addend[k][l]
        return result_matrix


def list_to_matrix_2d(input_l):
    if (len(input_l) ** 0.5) == int(len(input_l) ** 0.5):
        matrix_len = int(len(input_l) ** 0.5)
        result_matrix = []
        for i in range(matrix_len):
            empty_matrix = []
            for j in range(matrix_len):
                empty_matrix += [0]
            result_matrix += [empty_matrix]
            del empty_matrix
        for k in range(len(result_matrix)):
            for l in range(len(result_matrix[0])):
                result_matrix[l][k] = input_l[0]
                del input_l[0]
        return result_matrix

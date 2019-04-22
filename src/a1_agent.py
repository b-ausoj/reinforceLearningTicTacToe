from src import neural_network as n
import random
# The agent


class Agent:

    def __init__(self, nodes, learningrate, saving):
        inputnodes, hiddennodes, outputnodes = nodes
        new_weights, saving_place = saving
        self.policy_network = n.NeuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate, new_weights, saving_place)
    # neuronale netz initialisieren

    def query_board(self, board):
        return self.policy_network.query(board)

    def learn(self, inputlist, targetlist):
        self.policy_network.train(inputlist, targetlist)

    def choose_move(self, board, epsilon):
        if random.random() > epsilon:  # schauen, was random.random gibt, weil müsste zahl zwischen 0 und 1 geben
            return self.policy_network.query(board).index(max(self.policy_network.query(board)))
        else:
            return random.randint(0,8)

    def choose_move_passive(self, board):
        q_values = self.policy_network.query(board)
        for i in range(len(q_values)):
            if board[i] != 0:
                q_values[i] = 0
        return q_values.index(max(q_values))

        # gibt als return nur einen möglichen zug

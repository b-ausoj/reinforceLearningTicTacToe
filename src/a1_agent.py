from src import neural_network as n
import random
# The agent


class Agent:

    def __init__(self, nodes, learningrate, saving):
        inputnodes, hiddennodes1, hiddennodes2, hiddennodes3, outputnodes = nodes
        old_or_new, memory_l = saving
        self.policy_network = n.NeuralNetwork(inputnodes, hiddennodes1, hiddennodes2, hiddennodes3, outputnodes, learningrate, old_or_new, memory_l)

    def query_board(self, board):
        return self.policy_network.query(board)

    def learn(self, inputlist, targetlist):
        self.policy_network.train(inputlist, targetlist)

    def choose_move(self, board, epsilon):
        if random.random() > epsilon:
            return self.policy_network.query(board).index(max(self.policy_network.query(board)))
        else:
            return random.randint(0, 8)

    def choose_move_passive(self, board):
        q_values = self.policy_network.query(board)
        for i in range(len(q_values)):
            if board[2][i] == 0.01:
                q_values[i] = -10
        return q_values.index(max(q_values))
        # gibt als return nur einen möglichen zug

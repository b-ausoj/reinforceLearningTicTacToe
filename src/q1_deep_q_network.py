# import the agent and envornment
# import the other things
import random, copy
from src import a1_agent as a, e1_environment as e


def dqn(episodes, epsilons, update_every, nodes, learning_rate, discount_factor, saving1, saving2):

    epsilon_start, epsilon_end, epsilon_decay = epsilons

    agent_1 = a.Agent(nodes, learning_rate, saving1)
    agent_2 = a.Agent(nodes, learning_rate, saving2)  # ich trainiere 2 paralle

    epsilon = epsilon_start

    for episode in range(episodes):
        print(episode)  # zum löschen

        board, turn = e.set_up()
        game_playing = True

        while game_playing:

            if turn == "agent_1":

                move = agent_1.choose_move(board, epsilon)
                board_next, reward, game_playing = e.make_move(move, board, 0)
                # falsch:(die bestrafung gibt es erst beim lernen, wenn der loss ausgerechnet wird)
                # wenn ein falscher move, dann game_playing = false und board_next ebenfals = false
                # beim lernen: zuerst schauen, ob board_next = false ist, dann kein max q_next
                # sieg: reward = 1, falshcer zug: reward = -1, sosnst: reward = 0

                if game_playing:
                    board_next_e, reward, useless = e.make_move(agent_2.choose_move_passive(board_next), board_next, 1)
                else:
                    board_next_e = copy.deepcopy(board_next)

                q_values = agent_1.policy_network.query(board)
                if not board_next_e:
                    q_value_target = reward
                else:
                    q_value_target = reward + discount_factor * max(agent_1.target_network.query(board_next_e))
                q_values[move] = q_value_target
                agent_1.learn(board, q_values)

                board = copy.deepcopy(board_next)
                turn = "agent_2"

            else:

                move = agent_2.choose_move(board, epsilon)
                board_next, reward, game_playing = e.make_move(move, board, 1)
                # falsch:(die bestrafung gibt es erst beim lernen, wenn der loss ausgerechnet wird)
                # wenn ein falscher move, dann game_playing = false und board_next ebenfals = false
                # beim lernen: zuerst schauen, ob board_next = false ist, dann kein max q_next
                # sieg: reward = 1, falshcer zug: reward = -1, sosnst: reward = 0

                if game_playing:
                    board_next_e, reward, useless = e.make_move(agent_1.choose_move_passive(board_next), board_next, 0)
                else:
                    board_next_e = copy.deepcopy(board_next)

                q_values = agent_2.policy_network.query(board)
                if not board_next_e:
                    q_value_target = reward
                else:
                    q_value_target = reward + discount_factor * max(agent_2.target_network.query(board_next_e))
                q_values[move] = q_value_target
                agent_2.learn(board, q_values)

                board = copy.deepcopy(board_next)
                turn = "agent_1"

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        agent_1.update_target_network(episode, update_every), agent_2.update_target_network(episode, update_every)
        if (episode % 999) == 0:
            agent_1.policy_network.save_weights(), agent_2.policy_network.save_weights()


# diese werte können/müssen noch verändert werden
Episodes = 1000
Epsilons = 1.0, 0.01, 0.95  # epsilon: (start, end, decay)
Update_every = 5  # Target Network updating
Nodes = 27, 81, 81, 27, 9  # Anzahl Knoten, momentan 5 Schichten, erste Schicht muss 27 sein, letzte 9
Learning_rate = 0.001
Discount_factor = 0.9
Saving_weights_1 = "old", "../learned_data/nn_weights_agent1.pkl"  # Gewichte für Agent 1
Saving_weights_2 = "old", "../learned_data/nn_weights_agent2.pkl"  # Gewichte für Agent 2

test = a.Agent(Nodes, Learning_rate, Saving_weights_1)
print(test.policy_network.query([[0.99, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01],
                                 [0.01, 0.01, 0.01, 0.99, 0.99, 0.01, 0.99, 0.01, 0.01],
                                 [0.01, 0.99, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.99]]))

#test.policy_network.train([[0.99, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01], [0.01, 0.01, 0.01, 0.99, 0.99, 0.01, 0.99, 0.01, 0.01], [0.01, 0.99, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.99]], [-0.11899158002423849, 0.99, -0.203401494098854, -0.41678819440276427, -0.08159857195115938, -0.08859334844187128, -0.6790027384679047, -0.1428038058139509, -0.1092761852954447])

dqn(Episodes, Epsilons, Update_every, Nodes, Learning_rate, Discount_factor, Saving_weights_1, Saving_weights_2)

test = a.Agent(Nodes, Learning_rate, Saving_weights_1)
print(test.policy_network.query([[0.99, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01],
                                 [0.01, 0.01, 0.01, 0.99, 0.99, 0.01, 0.99, 0.01, 0.01],
                                 [0.01, 0.99, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.99]]))
#test.policy_network.save_weights()

# Leeres Brett darf nicht 0, 0 ... sein, neuronale Netz gibt sonst keinen Output
# Input mit min 18 nodes, also für x 9 und für o 9, vielleicht auch 9 für die leeren felder (leer = 1)
# das batch learning ist noch nicht das richtige batch learning (maches falsch)
# clone netzwerk
# gewichtsänderung muss vielleicht auch überarbeitet werden
# idee:das netz soll gegen einen clone von sich selbst spielen

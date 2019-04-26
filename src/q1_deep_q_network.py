# import the agent and envornment
# import the other things
import random
from src import a1_agent as a, e1_environment as e


def dqn(episodes, epsilons, replay_memory_size, batch_size, update_every, nodes, learning_rate, discount_factor, saving1, saving2):

    epsilon_start, epsilon_end, epsilon_decay = epsilons

    replay_memory_1, replay_memory_2 = [], []  # Replay memory

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

                if not game_playing:
                    replay_memory_1.append([board, move, reward, board_next])

                else:
                    board_next_e, reward, useless = e.make_move(agent_2.choose_move_passive(board_next), board_next, 1)
                    replay_memory_1.append([board, move, reward, board_next_e])

                if len(replay_memory_1) > replay_memory_size:
                    replay_memory_1.pop(0)

                if episode > 100:  # Auch noch etwas, was man ändern könnte, aber glaubs nicht viel ausmacht
                    for i in range(batch_size):
                        e_board, e_move, e_reward, e_board_next = random.choice(replay_memory_1)
                        q_values = agent_1.policy_network.query(e_board)
                        # q_value_output = q_values[e_move]
                        if not e_board_next:
                            q_value_target = e_reward
                        else:
                            q_value_target = e_reward + discount_factor * max(agent_1.target_network.query(e_board_next))
                        # output_error = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                        # output_error[e_move] = q_value_target - q_value_output  # braucht es nicht wegen backprop. im neural_network.py, wenn gradient descent wie im tutorial beschireben, würde ichs brauchen
                        q_values[e_move] = q_value_target
                        agent_1.learn(e_board, q_values)

                # now the learning from the replay memory comes
                    # beim lernen muss das mit dem target network clone gemacht werden

                if board_next != False:  # muss so sein
                    board = list(board_next)
                turn = "agent_2"

            else:

                move = agent_2.choose_move(board, epsilon)
                board_next, reward, game_playing = e.make_move(move, board, 1)
                # falsch:(die bestrafung gibt es erst beim lernen, wenn der loss ausgerechnet wird)
                # wenn ein falscher move, dann game_playing = false und board_next ebenfals = false
                # beim lernen: zuerst schauen, ob board_next = false ist, dann kein max q_next
                # sieg: reward = 1, falshcer zug: reward = -1, sosnst: reward = 0

                if not game_playing:
                    replay_memory_2.append([board, move, reward, board_next])

                else:
                    board_next_e, reward, useless = e.make_move(agent_1.choose_move_passive(board_next), board_next, 0)
                    replay_memory_2.append([board, move, reward, board_next_e])

                if len(replay_memory_2) > replay_memory_size:
                    replay_memory_2.pop(0)

                if episode > 100:
                    for i in range(batch_size):
                        e_board, e_move, e_reward, e_board_next = random.choice(replay_memory_2)
                        q_values = agent_2.policy_network.query(e_board)
                        if not e_board_next:
                            q_value_target = e_reward
                        else:
                            q_value_target = e_reward + discount_factor * max(agent_2.target_network.query(e_board_next))
                        q_values[e_move] = q_value_target
                        agent_2.learn(e_board, q_values)

                if board_next != False:  # muss so sein
                    board = list(board_next)
                turn = "agent_1"

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        agent_1.update_target_network(episode, update_every), agent_2.update_target_network(episode, update_every)
        if (episode % 1000) == 0:
            agent_1.policy_network.save_weights(), agent_2.policy_network.save_weights()


Episodes = 200000
Epsilons = 1.0, 0.01, 0.95  # epsilon: (start, end, decay)
Replay_memory_size = 10000  # Anzahl an Spielbeispielen, mit denen Trainiert wird
Batch_size = 10  # Anzahl Beispiele, die aufs Mal trainiert werden
Update_every = 5  # Target Network updating
Nodes = 27, 81, 81, 27, 9  # Anzahl Knoten, momentan 5 Schichten, erste Schicht muss 27 sein, letzte 9
Learning_rate = 0.01
Discount_factor = 0.9
Saving_weights_1 = "old", "../learned_data/nn_weights_agent1.pkl"  # Gewichte für Agent 1
Saving_weights_2 = "old", "../learned_data/nn_weights_agent2.pkl"  # Gewichte für Agent 2

# diese werte können/müssen noch verändert werden

dqn(Episodes, Epsilons, Replay_memory_size, Batch_size, Update_every, Nodes, Learning_rate, Discount_factor, Saving_weights_1, Saving_weights_2)

# test = a.Agent(Nodes, Learning_rate, Saving_weights_1)
# print(test.policy_network.query([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
#                                  [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
#                                  [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]]))

# Leeres Brett darf nicht 0, 0 ... sein, neuronale Netz gibt sonst keinen Output
# Input mit min 18 nodes, also für x 9 und für o 9, vielleicht auch 9 für die leeren felder (leer = 1)
# das batch learning ist noch nicht das richtige batch learning (maches falsch)
# clone netzwerk
# gewichtsänderung muss vielleicht auch überarbeitet werden
# idee:das netz soll gegen einen clone von sich selbst spielen

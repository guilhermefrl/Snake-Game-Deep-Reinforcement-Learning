import os as operative_system
operative_system.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from snake_game import SnakeGame
import tensorflow as tf
import numpy as np
from tensorflow import keras
from collections import deque
import random
from math import sqrt
import matplotlib.pyplot as plt


# O treino vai ser 900 episódios, onde cada episódio corresponde a um jogo completo
train_episodes = 900

# Criar jogo da Snake
env = SnakeGame(14, 14, border=1)


# Modelo para o agente
def agent(state_shape, action_shape):

    # Taxa de Aprendizagem
    learning_rate = 0.001
    # Usar o inicializador HeUniform, para inicializar os pesos usando uma distribuição uniforme
    init = tf.keras.initializers.HeUniform()

    # Criar um modelo sequencial
    model = keras.Sequential()

    #Modelo
    model.add(keras.layers.Input(shape=(state_shape.shape[0],state_shape.shape[1] ,state_shape.shape[2])))

    # Camada convolucional 2D, com 32 filtros, kernel 2x2 e same padding
    model.add(keras.layers.Conv2D(32, (2, 2), padding="same"))
    # Ativação ReLU
    model.add(keras.layers.Activation("relu"))

    # Camada convolucional 2D, com 64 filtros, kernel 3x3 e same padding
    model.add(keras.layers.Conv2D(64, (2, 2), padding="same"))
    # Ativação ReLU
    model.add(keras.layers.Activation("relu"))

    # Camada convolucional 2D, com 64 filtros, kernel 3x3 e same padding
    model.add(keras.layers.Conv2D(64, (2, 2), padding="same"))
    # Ativação ReLU
    model.add(keras.layers.Activation("relu"))
    # MaxPooling
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Passar input tridimensional para um vetor de uma dimensão
    model.add(keras.layers.Flatten())

    # Camada densa com 64 neurónios e ativação ReLU
    model.add(keras.layers.Dense(64, activation='relu', kernel_initializer=init))

    # Camada densa com 64 neurónios e ativação ReLU
    model.add(keras.layers.Dense(64, activation='relu', kernel_initializer=init))

    # Camada densa com o número de neurónios igual ao número de ações posíveis, e ativação linear
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))

    # Compilar o modelo
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

    return model


# Função para o treino
def train(env, replay_memory, model, target_model, done):

    # Discount Factor
    discount_factor = 0.15
    # Tamanho do batch
    batch_size = 64 * 2

    # Fazer sample aleatóriamente para criar um mini-batch
    mini_batch = random.sample(replay_memory, batch_size)

    # Obter os estados atuais do mini-batch
    current_states = np.array([transition[0] for transition in mini_batch])
    # Prever os Q-values atuais, de acordo com os estados atuais
    current_qs_list = model.predict(current_states)

    # Obter os estados futuros do mini-batch
    new_current_states = np.array([transition[3] for transition in mini_batch])
    # Prever os Q-values futuros, de acordo com os estados futuros
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []

    # Para cada mini-batch, obtém a experiência respetiva
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):

        # Se não for um estado terminal
        if not done:
            # Computar o máximo dos Q-values, apartir dos estados futuros.
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])

        # Caso seja um estado terminal
        else:
            # Definir a recompensa atual como o máximo dos Q-values
            max_future_q = reward
        
        # Atualizar o Q-value, para cada ação
        current_qs = current_qs_list[index]
        current_qs[action] = max_future_q

        X.append(observation)
        Y.append(current_qs)

    # Fazer Fit do modelo
    history = model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

    return history.history['accuracy']


# Heurística - Distância euclidiana
def heuristic(x_position, y_position, x_goal, y_goal):

    # Calcular a distância euclidiana
    result = sqrt(((x_goal-x_position)**2) + ((y_goal-y_position)**2))
    
    return result


# Heurística para escolher a ação
def heuristic_action(env):

    # Informações do estado
    total_score, food_coordinates, head, tail, direction = env.get_state()

    distances = []
    coordinates = []
    count = 2
    invalid_action = 0

    # Ações possíveis (-1: Virar à esquerda; 0: Seguir em frente; 1: Virar à direita)
    ACTIONS = [-1, 0, 1]

    # Calcular as coordenadas do próximo estado, ao aplicar as várias ações
    # Norte ou Sul
    if (direction == 0 or direction == 2):
        # Norte
        aux = 1

        # Sul
        if (direction == 2):
            aux = -1

        # Esquerda
        x1 = head[1] - aux
        y1 = head[0]

        # Continuar em frente
        x2 = head[1]
        y2 = head[0] - aux

        # Direita
        x3 = head[1] + aux
        y3 = head[0]

    # Este ou Oeste
    elif (direction == 1 or direction == 3):

        # Este
        aux = 1

        # Oeste
        if (direction == 3):
            aux = -1

        # Esquerda
        x1 = head[1]
        y1 = head[0] - aux

        # Continuar em frente
        x2 = head[1] + aux
        y2 = head[0]

        # Direita
        x3 = head[1]
        y3 = head[0] + aux

    # Calcular distâncias
    distance1 = heuristic(x1, y1, food_coordinates[0][1], food_coordinates[0][0])
    distances.append(distance1)

    distance2 = heuristic(x2, y2, food_coordinates[0][1], food_coordinates[0][0])
    distances.append(distance2)

    distance3 = heuristic(x3, y3, food_coordinates[0][1], food_coordinates[0][0])
    distances.append(distance3)

    coordinates.append((y1, x1))
    coordinates.append((y2, x2))
    coordinates.append((y3, x3))

    # Verificar se próximo passo é a cauda da cobra
    for index in reversed(coordinates):
        
        # Caso uma ação seja inválida
        if (index in tail):
            invalid_action += 1

            # Remover ação
            if (invalid_action <= 2):
                distances.pop(count)
                ACTIONS.pop(count)
        
            # Caso todas as ações sejam inválidas
            if (invalid_action == 3):
                # Retornar uma ação aleatória
                return random.choice([-1, 0, 1])

        count -=1
            
    # Encontrar a ação com menor distância
    min_distance = min(distances)
    index = distances.index(min_distance)

    # Escolher ação
    action = ACTIONS[index]
    
    return action


# Função para gerar exemplos
def generate_examples(replay_memory):

    replay_memory_size = 55000
    points_with_zero = 0

    while len(replay_memory) < replay_memory_size:
        
        total_training_rewards = 0

        # Fazer reset do ambiente
        observation, reward, done, info = env.reset()

        while not done or len(replay_memory) < replay_memory_size:

            epsilon = np.random.rand()

            if epsilon <= 0.05 :
                action = random.choice([-1,0,1])
            else:
                # Calcular heurística
                action = heuristic_action(env)

            # Fazer a ação escolhida
            new_observation, reward, done, info = env.step(action)

            # Fazer append da última experiência à replay memory
            replay_memory.append([observation, int(action), reward, new_observation, done])

            # Atualizar a observação atual
            observation = new_observation

            # Atualizar o número toral de rewards
            total_training_rewards += reward

            # Se chegarmos ao final de um jogo
            if done:

                break

    return replay_memory


# Main
def main():

    # Definir o valor do epsilon
    epsilon = 0.05

    accuracy = []
    num = []
    score_graph_x = []
    score_graph_y = []
    apples = []
    gamesteps = 0
    num_games = 0

    # Ações possíveis (-1: Virar à esquerda; 0: Seguir em frente; 1: Virar à direita)
    ACTIONS = [-1, 0, 1]

    # Modelo principal para o agente
    model = agent(env.board_state(), 3)

    # Cópia do modelo principal
    target_model = agent(env.board_state(), 3)
    # Copiar os pesos do modelo principal
    target_model.set_weights(model.get_weights())

    # Criar o replay memory e definir o tamanho máximo
    replay_memory = deque(maxlen=55000)

    # Gerar exemplos
    replay_memory = generate_examples(replay_memory)

    steps_to_update_target_model = 0

    # Para cada episódio
    for episode in range(train_episodes):

        total_training_rewards = 0
        num_games += 1

        # Fazer reset do ambiente
        observation, reward, done, info = env.reset()
        # Fazer reset da flag
        done = False

        gamesteps = 0

        # Começar a iterar até que o episódio termine
        while not done:

            steps_to_update_target_model += 1

            gamesteps += 1

            # Gerar um número aleatório
            random_number = np.random.rand()

            # Dependendo do valor de epsilon:

            # Se o número aleatório menor ou igual a epsilon
            if random_number <= epsilon:
                # Escolher uma ação aleatória
                action = random.choice(ACTIONS)
            
            # Caso contrário
            else:
                # Fazer a ação prevista pelo modelo. Ou seja:

                # Fazer reshape das observações
                reshaped = observation.reshape((1, observation.shape[0], observation.shape[1], observation.shape[2]))
                # Prever os Q-values, para cada ação, neste estado
                predicted = model.predict(reshaped).flatten()
                # Escolher a ação que correspode ao máximo dos Q-values
                action = np.argmax(predicted)
                action -= 1

            # Fazer a ação escolhida
            new_observation, reward, done, info = env.step(action)

            # Fazer append da última experiência à replay memory
            replay_memory.append([observation, action, reward, new_observation, done])

            # Só começamos o treino se tivermos pelo menos quatro passos
            if (steps_to_update_target_model % 4 == 0 or done):
                # Treinar o modelo
                acc = train(env, replay_memory, model, target_model, done)

                accuracy.append(acc)
                num.append(episode)

            # Atualizar a observação atual
            observation = new_observation
            # Atualizar o número toral de rewards
            total_training_rewards += reward

            # Se chegarmos ao final de um episódio, ou seja, que o jogo terminou
            if done:
                # Fazer print dos rewards
                print('Rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                # Incrementar o número total de rewards
                total_training_rewards += 1

                # Depois de 4 episódios, atualizar o Target Model
                if steps_to_update_target_model >= 4:
                    print('Copying main network weights to the target network weights')
                    # Copiar os pesos do modelo principal
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0

                score_graph_x.append(episode)
                score_graph_y.append(gamesteps)
                apples.append(total_training_rewards)

                break

    """Plot the graphs"""
    plt.plot(score_graph_x, score_graph_y)
    plt.xlabel('episode')
    plt.ylabel('steps')
    plt.title('Step scores')      
    plt.show()

    plt.plot(num, accuracy)
    plt.title('Agent accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()  
    
    plt.plot(score_graph_x, apples)
    plt.title('Apples score')
    plt.ylabel('apples')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()
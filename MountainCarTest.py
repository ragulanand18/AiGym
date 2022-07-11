import gym
import neat
import pickle, time
# import visuvalize
# from graphviz import Source

import platform
plt = platform.system()


config_path = "C:\\Users\\Ragul\\PycharmProjects\\AI\\MountainCar.txt"
finished_model_path = "C:\\Users\\Ragul\\PycharmProjects\\AI\\MountainCar.pkl"

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

with open(finished_model_path, "rb") as input_file:
    genome = pickle.load(input_file)


def run():
    env = gym.make('MountainCar-v0')
    done = False
    observation = env.reset()
    net = neat.nn.feed_forward.FeedForwardNetwork.create(genome, config)
    while not done:
        env.render()
        time.sleep(0.01)
        action = net.activate(observation)
        observation, reward, _, _ = env.step(action.index(max(action)))

    dot = visuvalize.draw_net(config, genome, False, show_disabled=False)
    env.close()
    s = Source(dot, filename='MountainCarNet', format='png')
    s.view()

run()


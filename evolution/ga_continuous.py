from __future__ import annotations

import copy
import os
import time
from collections import namedtuple
from typing import List, Tuple

import gym
import torch
from gym import wrappers
from matplotlib import pyplot as plt

import torch.nn.functional as F
import numpy as np
from numpy.core.records import ndarray
from torch import Tensor


class Agent:
    def __init__(self,
                 observation_space: int,
                 action_space: int,
                 layers: int,
                 neurons: int,
                 weights: Tensor = None,
                 biases: Tensor = None
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.layers = layers
        self.neurons = neurons
        self.structure = [(observation_space, neurons)] + (layers - 1) * [(neurons, neurons)] + [
            (neurons, action_space)]
        self.total_params = sum([a * b for a, b in self.structure])
        if weights is None:
            weights = torch.randn(size=(1, self.total_params)).flatten()
        if biases is None:
            biases = torch.randn(size=(1, layers + 1)).flatten()
        self.weights = weights
        self.biases = biases

    def __call__(self, x: Tensor):
        prev_pointer = 0
        for i, (first, last) in enumerate(self.structure):
            next_pointer = prev_pointer + first * last
            w = self.weights[prev_pointer:next_pointer].reshape((last, first))
            b = self.biases[i]
            x = F.linear(x, weight=w, bias=b)
            if i < len(self.structure) - 1:
                x = torch.relu(x)
            prev_pointer = next_pointer
        return torch.tanh(x)


class GA:
    def __init__(self,
                 population_count: int,
                 observation_space: int,
                 action_space: int,
                 num_of_layers: int,
                 num_of_neurons: int
    ):
        self.population_count = population_count
        self.population = []
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_of_layers = num_of_layers
        self.num_of_neurons = num_of_neurons

    def create_population(self, count: int) -> List[Agent]:
        return [Agent(self.observation_space, self.action_space, self.num_of_layers, self.num_of_neurons)
                for _ in range(count)]

    def perform_recombination(self, evaluation_scores: EvaluationScores, selection_count: float = 0.5) -> List[Agent]:
        new_population = []
        pop_portion = int(self.population_count * selection_count)
        chosen_population = evaluation_scores.perform_selection(pop_portion)
        for i in range(0, len(chosen_population), 2):
            first_agent = chosen_population[i].agent
            second_agent = chosen_population[i + 1].agent
            first_agent_weights_new = []
            second_agent_weights_new = []
            prev_pointer = 0
            for j, (first, last) in enumerate(first_agent.structure):
                next_pointer = prev_pointer + first * last
                crossover_index = np.random.randint(prev_pointer + 1, next_pointer - 1)
                first_agent_weights_new.extend(self._mutate(first_agent.weights[prev_pointer:crossover_index]))
                first_agent_weights_new.extend(self._mutate(second_agent.weights[crossover_index:next_pointer]))
                second_agent_weights_new.extend(self._mutate(second_agent.weights[prev_pointer:crossover_index]))
                second_agent_weights_new.extend(self._mutate(first_agent.weights[crossover_index:next_pointer]))
                prev_pointer = next_pointer
            first_agent_weights_new = torch.tensor(first_agent_weights_new)
            second_agent_weights_new = torch.tensor(second_agent_weights_new)
            first_agent_biases_new, second_agent_biases_new = \
                self._mix_biases(first_agent.biases, second_agent.biases)

            new_population.extend([
                Agent(self.observation_space, self.action_space, self.num_of_layers, self.num_of_neurons,
                      weights=first_agent_weights_new,
                      biases=first_agent_biases_new),
                Agent(self.observation_space, self.action_space, self.num_of_layers, self.num_of_neurons,
                      weights=second_agent_weights_new,
                      biases=first_agent_biases_new)
            ])

        # create the missing population part
        new_population.extend(self.create_population(self.population_count - len(new_population)))

        return new_population

    def _mix_biases(self, first: Tensor, second: Tensor) -> Tuple[Tensor, Tensor]:
        for i in range(len(first)):
            if np.random.random() < 0.01:
                first[i], second[i] = second[i], first[i]
        return self._mutate(first), self._mutate(second)

    @staticmethod
    def _mutate(weights: Tensor) -> Tensor:
        for i in range(len(weights)):
            if np.random.random() < 0.01:
                weights[i] += torch.randn(1).item()
        return weights


class EvaluationScores:
    EvaluationScore = namedtuple('EvaluationScore', ['agent', 'score'])

    def __init__(self):
        self.items = []

    def insert_score(self, agent: Agent, score: int):
        self.items.append(self.EvaluationScore(agent, score))

    def perform_selection(self, selection_num: int) -> List[EvaluationScore]:
        scores = np.asarray([es.score for es in self.items]) + np.random.normal(0, 1e-10, len(self.items))
        probabilities = self._create_prob_map(scores)
        chosen_indexes = np.random.choice(list(range(len(self.items))), size=selection_num, p=probabilities,
                                          replace=False)
        chosen_population = [item for i, item in enumerate(self.items) if i in chosen_indexes]
        return sorted(chosen_population, key=lambda item: item.score)

    def get_best(self):
        return max(self.items, key=lambda item: item.score)

    @staticmethod
    def _create_prob_map(scores: ndarray) -> ndarray:
        scores_copy = scores[:]
        # min max normalization because of +- values
        scores[:] = (scores_copy - scores_copy.min()) / (scores_copy.max() - scores_copy.min() + 1e-10)
        prob_maps = np.zeros(len(scores))
        total = sum(scores)
        for i, prob in enumerate(prob_maps):
            prob_maps[i] = scores[i] / total
        return prob_maps


class Evaluation:
    def __init__(self, environment):
        self.environment = environment

    def evaluate_population(self, population: List[Agent]) -> EvaluationScores:
        es = EvaluationScores()
        for i in range(len(population)):
            score = self.evaluate_agent(population[i])
            es.insert_score(population[i], score)
        return es

    def evaluate_agent(self, agent: Agent) -> int:
        state = self.environment.reset()
        rewards_sum = 0

        while True:
            state = torch.from_numpy(state).flatten()
            action = agent(state)
            state, reward, episode_done, _ = self.environment.step((action.item(),))
            rewards_sum += reward

            if episode_done:
                return rewards_sum


if __name__ == '__main__':
    env_name = 'MountainCarContinuous-v0'
    population_count = 100
    epochs = 200
    env = gym.make(env_name)
    video_path = os.path.join(os.getcwd(), 'videos', f'{env_name}_{time.strftime("%d-%m-%Y_%H-%M-%S")}')
    env = wrappers.Monitor(env, video_path, resume=True, video_callable=lambda count: count % 50 == 0)

    action_space = 1
    observation_space = env.observation_space.shape[0]
    num_of_neurons = 256
    num_of_layers = 2

    ga = GA(population_count, observation_space, action_space, num_of_layers, num_of_neurons)
    population = ga.create_population(population_count)
    evaluation = Evaluation(env)
    top_rewards = []
    best_agent = None
    best_score = float('-inf')
    for i in range(epochs):
        if i % 20 == 0:
            print('epoch', i)

        evaluation_scores = evaluation.evaluate_population(population)
        best_agent_score = evaluation_scores.get_best()
        top_rewards.append(best_agent_score.score)
        if best_agent_score.score > best_score:
            best_agent = copy.deepcopy(best_agent_score.agent)
            best_score = best_agent_score.score
            print('New best score:', best_score)
        population = ga.perform_recombination(evaluation_scores)

    plt.cla()
    plt.plot(top_rewards)
    plt.show()

    # evaluation of the best agent
    best_agent_reward = []
    for _ in range(100):
        episode_reward = evaluation.evaluate_agent(agent=best_agent)
        best_agent_reward.append(episode_reward)

    plt.cla()
    plt.plot(best_agent_reward)
    plt.show()

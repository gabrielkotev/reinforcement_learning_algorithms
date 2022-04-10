from __future__ import annotations

import copy
import json
import threading
import time
from json import JSONDecodeError
from typing import Tuple

import gym
import numpy as np
from gym import wrappers
from numpy.core.records import ndarray


ENV_NAME = 'BipedalWalker-v3'
AGENT_SAVE_PATH = f'c:\\ars\\{ENV_NAME}.json'


class Agent:
    def __init__(
            self,
            p: int,
            n: int,
            *,
            look_back_num: int = 4,
            step_size: float = 0.02,
            exploration_noise: float = 0.1,
            num_of_directions: int = 16,
            num_of_top_perform: float = 8
    ):
        self.look_back_num = look_back_num
        self.num_of_params = n * look_back_num
        self.weights = np.zeros((p, self.num_of_params))
        self.mean = np.zeros(self.num_of_params)
        self.mean_diff = np.zeros(self.num_of_params)
        self.var = np.zeros(self.num_of_params)
        self.deltas = np.zeros((p, self.num_of_params))
        self.step_size = step_size
        self.exploration_noise = exploration_noise
        self.num_of_directions = num_of_directions
        self.num_of_top_perform = num_of_top_perform
        self.counter = 0

    def get_action(self, state_input: ndarray, exploration: bool = True) -> ndarray:
        input = self._normalize(state_input)
        if exploration:
            return (self.weights + self.exploration_noise * self.deltas).dot(input)
        else:
            return self.weights.dot(input)

    def clone_and_modify(self, *, deltas: ndarray) -> Agent:
        new_agent = copy.deepcopy(self)
        new_agent.deltas = deltas
        return new_agent

    def create_rollouts(self) -> Tuple[Tuple[Agent]]:
        rollouts = []
        for _ in range(self.num_of_directions):
            rnd = np.random.randn(*self.weights.shape)
            rollouts.append((self.clone_and_modify(deltas=rnd), self.clone_and_modify(deltas=-rnd)))
        return tuple(rollouts)

    def update_weights(self, scores: ndarray, agents: Tuple[Tuple[Agent]]) -> None:
        total_scores = np.asarray(tuple(max(x[0], x[1]) for x in scores))
        best_in_iter = np.argsort(total_scores)[-self.num_of_top_perform:]
        top_rewards = np.asarray([s for k, s in enumerate(scores) if k in best_in_iter])
        top_deltas = np.asarray([s[0].deltas for k, s in enumerate(agents) if k in best_in_iter])

        std_rewards = scores.flatten().std()
        d_sum = np.sum(tuple((rewards_pair[0] - rewards_pair[1]) * pop_weights for rewards_pair, pop_weights in
                             zip(top_rewards, top_deltas)), axis=0)
        self.weights += self.step_size * d_sum / (self.num_of_top_perform * std_rewards)

    def _normalize(self, state_input: ndarray) -> float:
        self.counter += 1
        last_mean = self.mean.copy()
        self.mean += (state_input - self.mean) / self.counter
        self.mean_diff += (state_input - last_mean) * (state_input - self.mean)
        self.var = self.mean_diff / self.counter
        return (state_input - self.mean) / np.sqrt(self.var + 1e-5)

    def save_to_file(self) -> None:

        _obj_to_save = {key:(value.tolist() if type(value) == ndarray else value)
                        for key, value in self.__dict__.items()}

        with open(AGENT_SAVE_PATH, 'w') as outfile:
            outfile.write(json.dumps(_obj_to_save))

    @classmethod
    def instance(cls, **kwargs):
        properties = dict()
        try:
            with open(AGENT_SAVE_PATH, 'r') as input_file:
                properties = json.load(input_file)
        except (FileNotFoundError, JSONDecodeError):
            pass

        inst = cls(**kwargs)
        for key in properties:
            value = properties[key]
            value = np.asarray(value) if type(value) == list else value
            setattr(inst, key, value)

        return inst


class Trainer:
    def __init__(self, env_name: str, num_of_directions: int = 16, **kwargs):
        self.num_of_directions = num_of_directions
        self.environments = tuple((gym.make(env_name), gym.make(env_name)) for _ in range(num_of_directions))
        self.agent = Agent.instance(
            p=self.environments[0][0].action_space.shape[0],
            n=self.environments[0][0].observation_space.shape[0],
            **kwargs
        )
        # Used for saving the scores in parallel execution
        self.lock = threading.Lock()

    def train(self) -> float:
        rollouts = self.agent.create_rollouts()
        epoch_scores = self._parallel_rollouts_evaluation(rollouts)
        self.agent.update_weights(epoch_scores, rollouts)
        return self._evaluate_agent()

    def _parallel_rollouts_evaluation(self, rollouts: Tuple[Tuple[Agent]]) -> ndarray:
        workers = []
        epoch_scores = np.zeros((self.num_of_directions, 2))
        for i, rollouts_tuple in enumerate(rollouts):
            for j, rollout in enumerate(rollouts_tuple):
                t = threading.Thread(target=self._evaluate_rollout, args=(i, j, rollout, epoch_scores))
                workers.append(t)
                t.start()

        for t in workers:
            t.join()

        return epoch_scores

    def _evaluate_rollout(self, i: int, j: int, rollout: Agent, epoch_scores: ndarray):
        env = self.environments[i][j]
        state = env.reset()
        score = 0
        state = np.concatenate([state] * rollout.look_back_num)
        for step in range(2000):
            action = rollout.get_action(state)
            _state, reward, done, _ = env.step(action)
            state = np.concatenate([state, _state])[len(_state):]
            if reward < 0:
                reward = max(reward, -1)
            else:
                # normalization between 0 and 2 -> (x - min) / (max - min)
                reward = reward / 2
            # reward = min(max(reward, -1), 1)
            score += reward
            if done:
                break
        self.lock.acquire()
        epoch_scores[i][j] = score
        self.lock.release()

    def _evaluate_agent(self) -> float:
        env = self.environments[0][0]
        state = env.reset()
        state = np.concatenate([state] * self.agent.look_back_num)
        rewards = 0
        for step in range(2000):
            action = self.agent.get_action(state, exploration=False)
            _state, reward, done, _ = env.step(action)
            rewards += reward
            state = np.concatenate([state, _state])[len(_state):]
            if done:
                break
        return rewards


if __name__ == '__main__':
    epochs = 20000
    video_path = f'c:/videos/{ENV_NAME}_{time.strftime("%d-%m-%Y_%H-%M-%S")}'

    env = gym.make(ENV_NAME)
    env = wrappers.Monitor(env, video_path, force=True, video_callable=lambda episode: True)
    trainer = Trainer(ENV_NAME)

    top_agent_score = -np.inf
    for i in range(epochs):
        agent_evaluation = trainer.train()
        print('epoch:', i, 'score:', agent_evaluation)
        if agent_evaluation > top_agent_score:
            print('best score reached:', agent_evaluation)
            trainer.agent.save_to_file()
            top_agent_score = agent_evaluation
            agent = trainer.agent
            state = env.reset()
            score = 0
            state = np.concatenate([state] * agent.look_back_num)
            for step in range(2000):
                action = agent.get_action(state, exploration=False)
                _state, reward, done, _ = env.step(action)
                state = np.concatenate([state, _state])[len(_state):]
                score += reward
                if done:
                    break
            print('score from simulation:', score)

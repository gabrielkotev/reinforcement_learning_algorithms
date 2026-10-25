import logging
import multiprocessing
import os
import pathlib
import subprocess
import time
from datetime import datetime
from json import JSONDecodeError
from multiprocessing import Process

import mss
import ray

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import json
import copy
import random
from collections import namedtuple, deque
from typing import List, Tuple

import numpy as np
from numpy import ndarray


# Create a logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = logging.FileHandler('app.log')
console_handler = logging.StreamHandler()

# Set level for handlers
file_handler.setLevel(logging.DEBUG)
console_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

ENV_NAME = 'hexapod4l'
AGENT_SAVE_DIR = ENV_NAME + '_walk/'
AGENT_SAVE_PATH = AGENT_SAVE_DIR + 'hexapod_ars.json'
SCORES_SAVE_PATH = AGENT_SAVE_DIR + 'scores.txt'
os.makedirs(AGENT_SAVE_DIR, exist_ok=True)

NUM_OF_DIRECTION = 32
NUM_OF_TOP_PERFORM = 8
ENV_MAX_STEPS = 100

class Agent:
    def __init__(
            self,
            p: int,
            n: int,
            *,
            look_back_num: int = 3,
            step_size: float = 0.03,
            exploration_noise: float = 0.5,
            num_of_directions: int = NUM_OF_DIRECTION,
            num_of_top_perform: float = NUM_OF_TOP_PERFORM
    ):
        self.look_back_num = look_back_num
        self.num_of_params = n * look_back_num
        self.weights = np.zeros((p, self.num_of_params))
        self.mean = np.zeros(self.num_of_params)
        self.m2 = np.zeros(self.num_of_params)
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

    def clone_and_modify(self, *, deltas: ndarray):
        new_agent = copy.deepcopy(self)
        new_agent.deltas = deltas
        return new_agent

    def create_rollouts(self):
        rollouts = []
        for _ in range(self.num_of_directions):
            rnd = np.random.randn(*self.weights.shape)
            rollouts.append((self.clone_and_modify(deltas=rnd), self.clone_and_modify(deltas=-rnd)))
        return tuple(rollouts)

    def update_weights(self, scores: ndarray, rollouts) -> None:
        total_scores = np.asarray(tuple(max(x[0], x[1]) for x in scores))
        best_in_iter = np.argsort(total_scores)[-self.num_of_top_perform:]
        top_rewards = np.asarray([s for k, s in enumerate(scores) if k in best_in_iter])
        top_deltas = np.asarray([s[0].deltas for k, s in enumerate(rollouts) if k in best_in_iter])

        std_rewards = top_rewards.flatten().std()
        d_sum = np.sum(tuple((rewards_pair[0] - rewards_pair[1]) * pop_weights for rewards_pair, pop_weights in
                             zip(top_rewards, top_deltas)), axis=0)
        self.weights += self.step_size * d_sum / (self.num_of_top_perform * std_rewards)

    def _normalize(self, state_input: ndarray) -> float:
        """
        Calculating the variance online by using Welford's online algorithm
        and normalizing according to the paper
        """
        self.counter += 1
        delta = state_input - self.mean
        self.mean += delta / self.counter
        delta2 = state_input - self.mean
        self.m2 += delta * delta2
        var = self.m2 / self.counter
        return (state_input - self.mean) / np.sqrt(var + 1e-10)

    def save_to_file(self, save_path = None) -> None:

        if not save_path:
            save_path = AGENT_SAVE_PATH

        _obj_to_save = {key: (value.tolist() if type(value) == ndarray else value)
                        for key, value in self.__dict__.items()}

        with open(save_path, 'w') as outfile:
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
    RolloutIndex = namedtuple('RolloutIndex', ['id', 'rollout'])

    def __init__(self, env_pool, num_of_directions: int = NUM_OF_DIRECTION, **kwargs):
        self.num_of_directions = num_of_directions
        self.env_pool = env_pool
        self.agent = Agent.instance(
            p=self.env_pool.action_space,
            n=self.env_pool.observation_space,
            **kwargs
        )

    def train(self, recording_file: str = None) -> float:
        rollouts = self.agent.create_rollouts()
        start_time = datetime.now()
        epoch_scores = self._parallel_rollouts_evaluation(rollouts)
        print(f'Execution time {datetime.now() -start_time} secs')
        self.agent.update_weights(epoch_scores, rollouts)
        return self._evaluate_agent(recording_file)

    def _parallel_rollouts_evaluation(self, rollouts: Tuple[Tuple[Agent]]) -> ndarray:
        epoch_scores = [None] * (self.num_of_directions * 2)
        agents = []
        _id = 0
        for rollouts_tuple in rollouts:
            agents.append(self.RolloutIndex(_id, rollouts_tuple[0]))
            agents.append(self.RolloutIndex(_id+1, rollouts_tuple[1]))
            _id += 2
        envs_to_work = self.env_pool.envs.copy()
        working_tasks = []
        agent_index = 0
        while True:
            ready_ids, _ = ray.wait([x[1] for x in working_tasks])
            for i, (_score, _agent_index, index_in_pool) in enumerate(ray.get(ready_ids)):
                epoch_scores[_agent_index] = _score
                working_tasks = [x for x in working_tasks if x[0] != _agent_index]
                envs_to_work.append(self.env_pool.envs[index_in_pool])
                del ready_ids[i]

            if agent_index >= len(agents) or None not in epoch_scores:
                if working_tasks:
                    continue
                break

            if len(envs_to_work) == 0:
                continue

            available_envs = len(envs_to_work)
            working_tasks.extend([(_agent.id, _env.evaluate_rollout.remote(_agent)) for _env, _agent in
                                  zip(envs_to_work, agents[agent_index: agent_index + available_envs])])
            agent_index += available_envs
            envs_to_work = []

        return np.asarray(epoch_scores).reshape((self.num_of_directions, 2))

    def _evaluate_agent(self, recording_file: str = None) -> float:
        return self.env_pool.verification_env.evaluate_rollout(self.agent, recording_file)


class Env:
    Leg = namedtuple('Leg', 'up mid low')
    num_of_legs = 4
    joints_per_leg = 2
    num_of_joints = num_of_legs * joints_per_leg
    # this is going to hold an information about the pair of the legs
    # that are going to execute with the same action
    # this can and should be modified inorder to check different combinations
    # We are going to get the pair index plus the joint values
    action_space = num_of_joints
    # we are going to get orientation and position and the first joint of each pair ( as they are the same )
    observation_space = 5 + num_of_joints

    def __init__(self, scene_name, index_in_pool, port, max_steps=ENV_MAX_STEPS):
        self.max_steps = max_steps
        self.index_in_pool = index_in_pool
        self.client = RemoteAPIClient(port=port)
        self.sim = self.client.require('sim')
        scene_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'scenes', scene_name + '.ttt')
        self.sim.loadScene(scene_path)

        self.top_handler = self.sim.getObject('/top')
        self.body = self.sim.getObject('/body')
        self.dest_handler = self.sim.getObject('/destination')
        self.floor_handler = self.sim.getObject('/Floor')
        self.handlers = []

        for i in range(self.num_of_legs):
            up = self.sim.getObject(f'/joint1_[{i}]')
            mid = self.sim.getObject(f'/joint1_[{i}]/joint2_')
            low = self.sim.getObject(f'/joint1_[{i}]/joint3_')
            self.handlers.append(self.Leg(up, mid, low))

        # setting the maximum velocity of the joints to 60 deg/s
        target_velocity = 1047
        for i in range(self.num_of_legs):
            self.sim.setJointTargetVelocity(self.handlers[i].up, target_velocity)
            self.sim.setJointTargetVelocity(self.handlers[i].mid, target_velocity)
            self.sim.setJointTargetVelocity(self.handlers[i].low, target_velocity)

        self.sim.setStepping(True)
        self.step_number = 0

    def step(self, action):
        pos_before = np.array(self.sim.getObjectPosition(self.top_handler)[:2]) # we want to take the first two coordinates
        dest_pos = np.array(self.sim.getObjectPosition(self.dest_handler)[:2])  # we want to take the first two coordinates
        before_len = np.sqrt((dest_pos[0] - pos_before[0]) ** 2 + (dest_pos[1] - pos_before[1]) ** 2)

        for i in range(self.num_of_legs):
            h_index = i * self.joints_per_leg
            _action = action[h_index:h_index + self.joints_per_leg]
            self.sim.setJointTargetPosition(self.handlers[i].up, np.clip(_action[0].item(), -0.5, 0.5))
            self.sim.setJointTargetPosition(self.handlers[i].mid, np.clip(_action[1].item(), -1, 0.5))
        self.sim.step()

        self.step_number += 1

        # we want to take the first two coordinates
        pos_after = np.array(self.sim.getObjectPosition(self.top_handler)[:2])
        after_len = np.sqrt((dest_pos[0] - pos_after[0]) ** 2 + (dest_pos[1] - pos_after[1]) ** 2)

        reward = before_len - after_len

        # more precise reward system  but working less good

        # move_vec = pos_after - pos_before
        # dest_vec = dest_pos - pos_before

        # if np.linalg.norm(move_vec) == 0 or np.linalg.norm(dest_vec) == 0:
        #     reward = 0.0
        # else:
        #     cos_theta = np.dot(move_vec, dest_vec) / (np.linalg.norm(move_vec) * np.linalg.norm(dest_vec))
        #     reward = (cos_theta + 1) * (before_len - after_len) / 2

        done = False
        if self.step_number >= self.max_steps or after_len <= 0.5:
            self.sim.stopSimulation()
            done = True

        if after_len <= 0.5:
            reward += self.max_steps - self.step_number

        return self.state(), reward, done

    def state(self):
        state = self.sim.getObjectPosition(self.body)[0:2] + self.sim.getObjectOrientation(self.body)
        for h in self.handlers:
            state.append(self.sim.getJointTargetPosition(h.up))
            state.append(self.sim.getJointTargetPosition(h.mid))
        return np.asarray(state)

    def reset(self):
        self.step_number = 0
        self.sim.stopSimulation()
        self.sim.startSimulation()
        for h in self.handlers:
            self.sim.setJointTargetPosition(h.up, 0)
            self.sim.setJointTargetPosition(h.mid, -0.5)
            self.sim.setJointTargetPosition(h.low, 2)
        self.sim.step()
        return self.state()

    def close(self):
        self.sim.stopSimulation()


@ray.remote
class RemoteEnv(Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_rollout(self, rollout_info) -> tuple[int, int, int]:
        rollout_index, rollout = rollout_info
        rollout = copy.deepcopy(rollout)
        state = self.reset()
        score = 0
        state = np.concatenate([state] * rollout.look_back_num)
        _mean = 0
        n = 0
        while True:
            n += 1
            action = rollout.get_action(state)
            _state, reward, done = self.step(action)
            state = np.concatenate([state, _state])[len(_state):]
            if reward < 0:
                reward = max(reward, -1)
            else:
                # normalization between 0 and 2 -> (x - min) / (max - min)
                reward = reward / 2
            score += reward
            _mean += (reward - _mean) / n
            if done or (_mean < 0 and n > 20) :
                return score, rollout_index, self.index_in_pool


class VerificationEnv(Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        monitor_index = 1  # Usually 1 is the primary monitor

        # Initialize screen capture
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_index]
        self.width = self.monitor["width"]
        self.height = self.monitor["height"]

        x = 550
        y = 300
        self.offset = f"+{x},{y}"
        self.geometry = f"{self.width - x}x{self.height - y - 100}"
        self.display = ":10.0"  # You should place your display here

    def _start_record(self, output_file):
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-video_size", self.geometry,
            "-framerate", "60",
            "-f", "x11grab",
            "-i", f"{self.display}{self.offset}",  # Use the correct display here
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "ultrafast",
            output_file
        ]
        self.process = subprocess.Popen(ffmpeg_cmd)

    def _complete_record(self):
        self.process.terminate()
        self.process.wait()

    def evaluate_rollout(self, agent, record_output_file: str = None):
        if record_output_file:
            self._start_record(record_output_file)
        state = self.reset()
        state = np.concatenate([state] * agent.look_back_num)
        rewards = 0
        for step in range(2000):
            action = agent.get_action(state, exploration=False)
            _state, reward, done = self.step(action)
            rewards += reward
            state = np.concatenate([state, _state])[len(_state):]
            if done:
                break
        if record_output_file:
            self._complete_record()
        return rewards



class EnvironmentPool:

    def __init__(self, num_of_envs, add_verification=True, verification_visual=True):
        if num_of_envs >= 100:
            raise ValueError('Cannot be 100 and more')
        self.num_of_envs = num_of_envs
        self.base_rpc_port = 23000
        self.rpc_ports = list(range(self.base_rpc_port, self.base_rpc_port + self.num_of_envs)) \
            if self.num_of_envs else [self.base_rpc_port]
        self.base_gws_port = 23200
        self.gws_ports = list(range(self.base_gws_port, self.base_gws_port + self.num_of_envs)) \
            if self.num_of_envs else [self.base_gws_port]

        self.add_verification = add_verification
        self.verification_visual = verification_visual

        self.envs = []
        self.verification_env = None

        self.num_of_joints = Env.num_of_joints
        self.action_space = Env.action_space
        self.observation_space = Env.observation_space

    def create(self):
        def run_subprocess(rpc_port, gws_port, is_verification, verification_visual):
            args = [
                '<coppeliasim_path>/coppeliaSim.sh',
                f'-GzmqRemoteApi.rpcPort={rpc_port}', f'-GwsRemoteApi.port={gws_port}']
            if not is_verification or (is_verification and not verification_visual):
                args.append('-h')
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = process.communicate()
            if stderr:
                logger.error(stderr)

        for rpc_port, gws_port in zip(self.rpc_ports, self.gws_ports):
            p = Process(target=run_subprocess, args=(rpc_port, gws_port, False, False))
            p.start()

        id = 0
        for rpc_port, gws_port in zip(self.rpc_ports, self.gws_ports):
            self.envs.append(RemoteEnv.remote(ENV_NAME, id, rpc_port))
            id += 1

        if not self.add_verification:
            return

        p = Process(target=run_subprocess, args=(self.rpc_ports[-1] + 1, self.gws_ports[-1] + 1, self.add_verification, self.verification_visual))
        p.start()
        self.verification_env = VerificationEnv(ENV_NAME, id + 1, self.rpc_ports[-1] + 1)

    def terminate(self):
        ray.get([env.close.remote() for env in self.envs])
        if self.verification_env:
            self.verification_env.close()
        subprocess.run(['killall', 'coppeliaSim'], capture_output=True, text=True)


if __name__ == '__main__':
    ray.init()
    random.seed(0)
    np.random.seed(0)

    num_cpus = multiprocessing.cpu_count()

    envs = EnvironmentPool(num_cpus, add_verification=True, verification_visual=False)
    envs.create()

    action_space = envs.action_space
    observation_space = envs.observation_space
    epochs = 2000

    trainer = Trainer(envs)

    try:
        top_agent_score = -np.inf
        for i in range(epochs):
            agent_evaluation = trainer.train()
            print('epoch:', i, 'score:', agent_evaluation)
            with open(SCORES_SAVE_PATH, 'a') as file:
                file.write(f'{agent_evaluation} ')
            if agent_evaluation > top_agent_score:
                print('best score reached:', agent_evaluation)
                trainer.agent.save_to_file(AGENT_SAVE_DIR + f'best_agent_{i}.json')
                top_agent_score = agent_evaluation
            if i % 100 == 0:
                trainer.agent.save_to_file(AGENT_SAVE_DIR + f'epoch_{i}.json')
            trainer.agent.save_to_file()
    finally:
        envs.terminate()


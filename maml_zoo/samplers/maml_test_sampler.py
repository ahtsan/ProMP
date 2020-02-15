from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.vectorized_env_executor import MAMLParallelEnvExecutor, MAMLIterativeEnvExecutor
from maml_zoo.logger import logger
from maml_zoo.utils import utils
from collections import OrderedDict

from pyprind import ProgBar
import numpy as np
import time
import itertools


class MAMLTestSampler(MAMLSampler):
    """Sampler for Meta-RL Meta-testing"""

    def obtain_samples(self, rollout_per_task, for_adapt, initial_hidden=None):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger

        Returns: 
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """

        # initial setup / preparation
        test_batch_size = self.meta_batch_size
        paths = []

        n_samples = 0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]

        total_samples = test_batch_size * rollout_per_task * self.max_path_length
        pbar = ProgBar(total_samples)

        policy = self.policy
        policy.reset(dones=[True] * test_batch_size)

        if not for_adapt:
            # if for testing, we want to always start with the hiddens after adaptation
            assert initial_hidden is not None
            policy._hidden_state = initial_hidden

        # initial reset of envs
        obses = self.vec_env.reset()
        
        while n_samples < total_samples:
            
            # execute policy
            obs_per_task = np.split(np.asarray(obses), test_batch_size)
            actions, agent_infos = policy.get_actions(obs_per_task)

            # step environments
            actions = np.concatenate(actions) # stack meta batch
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)

            #  stack agent_infos and if no infos were provided (--> None) create empty dicts
            agent_infos, env_infos = self._handle_info_dicts(agent_infos, env_infos)

            new_samples = 0
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                # append new samples to running paths
                if isinstance(reward, np.ndarray):
                    reward = reward[0]
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["dones"].append(done)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)

                # if running path is done, add it to paths and empty the running path
                if done:
                    paths.append(dict(
                        observations=np.asarray(running_paths[idx]["observations"]),
                        actions=np.asarray(running_paths[idx]["actions"]),
                        rewards=np.asarray(running_paths[idx]["rewards"]),
                        dones=np.asarray(running_paths[idx]["dones"]),
                        env_infos=utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    new_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = _get_empty_running_paths_dict()

            pbar.update(new_samples)
            n_samples += new_samples
            obses = next_obses
        pbar.stop()

        return paths


def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], rewards=[], dones=[], env_infos=[], agent_infos=[])

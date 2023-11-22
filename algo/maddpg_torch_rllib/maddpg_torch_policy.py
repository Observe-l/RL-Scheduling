import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from typing import List, Dict, Union, Optional

from copy import deepcopy
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import adjust_nstep
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.annotations import override
from .maddpg_torch_model import MADDPGConfig

torch, nn = try_import_torch()

class MADDPGPostprocessing:
    """Implements agentwise termination signal and n-step learning."""

    @override(Policy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # FIXME: Get done from info is required since agentwise done is not
        #  supported now.
        sample_batch[SampleBatch.TERMINATEDS] = self.get_done_from_info(
            sample_batch[SampleBatch.INFOS]
        )

        # N-step Q adjustments
        if self.config["n_step"] > 1:
            adjust_nstep(self.config["n_step"], self.config["gamma"], sample_batch)

        return sample_batch


class MADDPGTorchPolicy(MADDPGPostprocessing, TorchPolicyV2):
    def __init__(self, obs_space, act_space, config):
        # _____ Initial Configuration
        self.config = config

        # FIXME: Get done from info is required since agentwise done is not
        #  supported now.
        self.get_done_from_info = np.vectorize(lambda info: info.get("done", False))

        agent_id = config["agent_id"]
        if agent_id is None:
            raise ValueError("Must set `agent_id` in the policy config.")
        if type(agent_id) is not int:
            raise ValueError("Agent ids must be integers for MADDPG.")

        # _____ Environment Setting
        def _make_continuous_space(space):
            if isinstance(space, Box):
                return space
            elif isinstance(space, Discrete):
                return Box(low=np.zeros((space.n,)), high=np.ones((space.n,)))
            else:
                raise UnsupportedSpaceException(
                    "Space {} is not supported.".format(space)
                )
        
        # Get observation dictionary and 

        # Implement MADDPG using pytorch
        self.actors = [_Actor(dim_obs, dim_act) for i in range(n_agents)]
        self.critics = [_Critic(n_agents, dim_obs,
                               dim_act) for i in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0
    
        class _Critic(nn.Module):
            def __init__(self, dim_observation, dim_action):
                super(_Critic, self).__init__()
                self.dim_observation = dim_observation
                self.dim_action = dim_action
                obs_dim = dim_observation
                act_dim = dim_action

                self.FC1 = nn.Linear(obs_dim, 64)
                self.FC2 = nn.Linear(1024+act_dim, 512)
                self.FC3 = nn.Linear(512, 300)
                self.FC4 = nn.Linear(300, 1)

            # obs: batch_size * obs_dim
            def forward(self, obs, acts):
                result = nn.functional.relu(self.FC1(obs))
                combined = torch.cat([result, acts], 1)
                result = nn.functional.relu(self.FC2(combined))
                return self.FC4(nn.functional.relu(self.FC3(result)))


        class _Actor(nn.Module):
            def __init__(self, dim_observation, dim_action):
                super(_Actor, self).__init__()
                self.FC1 = nn.Linear(dim_observation, 500)
                self.FC2 = nn.Linear(500, 128)
                self.FC3 = nn.Linear(128, dim_action)

            # action output between -2 and 2
            def forward(self, obs):
                result = nn.functional.relu(self.FC1(obs))
                result = nn.functional.relu(self.FC2(result))
                result = nn.functional.tanh(self.FC3(result))
                return result
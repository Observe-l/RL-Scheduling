import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from typing import List, Dict, Type, Union, Optional

from copy import deepcopy
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
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

from ray.rllib.algorithms.ddpg.noop_model import TorchNoopModel
from .maddpg_torch_model import MADDPGConfig, MADDPGTorchModel

torch, nn = try_import_torch()


class MADDPGTorchPolicy(TorchPolicyV2):
    def __init__(self, obs_space, act_space, config):
        # _____ Initial Configuration
        self.config = dict(MADDPGConfig().to_dict(), **config)

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

        class _Critic(nn.Module):
            def __init__(self, obs_n, act_n):
                super(_Critic, self).__init__()
                obs_dim = [tmp_dim.shape[0] for tmp_dim in obs_n]
                act_dim = [tmp_dim.shape[0] for tmp_dim in act_n]

                self.FC1 = nn.Linear(sum(obs_dim)+sum(act_dim), 64)
                self.FC2 = nn.Linear(64, 64)
                self.FC3 = nn.Linear(64, 64)
                self.q_out = nn.Linear(64, 1)

            def forward(self, obs, acts):
                state = torch.cat(obs, dim=1)
                actions = torch.cat(acts, dim=1)
                x = torch.cat([state,actions],dim=1)
                x = nn.functional.relu(self.FC1(x))
                x = nn.functional.relu(self.FC2(x))
                x = nn.functional.relu(self.FC3(x))
                q_value = self.q_out(x)

                return q_value

        class _Actor(nn.Module):
            def __init__(self, obs_space, act_space):
                super(_Actor, self).__init__()
                obs_dim = obs_space.shape[0]
                act_dim = act_space.shape[0]
                self.FC1 = nn.Linear(obs_dim, 64)
                self.FC2 = nn.Linear(64, 64)
                self.FC3 = nn.Linear(64, 64)
                self.action_out = nn.Linear(64,act_dim)

            # action output between -2 and 2
            def forward(self, obs):
                x = nn.functional.relu(self.FC1(obs))
                x = nn.functional.relu(self.FC2(x))
                x = nn.functional.relu(self.FC3(x))
                actions = nn.functional.tanh(self.action_out(x))
                return actions
        
        # Get observation dictionary and actionary dictionary
        obs_dict = self.config["obs_space_dict"]
        act_dict = self.config["act_space_dict"]
        # Make continous space for current agent
        self.act = _make_continuous_space(act_space)
        self.obs = _make_continuous_space(obs_space)
        # Make continous space for all agents
        self.obs_n = [_make_continuous_space(obs_one) for obs_one in obs_dict.values()]
        self.act_n = [_make_continuous_space(act_one) for act_one in act_dict.values()]

        # Build critic network and actor network
        self.critic = _Critic(self.obs_n, self.act_n)
        self.actor = _Actor(self.obs, self.act)

        # Build the target network
        self.critic_target = _Critic(self.obs_n, self.act_n)
        self.actor_target = _Actor(self.obs, self.act)

        TorchPolicyV2.__init__(
            self,
            obs_space,
            act_space,
            self.config,
            max_seq_len=self.config["model"]["max_seq_len"]
        )


    @override(TorchPolicyV2)
    def make_model(self) -> ModelV2:
        self.config["model"]["multiagent"] = self.config["multiagent"] # Needed for critic obs_space and act_space
        if self.config["use_state_preprocessor"]:
            default_model = None  # catalog decides
            num_outputs = 256  # arbitrary
            self.config["model"]["no_final_linear"] = True
        else:
            default_model = TorchNoopModel
            num_outputs = np.prod(self.obs.shape)
        model =  ModelCatalog.get_model_v2(
            obs_space=self.obs,
            action_space=self.act,
            num_outputs=num_outputs,
            model_config=self.config["model"],
            framework=self.config["framework"],
            model_interface=MADDPGTorchModel,
            default_model=default_model,
            name="maddpg_model",
            actor_hidden_activation=self.config["actor_hidden_activation"],
            actor_hiddens=self.config["actor_hiddens"],
            critic_hidden_activation=self.config["critic_hidden_activation"],
            critic_hiddens=self.config["critic_hiddens"],
            twin_q=self.config["twin_q"],
            add_layer_norm=(
                self.config["exploration_config"].get("type") == "ParameterNoise"
            ),
        )

        self.target_models = ModelCatalog.get_model_v2(
            obs_space=self.obs,
            action_space=self.act,
            num_outputs=num_outputs,
            model_config=self.config["model"],
            framework=self.config["framework"],
            model_interface=MADDPGTorchModel,
            default_model=default_model,
            name="target_maddpg_model",
            actor_hidden_activation=self.config["actor_hidden_activation"],
            actor_hiddens=self.config["actor_hiddens"],
            critic_hidden_activation=self.config["critic_hidden_activation"],
            critic_hiddens=self.config["critic_hiddens"],
            twin_q=self.config["twin_q"],
            add_layer_norm=(
                self.config["exploration_config"].get("type") == "ParameterNoise"
            ),
        )

        return model

    @override(TorchPolicyV2)
    def loss(
        self, 
        model: ModelV2, 
        dist_class: type[TorchDistributionWrapper], 
        train_batch: SampleBatch
        ) -> Union[TensorType, List[TensorType]]:
        # Get batch data
        state_batch = train_batch[SampleBatch.OBS]
        actions_batch = train_batch[SampleBatch.ACTIONS]
        reward_batch = train_batch[SampleBatch.REWARDS]

        next_state_batch = train_batch[SampleBatch.NEXT_OBS]
        
        return super().loss(model, dist_class, train_batch)

    @override(TorchPolicyV2)
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
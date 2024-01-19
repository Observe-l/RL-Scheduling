import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np

from ray.rllib import Policy
# from ray.rllib.algorithms.ddpg.noop_model import TorchNoopModel
from ray.rllib.models import ModelV2
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.error import UnsupportedSpaceException
from .maddpg_torch_model import TorchNoopModel

def make_maddpg_models(policy: Policy) -> ModelV2:
    from .maddpg_torch_model import MADDPGTorchModel
    policy.config["model"]["policies"] = policy.config[
        "policies"
    ]  # Needed for critic obs_space and act_space
    if policy.config["use_state_preprocessor"]:
        default_model = None  # catalog decides
        num_outputs = 256  # arbitrary
        policy.config["model"]["no_final_linear"] = True
    else:
        default_model = TorchNoopModel
        num_outputs = np.prod(policy.observation_space.shape)

    policy.model = ModelCatalog.get_model_v2(
        obs_space=policy.observation_space,
        action_space=policy.action_space,
        num_outputs=num_outputs,
        model_config=policy.config["model"],
        framework=policy.config["framework"],
        model_interface=MADDPGTorchModel,
        default_model=default_model,
        name="maddpg_model",
        actor_hidden_activation=policy.config["actor_hidden_activation"],
        actor_hiddens=policy.config["actor_hiddens"],
        critic_hidden_activation=policy.config["critic_hidden_activation"],
        critic_hiddens=policy.config["critic_hiddens"],
        twin_q=policy.config["twin_q"],
        add_layer_norm=(
            policy.config["exploration_config"].get("type") == "ParameterNoise"
        ),
    )

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=policy.observation_space,
        action_space=policy.action_space,
        num_outputs=num_outputs,
        model_config=policy.config["model"],
        framework=policy.config["framework"],
        model_interface=MADDPGTorchModel,
        default_model=default_model,
        name="target_maddpg_model",
        actor_hidden_activation=policy.config["actor_hidden_activation"],
        actor_hiddens=policy.config["actor_hiddens"],
        critic_hidden_activation=policy.config["critic_hidden_activation"],
        critic_hiddens=policy.config["critic_hiddens"],
        twin_q=policy.config["twin_q"],
        add_layer_norm=(
            policy.config["exploration_config"].get("type") == "ParameterNoise"
        ),
    )

    return policy.model

def validate_spaces(
        policy: Policy,
        obs_space,
        action_space,
        config
    ) -> None:
    def _make_continuous_space(space):
        if isinstance(space, Box):
            return space
        elif isinstance(space, Discrete):
            return Box(low=np.zeros((space.n,)), high=np.ones((space.n,)))
        else:
            raise UnsupportedSpaceException(
                "Space {} is not supported.".format(space)
            )
    policy.observation_space = _make_continuous_space(obs_space)
    policy.action_space = _make_continuous_space(action_space)

    policy.obs_space_n = [
        _make_continuous_space(space) 
        for _, (_, space, _, _) in config["policies"].items()
    ]
    
    policy.act_space_n = [
        _make_continuous_space(space) 
        for _, (_, _, space, _) in config["policies"].items()
    ]
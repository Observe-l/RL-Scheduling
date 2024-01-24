import logging
from typing import Dict, List, Optional, Type, Union
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.deprecation import (
    DEPRECATED_VALUE,
    Deprecated,
    ALGO_DEPRECATION_WARNING,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

torch, nn = try_import_torch()

def _make_continuous_space(space):
    if isinstance(space, Box):
        return space
    elif isinstance(space, Discrete):
        return Box(low=np.zeros((space.n,)), high=np.ones((space.n,)))
    else:
        raise UnsupportedSpaceException("Space {} is not supported.".format(space))
    
class MADDPGConfig(AlgorithmConfig):
    """Defines a configuration class from which a MADDPG Algorithm can be built.

    Example:
        >>> from ray.rllib.algorithms.maddpg.maddpg import MADDPGConfig
        >>> config = MADDPGConfig()
        >>> print(config.replay_buffer_config)  # doctest: +SKIP
        >>> replay_config = config.replay_buffer_config.update(  # doctest: +SKIP
        ...     {
        ...         "capacity": 100000,
        ...         "prioritized_replay_alpha": 0.8,
        ...         "prioritized_replay_beta": 0.45,
        ...         "prioritized_replay_eps": 2e-6,
        ...     }
        ... )
        >>> config.training(replay_buffer_config=replay_config)   # doctest: +SKIP
        >>> config = config.resources(num_gpus=0)   # doctest: +SKIP
        >>> config = config.rollouts(num_rollout_workers=4)   # doctest: +SKIP
        >>> config = config.environment("CartPole-v1")   # doctest: +SKIP
        >>> algo = config.build()  # doctest: +SKIP
        >>> algo.train()  # doctest: +SKIP

    Example:
        >>> from ray.rllib.algorithms.maddpg.maddpg import MADDPGConfig
        >>> from ray import air
        >>> from ray import tune
        >>> config = MADDPGConfig()
        >>> config.training(n_step=tune.grid_search([3, 5]))  # doctest: +SKIP
        >>> config.environment(env="CartPole-v1")  # doctest: +SKIP
        >>> tune.Tuner(  # doctest: +SKIP
        ...     "MADDPG",
        ...     run_config=air.RunConfig(stop={"episode_reward_mean":200}),
        ...     param_space=config.to_dict()
        ... ).fit()
    """

    def __init__(self, algo_class=None):
        """Initializes a DQNConfig instance."""
        super().__init__(algo_class=algo_class or MADDPG)

        # fmt: off
        # __sphinx_doc_begin__
        # MADDPG specific config settings:
        self.agent_id = None
        self.use_local_critic = False
        self.use_state_preprocessor = False
        self.actor_hiddens = [64, 64]
        self.actor_hidden_activation = "relu"
        self.critic_hiddens = [64, 64]
        self.critic_hidden_activation = "relu"
        self.n_step = 1
        self.good_policy = "maddpg"
        self.adv_policy = "maddpg"
        self.replay_buffer_config = {
            "type": "MultiAgentReplayBuffer",
            # Specify prioritized replay by supplying a buffer type that supports
            # prioritization, for example: MultiAgentPrioritizedReplayBuffer.
            "prioritized_replay": DEPRECATED_VALUE,
            "capacity": int(1e6),
            # Force lockstep replay mode for MADDPG.
            "replay_mode": "lockstep",
        }
        self.compress_observations = None
        self.training_intensity = None
        self.num_steps_sampled_before_learning_starts = 1024 * 25
        self.critic_lr = 1e-2
        self.actor_lr = 1e-2
        self.target_network_update_freq = 0
        self.tau = 0.01
        self.actor_feature_reg = 0.001
        # self.grad_norm_clipping = 0.5
        self.grad_clip = 100

        # Changes to Algorithm's default:
        self.rollout_fragment_length = 100
        self.train_batch_size = 1024
        self.num_rollout_workers = 1
        self.min_time_s_per_iteration = 0
        self.min_sample_timesteps_per_iteration = 1000

        # torch-specific model configs
        self.twin_q = False
        self.policy_delay = 1
        self.smooth_target_policy = False
        self.use_huber = False
        self.huber_threshold = 1.0
        self.l2_reg = None

        # self.exploration_config = {
        #     # The Exploration class to use. In the simplest case, this is the name
        #     # (str) of any class present in the `rllib.utils.exploration` package.
        #     # You can also provide the python class directly or the full location
        #     # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
        #     # EpsilonGreedy").
        #     "type": "StochasticSampling",
        #     # Add constructor kwargs here (if any).
        # }
        self.exploration_config = {
            "type": "GaussianNoise",
            # For how many timesteps should we return completely random actions,
            # before we start adding (scaled) noise?
            "random_timesteps": 1000,
            # The stddev (sigma) to be used for the actions
            "stddev": 0.5,
            # The initial noise scaling factor.
            "initial_scale": 1.0,
            # The final noise scaling factor.
            "final_scale": 0.02,
            # Timesteps over which to anneal scale (from initial to final values).
            "scale_timesteps": 10000,
        }
        
        # fmt: on
        # __sphinx_doc_end__

        # === Evaluation ===
        # self.evaluation_interval = None
        # self.evaluation_num_episodes = None
        # self.learn_other_policies = None
        # # Extra configuration that disables exploration.
        # self.evaluation_config = {
        #     "explore": False
        # }
        



    @override(AlgorithmConfig)
    def training(
        self,
        *,
        agent_id: Optional[str] = NotProvided,
        use_local_critic: Optional[bool] = NotProvided,
        use_state_preprocessor: Optional[bool] = NotProvided,
        actor_hiddens: Optional[List[int]] = NotProvided,
        actor_hidden_activation: Optional[str] = NotProvided,
        critic_hiddens: Optional[List[int]] = NotProvided,
        critic_hidden_activation: Optional[str] = NotProvided,
        n_step: Optional[int] = NotProvided,
        good_policy: Optional[str] = NotProvided,
        adv_policy: Optional[str] = NotProvided,
        replay_buffer_config: Optional[dict] = NotProvided,
        training_intensity: Optional[float] = NotProvided,
        num_steps_sampled_before_learning_starts: Optional[int] = NotProvided,
        critic_lr: Optional[float] = NotProvided,
        actor_lr: Optional[float] = NotProvided,
        target_network_update_freq: Optional[int] = NotProvided,
        tau: Optional[float] = NotProvided,
        actor_feature_reg: Optional[float] = NotProvided,
        grad_norm_clipping: Optional[float] = NotProvided,
        **kwargs,
    ) -> "MADDPGConfig":
        """Sets the training related configuration.

        Args:
            agent_id: ID of the agent controlled by this policy.
            use_local_critic: Use a local critic for this policy.
            use_state_preprocessor: Apply a state preprocessor with spec given by the
                "model" config option (like other RL algorithms). This is mostly useful
                if you have a weird observation shape, like an image. Disabled by
                default.
            actor_hiddens: Postprocess the policy network model output with these hidden
                layers. If `use_state_preprocessor` is False, then these will be the
                *only* hidden layers in the network.
            actor_hidden_activation: Hidden layers activation of the postprocessing
                stage of the policy network.
            critic_hiddens: Postprocess the critic network model output with these
                hidden layers; again, if use_state_preprocessor is True, then the state
                will be preprocessed by the model specified with the "model" config
                option first.
            critic_hidden_activation: Hidden layers activation of the postprocessing
                state of the critic.
            n_step: N-step for Q-learning.
            good_policy: Algorithm for good policies.
            adv_policy: Algorithm for adversary policies.
            replay_buffer_config: Replay buffer config.
                Examples:
                {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentReplayBuffer",
                "capacity": 50000,
                "replay_sequence_length": 1,
                }
                - OR -
                {
                "_enable_replay_buffer_api": True,
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 50000,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
                "replay_sequence_length": 1,
                }
                - Where -
                prioritized_replay_alpha: Alpha parameter controls the degree of
                prioritization in the buffer. In other words, when a buffer sample has
                a higher temporal-difference error, with how much more probability
                should it drawn to use to update the parametrized Q-network. 0.0
                corresponds to uniform probability. Setting much above 1.0 may quickly
                result as the sampling distribution could become heavily “pointy” with
                low entropy.
                prioritized_replay_beta: Beta parameter controls the degree of
                importance sampling which suppresses the influence of gradient updates
                from samples that have higher probability of being sampled via alpha
                parameter and the temporal-difference error.
                prioritized_replay_eps: Epsilon parameter sets the baseline probability
                for sampling so that when the temporal-difference error of a sample is
                zero, there is still a chance of drawing the sample.
            training_intensity: If set, this will fix the ratio of replayed from a
                buffer and learned on timesteps to sampled from an environment and
                stored in the replay buffer timesteps. Otherwise, the replay will
                proceed at the native ratio determined by
                `(train_batch_size / rollout_fragment_length)`.
            num_steps_sampled_before_learning_starts: Number of timesteps to collect
                from rollout workers before we start sampling from replay buffers for
                learning. Whether we count this in agent steps  or environment steps
                depends on config.multi_agent(count_steps_by=..).
            critic_lr: Learning rate for the critic (Q-function) optimizer.
            actor_lr: Learning rate for the actor (policy) optimizer.
            target_network_update_freq: Update the target network every
                `target_network_update_freq` sample steps.
            tau: Update the target by \tau * policy + (1-\tau) * target_policy.
            actor_feature_reg: Weights for feature regularization for the actor.
            grad_norm_clipping: If not None, clip gradients during optimization at this
                value.

        Returns:
            This updated AlgorithmConfig object.
        """

        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if agent_id is not NotProvided:
            self.agent_id = agent_id
        if use_local_critic is not NotProvided:
            self.use_local_critic = use_local_critic
        if use_state_preprocessor is not NotProvided:
            self.use_state_preprocessor = use_state_preprocessor
        if actor_hiddens is not NotProvided:
            self.actor_hiddens = actor_hiddens
        if actor_hidden_activation is not NotProvided:
            self.actor_hidden_activation = actor_hidden_activation
        if critic_hiddens is not NotProvided:
            self.critic_hiddens = critic_hiddens
        if critic_hidden_activation is not NotProvided:
            self.critic_hidden_activation = critic_hidden_activation
        if n_step is not NotProvided:
            self.n_step = n_step
        if good_policy is not NotProvided:
            self.good_policy = good_policy
        if adv_policy is not NotProvided:
            self.adv_policy = adv_policy
        if replay_buffer_config is not NotProvided:
            self.replay_buffer_config = replay_buffer_config
        if training_intensity is not NotProvided:
            self.training_intensity = training_intensity
        if num_steps_sampled_before_learning_starts is not NotProvided:
            self.num_steps_sampled_before_learning_starts = (
                num_steps_sampled_before_learning_starts
            )
        if critic_lr is not NotProvided:
            self.critic_lr = critic_lr
        if actor_lr is not NotProvided:
            self.actor_lr = actor_lr
        if target_network_update_freq is not NotProvided:
            self.target_network_update_freq = target_network_update_freq
        if tau is not NotProvided:
            self.tau = tau
        if actor_feature_reg is not NotProvided:
            self.actor_feature_reg = actor_feature_reg
        if grad_norm_clipping is not NotProvided:
            self.grad_norm_clipping = grad_norm_clipping

        return self

    @override(AlgorithmConfig)
    def validate(self) -> None:
        """Adds the `before_learn_on_batch` hook to the config.

        This hook is called explicitly prior to `train_one_step()` in the
        `training_step()` methods of DQN and APEX.
        """
        # Call super's validation method.
        super().validate()

        def f(batch, workers, config):
            policies = dict(
                workers.local_worker().foreach_policy_to_train(lambda p, i: (i, p))
            )
            return before_learn_on_batch(batch, policies, config["train_batch_size"])

        self.before_learn_on_batch = f


def before_learn_on_batch(multi_agent_batch, policies, train_batch_size):
    samples = {}

    def sampler(policy, obs):
        return policy.compute_actions(obs)[0]
    
    # Modify keys.
    for pid, p in policies.items():
        i = p.config["agent_id"]
        keys = multi_agent_batch.policy_batches[pid].keys()
        keys = ["_".join([k, str(i)]) for k in keys]
        samples.update(dict(zip(keys, multi_agent_batch.policy_batches[pid].values())))
        new_obs = samples["new_obs_{}".format(i)]
        new_act = sampler(p, new_obs)
        samples.update({"new_actions_{}".format(i): new_act})

    # Share samples among agents.
    policy_batches = {pid: SampleBatch(samples) for pid in policies.keys()}
    return MultiAgentBatch(policy_batches, train_batch_size)


class MADDPG(DQN):
    @classmethod
    @override(DQN)
    def get_default_config(cls) -> AlgorithmConfig:
        return MADDPGConfig()

    @classmethod
    @override(DQN)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        from .maddpg_torch_policy import MADDPGTorchPolicy
        return MADDPGTorchPolicy

class MADDPGTorchModel(TorchModelV2, nn.Module):
    """
    Extension of TorchModelV2 for MADDPG
    Note that the critic takes in the joint state and action over all agents
    Data flow:
        obs -> forward() -> model_out
        model_out -> get_policy_output() -> pi(s)
        model_out, actions -> get_q_values() -> Q(s, a)
        model_out, actions -> get_twin_q_values() -> Q_twin(s, a)
    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass.
    """

    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        # Extra MADDPGActionModel args:
        actor_hiddens: List[int] = [256, 256],
        actor_hidden_activation: str = "relu",
        critic_hiddens: List[int] = [256, 256],
        critic_hidden_activation: str = "relu",
        twin_q: bool = False,
        add_layer_norm: bool = False,
    ):

        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )

        self.bounded = np.logical_and(self.action_space.bounded_above,
                                      self.action_space.bounded_below).any()
        self.action_dim = np.product(self.action_space.shape)

        # Build the policy network.
        self.policy_model = nn.Sequential()
        ins = int(np.product(observation_space.shape))
        self.obs_ins = ins
        activation = get_activation_fn(actor_hidden_activation, framework="torch")
        for i, n in enumerate(actor_hiddens):
            self.policy_model.add_module(
                "action_{}".format(i),
                SlimFC(
                    ins,
                    n,
                    initializer=torch.nn.init.xavier_uniform_,
                    activation_fn=activation,
                ),
            )
            # Add LayerNorm after each Dense.
            if add_layer_norm:
                self.policy_model.add_module(
                    "LayerNorm_A_{}".format(i), nn.LayerNorm(n)
                )
            ins = n

        self.policy_model.add_module(
            "action_out",
            SlimFC(
                ins,
                self.action_dim,
                initializer=torch.nn.init.xavier_uniform_,
                activation_fn=None,
            ),
        )

        # Use sigmoid to scale to [0,1], but also double magnitude of input to
        # emulate behaviour of tanh activation used in DDPG and TD3 papers.
        # After sigmoid squashing, re-scale to env action space bounds.
        class _Lambda(nn.Module):
            def __init__(self_):
                super().__init__()
                low_action = nn.Parameter(
                    torch.from_numpy(self.action_space.low).float())
                low_action.requires_grad = False
                self_.register_parameter("low_action", low_action)
                action_range = nn.Parameter(
                    torch.from_numpy(self.action_space.high -
                                     self.action_space.low).float())
                action_range.requires_grad = False
                self_.register_parameter("action_range", action_range)

            def forward(self_, x):
                sigmoid_out = nn.Sigmoid()(2.0 * x)
                squashed = self_.action_range * sigmoid_out + self_.low_action
                return squashed

        # Only squash if we have bounded actions.
        if self.bounded:
            self.policy_model.add_module("action_out_squashed", _Lambda())

        # Build MADDPG Critic and Target Critic

        obs_space_n = [
            _make_continuous_space(space)
            for _, (_, space, _, _) in model_config["policies"].items()
        ]
        act_space_n = [
            _make_continuous_space(space)
            for _, (_, _, space, _) in model_config["policies"].items()
        ]
        self.critic_obs = np.sum([obs_space.shape[0] for obs_space in obs_space_n])
        self.critic_act = np.sum([act_space.shape[0] for act_space in act_space_n])

        # Build the Q-net(s), including target Q-net(s).
        def build_q_net(name_):
            activation = get_activation_fn(critic_hidden_activation, framework="torch")
            q_net = nn.Sequential()
            ins = self.critic_obs + self.critic_act
            for i, n in enumerate(critic_hiddens):
                q_net.add_module(
                    "{}_hidden_{}".format(name_, i),
                    SlimFC(
                        ins,
                        n,
                        initializer=nn.init.xavier_uniform_,
                        activation_fn=activation,
                    ),
                )
                ins = n

            q_net.add_module(
                "{}_out".format(name_),
                SlimFC(
                    ins,
                    1,
                    initializer=torch.nn.init.xavier_uniform_,
                    activation_fn=None,
                ),
            )
            return q_net

        self.q_model = build_q_net("q")
        if twin_q:
            self.twin_q_model = build_q_net("twin_q")
        else:
            self.twin_q_model = None

        # self.view_requirements[SampleBatch.ACTIONS] = ViewRequirement(
        #     SampleBatch.ACTIONS
        # )
        # self.view_requirements["new_actions"] = ViewRequirement("new_actions")
        # self.view_requirements["t"] = ViewRequirement("t")
        # self.view_requirements[SampleBatch.NEXT_OBS] = ViewRequirement(
        #     data_col=SampleBatch.OBS, shift=1, space=self.obs_space
        # )

    def get_q_values(
        self, model_out_n: List[TensorType], act_n: List[TensorType]
    ) -> TensorType:
        """Return the Q estimates for the most recent forward pass.
        This implements Q(s, a).
        Args:
            model_out_n (List[Tensor]): obs embeddings from the model layers of each agent,
            of shape [BATCH_SIZE, num_outputs].
            actions (Tensor): Actions from each agent to return the Q-values for.
                Shape: [BATCH_SIZE, action_dim].
        Returns:
            tensor of shape [BATCH_SIZE].
        """
        model_out_n = torch.cat(model_out_n, -1)
        act_n = torch.cat(act_n, dim=-1)
        return self.q_model(torch.cat([model_out_n, act_n], -1))

    def get_twin_q_values(
        self, model_out_n: TensorType, act_n: TensorType
    ) -> TensorType:
        """Same as get_q_values but using the twin Q net.
        This implements the twin Q(s, a).
        Args:
            model_out_n (List[Tensor]): obs embeddings from the model layers of each agent,
            of shape [BATCH_SIZE, num_outputs].
            actions (Tensor): Actions from each agent to return the Q-values for.
                Shape: [BATCH_SIZE, action_dim].
        Returns:
            tensor of shape [BATCH_SIZE].
        """
        model_out_n = torch.cat(model_out_n, -1)
        print
        act_n = torch.cat(act_n, dim=-1)
        return self.twin_q_model(torch.cat([model_out_n, act_n], -1))

    def get_policy_output(self, model_out: TensorType) -> TensorType:
        """Return the action output for the most recent forward pass.
        This outputs the logits over the action space for discrete actions.
        Args:
            model_out (Tensor): obs embeddings from the model layers, of shape
                [BATCH_SIZE, num_outputs].
        Returns:
            tensor of shape [BATCH_SIZE, action_out_size]
        """
        return self.policy_model(model_out)

    def policy_variables(
        self, as_dict: bool = False
    ) -> Union[List[TensorType], Dict[str, TensorType]]:
        """Return the list of variables for the policy net."""
        if as_dict:
            return self.policy_model.state_dict()
        return list(self.policy_model.parameters())

    def q_variables(
        self, as_dict=False
    ) -> Union[List[TensorType], Dict[str, TensorType]]:
        """Return the list of variables for Q / twin Q nets."""
        if as_dict:
            return {
                **self.q_model.state_dict(),
                **(self.twin_q_model.state_dict() if self.twin_q_model else {}),
            }
        return list(self.q_model.parameters()) + (
            list(self.twin_q_model.parameters()) if self.twin_q_model else []
        )

class TorchNoopModel(TorchModelV2):
    """Trivial model that just returns the obs flattened.

    This is the model used if use_state_preprocessor=False."""

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        return input_dict["obs_flat"].float(), state
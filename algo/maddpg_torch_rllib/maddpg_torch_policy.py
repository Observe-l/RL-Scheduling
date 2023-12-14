import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from typing import Any, List, Dict, Tuple, Type, Union, Optional

from copy import deepcopy
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.policy.torch_mixins import TargetNetworkMixin
from ray.rllib.evaluation.postprocessing import adjust_nstep
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.torch_action_dist import (
    TorchDeterministic,
    TorchDirichlet,
    TorchDistributionWrapper,
)
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import override
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping, 
    concat_multi_gpu_td_errors,
    huber_loss, 
    l2_loss)
from ray.rllib.utils.typing import ModelConfigDict, ModelGradients, TensorType
from ray.rllib.utils.spaces.simplex import Simplex

from ray.rllib.algorithms.ddpg.noop_model import TorchNoopModel
from .maddpg_torch_model import MADDPGConfig, MADDPGTorchModel
from .utils import make_maddpg_models, validate_spaces

torch, nn = try_import_torch()

class ComputeTDErrorMixin:
    def __init__(self: TorchPolicyV2):
        def compute_td_error(
                obs_t, act_t, rew_t, obs_tp1, terminateds_mask
            ):
            input_dict = self._lazy_tensor_dict(
                SampleBatch(
                    {
                        SampleBatch.CUR_OBS: obs_t,
                        SampleBatch.ACTIONS: act_t,
                        SampleBatch.REWARDS: rew_t,
                        SampleBatch.NEXT_OBS: obs_tp1,
                        SampleBatch.TERMINATEDS: terminateds_mask,
                    }
                )
            )
            # Do forward pass on loss to update td errors attribute
            self.loss(self.model, None, input_dict)

            # Self.td_error is set within actor_critic_loss call.
            return self.model.tower_state["td_error"]

        self.compute_td_error = compute_td_error



class MADDPGTorchPolicy(TargetNetworkMixin, ComputeTDErrorMixin, TorchPolicyV2):
    def __init__(self, obs_space, act_space, config):
        # _____ Initial Configuration
        self.config = dict(MADDPGConfig().to_dict(), **config)
        self.global_step = 0

        # FIXME: Get done from info is required since agentwise done is not
        #  supported now.
        self.get_done_from_info = np.vectorize(lambda info: info.get("done", False))

        agent_id = config["agent_id"]
        if agent_id is None:
            raise ValueError("Must set `agent_id` in the policy config.")
        if type(agent_id) is not int:
            raise ValueError("Agent ids must be integers for MADDPG.")
        
        validate_spaces(self,obs_space,act_space,self.config)

        TorchPolicyV2.__init__(
            self,
            self.observation_space,
            self.action_space,
            self.config,
            max_seq_len=self.config["model"]["max_seq_len"]
        )

        ComputeTDErrorMixin.__init__(self)

        TargetNetworkMixin.__init__(self)


    @override(TorchPolicyV2)
    def make_model_and_action_dist(self) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
        model = make_maddpg_models(self)
        if isinstance(self.action_space, Simplex):
            distr_class = TorchDirichlet
        else:
            distr_class = TorchDeterministic

        return model, distr_class
    
    @override(TorchPolicyV2)
    def optimizer(self) -> List["torch.optim.Optimizer"]:
        '''
        Create separate optimizers for actor & critic losses.
        '''
        # Set epsilons to match tf.keras.optimizers.Adam's epsilon default.
        self._actor_optimizer = torch.optim.Adam(
            params=self.model.policy_variables(), lr=self.config["actor_lr"], eps=1e-7
        )

        self._critic_optimizer = torch.optim.Adam(
            params=self.model.q_variables(), lr=self.config["critic_lr"], eps=1e-7
        )
        # Return them in the same order as the respective loss terms are returned.
        return [self._actor_optimizer, self._critic_optimizer]
    
    @override(TorchPolicyV2)
    def apply_gradients(self, gradients: ModelGradients) -> None:
        '''
        For policy gradient, update policy net one time v.s.
        update critic net `policy_delay` time(s).
        '''
        if self.global_step % self.config["policy_delay"] == 0:
            self._actor_optimizer.step()
        
        self._critic_optimizer.step()

        # Increment global step & apply ops.
        self.global_step += 1

    @override(TorchPolicyV2)
    def action_distribution_fn(
        self,
        model: ModelV2,
        *,
        obs_batch: TensorType,
        state_batches: TensorType,
        is_training: bool = False,
        **kwargs
    ) -> Tuple[TensorType, type, List[TensorType]]:
        '''
        action_distribution, based on ddpg fuction
        from ray.rllib.algorithms.ddpg.ddpg_torch_policy
        '''
        model_out, _ = model(
            SampleBatch(obs=obs_batch[SampleBatch.CUR_OBS], _is_training=is_training)
        )
        dist_inputs = model.get_policy_output(model_out)

        if isinstance(self.action_space, Simplex):
            distr_class = TorchDirichlet
        else:
            distr_class = TorchDeterministic
        return dist_inputs, distr_class, []  # []=state out

    @override(TorchPolicyV2)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # N-step Q adjustments
        if self.config["n_step"] > 1:
            adjust_nstep(self.config["n_step"], self.config["gamma"], sample_batch)

        return sample_batch

    @override(TorchPolicyV2)
    def loss(
        self, 
        model: ModelV2, 
        dist_class: type[TorchDistributionWrapper], 
        train_batch: SampleBatch
        ) -> List[TensorType]:
        '''
        The actor and critic loss
        '''
        target_model = self.target_models[model]

        twin_q = self.config["twin_q"]
        gamma = self.config["gamma"]
        n_step = self.config["n_step"]
        use_huber = self.config["use_huber"]
        huber_threshold = self.config["huber_threshold"]
        l2_reg = self.config["l2_reg"]
        agent_id = self.config["agent_id"]
        n_agents = len(self.config["multiagent"]["policies"])

        input_dict = SampleBatch(
            obs=train_batch["_".join([SampleBatch.CUR_OBS, str(agent_id)])], _is_training=True
        )
        input_dict_next = SampleBatch(
            obs=train_batch["_".join([SampleBatch.NEXT_OBS, str(agent_id)])], _is_training=True
        )

        model_out_t, _ = model(input_dict, [], None)

        policy_t = model.get_policy_output(model_out_t)

        target_model_out_tp1, _ = target_model(input_dict_next, [], None)

        policy_tp1 = target_model.get_policy_output(target_model_out_tp1)

        # Action outputs.
        if self.config["smooth_target_policy"]:
            target_noise_clip = self.config["target_noise_clip"]
            clipped_normal_sample = torch.clamp(
                torch.normal(
                    mean=torch.zeros(policy_tp1.size()), std=self.config["target_noise"]
                ).to(policy_tp1.device),
                -target_noise_clip,
                target_noise_clip,
            )

            policy_tp1_smoothed = torch.min(
                torch.max(
                    policy_tp1 + clipped_normal_sample,
                    torch.tensor(
                        self.action_space.low,
                        dtype=torch.float32,
                        device=policy_tp1.device,
                    ),
                ),
                torch.tensor(
                    self.action_space.high, dtype=torch.float32, device=policy_tp1.device
                ),
            )
        else:
            # No smoothing, just use deterministic actions.
            policy_tp1_smoothed = policy_tp1

        obs_n = [
            train_batch["_".join([SampleBatch.CUR_OBS, str(id)])] for id in range(n_agents)
        ]
        act_n = [
            train_batch["_".join([SampleBatch.ACTIONS, str(id)])] for id in range(n_agents)
        ]
        next_obs_n = [
            train_batch["_".join([SampleBatch.NEXT_OBS, str(id)])] for id in range(n_agents)
        ]
        next_policy_n = [train_batch["new_actions_{}".format(id)] for id in range(n_agents)]
        next_policy_n[agent_id] = policy_tp1_smoothed
        rewards = train_batch["rewards_{}".format(agent_id)]
        dones = train_batch["dones_{}".format(agent_id)]

        if self.config["use_state_preprocessor"]:
            # Create all state preprocessors
            model_n = [
                ModelCatalog.get_model_v2(
                    obs_space,
                    act_space,
                    1,
                    self.config["model"],
                    default_model=TorchNoopModel,
                )
                for obs_space, act_space in zip(self.obs_space_n, self.act_space_n)
            ]
            # Get states from preprocessors
            model_out_n = [
                model.forward({SampleBatch.OBS: obs, "is_training": True}, [], None)[0]
                for model, obs in zip(model_n, obs_n)
            ]
            model_out_next_n = [
                model.forward({SampleBatch.OBS: next_obs, "is_training": True}, [], None)[0]
                for model, next_obs in zip(model_n, next_obs_n)
            ]
        else:
            model_out_n = obs_n
            model_out_next_n = next_obs_n

        # Q-values for given actions & observations in given current
        q_t = model.get_q_values(model_out_n, act_n)

        # Compute this here so policy_n can be modified without deepcopying act_n
        if twin_q:
            twin_q_t = model.get_twin_q_values(model_out_n, act_n)

        # Q-values for current policy (no noise) in given current state
        policy_n = act_n
        policy_n[agent_id] = policy_t

        q_t_det_policy = model.get_q_values(model_out_n, policy_n)

        actor_loss = -torch.mean(q_t_det_policy)

        # Target q-net(s) evaluation.
        q_tp1 = target_model.get_q_values(model_out_next_n, next_policy_n)

        if twin_q:
            twin_q_tp1 = target_model.get_twin_q_values(
                model_out_next_n, next_policy_n
            )

        q_t_selected = torch.squeeze(q_t, axis=len(q_t.shape) - 1)

        if twin_q:
            twin_q_t_selected = torch.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
            q_tp1 = torch.min(q_tp1, twin_q_tp1)

        q_tp1_best = torch.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
        q_tp1_best_masked = (~dones).float() * q_tp1_best

        q_t_selected_target = (rewards + gamma ** n_step * q_tp1_best_masked).detach()

        # Compute the error (potentially clipped).
        if twin_q:
            td_error = q_t_selected - q_t_selected_target
            twin_td_error = twin_q_t_selected - q_t_selected_target
            if use_huber:
                errors = huber_loss(td_error, huber_threshold) + huber_loss(
                    twin_td_error, huber_threshold
                )
            else:
                errors = 0.5 * (torch.pow(td_error, 2.0) + torch.pow(twin_td_error, 2.0))
        else:
            td_error = q_t_selected - q_t_selected_target
            if use_huber:
                errors = huber_loss(td_error, huber_threshold)
            else:
                errors = 0.5 * torch.pow(td_error, 2.0)

        critic_loss = torch.mean(errors)

        # Add l2-regularization if required.
        if l2_reg is not None:
            for name, var in model.policy_variables(as_dict=True).items():
                if "bias" not in name:
                    actor_loss += l2_reg * l2_loss(var)
            for name, var in model.q_variables(as_dict=True).items():
                if "bias" not in name:
                    critic_loss += l2_reg * l2_loss(var)

        model.tower_state["q_t"] = q_t
        model.tower_state["actor_loss"] = actor_loss
        model.tower_state["critic_loss"] = critic_loss
        model.tower_state["td_error"] = td_error

        return [actor_loss, critic_loss]
    
    @override(TorchPolicyV2)
    def extra_grad_process(
        self, optimizer: "torch.optim.Optimizer", loss: TensorType
    ) -> Dict[str, TensorType]:
        return apply_grad_clipping(self, optimizer, loss)
    
    @override(TorchPolicyV2)
    def extra_compute_grad_fetches(self) -> Dict[str, Any]:
        fetches = convert_to_numpy(concat_multi_gpu_td_errors(self))
        return dict({LEARNER_STATS_KEY: {}}, **fetches)
    
    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        q_t = torch.stack(self.get_tower_stats("q_t"))
        stats = {
            "actor_loss": torch.mean(torch.stack(self.get_tower_stats("actor_loss"))),
            "critic_loss": torch.mean(torch.stack(self.get_tower_stats("critic_loss"))),
            "mean_q": torch.mean(q_t),
            "max_q": torch.max(q_t),
            "min_q": torch.min(q_t)
        }
        return convert_to_numpy(stats)
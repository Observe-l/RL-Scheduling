import ray
import os
from ray import tune, air
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env.env_context import EnvContext
import optparse
import numpy as np
import time
from util.ray_env import Simple_Scheduling

# Define the policies
env_config = EnvContext(env_config={"path":"/home/lwh/Documents/Code/RL-Scheduling/result/ray_mappo"},worker_index=0)
env = Simple_Scheduling(env_config=env_config)

observation = env.observation_space
action = env.action_space
policies = {}
for agent_id, obs, act, i in zip(observation.keys(), observation.values(),action.values(),range(len(observation))):
    policies[f'{agent_id}'] = (
                                None,
                                obs,
                                act,
                                {},
                            )

env.stop_env()

def policy_mapping_fn(agent_id, episode, **kwargs):
    if agent_id < env.truck_num:
        return '0'
    else:
        return f'{agent_id}'

if __name__ == "__main__":
    ray.init()
    config = PPOConfig().to_dict()
    config.update({
        "env": Simple_Scheduling,
        "env_config": {"path":"/home/lwh/Documents/Code/RL-Scheduling/result/ray_ppo/"},
        "disable_env_checking":True,
        "num_workers": 30,
        "num_envs_per_worker": 1,
        "num_cpus_per_worker": 1,
        "num_gpus_per_worker": 1/30,
        "ignore_worker_failures":True,
        "recreate_failed_workers":True,
        "multiagent":{
            "policies":policies,
            "policy_mapping_fn":policy_mapping_fn,
        }
    })
    ray_dir = "/home/lwh/Documents/Code/RL-Scheduling/train_ray/"
    exp_name = "MAPPO"
    stop = {"training_iteration": 30}
    tunner = tune.Tuner(
        PPO,
        param_space=config,
        run_config=air.RunConfig(
            storage_path=ray_dir,
            log_to_file=True,
            local_dir=ray_dir,
            name=exp_name,
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=1,
                num_to_keep=2000,
                checkpoint_score_attribute='episode_reward_mean',
                checkpoint_at_end=True,
            ),
        )
    )
    result = tunner.fit()
    # best_checkpoint = result.get_best_result(metric='episode_reward_mean',mode='max',scope='avg').log_dir
    # print(best_checkpoint)
    # with open('best_checkpoint.txt','w') as f:
    #     f.write(str(best_checkpoint))
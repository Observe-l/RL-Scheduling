import ray
import os
from ray import tune, air
from gym.spaces import Discrete, Box
from ray.rllib.algorithms.maddpg.maddpg import MADDPGConfig,MADDPG
from ray.rllib.policy.policy import PolicySpec
import optparse
import numpy as np
import time
from util.ray_env import Simple_Scheduling

# Define the policies
env = Simple_Scheduling(env_config={"path":"/home/lwh/Documents/Code/RL-Scheduling/result/ray_maddpg/"})

observation = env.observation_space
action = env.action_space
policies = {}
for agent_id, obs, act, i in zip(observation.keys(), observation.values(),action.values(),range(len(observation))):
    policies[f'{agent_id}'] = (
                                None,
                                obs,
                                act,
                                {
                                    "agent_id":i,
                                    "use_local_critic": False,
                                    "obs_space_dict":observation,
                                    "act_space_dict":action,
                                }
                            )

env.stop_env()

def policy_mapping_fn(agent_id):
    return f'{agent_id}'

if __name__ == "__main__":
    ray.init()
    config = MADDPGConfig().to_dict()
    config.update({
        "env": Simple_Scheduling,
        "env_config": {"path":"/home/lwh/Documents/Code/RL-Scheduling/result/ray_maddpg/"},
        "disable_env_checking":True,
        "num_workers": 25,
        "num_cpus_per_worker": 1,
        "ignore_worker_failures":True,
        "recreate_failed_workers":True,
        "multiagent":{
            "policies":policies,
            "policy_mapping_fn":policy_mapping_fn,
        }
    })
    ray_dir = "/home/lwh/Documents/Code/RL-Scheduling/train_ray/"
    exp_name = "MADDPG"
    stop = {'episodes_total':2000}
    tunner = tune.Tuner(
        MADDPG,
        param_space=config,
        run_config=air.RunConfig(
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
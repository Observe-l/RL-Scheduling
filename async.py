import ray
import os
from ray import tune, air
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env.env_context import EnvContext
import optparse
import numpy as np
import time
from util.async_env import Simple_Scheduling


def policy_mapping_fn(agent_id, episode, **kwargs):
    return f'{agent_id}'

def get_parameters():
    opt = optparse.OptionParser(description="Basic parameters")
    opt.add_option("-n","--number",default=12,type=int,help="number of agents")
    options, args = opt.parse_args()
    return options

if __name__ == "__main__":
    options = get_parameters()
    # Define the policies
    env_config = EnvContext(env_config={"path":f"/home/lwh/Documents/Code/RL-Scheduling/result/MAPPO_async_vary_a_{options.number}",
                                        "agents":options.number},
                                        worker_index=0)
    env = Simple_Scheduling(env_config=env_config)

    observation = env.observation_space
    action = env.action_space
    policies = {}
    for agent_id, obs, act, i in zip(observation.keys(), observation.values(),action.values(),range(len(observation))):
        policies[f'{agent_id}'] = (None,obs,act,{"agent_id":i,})
    env.stop_env()

    ray.init()
    config = PPOConfig().to_dict()
    config.update({
        "env": Simple_Scheduling,
        "env_config": {"path":f"/home/lwh/Documents/Code/RL-Scheduling/result/MAPPO_async_vary_a_{options.number}",
                       "agents":options.number},
        "disable_env_checking":True,
        "framework":"torch",
        "num_workers": 32,
        "num_envs_per_worker": 1,
        "num_cpus_per_worker": 1,
        # "num_gpus_per_worker": 1/30,
        "ignore_worker_failures":True,
        "recreate_failed_workers":True,
        "multiagent":{
            "policies":policies,
            "policy_mapping_fn":policy_mapping_fn,
        }
    })
    exp_name = f"MAPPO_async_vary_a_{options.number}"
    stop = {'episodes_total':2500}
    tunner = tune.Tuner(
        PPO,
        param_space=config,
        run_config=air.RunConfig(
            log_to_file=True,
            name=exp_name,
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=1,
                num_to_keep=2500,
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
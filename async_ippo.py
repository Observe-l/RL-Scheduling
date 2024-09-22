import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env.env_context import EnvContext
from util.async_logistic import async_scheduling

def policy_mapping_fn(agent_id, episode, **kwargs):
    return f'{agent_id}'


if __name__ == "__main__":
    env_config = EnvContext(env_config={"algo":"async-iPPO","truck_num":20},worker_index=0)
    env = async_scheduling(env_config)
    observation = env.observation_space
    action = env.action_space
    policies = {}
    for agent_id, obs, act, i in zip(observation.keys(), observation.values(),action.values(),range(len(observation))):
        policies[f'{agent_id}'] = (None,obs,act,{})
    # env.stop_env()

    ray.init()
    config = PPOConfig().to_dict()
    config.update({
        "env": async_scheduling,
        "env_config": {"algo":"async-iPPO","truck_num":20},
        "disable_env_checking":True,
        "num_workers": 1,
        "num_envs_per_env_runner": 1,
        "num_cpus_per_env_runner": 20,
        "num_gpus_per_env_runner": 1,
        "ignore_env_runner_failures":True,
        "recreate_failed_env_runners":True,
        "multiagent":{
            "policies":policies,
            "policy_mapping_fn":policy_mapping_fn,
        }
    })

    exp_name = "async_iPPO"
    stop = {'timesteps_total':2500}
    tunner = tune.Tuner(
        PPO,
        param_space=config,
        run_config=air.RunConfig(
            log_to_file=True,
            name=exp_name,
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=1,
                num_to_keep=3,
                checkpoint_score_attribute='env_runners/episode_reward_mean',
                checkpoint_at_end=True,
            )
        )
    )
    result = tunner.fit()

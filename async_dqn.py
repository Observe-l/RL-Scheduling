import ray
from ray import tune, air
from ray.rllib.algorithms.dqn.dqn import DQN, DQNConfig
from ray.rllib.env.env_context import EnvContext
from util.async_env import async_scheduling

def policy_mapping_fn(agent_id, episode, **kwargs):
    return f'{agent_id}'


if __name__ == "__main__":
    env_config = EnvContext(env_config={"algo":"DQN"},worker_index=0)
    env = async_scheduling(env_config)
    observation = env.observation_space
    action = env.action_space
    policies = {}
    for agent_id, obs, act, i in zip(observation.keys(), observation.values(),action.values(),range(len(observation))):
        policies[f'{agent_id}'] = (None,obs,act,{})
    env.stop_env()

    ray.init()
    config = DQNConfig().to_dict()
    config.update({
        "env": async_scheduling,
        "env_config": {"algo":"DQN"},
        "disable_env_checking":True,
        "num_workers": 30,
        "num_envs_per_worker": 1,
        "num_cpus_per_worker": 1,
        "ignore_worker_failures":True,
        "recreate_failed_workers":True,
        "multiagent":{
            "policies":policies,
            "policy_mapping_fn":policy_mapping_fn,
        }
    })

    exp_name = "async_DQN"
    stop = {'episodes_total':2500}
    tunner = tune.Tuner(
        DQN,
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
            )
        )
    )
    result = tunner.fit()
'''
Random select.
'''
import random
from util.ray_env import Simple_Scheduling
from ray.rllib.env.env_context import EnvContext

env_config = EnvContext(env_config={"path":"/home/lwh/Documents/Code/RL-Scheduling/result/random"},worker_index=0)
env = Simple_Scheduling(env_config=env_config)
env.reset()

# Test 10 episode
EPI_BOUND = 10
while env.episode_num < EPI_BOUND:
    # 1 episode
    while env.done['__all__'] == False:
        truck_act = {i: random.randint(0,2) for i in range(env.truck_num)}
        factory_act = {i+env.truck_num: random.randint(0,1) for i in range(len(env.factory_agents))}
        tmp_act = {**truck_act, **factory_act}
        tmp_obs, tmp_rew, tmp_done, _ = env.step(tmp_act)
    
    env.reset()

env.stop_env()
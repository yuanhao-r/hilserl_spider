import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

# Prefer local project paths to avoid importing stale/global hil-serl installs.
_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
_WORKSPACE_ROOT = _REPO_ROOT.parent
for _p in [
    _REPO_ROOT / "serl_robot_infra",
    _REPO_ROOT / "serl_launcher",
    _REPO_ROOT / "examples",
    _WORKSPACE_ROOT / "test_teleop" / "python",
    _WORKSPACE_ROOT / "test_teleop" / "demo",
]:
    _p_str = str(_p)
    if _p.exists() and _p_str not in sys.path:
        sys.path.insert(0, _p_str)

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

def leave_z(state):
    # return state
    # print("state",state)
    state[:,:2] = 0
    state[:,3:] = 0

    # TODO: remove it after test.
    # state[:,2] = 0
    return state

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    
    obs, info = env.reset()
    obs['state'] = leave_z(obs['state'])

    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    episode_count = 0
    
    LOOP_DURATION = 0.1

    while success_count < success_needed:
        loop_start_time = time.time()

        actions = np.zeros(env.action_space.sample().shape) 
        time1 = time.time()
        next_obs, rew, done, truncated, info = env.step(actions)
        next_obs['state'] = leave_z(next_obs['state'])
        # print("original STEP-TIME = ",time.time()-time1,flush=True)
        # elapsed = time.time() - loop_start_time
        # if elapsed < LOOP_DURATION:
        #     # 如果运行太快，就睡够剩下的时间
        #     time.sleep(LOOP_DURATION - elapsed)
        
        # # 打印整个 STEP 的总耗时（现在应该是 0.1s 左右）
        # print(f"now STEP-TIME = {time.time() - loop_start_time:.4f}s", flush=True)
        returns += rew
        if "intervene_action" in info:
            actions = info["intervene_action"]
        actions_copy = copy.deepcopy(actions)
        actions_copy[3:] = 0.0
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions_copy,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )
        )
        trajectory.append(transition)
        
        pbar.set_description(f"Return: {returns}")

        obs = next_obs
        if done:
            episode_count += 1
            episode_steps = len(trajectory)
            episode_success = bool(info.get("succeed", False))

            if episode_success:
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                success_count += 1
                pbar.update(1)

            print(
                f"[Episode {episode_count}] "
                f"steps={episode_steps}, return={float(returns):.3f}, "
                f"succeed={episode_success}, truncated={truncated}, "
                f"successes={success_count}/{success_needed}"
            )
            trajectory = []
            returns = 0
            obs, info = env.reset()
            obs['state'] = leave_z(obs['state'])
            print(obs['state'])
            print("RESET-TIME = ",time.time()-time1,flush=True)

    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

if __name__ == "__main__":
    app.run(main)

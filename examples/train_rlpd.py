#!/usr/bin/env python3

import glob
import time
import threading
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl

from jax.sharding import NamedSharding, Mesh
from jax.sharding import PartitionSpec as P

from collections import deque
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 25, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean(
    "show_runtime_q",
    False,
    "Overlay critic Q estimates on the live observation display during actor runs.",
)
flags.DEFINE_integer(
    "show_runtime_q_every",
    1,
    "Compute/display runtime Q every N actor steps when --show_runtime_q is enabled.",
)

flags.DEFINE_boolean("wandb", False, "Use wandb to log training process.")
flags.DEFINE_string("wandb_log_path", '/home/wandb_data', "Path to save wandb log.")
flags.DEFINE_string("wandb_mode", 'offline', "Wandb log mode(online/offline).")

flags.DEFINE_multi_string("replay_intervention_path", None, "Path to the replay data.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_boolean(
    "save_local_plots",
    True,
    "Save local metric plots (both SVG and PNG) from learner stats callback.",
)
flags.DEFINE_string(
    "local_plot_dir",
    "",
    "Directory to save local metric plots. Default: <checkpoint_path>/local_plots",
)
flags.DEFINE_multi_string(
    "local_plot_keys",
    [
        "environment/episode/r",
        "environment/episode/l",
        "environment/episode/t",
        "environment/episode/intervention_steps",
        "environment/episode/intervention_count",
        "environment/episode/intervention_rate",
        "task/recent_success_rate",
    ],
    "Metric keys to export to local plots.",
)
flags.DEFINE_integer(
    "local_plot_max_points",
    5000,
    "Max history points kept per metric for local plotting.",
)


devices = jax.local_devices()
num_devices = len(devices)
mesh = Mesh(devices, axis_names=('device',))
sharding = NamedSharding(mesh, P())


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def _flatten_dict(d: dict, parent_key: str = "") -> dict:
    flat = {}
    for key, value in d.items():
        new_key = f"{parent_key}/{key}" if parent_key else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, new_key))
        else:
            flat[new_key] = value
    return flat


def _to_scalar_float(value):
    if isinstance(value, (bool, int, float, np.number)):
        return float(value)
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.reshape(-1)[0])
        return None
    return None


def _format_runtime_q(value) -> str:
    scalar = _to_scalar_float(value)
    return "None" if scalar is None else f"{scalar:.4f}"


def _set_runtime_q_overlay(env, step: int, q_info: dict | None) -> None:
    base_env = getattr(env, "unwrapped", env)
    setter = getattr(base_env, "set_display_overlay", None)
    if setter is None:
        return
    if not q_info:
        setter(None)
        return
    setter(
        [
            f"step={step}",
            #(
                f"Qmin={_format_runtime_q(q_info.get('q_min'))} ",
                f"Qmean={_format_runtime_q(q_info.get('q_mean'))} ",
                f"Qmax={_format_runtime_q(q_info.get('q_max'))}",
        #    ),
        ]
    )


def _prepare_runtime_q_actions(observations: dict, actions) -> np.ndarray:
    action_array = np.asarray(actions)
    if action_array.ndim == 2 and action_array.shape[0] == 1:
        return action_array[0]
    return action_array


class LocalMetricPlotter:
    def __init__(self, output_dir: str, keys: list[str], max_points: int = 5000):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.keys = [str(k) for k in keys] if keys else []
        self.max_points = max(100, int(max_points))
        self.history = {k: {"x": [], "y": []} for k in self.keys}
        self.lock = threading.Lock()
        self.enabled = True
        self._plt = None
        self._load_matplotlib()

    def _load_matplotlib(self):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            self._plt = plt
        except Exception as e:
            self.enabled = False
            print(f"[LocalPlotter] disabled: matplotlib import failed: {e}", flush=True)

    @staticmethod
    def _safe_metric_name(metric_key: str) -> str:
        return metric_key.replace("/", "__")

    def _trim(self, arr: list):
        if len(arr) > self.max_points:
            del arr[: len(arr) - self.max_points]

    def update(self, payload: dict, step: int):
        if (not self.enabled) or (self._plt is None):
            return
        if not isinstance(payload, dict):
            return

        flat = _flatten_dict(payload)
        touched = []
        with self.lock:
            for key in self.keys:
                if key not in flat:
                    continue
                val = _to_scalar_float(flat[key])
                if val is None:
                    continue
                self.history[key]["x"].append(int(step))
                self.history[key]["y"].append(val)
                self._trim(self.history[key]["x"])
                self._trim(self.history[key]["y"])
                touched.append(key)

        for key in touched:
            self._save_metric_plot(key)

    def _save_metric_plot(self, key: str):
        with self.lock:
            xs = self.history[key]["x"].copy()
            ys = self.history[key]["y"].copy()
        if len(xs) == 0:
            return

        fig, ax = self._plt.subplots(figsize=(8.0, 4.8), dpi=160)
        ax.plot(xs, ys, linewidth=1.6)
        ax.set_title(key)
        ax.set_xlabel("learner_step")
        ax.set_ylabel(key.split("/")[-1])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe_name = self._safe_metric_name(key)
        svg_path = os.path.join(self.output_dir, f"{safe_name}.svg")
        png_path = os.path.join(self.output_dir, f"{safe_name}.png")
        fig.savefig(svg_path, format="svg")
        fig.savefig(png_path, format="png")
        self._plt.close(fig)


##############################################################################
def leave_z(state):
    state[:,:2] = 0
    state[:,3:] = 0

    # TODO: remove it after test.
    # state[:,2] = 0
    return state

def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        time_consumed = 0.0
        skip_regrasp = False
        # for episode in range(FLAGS.eval_n_trajs):
        episode = 0
        while episode < FLAGS.eval_n_trajs:
            obs, _ = env.reset(options = {
                        "skip_regrasp": skip_regrasp
                    })
            # obs['state'] *= 0.0
            obs['state'] = leave_z(obs['state'])

            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=False,
                    seed=key
                )
                actions = np.asarray(jax.device_get(actions))

                next_obs, reward, done, truncated, info = env.step(actions)
                # print("state_rlpd22:",obs['state'],flush=True)
                # next_obs['state'] *= 0.0
                next_obs['state'] = leave_z(next_obs['state'])

                obs = next_obs

                if done:
                    dt = time.time() - start_time
                    time_consumed += dt
                    if reward:
                        time_list.append(dt)
                        print(dt)

                    if info['succeed']:
                        success_counter += 1 # reward # TODO if other reward

                    skip_regrasp = info.get("skip_regrasp", False)
                    if skip_regrasp:
                        episode -= 1
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

            episode += 1
    
        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {time_consumed / FLAGS.eval_n_trajs}")

        env.go_to_rest()
        # env.go_to_reset()
        return  # after done eval, return and exit
    
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:])
        + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []
    suceess_q = deque(maxlen=FLAGS.eval_n_trajs)
    task_info = {}

    obs, _ = env.reset()
    # obs['state'] *= 0.0
    obs['state'] = leave_z(obs['state'])
    # print("state_rlpd33:",obs['state'],flush=True)
    done = False

    print(type(env))

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))
            if FLAGS.show_runtime_q and step % max(1, FLAGS.show_runtime_q_every) == 0:
                q_actions = _prepare_runtime_q_actions(obs, actions)
                q_info = jax.device_get(
                    agent.evaluate_actions_q(
                        observations=jax.device_put(obs),
                        actions=jax.device_put(q_actions),
                    )
                )
                _set_runtime_q_overlay(env, step, q_info)

        # Step environment
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(actions)
            # next_obs['state'] *= 0.0
            next_obs['state'] = leave_z(next_obs['state'])
            # print_green(f"state_rlpd111{obs['state']}.")

            # print("state_rlpd444:",obs['state'],flush=True)
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            # override the action with the intervention action
            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False
            # print("action: ",actions,flush=True)
            running_return += reward
            actions_copy = copy.deepcopy(actions)
            actions_copy[3:] = 0.0
            # print("action_copy: ",actions_copy,flush=True)

            transition = dict(
                observations=obs,
                actions=actions_copy,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            if 'grasp_penalty' in info:
                transition['grasp_penalty']= info['grasp_penalty']
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))
            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs
            if done or truncated:
                succeed = float(info.get("succeed", False))
                suceess_q.append(succeed) # TODO if other reward
                if len(suceess_q) >= suceess_q.maxlen * 0.5:
                    task_info['recent_success_rate'] = sum(suceess_q) / len(suceess_q)
                    task_stats = {"task": task_info}  # send stats to the learner to log
                    client.request("send-stats", task_stats)
                    print("recent_success_rate: ", task_info['recent_success_rate'], flush=True)

                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                ###加入介入率###
                ep_len = info.get("episode", {}).get("l", 0)
                # print("episode total steps:", ep_len, flush=True)
                ep_len = int(np.asarray(ep_len).reshape(-1)[0]) if np.size(ep_len) else int(ep_len)
                info["episode"]["intervention_rate"] = float(intervention_steps) / max(1, ep_len)

                stats = {"environment": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}, last state: {obs['state'][0][0]:.2f}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                obs, _ = env.reset(
                    options = {
                        "skip_regrasp": info.get("skip_regrasp", False)
                    })

                # print("state_rlpd55:",obs['state'],flush=True)
                # obs['state'] *= 0.0
                obs['state'] = leave_z(obs['state'])

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            # dump to pickle file
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path):
                os.makedirs(demo_buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(
                os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
            ) as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:])
        + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step

    local_plotter = None
    # if FLAGS.save_local_plots:
    #     if FLAGS.local_plot_dir:
    #         local_plot_dir = FLAGS.local_plot_dir
    #     elif FLAGS.checkpoint_path:
    #         local_plot_dir = os.path.join(
    #             os.path.abspath(FLAGS.checkpoint_path), "local_plots"
    #         )
    #     else:
    #         local_plot_dir = os.path.abspath("./local_plots")
    #     local_plotter = LocalMetricPlotter(
    #         output_dir=local_plot_dir,
    #         keys=list(FLAGS.local_plot_keys),
    #         max_points=FLAGS.local_plot_max_points,
    #     )
    #     print(
    #         f"[LocalPlotter] enabled. Exporting SVG+PNG to: {local_plot_dir}",
    #         flush=True,
    #     )

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        if local_plotter is not None:
            local_plotter.update(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding,
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding,
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    
    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
        replay_intervention_files=FLAGS.replay_intervention_path,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':   
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            resnet_param_fixed=config.resnet_param_fixed,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            resnet_param_fixed=config.resnet_param_fixed,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            resnet_param_fixed=config.resnet_param_fixed,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree.map(jnp.array, agent), sharding
    )

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        ckpt_number = os.path.basename(
            checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        )[11:]
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    def create_replay_buffer_and_wandb_logger(use_wandb=True):
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        if use_wandb:
            # set up wandb and logging
            wandb_logger = make_wandb_logger(
                project="hil-serl",
                description=FLAGS.exp_name,
                debug=FLAGS.debug,
                local_output_dir=FLAGS.wandb_log_path,
                mode=FLAGS.wandb_mode,
            )
            return replay_buffer, wandb_logger
        else:
            return replay_buffer, None

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding)
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger(use_wandb=FLAGS.wandb)
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding)
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)

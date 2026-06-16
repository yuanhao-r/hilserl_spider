# !/usr/bin/env python3

from functools import partial

import jax
from jax import nn
import jax.numpy as jnp

from agentlace.trainer import TrainerConfig

from serl_launcher.common.typing import Batch, PRNGKey
from serl_launcher.common.wandb import WandBLogger
from serl_launcher.agents.continuous.bc import BCAgent
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.vision.data_augmentations import batched_random_crop, color_transform

##############################################################################


def make_bc_agent(
    seed, 
    sample_obs, 
    sample_action, 
    image_keys=("image",), 
    encoder_type="resnet-pretrained"
):
    return BCAgent.create(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [512, 512, 512],
            "dropout_rate": 0.25,
        },
        policy_kwargs={
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        use_proprio=True,
        encoder_type=encoder_type,
        image_keys=image_keys,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )


def make_sac_pixel_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    resnet_param_fixed=True,
):
    agent = SACAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
        resnet_param_fixed=resnet_param_fixed,
    )
    return agent


def make_sac_pixel_agent_hybrid_single_arm(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    resnet_param_fixed=True,
):
    agent = SACAgentHybridSingleArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
        resnet_param_fixed=resnet_param_fixed,
    )
    return agent


def make_sac_pixel_agent_hybrid_dual_arm(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image",),
    encoder_type="resnet-pretrained",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    resnet_param_fixed=True,
):
    agent = SACAgentHybridDualArm.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        grasp_critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
        resnet_param_fixed=resnet_param_fixed
    )
    return agent


def linear_schedule(step):
    init_value = 10.0
    end_value = 50.0
    decay_steps = 15_000


    linear_step = jnp.minimum(step, decay_steps)
    decayed_value = init_value + (end_value - init_value) * (linear_step / decay_steps)
    return decayed_value
    
@partial(
    jax.jit,
    static_argnames=(
        "brightness",
        "contrast",
        "saturation",
        "hue",
        "to_grayscale_prob",
        "color_jitter_prob",
        "apply_prob",
        "shuffle",
        "num_batch_dims",
    ),
)
def batched_color_transform(
    img,
    rng,
    *,
    brightness=0.3,
    contrast=0.35,
    saturation=0.3,
    hue=0.05,
    to_grayscale_prob=0.0,
    color_jitter_prob=0.8,
    apply_prob=0.8,
    shuffle=True,
    num_batch_dims: int = 1,
):
    original_shape = img.shape
    original_dtype = img.dtype
    img = jnp.reshape(img, (-1, *img.shape[num_batch_dims:]))
    rngs = jax.random.split(rng, img.shape[0])

    def transform_one(image, key):
        image_float = image.astype(jnp.float32) / 255.0
        image_float = color_transform(
            image_float,
            key,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            to_grayscale_prob=to_grayscale_prob,
            color_jitter_prob=color_jitter_prob,
            apply_prob=apply_prob,
            shuffle=shuffle,
        )
        image_out = jnp.clip(image_float * 255.0, 0.0, 255.0)
        return image_out.astype(original_dtype)

    img = jax.vmap(transform_one, in_axes=(0, 0), out_axes=0)(img, rngs)
    return jnp.reshape(img, original_shape)


def make_batch_augmentation_func(image_keys) -> callable:

    def data_augmentation_fn(rng, observations):
        rngs = jax.random.split(rng, len(image_keys) * 2)
        for pixel_key in image_keys:
            crop_rng, color_rng = rngs[:2]
            rngs = rngs[2:]
            pixels = batched_random_crop(
                observations[pixel_key], crop_rng, padding=4, num_batch_dims=2
            )
            # pixels = batched_color_transform(
            #     pixels,
            #     color_rng,
            #     brightness=0.2,
            #     contrast=0.2,
            #     saturation=0.2,
            #     hue=0.05,
            #     to_grayscale_prob=0.0,
            #     color_jitter_prob=0.8,
            #     apply_prob=0.8,
            #     shuffle=True,
            #     num_batch_dims=2,
            # )
            observations = observations.copy(
                add_or_replace={
                    pixel_key: pixels,
                }
            )
        return observations
    
    def augment_batch(batch: Batch, rng: PRNGKey) -> Batch:
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = data_augmentation_fn(next_obs_rng, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "next_observations": next_obs,
            }
        )
        return batch
    
    return augment_batch


def make_trainer_config(port_number: int = 5588, broadcast_port: int = 5589):
    return TrainerConfig(
        port_number=port_number,
        broadcast_port=broadcast_port,
        request_types=["send-stats"],
    )


def make_wandb_logger(
    project: str = "hil-serl",
    description: str = "serl_launcher",
    debug: bool = False,
    local_output_dir: str = "",
    mode: str = "online",
):
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "exp_descriptor": description,
            "tag": description,
            "local_output_dir": local_output_dir,
            "mode": mode,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        debug=debug,
    )
    return wandb_logger

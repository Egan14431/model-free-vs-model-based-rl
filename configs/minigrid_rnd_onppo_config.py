from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 10
task = 'MiniGrid-DoorKey-8x8-v0'
seed = 6

minigrid_ppo_rnd_config = dict(
    exp_name=f'experiments/minigrid/{task}_rnd_onppo_seed{seed}',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        env_id=task,
        max_step=300,
        stop_value=20,  
        full_obs=False,
        onehot_obs = False,
        move_bonus = False,
        flat_obs=True,
    ),
    reward_model=dict(
        intrinsic_reward_type='add',
        # intrinsic_reward_weight means the relative weight of RND intrinsic_reward.
        # Specifically for sparse reward env MiniGrid, in this env,
        # if reach goal, the agent get reward ~1, otherwise 0,
        # We could set the intrinsic_reward_weight approximately equal to the inverse of max_episode_steps.
        # Please refer to rnd_reward_model for details.
        intrinsic_reward_weight=0.003,  # 1/300
        learning_rate=3e-4,
        obs_shape=2835,#192,#2835,
        batch_size=320,
        update_per_collect=50,
        clear_buffer_per_iters=int(1e3),
        obs_norm=False,
        obs_norm_clamp_max=5,
        obs_norm_clamp_min=-5,
        extrinsic_reward_norm=True,
        extrinsic_reward_norm_max=1,
    ),
    policy=dict(
        recompute_adv=True,
        cuda=True,
        action_space='discrete',
        model=dict(
            obs_shape=2835,
            action_shape=7,
            action_space='discrete',
            encoder_hidden_size_list=[256, 128, 64, 64],
            critic_head_hidden_size=64,
            actor_head_hidden_size=64,
        ),
        learn=dict(
            epoch_per_collect=10,
            update_per_collect=1,
            batch_size=320,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
        ),
        collect=dict(
            collector_env_num=collector_env_num,
            n_sample=3200,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, )),
    ),
)
minigrid_ppo_rnd_config = EasyDict(minigrid_ppo_rnd_config)
main_config = minigrid_ppo_rnd_config
minigrid_ppo_rnd_create_config = dict(
    env=dict(
        type='minigrid',
        import_names=['dizoo.minigrid.envs.minigrid_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
    reward_model=dict(type='rnd'),
)
minigrid_ppo_rnd_create_config = EasyDict(minigrid_ppo_rnd_create_config)
create_config = minigrid_ppo_rnd_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_reward_model_onpolicy
    serial_pipeline_reward_model_onpolicy([main_config, create_config], seed=seed, max_env_step=int(10e6))

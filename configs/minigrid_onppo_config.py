from easydict import EasyDict

collector_env_num = 8
task = 'MiniGrid-DoorKey-8x8-v0'
seed = 84
max_env_step = 2e5


minigrid_ppo_config = dict(
    exp_name=f'experiments/minigrid/{task}_onppo_seed{seed}',  
    env=dict(
        collector_env_num=8,
        evaluator_env_num=10,
        n_evaluator_episode=10,
        env_id=task,
        max_step=300,
        stop_value=10,    
        full_obs=False, 
        onehot_obs = False,
        move_bonus = False,
        flat_obs=True,
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        action_space='discrete',
        model=dict(
            obs_shape=2835,
            action_shape=7,
            action_space='discrete',
            encoder_hidden_size_list=[256, 128, 64, 64],
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
            n_sample=int(3200),
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=500, )),    # NOTE Added, was 1000
    ),
)
minigrid_ppo_config = EasyDict(minigrid_ppo_config)
main_config = minigrid_ppo_config
minigrid_ppo_create_config = dict(
    env=dict(
        type='minigrid',
        import_names=['dizoo.minigrid.envs.minigrid_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
minigrid_ppo_create_config = EasyDict(minigrid_ppo_create_config)
create_config = minigrid_ppo_create_config

if __name__ == "__main__":
    seeds = [20,30,40,50,60]
    for seed in seeds:
        main_config.exp_name = f'experiments/minigrid/{task}_onppo_seed{seed}'
        from ding.entry import serial_pipeline_onpolicy
        serial_pipeline_onpolicy([main_config, create_config], seed=seed, max_env_step=max_env_step)
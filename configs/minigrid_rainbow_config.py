from easydict import EasyDict

collector_env_num = 10
evaluator_env_num = 10
task = 'MiniGrid-DoorKey-8x8-v0'
seed = 999
max_env_step = 2e5

minigrid_rainbow_config = dict(
    exp_name=f'experiments/minigrid/{task}_rainbow_seed{seed}',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        env_id=task,       
        max_step=300,
        stop_value=20,  # run fixed env_steps
        full_obs=False,
        onehot_obs = False,
        move_bonus = False,
        flat_obs=True,
    ),
    policy=dict(
        cuda=True,
        priority=True,
        priority_IS_weight=True,
        model=dict(
            obs_shape=2835,   
            action_shape=7,
            encoder_hidden_size_list=[256, 128, 64],
            v_min=-1,
            v_max=1,
            n_atom=51,
        ),
        nstep=3,
        discount_factor=0.99,
        noisy_net=True,
        learn=dict(
            update_per_collect=20,
            batch_size=64,
            learning_rate=0.0003,
            target_update_freq=100,
        ),
        collect=dict(n_sample=40, unroll_len=1,),
        eval=dict(evaluator=dict(eval_freq=1000, )),
        other=dict(
            eps=dict(      
                type='linear',
                start=0.95,
                end=0.05,
                decay=2e4,
            ),
            replay_buffer=dict(replay_buffer_size=20000),
        ),
    ),
)

minigrid_rainbow_config = EasyDict(minigrid_rainbow_config)
main_config = minigrid_rainbow_config
minigrid_rainbow_create_config = dict(
    env=dict(
        type='minigrid',
        import_names=['dizoo.minigrid.envs.minigrid_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='rainbow'),
)
minigrid_rainbow_create_config = EasyDict(minigrid_rainbow_create_config)
create_config = minigrid_rainbow_create_config

if __name__ == "__main__":
    seeds = [20,30,40,50,60]
    for seed in seeds:
        main_config.exp_name = f'experiments/minigrid/{task}_rainbow_seed{seed}'
        from ding.entry import serial_pipeline
        serial_pipeline([main_config, create_config], seed=seed, max_env_step=max_env_step)
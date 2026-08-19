from easydict import EasyDict
seed = 20
task = 'MsPacmanNoFrameskip-v4'
max_env_step = 1e6


qbert_rainbow_config = dict(
    exp_name=f'experiments/atari/{task}_rainbow_seed{seed}',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=10,
        n_evaluator_episode=10,
        stop_value=30000,
        env_id=task,
        #'ALE/Qbert-v5' is available. But special setting is needed after gym make.
        frame_stack=4
    ),
    policy=dict(
        cuda=True,
        priority=True,
        priority_IS_weight=True,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=9,
            encoder_hidden_size_list=[128, 128, 512],
            v_min=-10,
            v_max=10,
            n_atom=51,
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=25,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
            iqn=False,
        ),
        collect=dict(n_sample=100, ),
        eval=dict(evaluator=dict(eval_freq=4000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.05,
                end=0.05,
                decay=1000000,
            ),
            replay_buffer=dict(replay_buffer_size=100_000, ),
        ),
    ),
)
main_config = EasyDict(qbert_rainbow_config)

qbert_rainbow_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='rainbow'),
)
create_config = EasyDict(qbert_rainbow_create_config)

if __name__ == '__main__':
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=seed, max_env_step=max_env_step)

from easydict import EasyDict
max_env_step = 2e5

mtcar_ppo_config = dict(
    exp_name='cartpole_ppo_seed999',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=10,
        n_evaluator_episode=10,
        stop_value=2000,
    ),
    policy=dict(
        cuda=True,
        action_space='discrete',
        model=dict(
            obs_shape=2,
            action_shape=3,
            action_space='discrete',
            encoder_hidden_size_list=[64, 64, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            # Path to the pretrained checkpoint (ckpt).
            # If set to an empty string (''), no pretrained model will be loaded.
            # To load a pretrained ckpt, specify the path like this:
            # learner=dict(hook=dict(load_ckpt_before_run='/path/to/your/ckpt/iteration_100.pth.tar')),

            # If True, the environment step count (collector.envstep) and training iteration (train_iter)
            # will be loaded from the pretrained checkpoint, allowing training to resume seamlessly
            # from where the ckpt left off.
            resume_training=False,
        ),
        collect=dict(
            n_sample=2048,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
    ),
)

mtcar_ppo_config = EasyDict(mtcar_ppo_config)
main_config = mtcar_ppo_config
mtcar_ppo_create_config =  dict(
    env=dict(
        type='mountain_car',
        import_names=['dizoo.classic_control.mountain_car.envs.mtcar_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
)
mtcar_ppo_create_config = EasyDict(mtcar_ppo_create_config)
create_config = mtcar_ppo_create_config

if __name__ == "__main__":
    seeds = [20,30,40,50,60]
    for seed in seeds:
        main_config.exp_name = f'experiments/mtcar/mtcar_ppo_seed{seed}'
        from ding.entry import serial_pipeline_onpolicy
        serial_pipeline_onpolicy((main_config, create_config), seed=seed, max_env_step=max_env_step)
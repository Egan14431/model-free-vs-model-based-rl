from easydict import EasyDict

env_id = 'MiniGrid-DoorKey-8x8-v0'
max_env_step = int(1e5)
max_episode_steps = 300

seed = 999
collector_env_num = 8
n_episode = 8
evaluator_env_num = 10
num_simulations = 50
update_per_collect = 200
batch_size = 256
replay_ratio = 0.25
reanalyze_ratio = 0
td_steps = 5
policy_entropy_weight = 0.  # 0.005
threshold_training_steps_for_final_temperature = int(5e5)
eps_greedy_exploration_in_collect = False


minigrid_muzero_config = dict(
    exp_name=f'data_muzero/{env_id}_muzero_ns{num_simulations}_upc{update_per_collect}_rer{reanalyze_ratio}_'
             f'collect-eps-{eps_greedy_exploration_in_collect}_temp-final-steps-{threshold_training_steps_for_final_temperature}_pelw{policy_entropy_weight}_seed{seed}',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        continuous=False,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        max_step=max_episode_steps,
    ),
    policy=dict(
        model=dict(
            observation_shape=2835,
            action_space_size=7,
            model_type='mlp',
            lstm_hidden_size=256,
            latent_state_dim=512,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            self_supervised_learning_loss=True,  # NOTE: default is False.
        ),
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            decay=int(2e5),
        ),
        policy_entropy_weight=policy_entropy_weight,
        td_steps=td_steps,
        manual_temperature_decay=True,
        threshold_training_steps_for_final_temperature=threshold_training_steps_for_final_temperature,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=300,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        # Added the below after PR #497
        optim_type='AdamW',#'Adam', # NOTE: Changed as per PR497
        weight_decay=1e-4,          # NOTE: Added for above. Remove if using Adam
        use_priority=True,
        use_max_priority_for_new_data=True,
        priority_prob_alpha=0.6,
        priority_prob_beta=0.4,

        piecewise_decay_lr_scheduler=False,
        learning_rate=0.003,
        ssl_loss_weight=2,  # NOTE: default is 0.
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(1000),
        replay_buffer_size=int(1e5),  # the size/capacity of replay_buffer, in the terms of transitions. NOTE: Changed to 1e5
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)

minigrid_muzero_config = EasyDict(minigrid_muzero_config)
main_config = minigrid_muzero_config

minigrid_muzero_create_config = dict(
    env=dict(
        type='minigrid_lightzero',
        import_names=['zoo.minigrid.envs.minigrid_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
    collector=dict(
        type='episode_muzero',
        import_names=['lzero.worker.muzero_collector'],
    )
)
minigrid_muzero_create_config = EasyDict(minigrid_muzero_create_config)
create_config = minigrid_muzero_create_config

if __name__ == "__main__":
    
    seeds = [20,30,40,50,60]
    for seed in seeds:
        main_config.exp_name = f'experiments/minigrid/logs/{env_id}_muzero_ns{num_simulations}_upc{update_per_collect}_seed{seed}'

        from lzero.entry import train_muzero
        train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)

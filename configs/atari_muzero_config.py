from easydict import EasyDict
from zoo.atari.config.atari_env_action_space_map import atari_env_action_space_map
norm_type = 'BN'
env_id = 'PongNoFrameskip-v4'  # You can specify any Atari game here
action_space_size = atari_env_action_space_map[env_id]


collector_env_num = 8
n_episode = 8
evaluator_env_num = 10
num_simulations = int(50)   
update_per_collect = None   # NOTE: See Appendix D Data Generation of MuZero paper
replay_ratio = 0.25
batch_size = 256
max_env_step = int(1e6)
reanalyze_ratio = 0.
eps_greedy_exploration_in_collect = False
seed = 20


atari_muzero_config = dict(
    exp_name=f'experiments/atari/logs/{env_id[:-14]}_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_seed{seed}_priority',
    env=dict(
        stop_value=int(1e6),
        env_id=env_id,
        observation_shape=(4, 96, 96),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        collect_max_episode_steps=int(5e3),
        eval_max_episode_steps=int(5e3),
    ),
    policy=dict(
        model=dict(
            observation_shape=(4, 96, 96),
            frame_stack_num=4,
            action_space_size=6,
            downsample=True,
            self_supervised_learning_loss=False,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            use_sim_norm=True,
            model_type='conv'
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=200,
        random_collect_episode_num=0,
        eps=dict(
            eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
            # need to dynamically adjust the number of decay steps 
            # according to the characteristics of the environment and the algorithm
            type='linear',
            start=1.,
            end=0.05,
            decay=int(1e5),
        ),
        use_augmentation=False,
        use_priority=True,       
        priority_prob_alpha=0.6, # NOTE: Default 0.6
        priority_prob_beta=0.4,  # NOTE: Default 0.4
        replay_ratio=replay_ratio,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='SGD',
        piecewise_decay_lr_scheduler=True,
        learning_rate=0.2,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        ssl_loss_weight=0,  
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        mcts_ctree=False,
        td_steps=5,            # NOTE: Default is 5 but original MuZero paper uses 10 without Reanalyze
        value_loss_weight=0.25,  # NOTE: Reanalyze uses 0.25, original uses 1. If unstable change back to 0.25
        manual_temperature_decay=True,
        threshold_training_steps_for_final_temperature=int(6e4),
    ),
)
atari_muzero_config = EasyDict(atari_muzero_config)
main_config = atari_muzero_config

atari_muzero_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
atari_muzero_create_config = EasyDict(atari_muzero_create_config)
create_config = atari_muzero_create_config

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=seed, max_env_step=max_env_step)#, model_path=model_path)

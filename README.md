# model-free-vs-model-based-rl

This repository contains the experimental configurations, output data, and agent videos produced as part of the MSc dissertation:

**A Comparison of Model-free and Model-based Deep Reinforcement Learning across Benchmark Environments**

## Repository Contents

### `configs/`

Contains the configuration files used to run the experiments.

These files provide the experimental settings used for the different algorithms and environments, including the configurations used for:

* **MuZero**
* **PPO**
* **Rainbow DQN**

The configurations correspond to the experiments described in the dissertation. They must be used in conjunction with the DI-Engine and LightZero frameworks.

### `outputs/`

Contains the CSV output files generated from the experiments.

These files contain the recorded experimental results used for analysing algorithm performance and producing the results presented in the dissertation.

The output files are provided as the raw experimental results rather than as a separate analysis or software package.

### `videos/`

Contains videos showing agents trained during the experiments interacting with their respective environments.

These videos provide qualitative examples of the behaviours learned by the agents and supplement the quantitative results reported in the dissertation.

## Environments

The experiments cover three reinforcement learning domains:

* **Classic Control**

  * CartPole
  * Mountain Car
* **Atari**

  * Ms. Pac-Man
  * Pong
* **MiniGrid**

  * Empty-8x8
  * DoorKey-5x5
  * DoorKey-6x6
  * DoorKey-8x8
  * KeyCorridorS3R3

## Algorithms

The experiments compare:

| Algorithm   | Approach                           |
| ----------- | ---------------------------------- |
| MuZero      | Model-based reinforcement learning |
| PPO         | Model-free reinforcement learning  |
| Rainbow DQN | Model-free reinforcement learning  |

## Purpose of the Repository

This repository is intended as a supporting archive for the dissertation experiments. It is not intended to be a standalone implementation or reproduction framework.

The files are provided to document:

1. The configurations used for the experiments.
2. The experimental results produced by those configurations.
3. Examples of the behaviour exhibited by the trained agents.

Together, these files provide supporting evidence for the experimental results and conclusions presented in the dissertation.


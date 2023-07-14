# RL Assignments

Template code for reinforcement learning assignments.

## Getting Started

Install requirements:

```
conda create -n rl python=3.8
conda activate rl
pip install -r requirements.txt
```



and it should be good to go.


## Train Models

To train a policy, run the following command:

``python run.py --env_config data/envs/dm_cheetah.yaml --agent_config a2/dm_cheetah_cem_agent.yaml --mode train --log_file output/log.txt --out_model_file output/model.pt --visualize``

- `--env_config` specifies the configuration file for the environment.
- `--agent_config` specifies configuration file for the agent.
- `--visualize` enables visualization. Rendering should be disabled for faster training.
- `--log_file` specifies the output log file, which will keep track of statistics during training.
- `--out_model_file` specifies the output model file, which contains the model parameters.

## Test Models

To load a trained model, run the following command:

``python run.py --env_config data/envs/dm_cheetah_env.yaml --agent_config a2/dm_cheetah_cem_agent.yaml --mode test --model_file data/models/dm_cheetah_ppo_model.pt --visualize``

- `--model_file` specifies the `.pt` file that contains parameters for the trained model. Pretrained models are available in `data/models/`.


## Visualizing Training Logs

During training, a tensorboard `events` file will be saved the same output directory as the log file. The log can be viewed with:

``tensorboard --logdir=output/ --port=6006 --bind_all``


The output log `.txt` file can also be plotted using the plotting script in `tools/plot_log/plot_log.py`.


## Running Command

1. 
2. For Cross Entropy Method under DeepMind cheetah environment:
python run.py  --mode test --env_config data/envs/dm_cheetah.yaml --agent_config a2/dm_cheetah_cem_agent.yaml --model_file a2/cheetah_cem_model.pt --visualize 
3. For Policy Gradient Method under DeepMind cheetah environment:
python run.py  --mode test --env_config data/envs/dm_cheetah.yaml --agent_config a2/dm_cheetah_pg_agent.yaml --model_file a2/cheetah_pg_model.pt --visualize
4. For Q-learning method under Atari Pong environment:
python run.py --mode test --env_config data/envs/atari_pong.yaml --agent_config a3/atari_dqn_agent.yaml --model_file a3/atari_pong_dqn_model.pt --test_episodes 20 --visualize
5. For Q-learning method under Atari Breakout environment:
python run.py --mode test --env_config data/envs/atari_breakout.yaml --agent_config a3/atari_dqn_agent.yaml --model_file a3/atari_breakout_dqn_model.pt --test_episodes 20 --visualize
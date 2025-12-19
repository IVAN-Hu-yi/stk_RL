
from pathlib import Path
from pystk2_gymnasium import AgentSpec
from functools import partial
import torch
import inspect
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

# Note the use of relative imports
from .actors import Actor
from .pystk_actor import env_name, get_wrappers, player_name
from config import ppoconfig
from buffers.sequenceBuffer import sequenceReplayBuffer
from buffers.types import Transition, Batch
try:
    from tqdm.auto import tqdm
except Exception:
    from tqdm import tqdm


seq_obs_keys = ['items_position', 'items_type', 'karts_position', 'paths_distance', 'paths_end', 'paths_start', 'paths_width']
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


if __name__ == "__main__":
    # Setup the environment
    make_stkenv = partial(
        make_env,
        env_name,
        wrappers=get_wrappers(),
        render_mode=None,
        autoreset=True,
        agent=AgentSpec(use_ai=False, name=player_name),
    )

    env_agent = ParallelGymAgent(make_stkenv, 1)
    env = env_agent.envs[0]

    # (2) Learn
    # get raw observation dimensions
    seq_raw_dims = {key:None for key in seq_obs_keys}
    for key, space in env.observation_space.items():
        if key in seq_obs_keys:
            seq_raw_dims[key] = space.feature_space.shape
    dfconfig = ppoconfig(device, seq_raw_dims).config
    print(f"Using device: {device}")
    print(f"PPO Config: {dfconfig}")

    actor = Actor(None, None, 'PPO', dfconfig, device)
    buffer = sequenceReplayBuffer(**dfconfig['ReplayBuffer'])
    train_cofig = dfconfig['training_config'] 
    pbar = tqdm(total=train_cofig['max_epochs']*1500, desc=f"Training Actor")
    
    # counters
    episode_reward = 0.0
    episode_counts = 0
    episode_step = 0

    # pre-allocation for Transition class
    log_prob = None
    value = None
    batch = None

    obs, _ = env.reset()
    global_step = 0

    while episode_counts < train_cofig['max_epochs']:

        # ---------------------------------------------------------
        # Interact with the environment
        # ---------------------------------------------------------
        action, log_prob, _ = actor.algo.select_action(obs, eval_mode=False)

        next_obs, reward, terminated, truncated, _ = env.step(action)

        # ---------------------------------------------------------
        # Store Transition in Replay Buffer 
        # ---------------------------------------------------------
        transition = {
            "states": obs,
            "action": action,
            "reward": reward,
            "next_states": next_obs,
            "truncated": truncated,
            "terminated": terminated,
            "log_prob": log_prob,
        }
        t = Transition(**transition)
        buffer.add(t)

        obs = next_obs
        episode_step += 1
        global_step += 1

        # ---------------------------------------------------------
        # Train the agent 
        # ---------------------------------------------------------

        if global_step % train_cofig['update_interval'] == 0:
            batch = buffer.sample(train_cofig['batch_size'])
            metrics = actor.algo.update(batch)
            buffer.clear()
            batch_reward = batch.rewards.mean().item()
            pbar.set_postfix({
                'Episodes': f'{episode_counts}/{train_cofig["max_epochs"]}',
                'Buffer': f'{buffer.size:,}',
                'current reward': f'{batch_reward:.2f}',
                'Batch Avg Reward': f'{batch_reward:.2f}',
            })

        if terminated or truncated:
            episode_counts += 1
            episode_step = 0
            obs, _ = env.reset()

        # ---------------------------------------------------------
        # Handle progress bar
        # ---------------------------------------------------------
        pbar.update(1)
        pbar.set_postfix({
            'Episodes': f'{episode_counts}/{train_cofig["max_epochs"]}',
            'Buffer': f'{buffer.size:,}',
            'current reward': f'{reward:.2f}',
        })
    

    # (3) Save the actor sate
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    torch.save(actor.state_dict(), mod_path / "pystk_actor.pth")

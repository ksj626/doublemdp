"""Original BDPO data adapters, independent of the legacy D4RL runtime.
D4RL transition/reward processing extracted verbatim from pinned FlowRL (MIT).
Only saved rewards enter agent training; analytic toy helpers are evaluation-only.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
def compute_returns(reward, end):
    return_ = []
    ep_return = 0
    for i in range(len(reward)):
        ep_return += reward[i]
        if end[i] or i == len(reward) - 1:
            return_.append(ep_return)
            ep_return = 0
    return np.array(return_)

def normalize_reward(reward, end, method="none"):
    if method == "none" or method is None:
        return reward
    elif method == "iql_mujoco":
        returns = compute_returns(reward, end)
        max_return = np.max(returns)
        min_return = np.min(returns)
        factor = 1000.0 / (max_return - min_return)
        print(f"norm reward {method}: return max = {max_return}, return min = {min_return}, factor = {factor}")
        return reward * factor
    elif method == "white":
        r_mean, r_std = np.mean(reward), np.std(reward)
        print(f"norm reward {method}: mean = {r_mean}, std = {r_std}")
        return (reward - r_mean) / r_std
    elif method == "iql_antmaze":
        return reward - 1.0
    elif method == "cql_antmaze":
        return (reward - 0.5) * 4.0
    elif method == "antmaze100":
        return reward * 100.0
    else:
        raise NotImplementedError(f"Reward normalization method {method} not implemented")

def qlearning_dataset(env, dataset=None, terminate_on_end: bool=False, discard_last: bool=True, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            next_actions: An N x dim_action array of next actions.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    next_action_ = []
    reward_ = []
    done_ = []
    end_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)  # Thus, the next_obs for the last timestep is totally false
        action = dataset['actions'][i].astype(np.float32)
        new_action = dataset['actions'][i+1].astype(np.float32)  # Thus, the next_action for the last timestep is totally false
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        end = False
        episode_step += 1

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps)
        if final_timestep:
            if not done_bool:
                if not terminate_on_end:
                    if discard_last:
                        episode_step = 0
                        end_[-1] = True
                        continue
                else:
                    done_bool = True
        if final_timestep or done_bool:
            end = True
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        next_action_.append(new_action)
        reward_.append(reward)
        done_.append(done_bool)
        end_.append(end)

    end_[-1] = True   # the last traj will be ended whatsoever
    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'next_actions': np.array(next_action_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        "ends": np.array(end_)
    }


def centers_8gaussians():
    s = 1 / np.sqrt(2)
    return np.asarray([(0,1),(-s,s),(-1,0),(-s,-s),(0,-1),(s,-s),(1,0),(s,s)]) * 4 / 1.414


def reward_8gaussians(actions):
    """Evaluation/data-generation oracle; never used by BDPOAgent.update."""
    distance = ((np.asarray(actions)[..., None, :] - centers_8gaussians())**2).sum(-1)
    return distance.argmin(-1).astype(np.float32) / 7.


def density_8gaussians(actions):
    variance = (.5 / 1.414)**2
    squared = ((np.asarray(actions)[..., None, :] - centers_8gaussians())**2).sum(-1)
    return np.exp(-squared / (2 * variance)).mean(-1) / (2*np.pi*variance)


def generate_8gaussians(size=1000000, seed=0):
    """Same order and RandomState draws as FlowRL inf_train_gen(8gaussians)."""
    rng = np.random.RandomState(seed)
    centers = centers_8gaussians() * 1.414
    actions = np.empty((size, 2), np.float32)
    for i in range(size):
        point = rng.randn(2) * .5
        point += centers[rng.randint(8)]
        actions[i] = point
    actions /= 1.414
    return {'observations': np.zeros((size, 1), np.float32), 'actions': actions,
            'rewards': reward_8gaussians(actions), 'masks': np.zeros(size, np.float32),
            'terminals': np.ones(size, bool),
            'next_observations': np.zeros((size, 1), np.float32)}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_d4rl_hdf5(path, norm_reward='iql_mujoco', clip_eps=1e-5, norm_obs=False):
    """Read an original raw HDF5; preserve shifted next observations and timeouts.

    Timeout rows are discarded exactly as in FlowRL, with the previous retained
    transition marked as an episode boundary without changing its bootstrap mask.
    The final raw row is excluded, matching the pinned source.
    """
    import h5py
    with h5py.File(path, 'r') as stream:
        raw = {name: np.asarray(stream[name]) for name in
               ('observations','actions','rewards','terminals','timeouts') if name in stream}
    processed = qlearning_dataset(SimpleNamespace(_max_episode_steps=1000), dataset=raw)
    obs, next_obs = processed['observations'], processed['next_observations']
    mean, std = np.zeros(obs.shape[-1], np.float32), np.ones(obs.shape[-1], np.float32)
    if norm_obs:
        mean, std = obs.mean(0), obs.std(0)
        std[std == 0] = 1
        obs, next_obs = (obs-mean)/std, (next_obs-mean)/std
    # Preserve the upstream float32 column shape during episode accumulation.
    rewards = normalize_reward(processed['rewards'][..., None], processed['ends'], norm_reward)[:, 0]
    result = {'observations': obs, 'actions': np.clip(processed['actions'], -(1-clip_eps), 1-clip_eps),
              'rewards': rewards, 'masks': 1-processed['terminals'].astype(np.float32),
              'terminals': processed['ends'], 'next_observations': next_obs}
    return result, {'source_sha256': file_sha256(path), 'raw_size': len(raw['rewards']),
                    'size': len(rewards), 'norm_reward': norm_reward,
                    'clip_eps': clip_eps, 'obs_mean': mean.tolist(), 'obs_std': std.tolist(),
                    'timeout_policy': 'discard timeout row; previous boundary retains bootstrap'}


class ScanSampler:
    """Upstream scan-with-reshuffle semantics, with checkpointable private RNG."""
    def __init__(self, dataset, seed=0):
        self.dataset = dataset
        self.rng = np.random.RandomState(seed)
        self.indices = np.arange(len(dataset['actions']))
        self.rng.shuffle(self.indices)
        self.position = 0

    def sample(self, batch_size):
        if self.position + batch_size >= len(self.indices):
            self.rng.shuffle(self.indices)
            self.position = 0
        indices = self.indices[self.position:self.position+batch_size]
        self.position += batch_size
        return {key: value[indices] for key, value in self.dataset.items()}

    def state_dict(self):
        return {'indices': self.indices, 'position': self.position, 'rng': self.rng.get_state()}

    def load_state_dict(self, state):
        self.indices, self.position = state['indices'], state['position']
        self.rng.set_state(state['rng'])

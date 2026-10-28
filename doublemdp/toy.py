"""Saved offline toy data; reward/dynamics oracles are never called by updates."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .data import save_dataset


@dataclass(frozen=True)
class Bandit:
    contexts: int = 3
    radius: float = .14
    gap: bool = False

    @property
    def centers(self):
        theta = np.arange(8) * np.pi / 4
        return .7 * np.stack([np.cos(theta), np.sin(theta)], -1)

    @property
    def weights(self):
        weights = np.array([.27, .23, .19, .14, .08, .05, .025, .015])
        if self.gap:
            weights[-1] = 0
        return weights / weights.sum()

    def density(self, actions):
        a = np.asarray(actions)
        inside = np.linalg.norm(a[..., None, :] - self.centers, axis=-1) <= self.radius
        return (inside * self.weights).sum(-1) / (np.pi * self.radius**2)

    def reward(self, actions, context=0):
        """Evaluation/dataset-generation oracle, never an agent training callback."""
        a = np.asarray(actions)
        centers = self.centers
        rewards = np.array([.05, .1, .12, .15, .2, .25, .7, 1.])
        context = np.asarray(context, dtype=int)
        mode = np.argmin(np.linalg.norm(a[..., None, :] - centers, axis=-1), axis=-1)
        heights = rewards[(mode - context) % 8]
        # High reward extends outside support; extrapolation is measurable.
        distance = np.min(np.linalg.norm(a[..., None, :] - centers, axis=-1), axis=-1)
        return heights * np.exp(-.5 * (distance / .20)**2)

    def sample(self, count, seed=0):
        rng = np.random.RandomState(seed)
        context = rng.randint(self.contexts, size=count)
        mode = rng.choice(8, count, p=self.weights)
        theta = rng.uniform(0, 2*np.pi, count)
        radius = self.radius * np.sqrt(rng.uniform(size=count))
        actions = self.centers[mode] + radius[:, None] * np.stack([np.cos(theta), np.sin(theta)], -1)
        obs = np.eye(self.contexts)[context]
        return {"observations": obs.astype("f"), "next_observations": obs.astype("f"),
                "actions": actions.astype("f"), "rewards": self.reward(actions, context).astype("f"),
                "masks": np.zeros(count, "f"), "terminals": np.ones(count, "f")}

    def reference_grid(self, context=0, eta=.06, resolution=161):
        if eta <= 0:
            raise ValueError("eta must be positive")
        axis = np.linspace(-1, 1, resolution)
        x, y = np.meshgrid(axis, axis)
        a = np.stack([x, y], -1)
        beta, reward = self.density(a), self.reward(a, context)
        unnormalized = beta * np.exp((reward - reward.max()) / eta)
        mass = unnormalized / unnormalized.sum()
        return {"axis": axis, "actions": a, "behavior_density": beta, "reward": reward,
                "reference_mass": mass, "eta": np.array(eta), "context": np.array(context)}


def segments_hit_boxes(starts, ends, boxes):
    """Exact slab intersection against closed rectangles, batched over segments."""
    starts, ends = np.broadcast_arrays(np.asarray(starts), np.asarray(ends))
    delta = ends - starts
    hit = np.zeros(starts.shape[:-1], dtype=bool)
    for low_x, low_y, high_x, high_y in boxes:
        low, high = np.array([low_x, low_y]), np.array([high_x, high_y])
        parallel = np.abs(delta) < 1e-14
        safe_delta = np.where(parallel, 1., delta)
        ta, tb = (low - starts) / safe_delta, (high - starts) / safe_delta
        near = np.where(parallel, -np.inf, np.minimum(ta, tb))
        far = np.where(parallel, np.inf, np.maximum(ta, tb))
        outside_parallel = np.any(parallel & ((starts < low) | (starts > high)), axis=-1)
        entry = np.maximum(0., near.max(-1))
        leave = np.minimum(1., far.min(-1))
        hit |= (entry <= leave) & ~outside_parallel
    return hit


class PointMaze:
    """Continuous continuing-task maze with true success terminals and logging timeouts.

    Geometric shaping supplies goal/map knowledge. Timeouts retain potential and
    bootstrap; no hidden finite-horizon reward or remaining-time dependence.
    """
    boxes = np.array([[.42, .18, .48, .38], [.42, .44, .48, .78],
                      [.57, .48, .80, .51], [.77, .48, .80, .70], [.57, .67, .80, .70]])
    goal = np.array([.91, .55], dtype=np.float32)
    start = np.array([.16, .55], dtype=np.float32)
    action_scale = .06
    # Keep the near-goal probe outside the genuine terminal region (radius .055).
    probes = np.array([[.32,.48], [.39,.41], [.52,.85], [.84,.55]], dtype=np.float32)

    def __init__(self, *, max_steps=150, gamma=.99, reward="sparse"):
        from gymnasium.spaces import Box
        self.observation_space = Box(0., 1., (2,), dtype=np.float32)
        self.action_space = Box(-1., 1., (2,), dtype=np.float32)
        self.max_steps, self.gamma, self.reward_mode = max_steps, gamma, reward
        if reward not in ("sparse", "dense"):
            raise ValueError(reward)
        self.position = self.start.copy()
        self.elapsed = 0
        self.rng = np.random.RandomState(0)

    @property
    def unwrapped(self):
        return self

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.position = np.array((options or {}).get("position", self.start), dtype=np.float32)
        self.elapsed = 0
        return self.position.copy(), {}

    def potential(self, positions):
        return -np.linalg.norm(np.asarray(positions) - self.goal, axis=-1) / np.sqrt(2)

    def transition(self, positions, actions):
        positions = np.asarray(positions)
        target = positions + self.action_scale * np.clip(actions, -1, 1)
        collision = segments_hit_boxes(positions, target, self.boxes)
        collision |= np.any((target < 0) | (target > 1), axis=-1)
        successor = np.where(collision[..., None], positions, target)
        terminal = np.linalg.norm(successor - self.goal, axis=-1) < .055
        sparse = terminal.astype(float)
        dense = sparse + self.gamma * np.where(terminal, 0., self.potential(successor)) - self.potential(positions)
        return successor, sparse, dense, terminal, collision

    def step(self, action):
        successor, sparse, dense, terminal, collision = self.transition(self.position, action)
        self.position = successor.astype(np.float32)
        self.elapsed += 1
        truncated = self.elapsed >= self.max_steps and not bool(terminal)
        reward = sparse if self.reward_mode == "sparse" else dense
        return self.position.copy(), float(reward), bool(terminal), truncated, {
            "success": float(terminal), "collision": float(collision), "sparse_reward": float(sparse),
            "reward_dense": float(dense)}

    def close(self):
        pass

    def state_dict(self):
        return {"position": self.position.copy(), "elapsed": self.elapsed, "rng": self.rng.get_state()}

    def load_state_dict(self, state):
        self.position, self.elapsed = state["position"].copy(), state["elapsed"]
        self.rng.set_state(state["rng"])


def maze_dataset(episodes=500, seed=0, gap=False, gamma=.99):
    rng = np.random.RandomState(seed)
    env = PointMaze(gamma=gamma)
    routes = [np.array([[.34,.41], [.53,.41], [.88,.41], env.goal]),
              np.array([[.32,.85], [.85,.85], env.goal]),
              np.array([[.34,.41], [.53,.41], [.53,.59], [.70,.59], [.53,.59], [.53,.41], [.88,.41], env.goal])]
    rows = {k: [] for k in ("observations", "next_observations", "actions", "rewards", "rewards_dense", "terminals", "masks", "route")}
    for episode in range(episodes):
        route_id = rng.choice(3, p=[.20, .60, .20])
        route = routes[route_id]
        obs, _ = env.reset(seed=seed + episode)
        waypoint = 0
        # Fragments start on supported waypoints and terminate independently of success.
        if episode % 5 == 0:
            waypoint = rng.randint(len(route))
            initial = route[max(0, waypoint-1)] if waypoint else env.start
            obs, _ = env.reset(options={"position": initial})
        limit = int(rng.randint(12, 55)) if episode % 4 == 0 else env.max_steps
        for t in range(limit):
            while waypoint < len(route)-1 and np.linalg.norm(obs-route[waypoint]) < .035:
                waypoint += 1
            action = np.clip((route[waypoint] - obs) / env.action_scale + rng.normal(0, .12, 2), -1, 1)
            if rng.rand() < .06:
                action = rng.uniform(-1, 1, 2)
            nxt, reward, term, trunc, info = env.step(action)
            boundary = term or trunc or t == limit-1
            # Remove all transitions through the shortcut connector; remaining
            # fragments preserve their original bootstrap masks and rewards.
            removed = gap and bool(segments_hit_boxes(obs, nxt, [[.39,.36,.51,.47]]))
            if removed:
                if rows["terminals"]:
                    rows["terminals"][-1] = 1.
            else:
                for key, value in dict(observations=obs, next_observations=nxt, actions=action,
                                       rewards=reward, rewards_dense=info["reward_dense"],
                                       terminals=float(boundary), masks=float(not term), route=route_id).items():
                    rows[key].append(value)
            obs = nxt
            if boundary:
                break
        if rows["terminals"]:
            rows["terminals"][-1] = 1.
    return {k: np.asarray(v, dtype=np.float32) for k,v in rows.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["bandit", "maze"])
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--count", type=int, default=50000, help="Bandit samples or maze episodes")
    p.add_argument("--gap", action="store_true")
    args = p.parse_args()
    if args.kind == "bandit":
        arrays = Bandit(gap=args.gap).sample(args.count, args.seed)
        metadata = {"kind":"bandit", "terminal_bandit":True, "behavior":"eight finite-support uniform disks", "contexts":3}
    else:
        arrays = maze_dataset(args.count, args.seed, args.gap)
        metadata = {"kind":"maze", "gamma":.99, "timeout":"continuing; bootstrap mask=1", "shaping":"r + gamma*Phi(next)-Phi(current); Phi(terminal)=0"}
    metadata.update(seed=args.seed, gap=args.gap, reward_access="saved offline only", profile="toy_diagnostic")
    print(save_dataset(args.output, arrays, metadata))


if __name__ == "__main__":
    main()

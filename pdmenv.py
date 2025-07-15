import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PdMEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 telemetry_df: pd.DataFrame,
                 failures_df: pd.DataFrame,
                 maintenance_df: pd.DataFrame,
                 errors_df: pd.DataFrame,
                 machines_df: pd.DataFrame,
                 window_size: int = 24,
                 maintenance_cost: float = 10.0,
                 failure_penalty: float = 100.0,
                 running_reward: float = 1.0,
                 max_steps: int = None):  # ✅ yeni parametre
        super().__init__()
        for df in [telemetry_df, failures_df, maintenance_df, errors_df]:
            df['datetime'] = pd.to_datetime(df['datetime'])

        merged = telemetry_df.merge(
            failures_df[['datetime', 'machineID']].assign(failure=1),
            on=['datetime', 'machineID'], how='left'
        )
        merged['failure'] = merged['failure'].fillna(0).astype(int)

        merged = merged.merge(
            maintenance_df[['datetime', 'machineID']].assign(maintenance_event=1),
            on=['datetime', 'machineID'], how='left'
        )
        merged['maintenance_event'] = merged['maintenance_event'].fillna(0).astype(int)

        error_counts = (
            errors_df.groupby(['datetime', 'machineID'])
                     .size()
                     .reset_index(name='error_count')
        )
        merged = merged.merge(error_counts, on=['datetime', 'machineID'], how='left')
        merged['error_count'] = merged['error_count'].fillna(0).astype(int)

        merged = merged.merge(
            machines_df[['machineID', 'age']],
            on='machineID', how='left'
        )

        self.telemetry = merged.sort_values(['machineID', 'datetime']).reset_index(drop=True)
        self.window_size = window_size
        self.maintenance_cost = maintenance_cost
        self.failure_penalty = failure_penalty
        self.running_reward = running_reward
        self.max_steps = max_steps  # ✅ kaydet

        self.action_space = spaces.Discrete(2)
        num_features = (
            self.telemetry.shape[1] - 3
        )
        obs_shape = (window_size, num_features + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )

        self.current_step = None
        self.last_maintenance_step = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        #self.start_index = np.random.randint(self.window_size, len(self.telemetry) - self.max_steps)
        if self.max_steps is not None:
            max_start = len(self.telemetry) - self.max_steps
        else:
            max_start = len(self.telemetry) - 1
        self.start_index = np.random.randint(self.window_size, max_start)

        self.current_step = self.start_index
        self.last_maintenance_step = self.start_index
        obs = self._get_observation()
        return obs, {}

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"
        reward = 0.0
        if action == 1:
            reward -= self.maintenance_cost
            self.last_maintenance_step = self.current_step

        row = self.telemetry.iloc[self.current_step]
        if action == 0 and row['failure'] == 1:
            reward -= self.failure_penalty
        else:
            reward += self.running_reward

        self.current_step += 1
        #terminated = (self.current_step - self.start_index) >= self.max_steps  # ✅ yeni koşul
        if self.max_steps is not None:
            terminated = (self.current_step - self.start_index) >= self.max_steps
        else:
            terminated = self.current_step >= len(self.telemetry)

        truncated = False
        obs = None if terminated or truncated else self._get_observation()
        return obs, reward, terminated, truncated, {'step': self.current_step}

    def _get_observation(self):
        start = self.current_step - self.window_size
        window = self.telemetry.iloc[start:self.current_step]
        cols = [c for c in window.columns if c not in ['datetime', 'machineID']]
        data = window[cols].values
        time_since_maint = (
            np.arange(start, self.current_step) - self.last_maintenance_step
        ).reshape(-1, 1)
        obs = np.hstack([data, time_since_maint])
        return obs.astype(np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

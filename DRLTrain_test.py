"""
Arayüz olmadan yapılan bakım sayısını terminalden gösteren kod
import gymnasium as gym
import pandas as pd
import numpy as np
import torch
from pdmenv import PdMEnv
import matplotlib.pyplot as plt

def load_data():
    telemetry_df = pd.read_csv('telemetry.csv')
    failures_df = pd.read_csv('failures.csv')
    maintenance_df = pd.read_csv('maint.csv')
    errors_df = pd.read_csv('errors.csv')
    machines_df = pd.read_csv('machines.csv')
    return telemetry_df, failures_df, maintenance_df, errors_df, machines_df

class DQN(torch.nn.Module):
    def __init__(self, input_dim, n_actions):
        super(DQN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, env, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = env
        self.device = device

        sample_obs, _ = env.reset()
        self.input_dim = np.prod(sample_obs.shape)
        self.n_actions = env.action_space.n

        self.policy_net = DQN(self.input_dim, self.n_actions).to(device)
        #self.policy_net.load_state_dict(torch.load(model_path))
        self.policy_net.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        self.policy_net.eval()

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state.copy(), dtype=torch.float32, device=self.device).reshape(1, -1)
            return self.policy_net(state_tensor).argmax().item()

def test_model(env, agent, num_episodes=10):
    action_history = []
    failure_history = []
    maintenance_history = []
    reward_history = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        state = obs
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Kayıt tutma
            action_history.append(action)
            if 'failure' in info and info['failure']:
                failure_history.append(env.current_step)
            if action == 1:  # Bakım yapıldı
                maintenance_history.append(env.current_step)
            
            total_reward += reward
            state = next_obs
        
        reward_history.append(total_reward)
        print(f"Episode {ep+1}, Total Reward: {total_reward}")
    
    # Sonuçları görselleştirme
    plt.figure(figsize=(15, 10))
    
    # Eylemlerin zaman serisi
    plt.subplot(3, 1, 1)
    plt.plot(action_history, label='Actions (0=No action, 1=Maintenance)')
    plt.ylabel('Action')
    plt.title('Maintenance Actions Over Time')
    plt.legend()
    
    # Bakım ve arızaların işaretlenmesi
    plt.subplot(3, 1, 2)
    plt.plot(action_history, label='Actions')
    for m in maintenance_history:
        plt.axvline(x=m, color='g', alpha=0.3, label='Maintenance' if m == maintenance_history[0] else "")
    for f in failure_history:
        plt.axvline(x=f, color='r', alpha=0.3, label='Failure' if f == failure_history[0] else "")
    plt.ylabel('Action with Events')
    plt.legend()
    
    # Ödüller
    plt.subplot(3, 1, 3)
    plt.plot(reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.show()
    
    # İstatistikler
    print("\nTest Results Summary:")
    print(f"Total episodes: {num_episodes}")
    print(f"Average reward: {np.mean(reward_history):.2f}")
    print(f"Number of failures: {len(failure_history)}")
    print(f"Number of maintenance actions: {len(maintenance_history)}")
    print(f"Maintenance before failure: {sum(m < max(failure_history) for m in maintenance_history) if failure_history else 'N/A'}")

if __name__ == '__main__':
    # Verileri ve ortamı yükle
    telemetry_df, failures_df, maintenance_df, errors_df, machines_df = load_data()
    
    env = PdMEnv(
        telemetry_df,
        failures_df,
        maintenance_df,
        errors_df,
        machines_df,
        window_size=24,
        maintenance_cost=10.0,
        failure_penalty=100.0,
        running_reward=1.0,
        #max_steps=200
    )
    
    # Eğitilmiş modeli yükle
    model_path = 'dqn_pdm1.pth'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    #agent = DQNAgent(env, model_path)
    agent = DQNAgent(env, model_path, device=device)
    
    # Modeli test et
    test_model(env, agent, num_episodes=10)
"""
import pandas as pd
import numpy as np
import torch
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from pdmenv import PdMEnv

# === Model Tanımı ===
class DQN(torch.nn.Module):
    def __init__(self, input_dim, n_actions):
        super(DQN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_actions)
        )
    def forward(self, x):
        return self.net(x)

# === Model Yükleme ===
def load_model(env, path='dqn_pdm1.pth'):
    sample_obs, _ = env.reset()
    input_dim = np.prod(sample_obs.shape)
    n_actions = env.action_space.n

    model = DQN(input_dim, n_actions)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.eval()
    return model

# === Verileri Yükle ===
def load_data():
    telemetry_df = pd.read_csv('telemetry.csv')
    failures_df = pd.read_csv('failures.csv')
    maintenance_df = pd.read_csv('maint.csv')
    errors_df = pd.read_csv('errors.csv')
    machines_df = pd.read_csv('machines.csv')
    telemetry_df['datetime'] = pd.to_datetime(telemetry_df['datetime'])
    return telemetry_df, failures_df, maintenance_df, errors_df, machines_df

# === GUI + Grafik ===
def launch_gui(env, model, telemetry_df):
    root = tk.Tk()
    root.title("PdM Sensör İzleme")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=80)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    sensor_keys = ['volt', 'rotate', 'pressure', 'vibration']
    lines = {k: ax.plot([], [], label=k)[0] for k in sensor_keys}

    ax.set_ylim(telemetry_df[sensor_keys].min().min() - 5, telemetry_df[sensor_keys].max().max() + 5)
    ax.set_xlabel('Tarih')
    ax.set_ylabel('Sensör Değerleri')
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    fig.subplots_adjust(top=0.88)

    # Buton
    btn = tk.Button(root, text="Bakım Yapıldı, Devam Et", state="disabled")
    btn.pack(pady=10)

    paused = {'value': False}
    obs, _ = env.reset()
    window_size = 50  # Kaydırmalı pencere boyutu

    def on_continue():
        paused['value'] = False
        btn.config(state="disabled")

    btn.config(command=on_continue)

    def update(frame):
        if paused['value']:
            return

        nonlocal obs
        if frame >= len(telemetry_df):
            return

        now = telemetry_df['datetime'].iloc[frame]
        start_idx = max(0, frame - window_size)
        sub = telemetry_df.iloc[start_idx:frame + 1]
        x = sub['datetime']

        for key in sensor_keys:
            lines[key].set_data(x, sub[key])

        # 🛠️ Zaman aralığı kontrolü (tek zaman noktasına dikkat)
        if len(x) == 1:
            ax.set_xlim(x.iloc[0] - pd.Timedelta(hours=1), x.iloc[0] + pd.Timedelta(hours=1))
        else:
            ax.set_xlim(x.iloc[0], x.iloc[-1])

        ax.set_title(f'Tarih: {now}', pad=20)
        canvas.draw()

        # Bakım kararı
        with torch.no_grad():
            state_tensor = torch.tensor(obs, dtype=torch.float32).reshape(1, -1)
            action = model(state_tensor).argmax().item()

        # 🚨 current_step sınırı kontrolü
        if env.current_step >= len(telemetry_df):
            return

        obs_, _, terminated, truncated, _ = env.step(action)
        if not (terminated or truncated):
            obs = obs_

        if action == 1:
            paused['value'] = True
            btn.config(state="normal")
            ax.axvline(x=now, color='red', linestyle='--', linewidth=2)
            messagebox.showwarning("BAKIM", f"{now} tarihinde bakım önerildi.")

    ani = FuncAnimation(fig, update, frames=len(telemetry_df), interval=30)
    root.mainloop()

# === Ana Fonksiyon ===
if __name__ == '__main__':
    telemetry_df, failures_df, maintenance_df, errors_df, machines_df = load_data()

    env = PdMEnv(
        telemetry_df,
        failures_df,
        maintenance_df,
        errors_df,
        machines_df,
        window_size=24,
        maintenance_cost=10.0,
        failure_penalty=100.0,
        running_reward=1.0,
        max_steps=None  # sınırsız adım
    )

    model = load_model(env)
    launch_gui(env, model, telemetry_df)

# Gerekli kütüphaneler
import gymnasium as gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from pdmenv import PdMEnv # Özel olarak tanımlanmış kestirimci bakım ortamı

# Deneyim verilerini saklamak için bir namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Replay buffer: deneyimleri depolayan sınıf
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) # Sabit boyutlu kuyruk

    def push(self, *args):
        self.memory.append(Transition(*args)) # Yeni bir deneyim ekle

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        return batch # Rastgele batch seçimi

    def __len__(self):
        return len(self.memory)

# DQN (Deep Q-Network) tanımı
class DQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), # Giriş katmanı
            nn.ReLU(),
            nn.Linear(64, n_actions) # Çıkışta her aksiyon için bir Q-değeri
        )

    def forward(self, x):
        return self.net(x)


# Ajan sınıfı: çevreyle etkileşimi yöneten yapı
class DQNAgent:
    def __init__(self, env, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = env
        self.device = device

        # Ortamdan örnek bir gözlem alınarak giriş boyutu ve eylem sayısı belirlenir
        sample_obs, _ = env.reset()
        self.input_dim = np.prod(sample_obs.shape)
        self.n_actions = env.action_space.n

        # Politika ağı ve hedef ağı
        self.policy_net = DQN(self.input_dim, self.n_actions).to(device)
        self.target_net = DQN(self.input_dim, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizasyon ve hiperparametreler
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayMemory(10000)
        self.batch_size = 32
        self.gamma = 0.99 # İndirgeme faktörü
        self.eps_start = 1.0 # Başlangıç keşif oranı
        self.eps_end = 0.01 # Minimum keşif oranı
        self.eps_decay = 0.0001
        self.steps_done = 0
        self.update_target_every = 1000 # Kaç adımda bir hedef ağı güncellensin
        self.writer = SummaryWriter() # TensorBoard için

    # Eylem seçimi (epsilon-greedy)
    def select_action(self, state, training=True):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        np.exp(-1. * self.steps_done * self.eps_decay)
        self.steps_done += 1

        # Keşif ya da sömürü kararı
        if training and sample < eps_threshold:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state.copy(), dtype=torch.float32, device=self.device).reshape(1, -1)
                return self.policy_net(state_tensor).argmax().view(1, 1)

    # Ağı optimize etme işlemi (geri yayılım)
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Aksiyonları uygun forma dönüştür
        actions = np.array([a.item() if isinstance(a, torch.Tensor) else a for a in batch.action])
        action_batch = torch.tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1)

        # Gözlemleri tensöre çevir
        states = np.array([np.array(s, copy=True) for s in batch.state])
        next_states = [np.array(s, copy=True) for s in batch.next_state if s is not None]

        state_batch = torch.tensor(states, dtype=torch.float32, device=self.device).reshape(self.batch_size, -1)

        non_final_mask = torch.tensor([s is not None for s in batch.next_state],
                                      device=self.device, dtype=torch.bool)

        if next_states:
            non_final_next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
            non_final_next_states = non_final_next_states.reshape(len(next_states), -1)
        else:
            non_final_next_states = None

        # Ödüller
        rewards = np.array([r.item() if isinstance(r, torch.Tensor) else r for r in batch.reward])
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Q-değerlerini hesapla
        q_values = self.policy_net(state_batch)
        state_action_values = q_values.gather(1, action_batch)

        # Hedef Q-değerleri
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_next_states is not None:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Kayıp fonksiyonu (Mean Squared Error)
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient değerlerini sınırlama (gradient clipping)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    # Ajanın eğitim süreci
    def train(self, episodes, save_path='dqn_pdm1.pth'):
        rewards = []
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = obs
            total_reward = 0
            done = False

            current_epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                              np.exp(-1. * self.steps_done * self.eps_decay)

            while not done:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
                total_reward += reward
                done = terminated or truncated

                next_state = next_obs if not done else None
                self.memory.push(state, action, next_state, reward, done)
                state = next_obs if not done else state

                if self.steps_done % 4 == 0:
                    loss = self.optimize_model()

                # Hedef ağı güncelle
                if self.steps_done % self.update_target_every == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            rewards.append(total_reward)
            # Her 100 bölümde bir durumu kaydet
            if ep % 100 == 0:
                print(f'Episode {ep}, Reward: {total_reward}, Epsilon: {current_epsilon:.2f}')
                torch.save(self.policy_net.state_dict(), save_path)
                if loss is not None:
                    self.writer.add_scalar('Loss/train', loss, ep)
                self.writer.add_scalar('Reward/train', total_reward, ep)

        # Eğitim performans grafiği
        plt.figure(figsize=(12, 6))
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('DQN Training Performance')
        plt.savefig('dqn_training.png')
        self.writer.close()

# Verileri yükleme fonksiyonu
def load_data():
    telemetry_df = pd.read_csv('telemetry.csv')
    failures_df = pd.read_csv('failures.csv')
    maintenance_df = pd.read_csv('maint.csv')
    errors_df = pd.read_csv('errors.csv')
    machines_df = pd.read_csv('machines.csv')
    return telemetry_df, failures_df, maintenance_df, errors_df, machines_df

# Ana program bloğu
if __name__ == '__main__':
    telemetry_df, failures_df, maintenance_df, errors_df, machines_df = load_data()

    # Kestirimci bakım ortamının tanımı
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
        #max_steps= None  # Opsiyonel olarak adım sayısını sınırlama
    )

    # Ajan oluşturulup eğitilir
    agent = DQNAgent(env)
    agent.train(episodes=1000)

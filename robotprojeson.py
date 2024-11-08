import gymnasium as gym
import time
import numpy as np
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv

def train_and_evaluate(env_name, algorithm, total_iterations=10):
    print(f"\nOrtam: {env_name}, Algoritma: {algorithm.__name__} ile eğitim başlıyor...")

    total_timesteps_per_iteration = 1024
    all_rewards = []
    all_times = []

    # Belirtilen ortamı oluştur ve sarmalla
    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])

    if algorithm in [SAC, DDPG]:
        model = algorithm(
            'MlpPolicy', env,
            buffer_size=1000,       # Deneyim tamponunun boyutu
            learning_rate=0.005,    # Öğrenme oranı, modeli daha hızlı eğitmek için artırıldı
            gamma=0.9,              # Kısa vadeli ödüllere daha fazla odaklanıyor
            batch_size=32,          # Hızlı güncellemeler için toplu öğrenme boyutu azaltıldı
            verbose=1,              # Eğitim sürecindeki ayrıntıları gösterir
        )
    else:
        model = algorithm(
            'MlpPolicy', env,
            learning_rate=0.005,
            gamma=0.9,
            batch_size=32,
            n_steps=256,
            verbose=1,
        )

    # Eğitim süresini başlat
    start_time = time.time()
    total_rewards = []


    for iteration in range(total_iterations):
        try:
            obs = env.reset()  # Ortamı sıfırla
            total_reward = 0

            # Bir iterasyon boyunca adım atma döngüsü
            for _ in range(total_timesteps_per_iteration):
                action, _ = model.predict(obs)  # Eylemi tahmin et
                obs, reward, done, info = env.step(action)  # Ortamda bir adım at
                total_reward += reward[0]  # Ödülü topla (reward dizisi olduğundan indeksli al)

                if done:
                    obs = env.reset()  # Epizot bitince ortamı sıfırla

            # Belirli bir adım sayısı sonunda model öğrenme adımı gerçekleştirir
            model.learn(total_timesteps=total_timesteps_per_iteration)
            total_rewards.append(total_reward)  # Toplam ödülü kaydet
            elapsed_time = time.time() - start_time
            print(
                f"İterasyon: {iteration + 1}/{total_iterations}, "
                f"Geçen Süre: {elapsed_time:.2f} saniye, Toplam Ödül: {total_reward:.2f}")

        except Exception as e:
            print(f"{iteration + 1}. iterasyonda hata oluştu: {e}")
            break

    end_time = time.time()
    total_elapsed_time = end_time - start_time
    average_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
    all_rewards.append(total_rewards)
    all_times.append(total_elapsed_time)

    # Tüm çalıştırmaların sonuçlarının analizi
    avg_reward_per_run = [np.mean(rewards) for rewards in all_rewards]
    median_reward_per_run = [np.median(rewards) for rewards in all_rewards]
    best_reward_per_run = [np.max(rewards) for rewards in all_rewards]
    avg_time_per_run = np.mean(all_times)

    print(
        f"\n{env_name} - {algorithm.__name__} Sonuçları:\n"
        f"Toplam Ortalama Süre: {avg_time_per_run:.2f} saniye\n"
        f"Ortalama Ödül: {np.mean(avg_reward_per_run):.2f}, Medyan Ödül: {np.median(median_reward_per_run):.2f}, "
        f"En İyi Ödül: {np.max(best_reward_per_run):.2f}\n"
    )

    return avg_reward_per_run, median_reward_per_run, best_reward_per_run, avg_time_per_run

envs = ["MountainCarContinuous-v0", "Pusher-v4"]
algorithms = [PPO, SAC, DDPG]

all_avg_rewards = {}
all_median_rewards = {}
all_best_rewards = {}
all_avg_times = {}

# Her ortam ve algoritma kombinasyonunda eğitim ve değerlendirme yap
for env_name in envs:
    for algorithm in algorithms:
        avg_rewards, median_rewards, best_rewards, avg_time = train_and_evaluate(env_name, algorithm)
        all_avg_rewards[(env_name, algorithm.__name__)] = avg_rewards
        all_median_rewards[(env_name, algorithm.__name__)] = median_rewards
        all_best_rewards[(env_name, algorithm.__name__)] = best_rewards
        all_avg_times[(env_name, algorithm.__name__)] = avg_time

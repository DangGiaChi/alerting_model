import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_synthetic_timeseries(n_samples=10000, seed=42):
    np.random.seed(seed)
    
    time = np.arange(n_samples)
    
    cpu_base = 50 + 10 * np.sin(2 * np.pi * time / 500)
    cpu_noise = np.random.normal(0, 8, n_samples)
    cpu = cpu_base + cpu_noise
    
    memory_base = 60 + 5 * np.sin(2 * np.pi * time / 300)
    memory_noise = np.random.normal(0, 5, n_samples)
    memory = memory_base + memory_noise
    
    request_base = 100 + 20 * np.sin(2 * np.pi * time / 400)
    request_noise = np.random.normal(0, 15, n_samples)
    requests = request_base + request_noise
    
    incidents = np.zeros(n_samples)
    
    n_incidents = 20
    incident_indices = np.random.choice(range(200, n_samples - 200), n_incidents, replace=False)
    
    for i, idx in enumerate(incident_indices):
        duration = np.random.randint(20, 50)
        start = idx
        end = min(idx + duration, n_samples)
        
        incident_type = np.random.choice(['gradual', 'sudden', 'unpredictable'], p=[0.6, 0.25, 0.15])
        
        if incident_type == 'gradual':
            buildup_duration = np.random.randint(15, 45)
            buildup_start = max(0, start - buildup_duration)
            buildup_time = np.arange(buildup_start, start)
            buildup_factor = (buildup_time - buildup_start) / (start - buildup_start)
            
            intensity = np.random.uniform(0.7, 1.3)
            cpu[buildup_start:start] += buildup_factor * (20 * intensity)
            memory[buildup_start:start] += buildup_factor * (15 * intensity)
            requests[buildup_start:start] += buildup_factor * (40 * intensity)
        
        elif incident_type == 'sudden':
            pass
        
        spike_intensity = np.random.uniform(0.8, 1.2)
        cpu[start:end] += np.random.uniform(25, 45, end - start) * spike_intensity
        memory[start:end] += np.random.uniform(20, 35, end - start) * spike_intensity
        requests[start:end] += np.random.uniform(50, 90, end - start) * spike_intensity
        
        incidents[start:end] = 1
    
    n_false_alarms = 30
    false_alarm_indices = np.random.choice(range(200, n_samples - 200), n_false_alarms, replace=False)
    
    for idx in false_alarm_indices:
        if incidents[idx] == 1 or any(incidents[max(0, idx-50):min(n_samples, idx+50)] == 1):
            continue
            
        duration = np.random.randint(8, 25)
        start = idx
        end = min(idx + duration, n_samples)
        
        spike_type = np.random.choice(['cpu_spike', 'memory_spike', 'request_spike', 'multi_spike'])
        
        if spike_type == 'cpu_spike':
            cpu[start:end] += np.random.uniform(15, 30, end - start)
        elif spike_type == 'memory_spike':
            memory[start:end] += np.random.uniform(12, 25, end - start)
        elif spike_type == 'request_spike':
            requests[start:end] += np.random.uniform(30, 60, end - start)
        else:
            cpu[start:end] += np.random.uniform(10, 20, end - start)
            memory[start:end] += np.random.uniform(8, 15, end - start)
    
    cpu = np.clip(cpu, 0, 100)
    memory = np.clip(memory, 0, 100)
    requests = np.clip(requests, 0, 500)
    
    df = pd.DataFrame({
        'timestamp': time,
        'cpu_usage': cpu,
        'memory_usage': memory,
        'request_rate': requests,
        'incident': incidents.astype(int)
    })
    
    return df


def visualize_data(df, save_path='visualizations/data_visualization.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    
    axes[0].plot(df['timestamp'], df['cpu_usage'], linewidth=0.5)
    axes[0].set_ylabel('CPU Usage (%)')
    axes[0].set_title('Synthetic Time Series with Incidents')
    
    axes[1].plot(df['timestamp'], df['memory_usage'], linewidth=0.5, color='orange')
    axes[1].set_ylabel('Memory Usage (%)')
    
    axes[2].plot(df['timestamp'], df['request_rate'], linewidth=0.5, color='green')
    axes[2].set_ylabel('Request Rate')
    
    axes[3].fill_between(df['timestamp'], 0, df['incident'], alpha=0.5, color='red')
    axes[3].set_ylabel('Incident')
    axes[3].set_xlabel('Time Step')
    axes[3].set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Visualization saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    df = generate_synthetic_timeseries(n_samples=10000)
    
    df.to_csv('timeseries_data.csv', index=False)
    print(f"Data saved to timeseries_data.csv")
    print(f"Total samples: {len(df)}")
    print(f"Incident samples: {df['incident'].sum()} ({100*df['incident'].mean():.2f}%)")
    print(f"\nFirst few rows:")
    print(df.head(10))
    
    visualize_data(df)

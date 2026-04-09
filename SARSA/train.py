import os
import sys
import numpy as np
import gymnasium as gym
from sumo_rl import SumoEnvironment
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 1. SETUP SUMO PATHS
# Verified (x86) path for your system
os.environ['SUMO_HOME'] = r'C:\Program Files (x86)\Eclipse\Sumo' 
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("SUMO_HOME not found.")

# 2. FILE PATH LOGIC - Updated for 4-way Intersection
BASE_PATH = r'C:\Users\VICTUS\Documents\AI_Project\Intelligent-Traffic-Signal-Control-using-Reinforcement-Learning\Temp\VS_code'

def get_valid_path(base, filename):
    path = os.path.join(base, filename)
    if os.path.exists(path):
        return path
    return None

# Downloading these files from the "Intersection Model" folder is required

#NET_FILE = get_valid_path(BASE_PATH, 'cross.net.xml')
#ROU_FILE = get_valid_path(BASE_PATH, 'cross.rou.xml')

NET_FILE = "./cross.net.xml"
ROU_FILE = "./cross.rou.xml"

if not NET_FILE or not ROU_FILE:
    print(f"CRITICAL ERROR: 'cross.net.xml' or 'cross.rou.xml' not found")
    sys.exit()

# 3. SARSA AGENT DEFINITION
class SARSAAgent:
    """
    SARSA (State-Action-Reward-State-Action) reinforcement learning agent.
    
    This agent uses a Q-table to learn optimal policies for traffic signal control.
    """
    def __init__(self, action_size):
        self.q_table = {}
        self.lr, self.gamma, self.epsilon = 0.1, 0.95, 1.0
        self.action_size = action_size

    def get_state_key(self, state):
        return tuple(np.round(state, 1))

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state_key])

    def learn(self, s, a, r, s_next, a_next):
        s_k, sn_k = self.get_state_key(s), self.get_state_key(s_next)
        if sn_k not in self.q_table: self.q_table[sn_k] = np.zeros(self.action_size)
        predict = self.q_table[s_k][a]
        target = r + self.gamma * self.q_table[sn_k][a_next]
        self.q_table[s_k][a] += self.lr * (target - predict)
    
    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        """Decay the exploration rate."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def save_model(self, filepath="sarsa_qtable.pkl"):
        """Saves the Q-table to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Model successfully saved to: {filepath}")

    def load_model(self, filepath="sarsa_qtable.pkl"):
        """Loads the Q-table from a file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Model successfully loaded from: {filepath}")
        else:
            print(f"Warning: No saved model found at {filepath}")


def evaluate_model(agent, net_file, rou_file, tripinfo_file="tripinfo.xml"):
    """
    Evaluates the trained agent and generates a tripinfo.xml file.
    """
    print(f"\n--- Starting Evaluation ---")
    print(f"Generating trip info file at: {tripinfo_file}")
    
    # Initialize a fresh environment specifically for evaluation
    eval_env = SumoEnvironment(
        net_file=net_file, 
        route_file=rou_file,
        use_gui=False, # Set to True if you want to watch the trained agent
        num_seconds=3600,
        single_agent=True,
        additional_sumo_cmd=f"--tripinfo-output {tripinfo_file}" # Tells SUMO to generate the XML
    )
    
    state, info = eval_env.reset()
    done = False
    total_eval_reward = 0
    
    # Save the original epsilon and set it to 0 for pure exploitation (no random actions)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = eval_env.step(action)
        
        done = terminated or truncated
        state = next_state
        total_eval_reward += reward
        
    # Clean up and restore
    agent.epsilon = original_epsilon
    eval_env.close()
    
    print(f"Evaluation completed! Total Reward: {total_eval_reward}")
    print(f"Trip info successfully saved to: {os.path.abspath(tripinfo_file)}\n")



# 4. START SIMULATION AND TRAINING
try:
    # Initialize environment with the 4-way intersection files
    env = SumoEnvironment(
        net_file=NET_FILE, 
        route_file=ROU_FILE,
        use_gui=False,
        num_seconds=3600,
        single_agent=True
    )
    
    # Environment will now detect the traffic light phases from cross.net.xml
    action_size = env.action_space.n
    agent = SARSAAgent(action_size=action_size)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='./sarsa_tensorboard')
    
    print("Starting SARSA Training...")
    episodes = 50
    reward_history = []
    
    for e in tqdm(range(episodes), desc="Training Episodes"):
        state, info = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        done = False
        
        while not done:
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            next_action = agent.choose_action(next_state)
            agent.learn(state, action, reward, next_state, next_action)
            
            state = next_state
            action = next_action
            total_reward += reward
        
        reward_history.append(total_reward)
        # Log to TensorBoard
        writer.add_scalar('Reward/Episode', total_reward, e + 1)
        writer.add_scalar('Epsilon', agent.epsilon, e + 1)
        # Decay epsilon
        agent.decay_epsilon()
    
    writer.close()
    env.close()
    
    print("Training completed successfully.")
    
    agent.save_model()
    evaluate_model(agent, NET_FILE, ROU_FILE, tripinfo_file="tripinfo_eval.xml")
    
except Exception as e:
    print(f"Error during simulation or training: {e}")
    # Ensure environment is closed if it was created
    try:
        env.close()
    except:
        pass

# 5. VISUALIZATION AND STATISTICS
if reward_history:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, episodes + 1), reward_history, marker='o', color='b', linestyle='-')
    plt.title('SARSA Agent Performance (4-Way Intersection)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()
    
    # Print statistics
    print("\nTraining Statistics:")
    print(f"Total Episodes: {len(reward_history)}")
    print(f"Average Reward: {np.mean(reward_history):.2f}")
    print(f"Max Reward: {np.max(reward_history):.2f}")
    print(f"Min Reward: {np.min(reward_history):.2f}")
    print(f"Std Dev: {np.std(reward_history):.2f}")
    if len(reward_history) > 10:
        print(f"Last 10 Episodes Average: {np.mean(reward_history[-10:]):.2f}")
else:
    print("No training data available for visualization.")
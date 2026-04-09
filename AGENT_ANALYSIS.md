# Comprehensive Analysis: SARSA vs Q-Learning vs DQN for Traffic Signal Control

## Overview
This analysis compares three reinforcement learning agents implemented in your traffic signal control system:
- **SARSA** (State-Action-Reward-State-Action) - `hello.py`
- **Q-Learning** (QL) - `ql_2way-single-intersection.py`
- **Deep Q-Network** (DQN) - `iiser-v3.ipynb`

---

## 1. REWARD FUNCTION

### SARSA (hello.py)
- **Default Reward**: Standard sumo-rl environment reward (not explicitly specified in code)
- **Type**: Single-agent default
- **Impact**: Generic reward lacks domain-specific optimization for emergency vehicles

### Q-Learning (ql_2way-single-intersection.py)
- **Reward Definition**: Controlled via `args.reward` parameter
  - Options: `"queue"`, `"wait"`, `"max-emergency"` (default)
- **Type**: Can be customized per use case
- **Impact**: 
  - `queue`: Penalizes vehicle queue length
  - `wait`: Penalizes waiting time
  - `max-emergency`: Prioritizes emergency vehicles
- **Density**: **Dense** - Reward given at every step

### DQN (iiser-v3.ipynb)
```python
def priority_reward_fn(ts):
    reward = 0
    for lane in ts.lanes:
        vehicles = traci.lane.getLastStepVehicleIDs(lane)
        for veh in vehicles:
            speed = traci.vehicle.getSpeed(veh)
            if speed < 0.1:  # Stopped
                v_type = traci.vehicle.getTypeID(veh)
                if v_type == "ambulance":
                    reward -= 50.0   # Severe penalty
                elif v_type == "bus":
                    reward -= 5.0    # Moderate penalty
                else:
                    reward -= 1.0    # Light penalty
    return reward
```
- **Type**: **Sparse-to-Dense Hybrid**
  - Dense in structure (every step evaluated)
  - Sparse feedback (only penalizes stopped vehicles)
- **Impact on Learning**:
  - ✅ **Advantages**: Immediately discourages stopping ambulances, faster emergency response learning
  - ❌ **Disadvantages**: May create local optima (constant movement even without purpose)

### Comparison Table
| Agent | Reward Type | Density | Priority Handling |
|-------|-----------|---------|------------------|
| SARSA | Generic | Dense | ❌ No |
| Q-Learning | Customizable | Dense | ✅ Yes (via "max-emergency") |
| DQN | Domain-Specific | Sparse-Dense | ✅ Yes (50x penalty for ambulances) |

---

## 2. MULTI-AGENT SETUP

### SARSA (hello.py)
```python
env = SumoEnvironment(
    net_file=NET_FILE, 
    route_file=ROU_FILE,
    use_gui=False,
    num_seconds=3600,
    single_agent=True  # 👈 SINGLE AGENT
)
```
- **Setup**: **Single-Agent**
- **Interaction**: N/A (only one traffic signal controlled)
- **Strategy**: Simple independent control

### Q-Learning (ql_2way-single-intersection.py)
```python
env = SumoEnvironment(
    net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
    route_file=args.route,
    out_csv_name=out_csv,
    use_gui=False,
    num_seconds=args.seconds,
    min_green=args.min_green,
    max_green=args.max_green,
    sumo_warnings=False,
)

for ts in env.ts_ids:  # Multiple traffic signals
    ql_agents[ts] = QLAgent(...)
```
- **Setup**: **Multi-Agent** (one agent per traffic signal)
- **Interaction Type**: **Independent Learners** (Cooperative by default)
  - Each agent learns independently
  - Agents don't directly share information
  - Implicit coordination through environment state
- **Strategy Change**:
  - Harder to converge (increased state space)
  - More realistic traffic scenarios
  - Potential for emergent behaviors

### DQN (iiser-v3.ipynb)
```python
env = sumo_rl.parallel_env(...)

for agent in env.possible_agents:
    obs_dim = env.observation_space(agent).shape[0]
    action_dim = env.action_space(agent).n
    q_nets[agent] = QNetwork(obs_dim, action_dim)
    optimizers[agent] = optim.Adam(q_nets[agent].parameters(), lr=learning_rate)
```
- **Setup**: **Multi-Agent Parallel Environment**
- **Interaction Type**: **Independent DQN Agents**
  - `parallel_env` allows simultaneous action selection
  - Independent Q-networks per traffic signal
  - Cooperative implicit coordination
- **Strategy**:
  - Parallel learning (faster training)
  - Scalable to larger networks
  - Can handle continuous state spaces

### Multi-Agent Comparison

| Aspect | SARSA | Q-Learning | DQN |
|--------|-------|-----------|-----|
| Agents | 1 | Multiple | Multiple |
| Coordination | N/A | Independent | Independent Parallel |
| State Space Growth | Minimal | Exponential | Sampled (via NN) |
| Communication | N/A | None | None |
| Scalability | High | Medium | High |

---

## 3. ACTIONS

### Available Actions
All three agents control **discrete traffic light phases**:
- **Action Space**: `n_phases` discrete actions (typically 4-8 phases)
- **Representation**: Integer index (0 to n_phases-1)
- **Physical Meaning**: Each action represents a different green light phase configuration

### SARSA (hello.py)
```python
class SARSAAgent:
    def __init__(self, action_size):
        self.action_size = action_size
    
    def choose_action(self, state):
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)  # Explore
        return np.argmax(self.q_table[state_key])  # Exploit
```
- **Type**: **Discrete**
- **Selection Strategy**: **ε-Greedy**
  - `epsilon = 1.0` (initially 100% exploration)
  - `epsilon_decay = 0.995` (gradual shift to exploitation)
  - `min_epsilon = 0.01` (always explore 1%)
- **Decision Logic**:
  - With probability `ε`: random action (exploration)
  - With probability `1-ε`: best known action (exploitation)

### Q-Learning (ql_2way-single-intersection.py)
```python
actions = {}
for ts in ql_agents.keys():
    if np.random.rand() < 0.5:
        # Bias toward serving worst lane
        actions[ts] = lane_to_phase.get(max_lane, 0)
    else:
        # RL decision
        actions[ts] = ql_agents[ts].act()

# Inside QLAgent:
actions[ts] = ql_agents[ts].act()
```
- **Type**: **Discrete**
- **Selection Strategy**: **Hybrid Heuristic + ε-Greedy**
  - 50% heuristic: serve the lane with maximum queue
  - 50% learned: RL agent action selection
  - Uses `EpsilonGreedy` exploration strategy
- **Advanced Feature**: **Domain Knowledge Integration**
  - Incorporates traffic domain expertise (serve worst lane)
  - Accelerates learning by reducing exploration space

### DQN (iiser-v3.ipynb)
```python
for agent in env.agents:
    obs_tensor = torch.FloatTensor(observations[agent]).unsqueeze(0)
    
    if random.random() < epsilon:
        actions[agent] = env.action_space(agent).sample()  # Random action
    else:
        with torch.no_grad():
            q_values = q_nets[agent](obs_tensor)
            actions[agent] = torch.argmax(q_values).item()  # Greedy
```
- **Type**: **Discrete**
- **Selection Strategy**: **ε-Greedy with Decay**
  - `epsilon_decay = 0.97` (aggressive annealing)
  - `epsilon_min = 0.05` (always explore 5%)
  - Updates per episode
- **Neural Network Decision**:
  - Q-network outputs Q-values for all actions
  - Selects action with maximum Q-value
  - Enables handling of high-dimensional state spaces

### Action Selection Comparison

| Aspect | SARSA | Q-Learning | DQN |
|--------|-------|-----------|-----|
| Type | Discrete | Discrete | Discrete |
| Strategy | ε-Greedy | Hybrid Heuristic+ε-Greedy | ε-Greedy |
| Initial ε | 1.0 | 0.3 | 1.0 |
| ε Decay | 0.995 | 0.9995 | 0.97 |
| Min ε | 0.01 | 0.005 | 0.05 |
| Domain Knowledge | ❌ No | ✅ Yes (heuristic) | ❌ No |
| Exploration Bias | Uniform Random | Weighted (worst lane) | Uniform Random |

---

## 4. OPTIMIZATION

### SARSA (hello.py)
```python
def learn(self, s, a, r, s_next, a_next):
    s_k = self.get_state_key(s)
    sn_k = self.get_state_key(s_next)
    if sn_k not in self.q_table: 
        self.q_table[sn_k] = np.zeros(self.action_size)
    
    predict = self.q_table[s_k][a]
    target = r + self.gamma * self.q_table[sn_k][a_next]  # Uses NEXT action taken
    self.q_table[s_k][a] += self.lr * (target - predict)
```

**Algorithm**: SARSA (On-policy)
- **Update Target**: Uses the **actual next action** `a_next`
- **Formula**: $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s',a') - Q(s,a)]$
- **Characteristics**:
  - ✅ **Stability**: More stable, doesn't overestimate
  - ✅ **Safe**: Learns actual exploratory behavior
  - ❌ **Convergence**: Slower (includes exploration in updates)
  - ❌ **Policy**: Can't learn greedy policy while exploring with ε-greedy

**Hyperparameters**:
- Learning Rate: `α = 0.1` (fixed)
- Discount Factor: `γ = 0.95` (95% weight on future rewards)
- Temporal Difference: Simple 1-step TD update

### Q-Learning (ql_2way-single-intersection.py)
```python
# QLAgent uses similar update pattern but off-policy
ql_agents[agent_id].learn(next_state=next_state, reward=reward)
```

**Algorithm**: Q-Learning (Off-policy)
- **Update Target**: Uses the **best possible next action**
- **Implied Formula**: $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$
- **Characteristics**:
  - ✅ **Optimality**: Can learn optimal policy while exploring with any strategy
  - ✅ **Speed**: Faster convergence (learns optimal directly)
  - ❌ **Stability**: Can overestimate Q-values (optimism bias)
  - ❌ **Variance**: Higher variance in early training

**Hyperparameters**:
- Learning Rate: `α = 0.3` (configurable)
- Discount Factor: `γ = 0.99` (configurable)
- Exploration: `EpsilonGreedy` with decay

### DQN (iiser-v3.ipynb)
```python
current_q = q_nets[agent](b_obs).gather(1, b_act)

with torch.no_grad():
    max_next_q = q_nets[agent](b_next_obs).max(1)[0].unsqueeze(1)
    target_q = b_rew + (gamma * max_next_q * (1 - b_done))

loss = loss_fn(current_q, target_q)
optimizers[agent].zero_grad()
loss.backward()
optimizers[agent].step()
```

**Algorithm**: Deep Q-Network (Off-policy)
- **Update Target**: Similar to Q-Learning but with neural network approximation
- **Formula**: $\mathcal{L} = (r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2$
- **Loss Function**: Mean Squared Error (MSE)

**Key Differences from Q-Learning**:
1. **Replay Buffer**: Experience stored and sampled randomly
   ```python
   replay_buffers[agent].append((observations[agent], actions[agent], 
                                 rewards[agent], next_observations[agent], 
                                 terminations[agent] or truncations[agent]))
   
   batch = random.sample(replay_buffers[agent], batch_size)
   ```
   - ✅ Reduces correlation between samples
   - ✅ Better sample efficiency
   - ❌ Memory overhead

2. **Neural Network Approximation**: Scales to high-dimensional states
3. **Batch Training**: Updates on 64 samples at once (vectorized)

**Hyperparameters**:
- Learning Rate: `lr = 1e-3` (Adam optimizer)
- Discount Factor: `γ = 0.99`
- Batch Size: `64`
- Replay Buffer Size: `10,000`
- Epsilon Decay: `0.97` (aggressive)

### Exploration-Exploitation Balance

#### SARSA
- Initial Phase: Heavy exploration (ε=1.0)
- Decay: `epsilon *= 0.995` per episode
- Final Phase: Minimal exploitation (ε≥0.01)
- **Timeline**: Slow transition, explores thoroughly

#### Q-Learning
- Initial Phase: Moderate exploration (ε=0.3)
- Decay: `epsilon *= 0.9995` per episode
- Final Phase: Minimal exploitation (ε≥0.005)
- **Timeline**: Very slow, exploits quickly
- **Hybrid**: Plus 50% integration of domain heuristic

#### DQN
- Initial Phase: Aggressive exploration (ε=1.0)
- Decay: `epsilon *= 0.97` per episode (episodes lasting longer)
- Final Phase: Minimal exploitation (ε≥0.05)
- **Timeline**: Fast transition to exploitation
- **Advantage**: Replay buffer reduces importance of aggressive exploitation

### Optimization Comparison

| Aspect | SARSA | Q-Learning | DQN |
|--------|-------|-----------|-----|
| **Algorithm** | On-Policy | Off-Policy | Off-Policy |
| **Bootstrap** | Current Action | Best Action | Best Action (NN) |
| **Stability** | High | Medium | Medium-High |
| **Convergence Speed** | Slow | Medium | Fast |
| **Complexity** | Low | Low | High |
| **Sample Efficiency** | Low | Low | High (replay buffer) |
| **State Scalability** | Low | Low | High |
| **Overestimation Bias** | None | Yes | Yes |
| **Convergence Guarantee** | ✅ Yes | ✅ Yes | ❌ No (approx) |

---

## 5. CONVERGENCE OF REWARD

### Definition
Convergence occurs when the average episode reward stabilizes over consecutive episodes.

### SARSA (hello.py)

**Convergence Conditions**:
1. Learning rate decay (implicit): Learning rate doesn't decay - **potential issue**
2. Exploration reduction: ε decays aggressively (ε *= 0.995)
3. Q-table stability: Q-values eventually stabilize

**Theoretical Guarantee**:
- ✅ **Guaranteed convergence** (tabular SARSA with ε-greedy)
- Condition: All state-action pairs must be visited infinitely often
- Requirements: Minimum ε > 0 ✅ (ε_min = 0.01), fixed small learning rate ✅ (α = 0.1)

**Expected Behavior**:
```
Episode 1-5:     High variance (exploration phase)
Episode 6-20:    Decreasing variance (learning)
Episode 20+:     Stabilized rewards (policy convergence)
```

**Convergence Time**: **Medium** - depends on state space size

### Q-Learning (ql_2way-single-intersection.py)

**Convergence Conditions**:
1. Learning rate: Decays implicitly with exploration reduction
2. Exploration: ε decays from 0.3 to 0.005 (very slow)
3. State space: Can be large (multiple agents)

**Theoretical Guarantee**:
- ✅ **Guaranteed convergence to optimal Q-values** (tabular Q-Learning)
- More specific guarantee than SARSA:
  - Learning off-policy from exploratory actions
  - Guaranteed to converge to $Q^*$ (optimal policy)
- Condition: All state-action pairs visited infinitely often

**Hybrid Strategy Impact**:
- Heuristic (50% of actions): Reduces exploration space
- Q-Learning (50% of actions): Accelerates learning
- **Result**: Faster practical convergence despite theoretical same guarantee

**Convergence Time**: **Quick to Medium** - accelerated by domain knowledge

### DQN (iiser-v3.ipynb)

**Convergence Conditions**:
1. Network initialization: Random weights
2. Exploration: ε decays 0.97 per episode
3. Replay buffer: Provides stable training targets
4. Batch size: 64 (smooths updates)

**Theoretical Guarantee**:
- ❌ **No convergence guarantee** (neural network approximation)
- Challenges:
  - Function approximation error
  - Overestimation from max operation
  - Off-policy corrections incomplete
- Practical convergence: Usually achieves near-optimal in finite episodes

**Stability Mechanisms**:
- Replay buffer: Breaks sample correlation
- Random batch sampling: Reduces catastrophic forgetting
- MSE loss: Smooth optimization landscape

**Convergence Time**: **Fast** - typically 50-100 episodes (as set)

### Convergence Metrics

```python
# All track: reward_history / all_episode_rewards
all_episode_rewards = []
for episode in range(episodes):
    episode_reward = ...
    all_episode_rewards.append(episode_reward)
```

**Convergence Detection Methods**:

| Agent | Method |
|-------|--------|
| SARSA | 10-episode moving average stabilizes |
| Q-Learning | Average reward > threshold for N consecutive episodes |
| DQN | Plotting confirms reward flattening (figure at end) |

### Convergence Challenges

#### SARSA
- ❌ On-policy: Must explore while converging (slower)
- ✅ Stable: Conservative updates prevent oscillation

#### Q-Learning
- ✅ Off-policy: Can exploit while learning
- ❌ Overestimation: May require more careful tuning
- ✅ Hybrid helps: Heuristic accelerates convergence direction

#### DQN
- ✅ Fast: Aggressive updates with neural network
- ❌ Unstable: Can diverge if hyperparameters wrong
- ✅ Replay buffer helps: Stabilizes optimization

---

## 6. AGENT FUNCTIONALITY

### SARSA Agent (hello.py)

**Core Loop**:
```python
while not done:
    next_state, reward, terminated, truncated, info = env.step(action)
    next_action = agent.choose_action(next_state)  # 1. Choose NEXT action
    agent.learn(state, action, reward, next_state, next_action)  # 2. Learn with both actions
    state = next_state
    action = next_action
    total_reward += reward
```

**Learning Flow**:
1. **Observe** current state
2. **Act** with ε-greedy policy
3. **Receive** reward and next state
4. **Choose** next action
5. **Update** Q-table using **actual next action**
6. **Move** to next state

**Key Characteristic**: Learns from actions actually taken (including exploration)

**Bias**: Conservative - tends to learn safe policies

### Q-Learning Agent (ql_2way-single-intersection.py)

**Learning Update**:
```python
ql_agents[agent_id].learn(next_state=next_state, reward=reward)
# Internally uses: max_a Q(s', a)
```

**Learning Flow**:
1. **Observe** current state
2. **Act** with hybrid strategy (50% heuristic, 50% ε-greedy)
3. **Receive** reward and next state
4. **Update** Q-values using **best possible next action** (not necessarily taken)
5. **Prepare** next action

**Key Characteristic**: Learns optimal policy despite exploratory actions

**Enhancement**: Heuristic action selection
- 50% of time: Choose phase serving lane with max queue
- 50% of time: Let QL agent decide
- Result: Accelerated learning + domain expertise

**Bias**: Optimistic - tends toward aggressive policies

### DQN Agent (iiser-v3.ipynb)

**Learning Architecture**:
```python
# 1. Store experience in replay buffer
replay_buffers[agent].append((obs, action, reward, next_obs, done))

# 2. Sample and train
if len(replay_buffers[agent]) > batch_size:
    batch = random.sample(replay_buffers[agent], batch_size)
    
    # 3. Forward pass
    current_q = q_nets[agent](b_obs).gather(1, b_act)
    max_next_q = q_nets[agent](b_next_obs).max(1)[0]
    target_q = b_rew + (gamma * max_next_q * (1 - b_done))
    
    # 4. Backward pass
    loss = loss_fn(current_q, target_q)
    loss.backward()
    optimizer.step()
```

**Learning Flow**:
1. **Collect** transitions in replay buffer
2. **Sample** mini-batch (64 experiences)
3. **Evaluate** current Q-values for taken actions
4. **Compute** target Q-values using best next actions
5. **Backpropagate** loss through neural network
6. **Update** network weights via Adam optimizer

**Key Differences**:
- Asynchronous updates (offline learning from buffer)
- Batch learning (vectorized, more stable)
- Network handles high-dimensional states
- Non-tabular (generalization to unseen states)

**Network Architecture**:
```python
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        self.fc1 = nn.Linear(obs_dim, 64)  # Input layer
        self.fc2 = nn.Linear(64, 64)        # Hidden layer (64 neurons)
        self.fc3 = nn.Linear(64, action_dim) # Output layer (one Q-value per action)
```

---

## 7. LEARNING RATES AND DISCOUNT FACTORS

### SARSA

**Learning Rate (α = 0.1)**:
```python
self.lr = 0.1
target = r + self.gamma * self.q_table[sn_k][a_next]
self.q_table[s_k][a] += self.lr * (target - predict)
# Update magnitude = 0.1 * TD_error
```

**Impact**:
- Fixed rate (no decay)
- ✅ Stable: Small updates prevent oscillation
- ❌ Slow convergence: Takes longer to reach estimates
- **Interpretation**: Blend 90% old knowledge + 10% new signal

**Discount Factor (γ = 0.95)**:
```python
self.gamma = 0.95
target = r + self.gamma * self.q_table[sn_k][a_next]
# Future rewards weighted at 95%
```

**Impact**:
- $Q(s,a) = r + 0.95 \cdot Q(s',a')$
- 95% importance to future rewards, 5% to immediate
- ✅ Clear horizon: ~20 steps ahead remain influential
- Suitable for traffic (immediate plus near-future coordination matters)

**Sensitivity**:
```
γ = 0.90 → Myopic (favor immediate reward)
γ = 0.95 → Balanced (our setting)
γ = 0.99 → Far-sighted (care about long-term)
```

### Q-Learning

**Learning Rate (α = 0.3)** - Configurable
```python
prs.add_argument("-a", dest="alpha", type=float, default=0.3)
# Used in QLAgent internally
```

**Impact**:
- **3x higher than SARSA** (0.3 vs 0.1)
- ✅ Faster updates: Off-policy allows aggressive learning
- ⚠️ Higher variance: Can oscillate more
- **Interpretation**: 70% old + 30% new signal

**Discount Factor (γ = 0.99)** - Configurable
```python
prs.add_argument("-g", dest="gamma", type=float, default=0.99)
```

**Impact**:
- Higher than SARSA: 99% vs 95%
- ✅ Long-term horizon: ~100 steps remain influential
- Traffic-adaptive: Accounts for signal phase timing across multiple cycles
- Suitable for coordinated multi-intersection scenarios

### DQN

**Learning Rate (lr = 1e-3 = 0.001)**:
```python
optimizers[agent] = optim.Adam(q_nets[agent].parameters(), lr=learning_rate)
# lr=1e-3 with Adam optimizer
```

**Nature**: Dynamic learning rate
```python
# Adam adjusts per-parameter learning rates based on:
# - First moment (gradient mean)
# - Second moment (gradient variance)
# - Adaptive per-parameter rates
```

**Impact**:
- **Much lower than Q-Learning**: 0.001 vs 0.3
- Necessity: Neural network optimization requires careful step sizes
- Adam advantage: Stabilizes training, adaptive per layer
- ✅ Prevents divergence: Network weights stay in reasonable range

**Discount Factor (γ = 0.99)**:
```python
target_q = b_rew + (gamma * max_next_q * (1 - b_done))
```

**Impact**: Same as Q-Learning (99%)

**Dynamic Adjustment**: None in current code
- ❌ Fixed throughout training
- Consideration: Could implement learning rate annealing for better convergence

### Parameter Sensitivity Analysis

| Parameter | SARSA | QL | DQN | Impact |
|-----------|-------|----|----|--------|
| **α (low)** | Stable | Stable | Stable | Slow learning |
| **α (high)** | Unstable | Unstable | Diverges | Fast but risky |
| **γ (low)** | Myopic | Myopic | Myopic | Ignores future |
| **γ (high)** | Long-term | Long-term | Long-term | Requires convergence |

### Recommended Ranges for Traffic Control

| Algorithm | Recommended α | Recommended γ | Rationale |
|-----------|---------------|---------------|-----------|
| SARSA | 0.05-0.2 | 0.95-0.98 | Conservative; next 15-30 steps matter |
| Q-Learning | 0.1-0.5 | 0.95-0.99 | Off-policy allows higher α; longer horizon |
| DQN | 1e-4 to 1e-2 (varies) | 0.99 | Network updates carefully; far-sighted |

---

## 8. STATE REPRESENTATION

### Observation Design

All agents ultimately work with **lane-based observations**:

```python
# Standard sumo-rl observation (before custom enhancement)
observation = [
    lane_1_density,
    lane_2_density,
    lane_3_density,
    lane_4_density,
]
# Density = number_of_vehicles / lane_capacity
```

### SARSA (hello.py)

**State Processing**:
```python
def get_state_key(self, state):
    return tuple(np.round(state, 1))  # Discretize [0,1] into 0.0, 0.1, 0.2, ..., 1.0
```

**Characteristics**:
- **Type**: **Continuous (input) → Discrete (tabular)**
- **Rounding**: Rounds to 1 decimal place
- **State Space Size**: $(10)^4 = 10,000$ possible states (for 4 lanes)
- **Example States**:
  ```
  (0.1, 0.2, 0.5, 0.3)  → Lane densities rounded to 0.1
  (0.0, 0.0, 1.0, 1.0)  → High congestion on lanes 3&4
  ```

**Advantages**:
- ✅ Interpretable: Each dimension represents lane density
- ✅ Tabular Q-learning applicable
- ✅ Low computational cost

**Disadvantages**:
- ❌ Information loss: Rounding loses precision
- ❌ Fixed resolution: Can't adapt to problem complexity
- ❌ Scalability: 8 lanes → 10^8 states (not practical)

### Q-Learning (ql_2way-single-intersection.py)

**State Processing**:
```python
# Uses sumo_rl.SumoEnvironment observation encoder
state, info = env.reset()

# Internally encodes:
starting_state = env.encode(initial_states[ts], ts)
next_state = env.encode(s[agent_id], agent_id)
```

**Characteristics**:
- **Type**: **Custom discrete encoding** (depends on sumo-rl version)
- **Fallback**: `str(initial_states[ts])` if encoder fails
- **Typically includes**:
  - Lane queue lengths (vehicles awaiting green)
  - Lane densities
  - Traffic light phase information
  - Optional: emergency vehicle presence

**Advantages**:
- ✅ Domain-specific encoding
- ✅ Captures traffic signal state
- ✅ Suitable for multi-agent coordination

**Disadvantages**:
- ❌ Decoder dependency: Black box in sumo_rl library
- ❌ Variable state space
- ⚠️ Fallback mechanism: String encoding if standard fails

### DQN (iiser-v3.ipynb)

**Custom Observation Function**:
```python
class PriorityObservationFunction(ObservationFunction):
    def __call__(self):
        obs = []
        density = self.ts.get_lanes_density()  # Lane densities
        obs.extend(density)
        
        for lane in self.ts.lanes:
            emergency_waiting = 0.0
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for veh in vehicles:
                if traci.vehicle.getTypeID(veh) == "ambulance":
                    emergency_waiting = 1.0
                    break
            obs.append(emergency_waiting)
        
        return np.array(obs, dtype=np.float32)
    
    def observation_space(self):
        return spaces.Box(low=0., high=1., shape=(len(self.ts.lanes) * 2,), dtype=np.float32)
```

**Structure**:
```
Observation = [density_lane1, density_lane2, ..., density_laneN,
               emergency_lane1, emergency_lane2, ..., emergency_laneN]
               
Example (4 lanes):
[0.45, 0.12, 0.89, 0.23, 1.0, 0.0, 0.0, 0.0]
 └─ Densities ┘         └─ Emergency presence ┘
```

**Characteristics**:
- **Type**: **Continuous** (no discretization)
- **Dimension**: `2 * num_lanes` (8 for 4 lanes)
- **Range**: [0, 1] (normalized)
- **Special Feature**: Emergency vehicle detection (binary flag per lane)

**State Space**:
- Infinite continuous space
- Neural network handles high-dimensional input
- ✅ No discretization loss
- ✅ Continuous problem representation

**Advantages**:
- ✅ Domain-informed: Includes emergency priority signal
- ✅ Linear features: Easy for NN to learn
- ✅ Scalable: Works with any number of lanes
- ✅ Continuous: Preserves natural state representation

**Disadvantages**:
- ❌ Requires neural network: Can't use tabular methods
- ❌ Training overhead: More complex optimization

### State Representation Comparison

| Aspect | SARSA | Q-Learning | DQN |
|--------|-------|-----------|-----|
| **Type** | Discrete | Discrete | Continuous |
| **Input** | Continuous → Rounded | Custom Encoding | Continuous |
| **State Space** | $10^{lanes}$ | Variable | Infinite |
| **Scalability** | ❌ Poor | ⚠️ Moderate | ✅ Good |
| **Domain Info** | ❌ Generic | ⚠️ Implicit | ✅ Explicit (Emergency) |
| **Interpretability** | ✅ High | ⚠️ Medium | ⚠️ Low (NN hidden) |
| **Algorithm Suitability** | Tabular | Tabular | Function approx |

### Observation Space Dimensions

```python
# SARSA: Implicitly large due to rounding
# Practical: ~10,000 (4 lanes × 10 bins each)

# Q-Learning: Depends on encoder
# Typical: Similar to SARSA or more complex

# DQN: Fixed and scalable
obs_dim = 2 * num_lanes  # 8 for 4 lanes, 16 for 8 lanes
print(f"Observation dimension: {obs_dim}")
```

---

## 9. EXPLORATION VS EXPLOITATION

### Fundamental Trade-off

**Exploration**: Trying new actions to discover better strategies
**Exploitation**: Using the best known strategy

### SARSA (hello.py)

**Strategy**: Epsilon-Greedy with Decay

```python
def choose_action(self, state):
    if np.random.rand() < self.epsilon:
        return np.random.randint(self.action_size)  # Random (explore)
    return np.argmax(self.q_table[state_key])  # Best Q-value (exploit)

def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
    self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
```

**Timeline**:
```
Episode  │ Epsilon │ Behavior
────────────────────────────────
0        │ 1.00   │ 100% exploration
5        │ 0.975  │ 98% exploration
50       │ 0.605  │ 61% exploration
100      │ 0.366  │ 37% exploration
500      │ 0.008  │ 0.8% exploration (mostly exploit)
```

**Pattern**: Very slow decay
- ✅ Thorough exploration: Discovers many strategies
- ❌ Late exploitation: Takes long to converge

**Decay Mathematics**:
- $\epsilon_t = 0.01 \text{ (min)} + (1.0 - 0.01) \times 0.995^t$
- Half-life: ~138 episodes (reaches ~50.5%)

### Q-Learning (ql_2way-single-intersection.py)

**Strategy**: Hybrid Heuristic + Epsilon-Greedy

```python
if args.fixed:
    # Pure fixed signal (no RL)
    pass
else:
    for ts in ql_agents.keys():
        if np.random.rand() < 0.2 and emergency_count >= 2:
            actions[ts] = 0  # Emergency priority
        else:
            if np.random.rand() < 0.5:
                actions[ts] = lane_to_phase.get(max_lane, 0)  # Heuristic: serve worst lane
            else:
                actions[ts] = ql_agents[ts].act()  # RL decision
```

**Three-Level Decision Process**:
1. **20% chance**: If ≥2 emergency vehicles, force phase 0
2. **50% chance** (left): Use heuristic (serve max queue lane)
3. **50% chance** (right): Use QL agent's ε-greedy decision

**Effective Strategy**:
```
15%  Emergency priority
42.5% Heuristic (50% × 85%)
42.5% Q-Learning (50% × 85%)
```

**Advantages**:
- ✅ Hybrid: Combines domain knowledge with learning
- ✅ Informed exploration: Heuristic guides toward good actions
- ✅ Safety: Always considers emergencies

**Trade-off**:
- Less pure RL learning (42.5% vs 100%)
- Faster practical convergence (domain knowledge helps)

**Exploration Strategy**: Implicit through EpsilonGreedy
```python
exploration_strategy=EpsilonGreedy(
    initial_epsilon=args.epsilon,      # Default 0.3
    min_epsilon=args.min_epsilon,      # Default 0.005
    decay=args.decay                   # Default 0.9995
)
```

- Initial ε: 0.3 (less aggressive than SARSA)
- Decay: 0.9995 (very slow, similar to SARSA)
- Min ε: 0.005 (almost pure exploitation end-state)

### DQN (iiser-v3.ipynb)

**Strategy**: Epsilon-Greedy with Aggressive Decay

```python
# Hyperparameters
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.97

# Update loop
for episode in tqdm(range(episodes), desc="Training Episodes"):
    # ... training ...
    epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Per-episode decay
```

**Timeline** (50 episodes):
```
Episode  │ Epsilon │ Behavior
────────────────────────────────
0        │ 1.00   │ 100% exploration
1        │ 0.97   │ 97% exploration
5        │ 0.859  │ 86% exploration
10       │ 0.738  │ 74% exploration
15       │ 0.632  │ 63% exploration
25       │ 0.462  │ 46% exploration
35       │ 0.337  │ 34% exploration
50       │ 0.195  │ 20% exploration (mostly exploit)
```

**Characteristics**:
- ✅ Fast transition: 3x faster decay than SARSA
- ✅ Rapid exploitation: Learns and applies quickly
- ❌ Less exploration: Misses some strategies
- ⚠️ Higher variance in early episodes

**Why Aggressive Decay Works**:
1. **Replay buffer**: Old good experiences remain available
2. **Neural network**: Generalizes from explored states
3. **Batch training**: Smooths updates, less noisy
4. **Off-policy**: Can exploit while replay buffer contains explorations

**Mathematical Comparison**:
```
Half-life (reaches ~50%):
SARSA:        ~138 episodes
Q-Learning:   ~600 episodes
DQN:          ~22 episodes (50 episodes total)
```

### Exploration Methods Employed

#### All Three
- ✅ **ε-Greedy**: Based on random number, choose random action
- ✅ **Decaying ε**: Reduce exploration over time
- ✅ **Minimum ε**: Never fully stop exploring

#### Q-Learning Only
- ✅ **Heuristic guidance**: Incorporate domain knowledge
- ✅ **Learned priorities**: Emergency vehicle detection

#### DQN Only
- ✅ **Implicit exploration**: Replay buffer prevents forgetting
- ✅ **Generalization**: NN learns from partially explored space
- ✅ **Batch effects**: Sampling smooths exploration

### Advanced Exploration Techniques NOT Used

❌ **Curiosity-driven exploration**: No intrinsic motivation/novelty detection
❌ **Boltzmann exploration**: Always ε-greedy, not softmax
❌ **Upper Confidence Bound (UCB)**: No bonus for uncertain actions
❌ **Thompson sampling**: No posterior sampling over Q-values
❌ **Intrinsic motivation**: No bonus for information gain

---

## 10. POLICY REPRESENTATION

### Definition
Policy: Mapping from states to actions (deterministic or stochastic)

### SARSA (hello.py)

**Explicit Policy Representation**:
```python
Q-table: Dictionary[State, Array[Q-values]]
self.q_table = {}
self.q_table[state_key] = np.zeros(self.action_size)

# Example for 4 lanes, 4 actions:
self.q_table[(0.1, 0.2, 0.5, 0.3)] = [12.5, -3.2, 8.1, 4.7]
                                       └─────────────────────┘
                                        Q-values for each action
```

**Policy Derivation**:
$$\pi(a|s) = \begin{cases} 1 & \text{if } a = \arg\max_a Q(s,a) \\ 0 & \text{otherwise} \end{cases}$$

**Deterministic Policy**: Greedy action selection → One best action per state

**Update Method**:
```python
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
# SARSA: a' is the ACTUAL next action taken
```

**Advantages**:
- ✅ **Interpretable**: Can inspect Q-values for any state
- ✅ **Explicit**: Actions directly readable
  ```python
  for state in agent.q_table:
      best_action = np.argmax(agent.q_table[state])
      print(f"In state {state}, take action {best_action}")
  ```
- ✅ **Efficient**: O(1) policy lookup

**Disadvantages**:
- ❌ **Memory**: Stores Q-values for all visited states
- ❌ **Scalability**: Infeasible for large/continuous state spaces
- ❌ **Generalization**: No knowledge transfer to unseen states

### Q-Learning (ql_2way-single-intersection.py)

**Policy Representation**: Q-table (same structure as SARSA)
```python
ql_agents[ts] = QLAgent(
    starting_state=starting_state,
    state_space=env.observation_space,
    action_space=env.action_space,
    alpha=args.alpha,
    gamma=args.gamma,
    exploration_strategy=EpsilonGreedy(...)
)
```

**Q-Learning Difference** (vs SARSA):
```python
# SARSA: target = r + γQ(s',a_taken)
# Q-Learning: target = r + γmax_a Q(s',a)
```

**Update Formula**:
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

**Policy Derivation**:
$$\pi(a|s) = \arg\max_a Q(s,a)$$

**Key Difference from SARSA**:
- Q-Learning learns **optimal policy** (greedy from Q-values)
- SARSA learns **exploratory policy** (includes randomness from ε-greedy)
- Both maintain separate exploration via ε-greedy, but Q-values differ

**Example Scenario**:
```
State s, two actions:
Action A: If taken (exploratory): leads to bad outcome
         Q_SARSA(s,A) = -10 (learns the actual bad outcome)
         Q_QL(s,A) = 5 (learns what best action would give from state s')

Action B: Best action after A
         Leads to +15

Q-Learning learns to exploit B despite exploring A
SARSA learns to avoid A
```

**Advantages**:
- ✅ **Optimal Policy**: Learns best possible actions
- ✅ **Deterministic**: Greedy from Q-table
- ✅ **Fast Convergence**: Off-policy bootstrap

**Disadvantages**:
- Same as SARSA (Q-table limitations)
- ❌ **Overestimation**: Max operator can overestimate Q-values

### DQN (iiser-v3.ipynb)

**Neural Network Policy Representation**:
```python
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Returns Q-values for all actions
```

**Policy Representation**: Parametric model $Q(s,a;\theta)$
- Network weights `θ` encode the policy
- Continuous approximation to Q-table

**Policy Derivation**:
```python
obs_tensor = torch.FloatTensor(observations[agent]).unsqueeze(0)
q_values = q_nets[agent](obs_tensor)  # Shape: (1, n_actions)
action = torch.argmax(q_values).item()  # Greedy selection
```

**Update Method**:
```python
Loss = (r + γ max_a Q(s',a;θ^-) - Q(s,a;θ))²
∂Loss/∂θ ∝ gradient → Update network weights
```

**Network Training**:
```python
for agent in env.agents:
    if len(replay_buffers[agent]) > batch_size:
        batch = random.sample(replay_buffers[agent], batch_size)
        b_obs, b_act, b_rew, b_next_obs, b_done = zip(*batch)
        
        # Compute current Q-values
        current_q = q_nets[agent](b_obs).gather(1, b_act)
        
        # Compute target Q-values
        with torch.no_grad():
            max_next_q = q_nets[agent](b_next_obs).max(1)[0].unsqueeze(1)
            target_q = b_rew + (gamma * max_next_q * (1 - b_done))
        
        # Update network
        loss = loss_fn(current_q, target_q)
        optimizers[agent].zero_grad()
        loss.backward()
        optimizers[agent].step()
```

**Advantages**:
- ✅ **Scalable**: Handles continuous/high-dim states
- ✅ **Generalizable**: Learned features transfer to unseen states
- ✅ **Efficient**: Scales with neural network, not state space
- ✅ **Implicit exploration**: Features learned from replay buffer

**Disadvantages**:
- ❌ **Black box**: Cannot directly interpret what network learned
- ❌ **Instability**: Can diverge, requires careful tuning
- ❌ **Convergence**: No theoretical guarantee (function approximation)

### Policy Update Comparison

| Aspect | SARSA | Q-Learning | DQN |
|--------|-------|-----------|-----|
| **Storage** | Q-table (dict) | Q-table (dict) | Neural network |
| **Lookup** | O(1) | O(1) | O(forward pass) |
| **Interpretability** | ✅ High | ✅ High | ❌ Low |
| **Scalability** | ❌ Poor | ❌ Poor | ✅ Good |
| **Generalization** | ❌ No | ❌ No | ✅ Yes |
| **Update** | Incremental | Incremental | Batch via backprop |
| **Convergence** | ✅ Guaranteed | ✅ Guaranteed | ⚠️ No guarantee |
| **Sample Efficiency** | Low | Low | High (replay buffer) |

### Policy Inspection Examples

**SARSA/QL**:
```python
# Check policy in specific state
state = (0.3, 0.2, 0.7, 0.1)
q_values = agent.q_table[state]  # [8.2, -1.5, 3.1, 6.9]
best_action = np.argmax(q_values)  # 0 (Q-value: 8.2)
print(f"Best action in state {state}: {best_action}")

# Visualize entire policy (if manageable)
for state, q_vals in agent.q_table.items():
    policy_action = np.argmax(q_vals)
    print(f"{state} → action {policy_action}")
```

**DQN**:
```python
# Check policy for specific observation
obs = np.array([0.45, 0.12, 0.89, 0.23, 1.0, 0.0, 0.0, 0.0])
obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
q_values = q_nets[agent](obs_tensor)  # [5.2, -2.1, 7.8, 4.3]
best_action = torch.argmax(q_values).item()  # 2
print(f"Best action: {best_action}")

# Cannot easily enumerate all possible actions:
# "State space" is continuous and infinite
```

---

## 11. STABILITY AND VARIANCE

### Sources of Instability

#### SARSA (hello.py)

**Stability Factors**:
- ✅ **On-policy**: Only learns from actual behavior (conservative)
- ✅ **Low learning rate**: α = 0.1 (small steps)
- ✅ **Continues exploration**: ε_min = 0.01 (never fully exploits)
- ✅ **Q-table bounds**: Finite state space naturally bounded

**Variance Characteristics**:
- **Reward variance**: High in early episodes (exploration)
- **Q-value variance**: Decreases with more updates
- **Policy variance**: Gradual change due to slow ε decay

**Stability Guarantee**:
```
If:
  - α ∈ [0, 1]
  - Σα_t = ∞ (sum of learning rates diverges → explore everything)
  - Σα_t² < ∞ (sum of squares converges → eventually converge)
  - All (s,a) visited infinitely often
  
Then: Q-table converges to true Q-values
```

Check for SARSA:
- ✅ Fixed α = 0.1 ✓
- ✅ ε_min = 0.01 ensures revisits ✓

**Risk**: Low - this is a reference algorithm for stability

#### Q-Learning (ql_2way-single-intersection.py)

**Stability Factors**:
- ⚠️ **Off-policy**: Aggressive bootstrap (max operator)
- ✅ **Higher learning rate**: α = 0.3 (faster updates, higher risk)
- ✅ **Heuristic stabilization**: 50% domain-guided actions
- ⚠️ **Overestimation**: Max operator can inflate Q-values

**Overestimation Problem**:
```python
target = r + gamma * max_a Q(s', a)
#                   └─ May overestimate if Q-values too high
```

**Example**:
```
True Q-values: [10, 8]
Estimated Q:   [12, 11]  (overestimated)
Max operation: Select 12  (worse: picks overestimated value)

Over time: Q-values drift upward unrealistically
```

**Mitigation in Code**:
- Heuristic guidance: 50% domain knowledge (reduces random exploration)
- Multi-agent environment: Multiple simultaneous learning signals

**Variance Characteristics**:
- **Reward variance**: Medium (guidance reduces randomness)
- **Q-value variance**: Higher than SARSA (aggressive bootstrap)
- **Policy variance**: Moderate (min ε = 0.005 very low)

**Stability Guarantee**:
```
Requires: αk and εk conditions (same as SARSA)
PLUS: α_k * (estimated_error) bounded
```

Check:
- ⚠️ Higher α may cause issues
- ✅ Heuristic partially mitigates

**Risk**: Medium - needs careful tuning

#### DQN (iiser-v3.ipynb)

**Potential Instabilities**:
1. **Function Approximation Error**: NN doesn't perfectly approximate Q-function
2. **Replay Buffer Stale Data**: Old transitions may not reflect current policy
3. **Overestimation**: Max operator (same as Q-Learning)
4. **Catastrophic Forgetting**: New learning overwrites old knowledge
5. **Deadly Triad**: Combination of:
   - Function approximation (NN)
   - Off-policy learning (DQN bootstraps)
   - Temporal difference (Q-based)

**Stabilization Techniques Used**:

```python
# 1. REPLAY BUFFER: Breaks temporal correlation
replay_buffers[agent] = deque(maxlen=replay_buffer_size)  # 10,000
batch = random.sample(replay_buffers[agent], batch_size)  # Sample 64

# Benefit: Different samples in each batch → diverse gradients
```

```python
# 2. BATCH LEARNING: Smoother optimization
for agent in env.agents:
    if len(replay_buffers[agent]) > batch_size:
        # Update on 64 samples at once (vectorized)
        # VS: Update on single transition (noisy)
```

```python
# 3. GRADIENT CLIPPING (not in code, but best practice):
# loss.backward()
# torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
# optimizer.step()
```

**Stabilization Techniques NOT Used** (could improve):
- ❌ **Target Network**: Separate network for computing targets
  ```python
  # Current: max_next_q = q_nets[agent](b_next_obs).max(1)[0]
  # Better: max_next_q = target_q_nets[agent](b_next_obs).max(1)[0]
  #        (update target networks every N steps)
  ```
- ❌ **Double DQN**: Use different network for action selection and evaluation
- ❌ **Dueling DQN**: Separate value and advantage streams
- ❌ **Prioritized Replay**: Sample important transitions more often

**Variance Characteristics**:
- **Reward variance**: High (learning curve noisy)
- **Loss variance**: Batch reduces variance, but bootstrap still present
- **Policy variance**: Can change rapidly (ε decay = 0.97/episode)

**Stability Guarantee**:
- ❌ No formal convergence guarantee (function approximation)
- Empirical convergence: Usually within 50 episodes (as set)

**Risk**: Medium-High without target networks
- Current implementation is functional but not production-grade
- May diverge on harder problems

### Variance Reduction Techniques

| Technique | SARSA | QL | DQN |
|-----------|-------|----|----|
| **Low learning rate** | ✅ 0.1 | ⚠️ 0.3 | ✅ 0.001 |
| **Replay buffer** | ❌ No | ❌ No | ✅ Yes |
| **Batch updates** | ❌ No | ❌ No | ✅ Yes (64) |
| **Target network** | ❌ No | ❌ No | ❌ No |
| **Double bootstrapping** | ❌ No | ❌ No | ❌ No |
| **Gradient clipping** | ❌ No | ❌ No | ❌ No |

### Stability Ranking

```
Stability (best to worst):
1. SARSA - Most stable, on-policy, low learning rate
2. Q-Learning with Heuristics - Medium, off-policy mitigated by domain knowledge
3. Basic DQN - Less stable, function approximation, but replay buffer helps
4. DQN (without target network) - Susceptible to divergence on harder problems
```

---

## 12. EXPLORATION TECHNIQUES

### Basic Technique: Epsilon-Greedy
All three agents use this as PRIMARY exploration method.

**Advanced Techniques Investigation**:

#### SARSA (hello.py)
```python
# Mechanism: Pure ε-greedy
if np.random.rand() < self.epsilon:
    return np.random.randint(self.action_size)  # Uniform random
else:
    return np.argmax(self.q_table[state_key])
```

**Advanced Techniques**: ❌ NONE
- No curiosity-driven exploration
- No intrinsic motivation
- No bootstrapping uncertainty
- No upper confidence bounds

**Assessment**: Basic but sufficient for traffic control domain

#### Q-Learning (ql_2way-single-intersection.py)
```python
# Primary: ε-greedy (in EpsilonGreedy class)

# Secondary: Domain Heuristic
if np.random.rand() < 0.5:
    actions[ts] = lane_to_phase.get(max_lane, 0)  # Serve worst lane
else:
    actions[ts] = ql_agents[ts].act()  # RL decision

# Tertiary: Emergency Priority
if emergency_count >= 2 and np.random.rand() < 0.2:
    actions[ts] = 0  # Fixed safe action
```

**Exploration Effect**:
1. **Heuristic guidance**: Biases exploration toward reasonable actions
2. **Emergency override**: Exploration-exploitation trade-off sacrificed for safety
3. **Multi-level**: Combines multiple exploration principles

**Assessment**: Practical and domain-aware
- ✅ Incorporates prior knowledge
- ✅ Safety override
- ⚠️ Reduces pure RL learning percentage

#### DQN (iiser-v3.ipynb)
```python
# Primary: ε-greedy
if random.random() < epsilon:
    actions[agent] = env.action_space(agent).sample()  # Random
else:
    q_values = q_nets[agent](obs_tensor)
    actions[agent] = torch.argmax(q_values).item()  # Greedy

# Secondary: Replay Buffer
replay_buffers[agent].append((obs, action, reward, next_obs, done))
batch = random.sample(replay_buffers[agent], batch_size)
# - Revisits old good experiences
# - Provides implicit exploration guidance
```

**Replay Buffer as Exploration Tool**:
```
Episode 1-10: Agent explores, stores diverse experiences
Episode 11+: Agent re-learns from diverse batch samples
Effect: Virtual exploration via replayed experiences
```

**Assessment**: Implicit exploration through experience replay
- ✅ Gentle exploration (batch smooths noise)
- ✅ Reuses past discoveries
- ⚠️ May get stuck in local optima

### Comparison of Exploration Sophistication

| Technique | SARSA | QL | DQN |
|-----------|-------|----|----|
| **ε-Greedy** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Uncertainty-based** | ❌ No | ❌ No | ❌ No |
| **Curiosity-driven** | ❌ No | ❌ No | ❌ No |
| **Domain heuristic** | ❌ No | ✅ Yes | ❌ No |
| **Experience replay** | ❌ No | ❌ No | ✅ Yes |
| **Intrinsic rewards** | ❌ No | ❌ No | ❌ No |

### Advanced Techniques NOT Implemented

#### 1. Uncertainty-Based Exploration
**Concept**: Explore states where uncertainty is high
```python
# Not used: Could compute Q-value std from ensemble
# std_dev = compute_uncertainty(state)
# bonus = β * std_dev
# action = argmax(q_values + bonus)
```

**Why skipped**: 
- Requires ensemble or bootstrap (not in code)
- Tabular methods don't have natural uncertainty

#### 2. Curiosity-Driven Exploration (ICM)
**Concept**: Reward for reducing prediction error
```python
# Not used: prediction_loss = ||forward_model(s,a) - s'||
# intrinsic_reward = prediction_loss
# total_reward = extrinsic + λ * intrinsic
```

**Why skipped**: 
- Complex to implement
- Traffic control already has dense rewards
- Not necessary for this domain

#### 3. Boltzmann Exploration
**Concept**: Softmax action selection (temperature-based)
```python
# Not used:
# probs = softmax(q_values / temperature)
# action = sample_from(probs)
```

**Why skipped**: 
- ε-greedy simpler and works well
- Tabular methods don't need softmax

#### 4. Upper Confidence Bound (UCB)
**Concept**: Optimistic action selection
```python
# Not used:
# ucb_bonus = sqrt(log(t) / N(s,a))
# action = argmax(q_values + c * ucb_bonus)
```

**Why skipped**: 
- Requires count tracking (N)
- Stationary environment (doesn't apply well)

### Impact Assessment

**Current Exploration Effectiveness**:

| Agent | Exploration Quality | Discovery Rate | Exploit Speed |
|-------|-------------------|-----------------|---------------|
| SARSA | Thorough | Slow (visits many states) | Slow (ε >0 long) |
| QL | Guided | Fast (heuristic) | Fast (domain helps) |
| DQN | Implicit | Medium (replay buffer) | Fast (ε decay 0.97) |

**Scenarios Where More Exploration Needed**:
- ❌ Very large state spaces (exponential states)
- ❌ Deceptive reward structures
- ❌ Sparse rewards (current: relatively dense)

**Current Adequacy**: 
✅ For traffic control, ε-greedy sufficient because:
1. State space not huge (~thousands to millions)
2. Reward dense (every step evaluated)
3. Problem time-bounded (episodes max 1000 seconds)

---

## 13. TRANSFER LEARNING

### Concept
Transfer learning: Reuse knowledge from one task/environment to another

### Current Implementation Status

#### SARSA (hello.py)
```python
# Model saving: NONE in visible code
# Models loaded:  NONE
# Transfer: ❌ NO TRANSFER LEARNING
```

**Limitation**: Entire Q-table discarded after training

#### Q-Learning (ql_2way-single-intersection.py)
```python
# Model saving: Implicit in sumo_rl
# Models loaded: May save to CSV via out_csv_name

# Potential:
out_csv = f"outputs/2way-single-intersection/{experiment_time}_..."
env.save_csv(out_csv, run)
```

**Status**: ⚠️ Saves training results, not directly transferable models
- CSV logs performance metrics
- Does not save Q-table weights
- ❌ Cannot directly transfer learned policy

#### DQN (iiser-v3.ipynb)
```python
# SAVING MODELS - EXPLICIT
for agent in env.possible_agents:
    model_path = Path("models") / "iiser_priority" / f"{agent}.pth"
    torch.save(q_nets[agent].state_dict(), str(model_path))  # Save weights

# Periodic saves
if (episode + 1) % save_interval == 0:
    for agent in env.possible_agents:
        model_path = Path("models") / "iiser_priority" / f"{agent}_ep{episode+1}.pth"
        torch.save(q_nets[agent].state_dict(), str(model_path))

# LOADING MODELS - IN EVALUATION
def evaluate_model():
    eval_q_nets[agent].load_state_dict(torch.load(
        str(Path("models") / "iiser_priority" / f"{agent}.pth")))
    eval_q_nets[agent].eval()
```

**Status**: ✅ TRANSFER-READY DESIGN
- Weights saved in `.pth` format
- Can be loaded into new networks
- Evaluation function demonstrates transfer

### Transfer Learning Opportunities

#### Scenario 1: Different Intersection Topology
**Source**: IISER intersection (4 lanes)
**Target**: Generic 3x3 grid intersection

**Feasibility**:

| Agent | Feasibility | Reason |
|-------|-----------|--------|
| SARSA | ❌ Low | Q-table tied to specific discretization; 3x3 grid has different state space |
| QL | ❌ Low | Same issue; Q-table mismatch |
| DQN | ⚠️ Medium | Neural network can generalize, but observation dimension would differ |

**Challenge**: Different number of lanes → different observation dimension

```python
# IISER: 4 lanes → 8-dimensional observation (2 per lane)
# New intersection: 8 lanes → 16-dimensional observation
# Network input layer expects 8, receives 16 → ERROR
```

**Solution for DQN**:
```python
# Fine-tuning approach:
# 1. Load old network
eval_q_nets[agent].load_state_dict(torch.load("models/iiser.pth"))

# 2. Add adapter layer for new dimension
# old_net: Linear(8) + hidden
# new_net: Adapter(16→8) + old_net

# 3. Train on new intersection with pre-trained hidden layer
# Lower learning rate, frozen early layers if possible
```

#### Scenario 2: Same Intersection, Different Vehicle Mix
**Source**: Balanced traffic (cars + buses + ambulances)
**Target**: Heavy truck traffic + few buses

**Feasibility**:

| Agent | Feasibility | Reason |
|-------|-----------|--------|
| SARSA | ❌ No | Observation space same, but optimal policy changes |
| QL | ❌ No | Same issue |
| DQN | ✅ High | Network learns features of vehicle types; can adapt |

**Transfer Strategy for DQN**:
```python
# 1. Load pre-trained model
model.load_state_dict(torch.load("models/iiser.pth"))

# 2. Fine-tune on new environment
# Keep most network frozen, fine-tune output layer:
for param in model.fc1.parameters():
    param.requires_grad = False  # Freeze early features
for param in model.fc2.parameters():
    param.requires_grad = False  # Freeze middle layer
for param in model.fc3.parameters():
    param.requires_grad = True   # Fine-tune output

# 3. Train with lower learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Lower than 1e-3
```

**Result**: Faster convergence on new vehicle mix (fewer episodes needed)

#### Scenario 3: Same Environment, Different Optimization Goals
**Source**: Priority for ambulances (current)
**Target**: Minimize average wait time (new objective)

**Feasibility**:

| Agent | Feasibility | Reason |
|-------|-----------|--------|
| SARSA/QL | ❌ No | Q-values directly tied to reward function; transfer doesn't apply |
| DQN | ⚠️ Medium | Network features may partially transfer, but reward substantially different |

**Challenge**: Reward function completely different
```python
# Old: priority_reward_fn() → Penalize ambulance stopping
# New: wait_reward_fn() → Penalize total waiting time
# These lead to opposite optimal policies in some cases
```

**Transfer Value**: Limited (different objectives fundamentally misaligned)

### Transfer Learning Comparison

| Aspect | SARSA | QL | DQN |
|-------|--------|----|----|
| **Model Saving** | ❌ No | ⚠️ Implicit | ✅ Yes |
| **Model Loading** | ❌ No | ❌ No | ✅ Yes |
| **Fine-tuning Ready** | ❌ No | ❌ No | ✅ Yes |
| **Feature Transfer** | ❌ None | ❌ None | ✅ Hidden layers |
| **Same environment** | ❌ No | ❌ No | ✅ Good |
| **Similar environment** | ❌ No | ❌ No | ⚠️ Possible |
| **Different objectives** | ❌ No | ❌ No | ❌ No |

### Recommended Transfer Strategy for DQN

**For new intersection (similar topology)**:
```python
# Step 1: Load source model
checkpoint = torch.load("models/iiser_priority/traffic_light_0.pth")
new_model.load_state_dict(checkpoint)

# Step 2: Retrain with lower learning rate
optimizer = optim.Adam(new_model.parameters(), lr=1e-4)  # 10x lower

# Step 3: Use pre-training as initialization
for episode in range(20):  # Fewer episodes needed (vs 50 from scratch)
    # ... training loop ...
```

**Expected Benefit**: ~30-50% faster convergence

---

## 14. PERFORMANCE METRICS

### Metrics Being Tracked

#### SARSA (hello.py)
```python
reward_history = []
for e in tqdm(range(episodes), desc="Training Episodes"):
    total_reward = 0
    # ... training ...
    reward_history.append(total_reward)
    
    writer.add_scalar('Reward/Episode', total_reward, e + 1)
    writer.add_scalar('Epsilon', agent.epsilon, e + 1)

# Post-training visualization
plt.plot(range(1, episodes + 1), reward_history, marker='o', color='b')
```

**Metrics**:
1. **Total Episode Reward**: Sum of all reward signals in episode
2. **Average Reward**: Mean across all episodes
3. **Epsilon (Exploration Rate)**: Decay tracking
4. **Min/Max Rewards**: Range of performance

**Statistics**:
```python
print(f"Average Reward: {np.mean(reward_history):.2f}")
print(f"Max Reward: {np.max(reward_history):.2f}")
print(f"Min Reward: {np.min(reward_history):.2f}")
print(f"Std Dev: {np.std(reward_history):.2f}")
print(f"Last 10 Episodes Average: {np.mean(reward_history[-10:]):.2f}")
```

#### Q-Learning (ql_2way-single-intersection.py)
```python
# Training results saved to CSV
env.save_csv(out_csv, run)

# Implicit metrics:
# - Traffic signal metrics per episode
# - CSV contains per-step data:
#   - Vehicle queues
#   - Wait times
#   - Signal phase changes
```

**Metrics** (CSV-based):
1. **Episode duration**: Simulation seconds
2. **Reward per episode**: Usually sum of wait times (inverse)
3. **Queue lengths**: Lane-specific metrics
4. **Wait times**: Vehicle wait duration

**Multi-run evaluation**:
```python
for run in range(1, args.runs + 1):
    # Train and save per-run results
    env.save_csv(out_csv, run)  # Creates timestamped outputs
```

#### DQN (iiser-v3.ipynb)
```python
all_episode_rewards = []

# During training
for episode in tqdm(range(episodes)):
    episode_reward = 0
    episode_losses = []
    
    while env.agents:
        # ... training ...
        episode_losses.append(loss.item())
    
    all_episode_rewards.append(episode_reward)
    
    # TensorBoard logging
    writer.add_scalar('Reward/Episode', episode_reward, episode)
    writer.add_scalar('Loss/Episode', avg_loss, episode)
    writer.add_scalar('Epsilon', epsilon, episode)

# Post-training visualization
def plot_rewards(rewards, window_size=5):
    plt.plot(rewards, label='Raw Episode Reward')
    smoothed_rewards = np.convolve(rewards, window, mode='valid')
    plt.plot(x_values, smoothed_rewards, label=f'Moving Average (Window=5)')
    plt.savefig('training_curve.png')
```

**Metrics**:
1. **Episode Reward**: Total reward (priority penalties summed)
2. **Episode Loss**: Average MSE loss across batches
3. **Epsilon**: Exploration rate decay
4. **Moving Average**: Smoothed reward (5-episode window)

### TensorBoard Logging

**DQN uses TensorBoard**:
```bash
tensorboard --logdir=runs/dqn_training
```

**Viewable Metrics**:
- Reward per episode (with trend line)
- Loss convergence (should decrease)
- Epsilon decay schedule

### Evaluation Phase

#### SARSA/Q-Learning
```python
# No explicit evaluation phase in provided code
# Metrics derived from training data only
```

#### DQN
```python
def evaluate_model():
    """Evaluate trained model with GUI visualization"""
    total_evaluation_reward = 0
    
    while env.agents:
        # ... greedy policy (no exploration) ...
        for agent in env.agents:
            total_evaluation_reward += rewards[agent]
    
    print(f"Simulation Finished. Final Total Reward: {total_evaluation_reward:.2f}")
```

**Evaluation Metrics**:
- **Total Evaluation Reward**: Performance of learned policy
- **Test episodes**: Separate from training (no exploration)

### Comparison Metrics Table

| Metric | SARSA | QL | DQN |
|--------|-------|----|----|
| **Episode Reward** | ✅ Tracked | ✅ Tracked | ✅ Tracked |
| **Epsilon Decay** | ✅ Tracked | ✅ Tracked | ✅ Tracked |
| **Loss/TD Error** | ❌ Not tracked | ❌ Not tracked | ✅ Tracked |
| **Moving Average** | ⚠️ Manual | ⚠️ Manual | ✅ Auto |
| **Evaluation Phase** | ❌ None | ❌ None | ✅ Yes |
| **Multi-run Compare** | ❌ None | ✅ Yes (multiple runs) | ❌ Single run |
| **Visualization** | ✅ Matplotlib | ✅ CSV + external | ✅ Matplotlib + TensorBoard |

### Performance Assessment Across Agents

**Learning Curve Quality**:

```
SARSA:       slow_start → gradual improvement → plateau (20+ episodes)
Q-Learning:  moderate_start → guided improvement → faster plateau (10-15 episodes)
DQN:         varies → can be smooth or noisy → fast plateau (15-30 episodes)
```

**Convergence Indicators**:

1. **Stable Reward**: Coefficient of variation < 10% over last 5 episodes
2. **Loss Minimization**: For DQN, loss approaching zero
3. **Exploration Reduction**: Epsilon near minimum
4. **Policy Consistency**: Action selection stable across similar states

### Recommendations for Better Metrics

**SARSA/QL**:
- ✅ Add moving average filter
- ✅ Track win rate (reward > threshold)
- ✅ Add evaluation phase

**DQN**:
- ✅ Track Q-value magnitude (should be relatively stable)
- ✅ Add confidence intervals for uncertainty
- ✅ Compare train vs test rewards

---

## 15. ALGORITHM LIMITATIONS

### SARSA (hello.py)

**Fundamental Limitations**:

1. **On-Policy Learning** ❌
   - Only learns from actions actually taken
   - Can't learn optimal policy while exploring
   - Converges to exploratory policy (suboptimal)

**Example**:
```
State s, two actions:
A: Explored (bad) → learns Q(s,A) = -10
B: Never explored → Q(s,B) remains high estimation

SARSA's policy: Takes A sometimes (exploration), B when exploiting
Optimal policy: Always take B

Outcome: SARSA learns non-optimal policy because it explores
```

2. **Tabular Limitations** ❌
   - State space explosion with continuous states
   - Cannot generalize to unseen states
   - 4 lanes with rounding: ~10,000 states
   - 8 lanes: ~100 million states
   - Practical limit: ~10-20 dimensions

3. **No Credit Assignment Across Distant Time Steps**
   - 1-step TD: Only connects adjacent states
   - Multi-step consequences lost

4. **Fixed Discretization** ❌
   - Rounding to 0.1 granularity fixed
   - Can't adapt resolution to problem
   - Information lost in quantization

5. **Slow Convergence** ❌
   - On-policy: Must explore while learning
   - ε decay = 0.995: Takes 600+ episodes to reach 95% exploitation
   - Practical convergence: 100+ episodes

**When SARSA Works Well**:
- ✅ Small discrete state spaces (< 1 million states)
- ✅ Off-policy data acceptable (conservative learning)
- ✅ Limited computational resources
- ✅ Stability critical (over optimality)

**When SARSA Fails**:
- ❌ Continuous or high-dimensional states
- ❌ Need optimal policy quickly
- ❌ Complex traffic with many vehicles

### Q-Learning (ql_2way-single-intersection.py)

**Advantages Over SARSA**:
- ✅ Off-policy: Can learn optimal policy while exploring
- ✅ Faster convergence: ~2x faster than SARSA

**Limitations**:

1. **Overestimation Bias** ⚠️
```python
target = r + gamma * max_a Q(s', a)
#                   └─ If Q-values overestimated, max is even worse
```

**Cascade Effect**:
```
Episode 1: Q(s,a) = 10 (overestimated)
Episode 2: Update using max(Q) = 10, creates new Q = 11
Episode 3: Q values drift upward: 12, 13, 14...
Result: Unrealistic Q-values → wrong policy priorities
```

**Severity**: High in sparse reward environments, medium in dense

**Mitigation in Q-Learning code**:
- Heuristic guidance: 50% domain knowledge reduces exploration variance
- Medium learning rate: α = 0.3 (not too aggressive)

2. **Tabular State Space Limitation** ❌
   - Same as SARSA: Scales poorly

3. **Convergence Not Guaranteed in Practice** ⚠️
   - Theoretical guarantee requires all (s,a) visited infinitely often
   - Multi-agent environment: Exponential state space
   - Practical: Some (s,a) never visited → no convergence for those

4. **Heuristic Bias** ⚠️
   - Current: 50% actions from heuristic
   - Impact: Reduces pure RL learning
   - Risk: May converge to heuristic-local optima, not global

**Example Scenario**:
```
Heuristic: Serve lane with max queue
True optimal: Sometimes leave lane waiting, handle emergency first

Result: QL learns heuristic bias, misses emergency opportunities
```

**When QL Works Well**:
- ✅ Medium state spaces (< 1 million)
- ✅ Dense rewards (prevent overestimation cascade)
- ✅ Moderate computation available
- ✅ Need near-optimal policy reasonably fast

**When QL Fails**:
- ❌ Sparse rewards (overestimation amplified)
- ❌ Very large state spaces
- ❌ Need guaranteed optimality

### DQN (iiser-v3.ipynb)

**Advantages Over Tabular**:
- ✅ Scales to continuous/high-dim states
- ✅ Generalization to unseen states
- ✅ Fast convergence (ε decay = 0.97)

**Limitations**:

1. **No Convergence Guarantee** ❌
```python
# Function approximation error:
# Neural network can never perfectly approximate Q-function
# Error accumulates through bootstrapping
```

**Why Theorems Fail**:
- Bellman operator assumes exact Q-values
- NN approximation introduces error
- Error feedback through max operation (amplifies)
- Off-policy + function approximation = "Deadly Triad"

2. **Overestimation Bias** ⚠️
   - Same max operator as Q-Learning
   - Solution not implemented: Double DQN

3. **Instability and Divergence Risk** ⚠️

**Sources**:
```python
# (a) Large learning rate
lr = 1e-3  # With Adam, can be unstable
# Fix: Lower to 1e-4 for harder problems

# (b) Large batch correlations
# Not happening: Using replay buffer (good)

# (c) No target network
# PROBLEMATIC: Current and target use same network
with torch.no_grad():
    max_next_q = q_nets[agent](b_next_obs).max(1)[0]
target_q = b_rew + (gamma * max_next_q)

# Better:
# Use separate target_q_nets[agent] that updates every N steps
```

4. **Catastrophic Forgetting** ❌
```python
# If new batch very different from training distribution:
# Network updates drastically → forgets old knowledge
# Replay buffer helps but doesn't eliminate
```

**Example**:
```
Buffer: Contains experiences from phase 1 (learning phase A)
Episode 50: Network encounters new situation (phase B)
New loss: Huge error, big gradient update
Result: Network fine-tunes on phase B, forgets phase A

Recovery: Slow (must wait for phase A back in mini-batch)
```

5. **Hyperparameter Sensitivity** ⚠️

**Critical Hyperparameters** (in code):
```python
learning_rate = 1e-3      # Too high → divergence, too low → slow
batch_size = 64            # Too low → noisy, too high → memory
epsilon_decay = 0.97       # Too fast → premature exploit, too slow → slow learning
replay_buffer_size = 10000 # Too small → high correlation, too large → memory/time
```

**Current Values**: Reasonable for this problem, but risky for larger problems

6. **Replay Buffer Stale Data** ⚠️
```python
replay_buffers[agent] = deque(maxlen=replay_buffer_size)  # 10,000 transitions
batch = random.sample(replay_buffers[agent], batch_size)  # Old data mixed with new
```

**Problem**:
```
Episode 1: Buffer contains experiences from old policy
Episode 50: Policy changed dramatically
Buffer still has 50 old experiences
Training on mixed old/new data → bias

Solution (not implemented):
- Prioritized experience replay: Sample important transitions more often
- Per-experience importance weights
```

7. **Agent Independence Assumption**

```python
# Current: Independent DQN per traffic signal
# Issue: Multi-agent environment
for agent in env.agents:
    # Each agent trains independently
    # Ignores coordinated effect on other agents
```

**Example**:
```
Agent 1 learns: "Green for north-south works"
Agent 2 doesn't see this learning
Both agents might learn: "Green north-south" ← Conflict!

Result: Agents interfere, suboptimal coordination
```

**Why not Fixed**: Requires cooperative learning (CTDE) algorithms (not trivial)

**When DQN Works Well**:
- ✅ High-dimensional state spaces (> 100 dimensions)
- ✅ Continuous observation spaces
- ✅ Sufficient computation (GPU recommended)
- ✅ Well-tuned hyperparameters
- ✅ Relatively simple environments

**When DQN Fails**:
- ❌ Sparse rewards (bootstrapping error amplified)
- ❌ Highly non-stationary environment (policy changes fast)
- ❌ Critical safety requirements (no convergence guarantee)
- ❌ Tiny computational budget
- ❌ Multi-agent coordination requirements

### Algorithm Comparison: Limitations

| Limitation | SARSA | QL | DQN |
|-----------|-------|----|----|
| **On-policy convergence** | ✅ | ❌ | ❌ |
| **Overestimation bias** | ❌ | ✅ | ✅ |
| **Tabular scalability** | ❌ (Poor) | ❌ (Poor) | ✅ (Good) |
| **Convergence guarantee** | ✅ | ✅ | ❌ |
| **Generalization** | ❌ | ❌ | ✅ |
| **Sample efficiency** | ❌ (Low) | ❌ (Low) | ✅ (High) |
| **Stability** | ✅ (High) | ⚠️ (Medium) | ⚠️ (Medium) |
| **Multi-agent coordination** | ⚠️ (None) | ⚠️ (Heuristic) | ⚠️ (Independent) |

### Domain-Specific Limitations (Traffic)

**All Three Agents**:
- ❌ **No explicit priority learning**: Rewards can override emergency rules
- ❌ **No phase constraints**: Can violate min-green/max-red times
- ❌ **No cooperative learning**: Agents don't explicitly share knowledge

**SARSA/QL Specific**:
- ❌ **Limited intersection size**: Can't scale beyond 4-8 lanes
- ❌ **No real-time adaptation**: Complete retraining needed for new environment

**DQN Specific**:
- ❌ **No structural constraints**: Learned policy can violate traffic laws
- ❌ **Requires massive computational budget**: Training 50+ episodes on full simulation

### Recommended Improvements

**All Algorithms**:
1. Add explicit safety constraints (never run all red simultaneously)
2. Implement multi-agent cooperative learning (QMIX, MADDPG)
3. Add domain-specific features (time of day, special events)

**SARSA/QL**:
1. Implement hierarchical learning (macro-actions for common patterns)
2. Add function approximation if scaling needed

**DQN**:
1. Implement target network (separate for stability)
2. Add prioritized experience replay
3. Use Double DQN (separate networks for action selection/evaluation)
4. Add explicit constraint violations as large negative rewards
5. Implement dueling architecture (value + advantage streams)

---

## Summary & Recommendations

### For Your Project

**Current Implementation Assessment**:
- ✅ **SARSA**: Stable baseline, limited scalability
- ✅ **Q-Learning**: Practical with domain guidance, faster convergence
- ⚠️ **DQN**: Powerful but requires careful tuning, not production-ready without target network

**Immediate Improvements**:

1. **For DQN**: Add target network (biggest stability boost)
2. **For All**: Add evaluation separate from training
3. **For Q-Learning**: Consider removing heuristic to test true RL performance
4. **For Traffic**: Add explicit emergency vehicle handling beyond reward shaping

**When to Use Each Algorithm**:
- **SARSA**: Teaching/research, small problems, maximum stability
- **Q-Learning**: Medium-scale traffic, incorporating domain knowledge
- **DQN**: Large-scale networks, generalization needed, sufficient compute available

### Best Practices Going Forward

1. **Always save model checkpoints** (all three)
2. **Maintain separate train/test phases** (validate generalization)
3. **Monitor multiple metrics** (not just reward)
4. **Implement early stopping** (when performance plateaus)
5. **Add multi-seed runs** (report mean ± std)

 
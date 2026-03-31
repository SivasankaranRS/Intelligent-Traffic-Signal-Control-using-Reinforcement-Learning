## 🚗 Environment Design Details

---

### 🚙 Vehicle Modeling

* [ ] Support multiple vehicle types:

  * [ ] Cars
  * [ ] Buses
  * [ ] Trucks
  * [ ] Emergency vehicles (e.g., ambulance)
* [ ] Assign different properties per vehicle type:

  * [ ] Maximum speed
  * [ ] Acceleration / deceleration
  * [ ] Length and lane occupancy
* [ ] Implement priority-based vehicles:

  * [ ] Emergency vehicles get highest priority
  * [ ] Ensure faster clearance at intersections
* [ ] Use color coding in SUMO GUI:

  * [ ] Different colors for each vehicle type
  * [ ] Highlight priority vehicles (e.g., ambulance in red)

---

### 🚦 Traffic Light System

* [ ] Model standard traffic signal phases:

  * [ ] Green (movement allowed)
  * [ ] Yellow (transition phase)
  * [ ] Red (stop phase)
* [ ] Define multiple phase configurations:

  * [ ] Straight vs turn movements
  * [ ] Multi-lane coordination
* [ ] Implement controllable signals via TraCI:

  * [ ] Switch phases programmatically
  * [ ] Adjust phase duration dynamically
* [ ] Ensure safety constraints:

  * [ ] Mandatory yellow phase between switches
  * [ ] Avoid conflicting green signals

---

### 🧠 RL State Representation (Important)

* [ ] Include traffic information:

  * [ ] Queue length per lane
  * [ ] Waiting time per lane
  * [ ] Number of vehicles approaching intersection
* [ ] Include priority awareness:

  * [ ] Presence of emergency vehicles
* [ ] Encode traffic light state:

  * [ ] Current phase (one-hot or index)

---

### 🎮 Action Space Design

* [ ] Define discrete actions:

  * [ ] Switch to next phase
  * [ ] Extend current green phase
* [ ] Ensure actions respect constraints:

  * [ ] No abrupt switching (must pass yellow phase)

---

### 🎯 Reward Design

* [ ] Minimize:

  * [ ] Total waiting time
  * [ ] Queue length
* [ ] Maximize:

  * [ ] Traffic throughput
* [ ] Add priority reward:

  * [ ] Bonus for clearing emergency vehicles faster
* [ ] Penalize:

  * [ ] Congestion buildup
  * [ ] Unnecessary switching

---

### 🔍 Additional Enhancements

* [ ] Add stochastic traffic flow (random arrivals)
* [ ] Vary traffic density (low, medium, high)
* [ ] Test robustness under different scenarios
* [ ] Log simulation data for analysis

---

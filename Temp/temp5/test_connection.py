import os
import sys
import traci

# 1. Start the simulation
# Use 'sumo' instead of 'sumo-gui' later for faster training
sumo_cmd = ["sumo-gui", "-c", "simulation.sumocfg", "--start"]
traci.start(sumo_cmd)

# 2. Automated Discovery (Crucial for large maps later!)
all_tls = traci.trafficlight.getIDList()
print(f"--- DISCOVERY ---")
print(f"Found {len(all_tls)} Traffic Light(s): {all_tls}")

# Pick the first light and see what it controls
if all_tls:
    tls_id = all_tls[0]
    controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
    print(f"Targeting Light: {tls_id}")
    print(f"Controlled Lanes: {controlled_lanes}")

# 3. Run a few steps to see cars
print(f"--- SIMULATION START ---")
for step in range(100):
    traci.simulationStep()
    
    # Check if any cars are waiting at the first controlled lane
    if all_tls:
        lane_to_check = controlled_lanes[0]
        queue = traci.lane.getLastStepHaltingNumber(lane_to_check)
        if queue > 0:
            print(f"Step {step}: {queue} vehicles waiting at {lane_to_check}")

traci.close()
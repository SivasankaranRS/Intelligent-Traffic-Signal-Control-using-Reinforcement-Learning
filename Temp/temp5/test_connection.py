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
# Instead of just watching, let's force a phase change
for step in range(1000):
    traci.simulationStep()

    if step % 20 == 0: # Every 20 steps, toggle the light
        # Get the current phase
        current_phase = traci.trafficlight.getPhase(tls_id)
        
        # Switch to the next phase (0, 1, 2, 3...)
        # Note: In SUMO, even numbers are usually Green, odd are Yellow
        new_phase = (current_phase + 1) % 4 
        traci.trafficlight.setPhase(tls_id, new_phase)
        print(f"Step {step}: Manually switched {tls_id} to phase {new_phase}")

traci.close()
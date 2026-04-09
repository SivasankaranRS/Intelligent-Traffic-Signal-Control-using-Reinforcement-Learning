import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
NET_FILE = os.path.join(CURRENT_DIR, "2way-single-intersection", "single-intersection.net.xml")
ROUTE_FILE = os.path.join(CURRENT_DIR, "2way-single-intersection", "single-intersection-vhvh.rou.xml")
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

from datetime import datetime
import argparse

# -------------------------------
# SUMO CHECK
# -------------------------------
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# -------------------------------
# GET VALID GREEN PHASES
# -------------------------------
def get_valid_phases(traffic_signal):
    phases = traffic_signal.phases
    return [i for i, p in enumerate(phases) if 'G' in p.state]

# -------------------------------
# SAFE STATE ENCODER
# -------------------------------
def safe_encode(env, state, ts_id):
    try:
        encoded = env.encode(state, ts_id)
        return tuple(encoded) if isinstance(encoded, (list, np.ndarray)) else encoded
    except (TypeError, IndexError):
        return tuple(np.round(state, 2)) if isinstance(state, (list, np.ndarray)) else str(state)

# -------------------------------
# PHASE SCORING FUNCTION
# -------------------------------
def score_phase(traffic_signal, phase_index, queues):
    phase = traffic_signal.phases[phase_index]
    state = phase.state

    score = 0
    total_queue = sum(queues) if len(queues) > 0 else 1

    for signal in state:
        if signal == 'G':
            score += total_queue / len(state)

    return score

# -------------------------------
# ACTION SELECTION
# -------------------------------
def choose_action(ts, agent, traffic_signal):
    queues = traffic_signal.get_lanes_queue()
    valid_phases = get_valid_phases(traffic_signal)

    # fallback safety
    if not valid_phases:
        return 0

    # -------------------
    # EMERGENCY PRIORITY
    # -------------------
    emergency_count = traffic_signal.get_emergency_vehicle_count()

    if emergency_count > 0:
        best_phase = max(valid_phases, key=lambda p: score_phase(traffic_signal, p, queues))
        return best_phase

    # -------------------
    # RL + HEURISTIC MIX
    # -------------------
    if np.random.rand() < 0.7:
        rl_action = agent.act()
        return valid_phases[rl_action % len(valid_phases)]
    else:
        return max(valid_phases, key=lambda p: score_phase(traffic_signal, p, queues))


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":

    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Q-Learning Single-Intersection"
    )

    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default="2way-single-intersection\\single-intersection-vhvh.rou.xml"
    )

    prs.add_argument("-a", dest="alpha", type=float, default=0.3)
    prs.add_argument("-g", dest="gamma", type=float, default=0.99)
    prs.add_argument("-e", dest="epsilon", type=float, default=0.3)
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005)
    prs.add_argument("-d", dest="decay", type=float, default=0.9995)

    prs.add_argument("-mingreen", dest="min_green", type=int, default=10)
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=30)

    prs.add_argument("-gui", action="store_true", default=False)
    prs.add_argument("-fixed", action="store_true", default=False)

    prs.add_argument("-s", dest="seconds", type=int, default=100000)
    prs.add_argument("-runs", dest="runs", type=int, default=1)

    args = prs.parse_args()

    experiment_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    out_csv = f"Intelligent-Traffic-Signal-Control-using-Reinforcement-Learning\\Q-Learning\\Outputs\\2way-single-intersection\\{experiment_time}_alpha{args.alpha}_gamma{args.gamma}_eps{args.epsilon}"

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    env = SumoEnvironment(
        net_file="2way-single-intersection\\single-intersection.net.xml",
        route_file=args.route,
        out_csv_name=out_csv,
        use_gui=False,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
        sumo_warnings=False,
    )

    for run in range(1, args.runs + 1):
        print(f"\n🚀 Starting Run {run}/{args.runs}")

        initial_states = env.reset()

        ql_agents = {}

        for ts in env.ts_ids:
            starting_state = safe_encode(env, initial_states[ts], ts)

            # determine valid phases dynamically
            valid_phases = get_valid_phases(env.traffic_signals[ts])

            ql_agents[ts] = QLAgent(
                starting_state=starting_state,
                state_space=env.observation_space,
                action_space=len(valid_phases) if len(valid_phases) > 0 else 4,
                alpha=args.alpha,
                gamma=args.gamma,
                exploration_strategy=EpsilonGreedy(
                    initial_epsilon=args.epsilon,
                    min_epsilon=args.min_epsilon,
                    decay=args.decay
                ),
            )

        done = {"__all__": False}
        step = 0

        while not done["__all__"]:
            actions = {}

            for ts in ql_agents.keys():
                traffic_signal = env.traffic_signals[ts]
                actions[ts] = choose_action(ts, ql_agents[ts], traffic_signal)

            s, r, done, _ = env.step(action=actions)

            for ts in ql_agents.keys():
                next_state = safe_encode(env, s[ts], ts)
                ql_agents[ts].learn(next_state=next_state, reward=r[ts])

            step += 1

            # -------------------
            # PROGRESS TRACKING
            # -------------------
            if step % 1000 == 0:
                print(f"Step: {step}")

        env.save_csv(out_csv, run)

    env.close()
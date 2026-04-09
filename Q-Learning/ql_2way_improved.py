"""
Q-Learning Traffic Signal Control (Improved - Lane Aware Emergency Version)
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import argparse
import numpy as np

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy


# =========================
# LOGGER
# =========================
class Logger:
    @staticmethod
    def setup_logger(name, log_file=None, level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)

        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(fh)

        return logger


# =========================
# CONFIG
# =========================
class TrafficConfig:
    def __init__(self, **kwargs):
        self.alpha = kwargs.get('alpha', 0.3)
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon', 0.3)
        self.min_epsilon = kwargs.get('min_epsilon', 0.005)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.9995)

        self.simulation_seconds = kwargs.get('seconds', 100000)
        self.min_green = kwargs.get('min_green', 10)
        self.max_green = kwargs.get('max_green', 30)

        # 🔥 UPDATED
        self.heuristic_prob = kwargs.get('heuristic_prob', 0.15)

        self.net_file = kwargs.get('net_file')
        self.route_file = kwargs.get('route_file')
        self.output_dir = kwargs.get('output_dir', 'outputs')

        self.num_runs = kwargs.get('runs', 1)
        self.use_gui = kwargs.get('use_gui', False)


# =========================
# AGENT
# =========================
class TrafficSignalAgent:
    def __init__(self, agent_id, env, config, logger):
        self.agent_id = agent_id
        self.env = env
        self.config = config
        self.logger = logger

        self.agent = None

        self.emergency_actions_taken = 0
        self.heuristic_actions_taken = 0
        self.rl_actions_taken = 0

        self._init_agent()

    def _init_agent(self):
        states = self.env.reset()
        state = states[self.agent_id]

        try:
            state = self.env.encode(state, self.agent_id)
        except:
            state = str(state)

        self.agent = QLAgent(
            starting_state=state,
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            alpha=self.config.alpha,
            gamma=self.config.gamma,
            exploration_strategy=EpsilonGreedy(
                initial_epsilon=self.config.epsilon,
                min_epsilon=self.config.min_epsilon,
                decay=self.config.epsilon_decay
            ),
        )

    def encode(self, state):
        try:
            return self.env.encode(state, self.agent_id)
        except:
            return str(state)

    # =========================
    # 🔥 UPDATED CORE LOGIC
    # =========================
    def select_action(self, traffic_signal):

        try:
            # =========================
            # 🚑 EMERGENCY PRIORITY (LANE AWARE)
            # =========================
            emergency_lanes = []

            for i, lane in enumerate(traffic_signal.lanes):
                vehicles = self.env.sumo.lane.getLastStepVehicleIDs(lane)

                for vid in vehicles:
                    vtype = self.env.sumo.vehicle.getTypeID(vid)

                    if "emergency" in vtype.lower():
                        emergency_lanes.append(i)

            if emergency_lanes:
                lane = emergency_lanes[0]

                # ⚠️ MODIFY THIS BASED ON YOUR NETWORK
                lane_to_phase = {
                    0: 0,
                    1: 1,
                    2: 2,
                    3: 3
                }

                self.emergency_actions_taken += 1
                return lane_to_phase.get(lane, 0)

            # =========================
            # 🚗 HEURISTIC (LOW PROBABILITY)
            # =========================
            if np.random.rand() < self.config.heuristic_prob:
                queues = traffic_signal.get_lanes_queue()
                max_lane = int(np.argmax(queues))

                phase = min(max_lane, self.env.action_space.n - 1)

                self.heuristic_actions_taken += 1
                return phase

            # =========================
            # 🧠 RL (MAIN)
            # =========================
            self.rl_actions_taken += 1
            return self.agent.act()

        except Exception as e:
            self.logger.warning(f"Action error: {e}")
            return 0

    def learn(self, next_state, reward):
        next_state = self.encode(next_state)
        self.agent.learn(next_state, reward)


# =========================
# TRAINER
# =========================
class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def run(self):
        env = SumoEnvironment(
            net_file=self.config.net_file,
            route_file=self.config.route_file,
            use_gui=self.config.use_gui,
            num_seconds=self.config.simulation_seconds,
            min_green=self.config.min_green,
            max_green=self.config.max_green,
        )

        agents = {
            ts: TrafficSignalAgent(ts, env, self.config, self.logger)
            for ts in env.ts_ids
        }

        for run in range(self.config.num_runs):
            self.logger.info(f"Run {run+1}")

            done = {"__all__": False}
            states = env.reset()

            while not done["__all__"]:
                actions = {}

                for ts in agents:
                    signal = env.traffic_signals[ts]
                    actions[ts] = agents[ts].select_action(signal)

                next_states, rewards, done, _ = env.step(actions)

                for ts in agents:
                    agents[ts].learn(next_states[ts], rewards[ts])

        env.close()


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-net", required=True)
    parser.add_argument("-route", required=True)
    parser.add_argument("-gui", action="store_true")

    args = parser.parse_args()

    if "SUMO_HOME" not in os.environ:
        sys.exit("SUMO_HOME not set")

    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

    logger = Logger.setup_logger("Traffic")

    config = TrafficConfig(
        net_file=args.net,
        route_file=args.route,
        use_gui=args.gui
    )

    trainer = Trainer(config, logger)
    trainer.run()


if __name__ == "__main__":
    main()
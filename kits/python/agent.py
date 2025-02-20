import logging
import os
from lux.utils import direction_to
import numpy as np
import torch
from process_observation import ProcessObservation
from ppo import ActorCriticNet, select_action, select_action_deterministic

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = (169, 24, 24)  
        self.num_actions = 3
        self.model = ActorCriticNet(self.input_shape, self.num_actions, self.device).to(self.device)
        self.model.load_model("/kaggle_simulations/agent/actor_critic_model_1300.pth", map_location='cpu')
        # self.model.load_model("actor_critic_model_1000.pth", map_location='cpu')

        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()
        self.P_O = ProcessObservation(self.team_id,self.opp_team_id)

    def act(self, step: int, obs, remainingOverageTime: int = 60):

        obs_data, _, _, _ = self.P_O.process_observation(obs)
        actions, _, _ = select_action_deterministic(self.model, obs_data)

        return actions

# luxai-s3 main.py main.py --output replay.json
# tar -czvf submission.tar.gz *
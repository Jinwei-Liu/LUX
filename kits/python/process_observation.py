import numpy as np
import copy
import logging
unit_num = 16

def reshape_obs(obs):
    """
    将 obs 对象转换为嵌套字典结构，按索引保留嵌套。
    """
    return {
        "units": {
            "position": obs.units.position,  # 单位位置
            "energy": obs.units.energy,      # 单位能量
        },
        "units_mask": obs.units_mask,         # 单位掩码
        "sensor_mask": obs.sensor_mask,       # 传感器掩码
        "map_features": {
            "energy": obs.map_features.energy,    # 地图能量
            "tile_type": obs.map_features.tile_type,  # 地图类型
        },
        "relic_nodes": obs.relic_nodes,           # 遗物节点
        "relic_nodes_mask": obs.relic_nodes_mask, # 遗物节点掩码
        "team_points": obs.team_points,           # 队伍积分
        "team_wins": obs.team_wins,               # 队伍胜利信息
        "steps": obs.steps,                       # 当前步骤
        "match_steps": obs.match_steps,           # 总步骤数
    }



# 地图能力和类别
class MapTracker:
    def __init__(self, space_size: int = 24,):
        """
        初始化地图跟踪器，包含能量图和地形类别图。
        :param space_size: 地图大小，默认为 24x24。
        """
        self.space_size = space_size
        self.energy_map = np.zeros((self.space_size, self.space_size), dtype=int)  # 能量图
        self.tile_type_map = np.full((self.space_size, self.space_size), -1, dtype=int)  # 地形类别图
        self.sensor_mask_int = np.zeros((self.space_size, self.space_size), dtype=int)  # 传感器掩码

    def _get_opposite(self, x: int, y: int) -> tuple[int, int]:
        """
        给定 (x, y)，返回地图对称位置 (x', y')。
        因此对称点即 (space_size-1 - y, space_size-1 - x)。
        """
        return (self.space_size - 1 - y, self.space_size - 1 - x)

    def update(self, obs: dict) -> None:
        """
        根据传感器掩码更新地图的能量和类别信息。
        :param obs: 包含传感器掩码、能量信息和地形类别信息的观察字典。
        """
        # 提取传感器掩码
        sensor_mask = np.array(obs["sensor_mask"], dtype=bool)  # shape (24, 24)
        self.sensor_mask_int =sensor_mask.astype(int)
        
        # 提取地图能量和地形类别
        map_energy = np.array(obs["map_features"]["energy"], dtype=int)  # shape (24, 24)
        map_tile_type = np.array(obs["map_features"]["tile_type"], dtype=int)  # shape (24, 24)

        # 遍历地图更新被传感器探测到的位置
        for y in range(self.space_size):
            for x in range(self.space_size):
                if sensor_mask[x, y]:  # 如果传感器掩码为 True
                    self.energy_map[x, y] = map_energy[x, y]  # 更新能量图
                    self.tile_type_map[x, y] = map_tile_type[x, y]  # 更新地形类别图
                    ox, oy = self._get_opposite(x,y)
                    self.energy_map[ox, oy] = map_energy[x, y]  # 更新能量图
                    self.tile_type_map[ox, oy] = map_tile_type[x, y]  # 更新地形类别图

    def get_map(self):
        """
        获取当前地图的能量和地形类别信息，编码之后的。
        :return: 返回能量图和地形类别图。
        """
        map_type = np.zeros((24, 24, 3), dtype=int)

        # 为每个类别创建独立的层
        for value in range(3):  # 0, 1, 2
            map_type[:, :, value] = (self.tile_type_map == value).astype(int)
        
        map_data = np.dstack((self.sensor_mask_int, self.energy_map, map_type))

        return map_data


# 用于遗物节点类别
class MinimalRelicRewardTracker:
    def __init__(self, 
                 space_size: int = 24,
                 max_relic_nodes: int = 6, 
                 relic_reward_range: int = 2):

        self.space_size = space_size
        self.max_relic_nodes = max_relic_nodes
        self.relic_reward_range = relic_reward_range
        
        # 存储每个格子的遗物/奖励状态: True/False/None
        self._nodes = [
            [
                {
                    'relic': None,   # 是否有遗物；None 表示未知
                    'reward': None,  # 是否有奖励；None 表示未知
                }
                for _ in range(space_size)
            ]
            for _ in range(space_size)
        ]

        # 是否已推断出地图上所有遗物 / 奖励
        self.all_relic_found = False
        self.all_reward_found = False

        # 记录上一回合我方总积分，以便推断本回合是否有积分增长
        self.last_team_points = 0

    def _get_opposite(self, x: int, y: int) -> tuple[int, int]:
        """
        给定 (x, y)，返回地图对称位置 (x', y')。
        因此对称点即 (space_size-1 - y, space_size-1 - x)。
        """
        return (self.space_size - 1 - y, self.space_size - 1 - x)

    def _set_relic_status(self, x: int, y: int, has_relic: bool):
        """
        将 (x,y) 节点以及它的对称点标记为有/无遗物。
        若原先已记录值与这次冲突，示例中简单覆盖，或可抛异常。
        """
        self._nodes[y][x]['relic'] = has_relic
        ox, oy = self._get_opposite(x, y)
        self._nodes[oy][ox]['relic'] = has_relic

    def _set_reward_status(self, x: int, y: int, has_reward: bool):
        """
        将 (x,y) 节点以及它的对称点标记为有/无奖励。
        """
        self._nodes[y][x]['reward'] = has_reward
        ox, oy = self._get_opposite(x, y)
        self._nodes[oy][ox]['reward'] = has_reward

    def _in_range_of_any_relic(self, x: int, y: int) -> bool:
        """
        简易判断：在以 (x,y) 为中心、曼哈顿距离 <= relic_reward_range
        的格子范围内，是否存在已确定 relic=True 的节点。
        """
        r = self.relic_reward_range
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.space_size and 0 <= ny < self.space_size:
                    if self._nodes[ny][nx]['relic'] is True:
                        return True
        return False

    def update(self, obs: dict, team_id: int) -> None:
        """
        - obs应包含:
          obs["team_points"][team_id]: int, 当回合我方累计积分
          obs["relic_nodes_mask"]: list/array[bool], 标记哪些遗物坐标本回合已确定
          obs["relic_nodes"]: list/array[(x, y)], 与上面 mask 对应
          (可选)obs["sensor_mask"]: 二维bool，标记可见区域
        
        - 通过本方法更新地图中的遗物/奖励信息。
        """
        # 1) 先计算本回合积分增加 (team_reward)
        current_points = obs["team_points"][team_id]
        team_reward = max(0, current_points - self.last_team_points)
        self.last_team_points = current_points

        # 2) 利用 relic_nodes_mask 直接设置遗物
        relic_mask = obs["relic_nodes_mask"]
        relic_coords = obs["relic_nodes"]  # 与 mask 一一对应
        for i, masked in enumerate(relic_mask):
            if masked:  # 观测到这个条目确实存在遗物
                x, y = relic_coords[i]
                self._set_relic_status(x, y, True)
 
        # 4) 判断是否遗物已找齐：若已知 True 的数量 >= max_relic_nodes
        #    则把尚未确定(None)的都标记为 False
        if not self.all_relic_found:
            found_relic_count = 0
            for row in self._nodes:
                for cell in row:
                    if cell['relic'] is True:
                        found_relic_count += 1
            if found_relic_count >= self.max_relic_nodes:
                # 如果大于了最大节点的那么剩下的都是没有遗物的  直接标记为 False
                for yy in range(self.space_size):
                    for xx in range(self.space_size):
                        if self._nodes[yy][xx]['relic'] is not True:
                            self._set_relic_status(xx, yy, False)
                self.all_relic_found = True
  

        # (a) 遗物节点外边的都是False，内部都是不确定
        for yy in range(self.space_size):
            for xx in range(self.space_size):
                if not self._in_range_of_any_relic(xx, yy):
                    self._set_reward_status(xx, yy, False)
                elif self._nodes[yy][xx]['reward'] is not True:
                    self._set_reward_status(xx, yy, None)
 
        # (b) 判断哪些节点是True
        unit_mask = np.array(obs['units_mask'][team_id])  # shape （16.）
        unit_positions = np.array(obs['units']['position'][team_id])  # shape（16，2） 
        filtered_positions = unit_positions[~np.all(unit_positions == -1, axis=1)]
        unique_positions = np.unique(filtered_positions, axis=0)
        team_reward_sum = 0

        if team_reward > 0:
            for x, y in unique_positions:
                if self._in_range_of_any_relic(x, y):
                    team_reward_sum += 1
            if team_reward == team_reward_sum:
                for x, y in unique_positions:
                    if self._in_range_of_any_relic(x, y):
                        self._set_reward_status(x, y, True)

    def get_relic_nodes(self):
        """
        获取当前确认有遗物的的图像。
        """
        coords = []
        for y in range(self.space_size):
            for x in range(self.space_size):
                if self._nodes[y][x]['relic'] is True:
                    coords.append((x, y))
        relic_map = np.zeros((self.space_size, self.space_size), dtype=int)

        for x,y in coords: 
            relic_map[x, y] = 1

        return relic_map

    def get_reward_nodes(self):
        """
        获取当前确认有奖励的所有坐标 (x, y)。
        """
        coords = []
        for y in range(self.space_size):
            for x in range(self.space_size):
                if self._nodes[y][x]['reward'] is True:
                    coords.append((x, y))

        reward_map = np.zeros((self.space_size, self.space_size), dtype=int)

        for x,y in coords: 
            reward_map[x, y] = 1

        return reward_map



class ProcessObservation:
    def __init__(self, team_id, opp_team_id):
        self.relic_reward_tracker = MinimalRelicRewardTracker()
        self.map_tracker = MapTracker()
        self.team_id = team_id
        self.opp_team_id = opp_team_id
        self.reward_relic_history = 0
        self.see_relic_history = 0
        self.see_relic_reward_history = 0
        self.explore_history = 0
        self.go_reward_return_history = np.zeros((unit_num), dtype=np.float32)
        self.unit_energy_history = {}
        self.position_self_last = np.zeros((24, 24, unit_num), dtype=np.int32)
        self.position_self_last_last = np.zeros((24, 24, unit_num), dtype=np.int32)
        self.position_opp_last = np.zeros((24, 24, unit_num), dtype=np.int32)
        self.position_opp_last_last = np.zeros((24, 24, unit_num), dtype=np.int32)
        self.energies_self_last = np.zeros((24, 24, unit_num), dtype=np.float32)
        self.energies_opp_last = np.zeros((24, 24, unit_num), dtype=np.float32)

    def process_observation(self, obs):
        
        # --------------------------------- 时间信息 ---------------------------------
        steps = np.array(obs['steps'])  # 当前步骤数(总数)
        match_steps = np.array(obs['match_steps'])  # 当前比赛步骤数（每局的数目）
        match_steps_state = np.ones((24, 24, 1), dtype=np.float32)
        steps_state = np.ones((24, 24, 1), dtype=np.float32)
        match_steps_state = match_steps_state * match_steps / 100.0
        steps_state = steps_state *steps /505.0

        if steps == 505:
            terminate = True
        else:
            terminate = False

        # 每一局开始都先清空历史奖励,并给出结束
        done = False
        if steps % 101 == 0 and steps != 0:
            done = True
        if steps % 101 == 1:
            self.position_self_last = np.zeros((24, 24, unit_num), dtype=np.int32)
            self.position_self_last_last = np.zeros((24, 24, unit_num), dtype=np.int32)
            self.position_opp_last = np.zeros((24, 24, unit_num), dtype=np.int32)
            self.position_opp_last_last = np.zeros((24, 24, unit_num), dtype=np.int32)
            self.energies_self_last = np.zeros((24, 24, unit_num), dtype=np.float32)
            self.energies_opp_last = np.zeros((24, 24, unit_num), dtype=np.float32)
            self.go_reward_return_history = np.zeros((unit_num), dtype=np.float32)
            self.reward_relic_history = 0
            self.unit_energy_history = {}
        # --------------------------------- 自身、对方位置和能量 ---------------------------------
        # 提取单位信息
        unit_mask = np.array(obs['units_mask'])  # shape （2，16）
        unit_positions = np.array(obs['units']['position'])  # shape（2，16，2
        unit_energies = np.array(obs['units']['energy'])  # shape （2，16）

        position_self = np.zeros((24, 24, unit_num), dtype=np.int32)
        energies_self = np.zeros((24, 24, unit_num), dtype=np.float32)

        position_opp = np.zeros((24, 24, unit_num), dtype=np.int32)
        energies_opp = np.zeros((24, 24, unit_num), dtype=np.float32)

        # 遍历每个单位，对自己的位置进行标记
        for unit_index in range(unit_num):
            # 检查 mask 是否激活
            if unit_mask[self.team_id, unit_index]:  # 如果存在单位
                # 获取单位的位置
                x, y = unit_positions[self.team_id, unit_index]
                position_self[x, y, unit_index] = 1  # 在对应位置置为 1
                energies_self[x, y, unit_index] = unit_energies[self.team_id, unit_index]
        energies_self = energies_self / 400.0  # 归一化，最多400
        # 遍历每个单位，对对方的位置进行标记
        for unit_index in range(unit_num):
            # 检查 mask 是否激活
            if unit_mask[self.opp_team_id, unit_index]:  # 如果存在单位
                # 获取单位的位置
                x, y = unit_positions[self.opp_team_id, unit_index]
                position_opp[x, y, unit_index] = 1  # 在对应位置置为 1
                energies_opp[x, y, unit_index] = unit_energies[self.opp_team_id, unit_index]
        energies_opp = energies_opp / 400.0 

        position_data = np.dstack((self.position_self_last_last,self.position_self_last,position_self,self.energies_self_last,energies_self,self.position_opp_last_last,self.position_opp_last,position_opp,self.energies_opp_last,energies_opp))
        self.position_self_last_last = self.position_self_last 
        self.position_self_last = position_self

        self.position_opp_last_last = self.position_opp_last
        self.position_opp_last = position_opp

        self.energies_self_last = energies_self
        self.energies_opp_last = energies_opp
        # --------------------------------- 地图特征 ---------------------------------
        self.map_tracker.update(obs)
        map_data = self.map_tracker.get_map()
        # --------------------------------- 遗物节点 ---------------------------------
        self.relic_reward_tracker.update(obs, self.team_id)
        relic_map = self.relic_reward_tracker.get_relic_nodes()
        reward_map = self.relic_reward_tracker.get_reward_nodes()

        # --------------------------------- 拼接数据 ---------------------------------
        obs_data = np.dstack((match_steps_state, steps_state, position_data, map_data, relic_map, reward_map))

        # --------------------------------- 团队信息计算奖励 ---------------------------------
        # 提取团队信息
        team_points = np.array(obs['team_points'])  # 当前团队得分 #shape (2, )
        team_wins = np.array(obs['team_wins'])  # 团队胜利次数 #shape (2, )

        # 队伍总得分的奖励
        reward_return_list = np.zeros((unit_num), dtype=np.float32)

        #------------------经过测试，探索未知给分的奖励并没有任何作用-------------

        # # 全部都有的奖励
        # if steps <= 101:
        #     # 探索未知位置的得分
        #     explore = map_data[:,:,2:].sum() * 5#每个位置给10分
        #     reward_return = explore - self.explore_history
        #     self.explore_history = explore
        #     reward_return_list += reward_return * np.ones((unit_num), dtype=np.float32)

        # 单个智能体有的奖励
        
        # # 只有第二轮有这个奖励
        # if steps > 101:

        #------------------课程二的内容-------------
        # 如果在奖励点就单独给奖励
        reward_map_copy = copy.deepcopy(reward_map) #防止多次领取奖励
        # 遍历每个单位
        for unit_index in range(unit_num):
            # 检查 mask 是否激活
            if unit_mask[self.team_id, unit_index]:  # 如果存在单位
                # 获取单位的位置
                x, y = unit_positions[self.team_id, unit_index]
                if reward_map_copy[x, y] == 1:
                    reward_return_list[unit_index] += 200 #单独给奖励
                    reward_map_copy[x, y] = 0
                # elif (reward_map_copy[x, y] == 0) and (reward_map[x, y] == 1):
                #     reward_return_list[unit_index] -= 50 #惩罚一下

        # 通过遗迹点设置奖励场
        reward_map_to_relic = np.zeros_like(relic_map, dtype=float)
        # 获取所有奖励点的位置
        reward_points = np.argwhere(relic_map == 1)

        # 计算每个点的奖励值
        rows, cols = reward_map_to_relic.shape
        for i in range(rows):
            for j in range(cols):
                # 计算当前点到所有奖励点的曼哈顿距离
                distances = np.abs(reward_points[:, 0] - i) + np.abs(reward_points[:, 1] - j)
                # 设置奖励值，距离越近奖励越高
                if distances.size > 0:
                    reward_map_to_relic[i, j] = 1 / (1 + np.min(distances)) * 10
                else:
                    # 处理 distances 为空的情况，例如设置默认值或跳过当前计算
                    reward_map_to_relic[i, j] = 0 

        # 遍历每个单位
        for unit_index in range(unit_num):
            # 检查 mask 是否激活
            if unit_mask[self.team_id, unit_index]:  # 如果存在单位
                # 获取单位的位置
                x, y = unit_positions[self.team_id, unit_index]
                reward_return_list[unit_index] += reward_map_to_relic[x, y] #奖励趋向于遗迹点

        # 偏向走向去遗迹点的奖励，通过差分奖励来实现，记录的是离奖励最近的智能体的距离，距离使用曼哈顿距离

        # #------------------课程一的内容-------------
        # # 预处理：提前计算所有奖励点的坐标
        # reward_locations = np.argwhere(reward_map == 1)  # 获取所有奖励坐标
        # go_reward_return = np.zeros((unit_num), dtype=np.float32)
 
        # for xx, yy in reward_locations:
        #     unit_reward = 0
        #     unit_index_record = []
        #     for unit_index in range(unit_num):
        #         if unit_mask[self.team_id, unit_index]:
        #             x, y = unit_positions[self.team_id, unit_index]
        #             # 计算到所有奖励点的曼哈顿距离
        #             distances = np.abs(xx-x) + np.abs(yy-y)
        #             unit_reward_now = 1 / (1 + distances) * 200
        #             if unit_reward_now > unit_reward:
        #                 unit_reward = unit_reward_now
        #                 unit_index_record = unit_index
        #     go_reward_return[unit_index_record] = unit_reward

        # reward_return_list += go_reward_return #- self.go_reward_return_history
        # self.go_reward_return_history = go_reward_return

        # #------------------能量变化给分-------------
        # # 能量变化得分 获得的能量-使用的能量
        # current_energies = unit_energies[self.team_id]  # 当前队伍所有单位的能量值
        # energy_reward_list = np.zeros((unit_num), dtype=np.float32)
        # for unit_index in range(unit_num):
        #     if unit_mask[self.team_id, unit_index]:  # 单位存活
        #         current_energy = current_energies[unit_index]
                
        #         # 首次出现或重生的情况
        #         if unit_index not in self.unit_energy_history or not self.unit_energy_history[unit_index][1]:
        #             self.unit_energy_history[unit_index] = (current_energy, True)  # 初始能量100
        #             continue
                    
        #         # 计算能量增量（排除初始值）
        #         last_energy, _ = self.unit_energy_history[unit_index]
        #         delta = current_energy - last_energy 
        #         energy_reward_list[unit_index] = delta
                
        #         # 更新历史记录
        #         self.unit_energy_history[unit_index] = (current_energy, True)
        #     else:  # 单位死亡
        #         if unit_index in self.unit_energy_history:
        #             # 标记为死亡状态，下次重生时重置
        #             self.unit_energy_history[unit_index] = (0, False) 

        # reward_return_list += energy_reward_list/10

        return  obs_data, reward_return_list/100, done, terminate


import os
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from lpsim import Match, Deck
from lpsim.agents import RandomAgent
from lpsim.network import HTTPServer
from lpsim.server.consts import DieColor
from lpsim.server.dice import Dice
# from RLMatch import RLMatch
from RLAgent import RLAgent

import numpy as np
import collections
import random

# hyper-parameters
Steps_Till_Backprop = 16
EPISODES = 5000            # 训练/测试幕数
EPSILON = 0.1             # epsilon-greedy
MEMORY_CAPACITY = 10000    # Experience Replay的容量
MIN_CAPACITY = 500         # 开始学习的下限
# MIN_CAPACITY = 20         # 开始学习的下限
Q_NETWORK_ITERATION = 10   # 同步target network的间隔
GAMMA = 0.98               # reward的折扣因子
BATCH_SIZE = 64
# BATCH_SIZE = 16
LR = 0.00025
ACTION_DIM = 20
STATE_DIM = 28

EACH_STEP_REWARD = 0.1
SWITCH_CHARACTOR_REWARD = -0.11
EACH_ROUND_REWARD = -3
WIN_REWARD = 10000

MODEL_PATH = 'log/lpsim/MiniMaxDQN/ckpt/36000.pth'
SAVING_IETRATION = 1000    # 保存Checkpoint的间隔
SAVE_PATH_PREFIX = './log/lpsim/MiniMaxDQN/'
TEST = False
# TEST = True

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda')

os.makedirs(f"{SAVE_PATH_PREFIX}/ckpt", exist_ok=True)

# DQN的模型部分
class DQN_Model(nn.Module):
    def __init__(self):
        super(DQN_Model, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, 512)
        self.fc2 = nn.Linear(512, ACTION_DIM)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Data:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

        

    def __iter__(self):
        # 返回一个迭代器，包含对象的属性值，方便zip函数调用
        # return iter([self.state, self.action, self.reward])    
        return iter([self.state, self.action, self.reward, self.next_state, self.done])    


class Memory:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def set(self, data):
        # TODO: 将数据添加到buffer中
        self.buffer.append(data)
        pass
    
    def get(self, batch_size):
        # TODO: 从buffer中随机采样batch_size大小的数据并返回
        batch = random.sample(self.buffer, batch_size)
        return batch
        pass

# RL-DQN Agent类
class RLDQN():
    def __init__(self):
        super(RLDQN, self).__init__()
        self.eval_net, self.target_net = DQN_Model().to(device), DQN_Model().to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = Memory(capacity=MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()


    def get_action_tensor(self, state):
        # print(f'state: {state}')
        state = torch.tensor(state, dtype=torch.float).to(device)
        action_tensor = self.eval_net.forward(state)
        return action_tensor
    
    def store_transition(self, data):
        self.memory.set(data)
        self.memory_counter += 1

    
    def learn(self):
        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.learn_step_counter % SAVING_IETRATION == 0:
            self.save_train_model(self.learn_step_counter)

        self.learn_step_counter += 1


        # 从Memory中随机采样一批数据
        batch = self.memory.get(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack([s.float() for s in states])
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.stack([s.float() for s in next_states])
        dones = torch.tensor(dones, dtype=torch.float).to(device)

        # 计算当前Q值和目标Q值
        q_eval = self.eval_net(states).gather(1, actions.unsqueeze(1))
        q_next = self.target_net(next_states).detach()
        q_target = rewards + GAMMA * q_next.max(1)[0] * (1 - dones)

        # 计算损失函数
        loss = self.loss_func(q_eval, q_target.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_train_model(self, epoch):
        torch.save(self.eval_net.state_dict(), f"{SAVE_PATH_PREFIX}ckpt/{epoch}.pth")

    def load_net(self, file):
        self.eval_net.load_state_dict(torch.load(file))
        self.target_net.load_state_dict(torch.load(file))


deck_string = '''
default_version:4.1
charactor:Fischl
charactor:Mona
charactor:Collei
Gambler's Earrings*2
Wine-Stained Tricorne*2
Vanarana
Timmie*2
Rana*2
Covenant of Rock
Wind and Freedom@4.0
The Bestest Travel Companion!*2
Changing Shifts*2
Toss-Up
Strategize*2
I Haven't Lost Yet!*2
Leave It to Me!
Calx's Arts*2
Adeptus' Temptation*2
Lotus Flower Crisp*2
Mondstadt Hash Brown*2
Tandoori Roast Chicken
'''

def main():
    rlDQN = RLDQN()

    writer = SummaryWriter(f'{SAVE_PATH_PREFIX}')



    agent_0 = RLAgent(player_idx = 0)  # 使用自定义的RLAgent
    # agent_1 = RandomAgent(player_idx = 1)
    agent_1 = RLAgent(player_idx = 1)  # 使用自定义的RLAgent


    if TEST:
        rlDQN.load_net(MODEL_PATH)

    deck0 = Deck.from_str(deck_string)
    deck1 = Deck.from_str(deck_string)

    for i in range(EPISODES):
        turns = 0

        print("EPISODE: ", i)
        ep_reward = 0
        ep_reward1 = 0

        match = Match()
        # match = RLMatch()

        match.set_deck([deck0, deck1])

        match.config.history_level = 10 # let history can be watched

        match.start()
        match.step()
        turns += 1

        player_idx = 0
        state_tensor_player0 = torch.tensor([
                match.round_number, 
                match.current_player, 
                match.player_tables[player_idx].charactors[0].hp,
                match.player_tables[player_idx].charactors[1].hp,
                match.player_tables[player_idx].charactors[2].hp,
                match.player_tables[player_idx].charactors[0].charge,
                match.player_tables[player_idx].charactors[1].charge,
                match.player_tables[player_idx].charactors[2].charge,
                match.player_tables[player_idx].charactors[0].is_alive,
                match.player_tables[player_idx].charactors[1].is_alive,
                match.player_tables[player_idx].charactors[2].is_alive,
                len(match.player_tables[player_idx].charactors[0].element_application), 
                len(match.player_tables[player_idx].charactors[1].element_application),
                len(match.player_tables[player_idx].charactors[2].element_application),
                len(match.player_tables[player_idx].hands), 
                len(match.player_tables[player_idx].dice.colors), 
                len(match.player_tables[1 - player_idx].hands), 
                len(match.player_tables[1 - player_idx].dice.colors), 
                match.player_tables[player_idx].active_charactor_idx, 
                match.player_tables[1 - player_idx].active_charactor_idx, 
                match.player_tables[player_idx].has_round_ended, 
                match.player_tables[1 - player_idx].has_round_ended, 
                match.player_tables[player_idx].arcane_legend, 
                match.player_tables[1 - player_idx].arcane_legend, 
                match.player_tables[player_idx].charge_satisfied, 
                match.player_tables[1 - player_idx].charge_satisfied, 
                match.player_tables[player_idx].plunge_satisfied, 
                match.player_tables[1 - player_idx].plunge_satisfied
                ]).to(device)
        player_idx1 = 1
        state_tensor_player1 = torch.tensor([
                match.round_number,
                match.current_player,
                match.player_tables[player_idx1].charactors[0].hp,
                match.player_tables[player_idx1].charactors[1].hp,
                match.player_tables[player_idx1].charactors[2].hp,
                match.player_tables[player_idx1].charactors[0].charge,
                match.player_tables[player_idx1].charactors[1].charge,
                match.player_tables[player_idx1].charactors[2].charge,
                match.player_tables[player_idx1].charactors[0].is_alive,
                match.player_tables[player_idx1].charactors[1].is_alive,
                match.player_tables[player_idx1].charactors[2].is_alive,
                len(match.player_tables[player_idx1].charactors[0].element_application),
                len(match.player_tables[player_idx1].charactors[1].element_application),
                len(match.player_tables[player_idx1].charactors[2].element_application),
                len(match.player_tables[player_idx1].hands),
                len(match.player_tables[player_idx1].dice.colors),
                len(match.player_tables[1 - player_idx1].hands),
                len(match.player_tables[1 - player_idx1].dice.colors),
                match.player_tables[player_idx1].active_charactor_idx,
                match.player_tables[1 - player_idx1].active_charactor_idx,
                match.player_tables[player_idx1].has_round_ended,
                match.player_tables[1 - player_idx1].has_round_ended,
                match.player_tables[player_idx1].arcane_legend,
                match.player_tables[1 - player_idx1].arcane_legend,
                match.player_tables[player_idx1].charge_satisfied,
                match.player_tables[1 - player_idx1].charge_satisfied,
                match.player_tables[player_idx1].plunge_satisfied,
                match.player_tables[1 - player_idx1].plunge_satisfied
                ]).to(device)


        old_round_number = match.round_number
        new_round_number = match.round_number
        old_round_number1 = match.round_number
        new_round_number1 = match.round_number

        mean_action_value = 0
        max_action_value = 0

        # while not match.is_game_end():
        while True:
            reward = 0
            reward1 = 0
            turns += 1
            done = False

            # if turns > 1024:
            #     print("turns > 1024")
            #     break

            # c1 = match.player_tables[0].charactors[0].hp    
            # print(f'c1: {c1}')

            # dice = match.player_tables[player_idx].dice
            # print(f'dice: {dice}')
            # print(f'dice.colors: {dice.colors}')
            # print(f'dice.colors_to_idx(): {dice.colors_to_idx(DieColor(x.upper()) for x in color_names)}')

            if match.winner == 0:
                reward += WIN_REWARD # 胜利奖励
            if match.winner == 1:
                reward1 += WIN_REWARD

            if match.need_respond(0):
                new_round_number = match.round_number
                
                action_tensor = rlDQN.get_action_tensor(state_tensor_player0).to(device) # 获取动作
                
                # 作为收敛参考
                mean_action_value = torch.mean(action_tensor)
                max_action_value = torch.max(action_tensor)

                ## 无需截断，由generate_response处理
                # reqs_count = len([x for x in match.requests if x.player_idx == 0])
                # do_action_tensor = action_tensor[:reqs_count] # 根据match.requests的长度进行截断
                response, action_index = agent_0.generate_response(match, action_tensor, EPSILON) # 生成回应
                match.respond(response) # 做出动作
                # print(f'response: {response}')
                match.step()

                # 计算奖励并存储到经验库
                next_state_tensor = torch.tensor([  # 可以根据你的需要更新状态Tensor
                    match.round_number, 
                    match.current_player,
                    match.player_tables[player_idx].charactors[0].hp,
                    match.player_tables[player_idx].charactors[1].hp,
                    match.player_tables[player_idx].charactors[2].hp,
                    match.player_tables[player_idx].charactors[0].charge,
                    match.player_tables[player_idx].charactors[1].charge,
                    match.player_tables[player_idx].charactors[2].charge,
                    match.player_tables[player_idx].charactors[0].is_alive,
                    match.player_tables[player_idx].charactors[1].is_alive,
                    match.player_tables[player_idx].charactors[2].is_alive,
                    len(match.player_tables[player_idx].charactors[0].element_application), 
                    len(match.player_tables[player_idx].charactors[1].element_application),
                    len(match.player_tables[player_idx].charactors[2].element_application),
                    len(match.player_tables[player_idx].hands), 
                    len(match.player_tables[player_idx].dice.colors), 
                    len(match.player_tables[1 - player_idx].hands), 
                    len(match.player_tables[1 - player_idx].dice.colors), 
                    match.player_tables[player_idx].active_charactor_idx, 
                    match.player_tables[1 - player_idx].active_charactor_idx, 
                    match.player_tables[player_idx].has_round_ended, 
                    match.player_tables[1 - player_idx].has_round_ended, 
                    match.player_tables[player_idx].arcane_legend, 
                    match.player_tables[1 - player_idx].arcane_legend, 
                    match.player_tables[player_idx].charge_satisfied, 
                    match.player_tables[1 - player_idx].charge_satisfied, 
                    match.player_tables[player_idx].plunge_satisfied, 
                    match.player_tables[1 - player_idx].plunge_satisfied
                ]).to(device)

                # action = response.number
                reward += EACH_STEP_REWARD # 每次行动的奖励
                # if new_round_number > old_round_number:
                if response.name == 'SwitchCharactorResponse':
                    reward += SWITCH_CHARACTOR_REWARD # 切换角色奖励
                if response.name == 'DeclareRoundEndResponse':
                    reward += EACH_ROUND_REWARD # 回合结束奖励
                if match.winner == 0:
                    reward += WIN_REWARD # 胜利奖励

                old_round_number = new_round_number
                done = match.is_game_end()

                # 存入经验库
                rlDQN.store_transition(Data(state_tensor_player0, action_index, reward, next_state_tensor, done)) # 存储经验

                ep_reward += reward
                
                if rlDQN.memory_counter > MIN_CAPACITY and turns % Steps_Till_Backprop == 0 and not TEST:
                    rlDQN.learn()

                if done:
                    print("episode: {} , the episode reward0 is {}".format(i, round(ep_reward, 8)))
                    print("episode: {} , the episode reward1 is {}".format(i, round(ep_reward1, 8)))
                    break
                state_tensor_player0 = next_state_tensor # 更新状态

            elif match.need_respond(1): # 1号玩家回合 零和博弈
                # match.respond(agent_1.generate_response(match))
                # match.step()
                new_round_number1 = match.round_number
                
                action_tensor1 = rlDQN.get_action_tensor(state_tensor_player1).to(device) # 获取动作
                ## 无需截断，由generate_response处理
                # reqs_count = len([x for x in match.requests if x.player_idx == 0])
                # do_action_tensor = action_tensor[:reqs_count] # 根据match.requests的长度进行截断
                response1, action_index1 = agent_1.generate_response(match, action_tensor1, EPSILON) # 生成回应
                match.respond(response1) # 做出动作
                match.step()


                # 计算奖励并存储到经验库
                next_state_tensor1 = torch.tensor([  # 可以根据你的需要更新状态Tensor
                    match.round_number,
                    match.current_player,
                    match.player_tables[player_idx1].charactors[0].hp,
                    match.player_tables[player_idx1].charactors[1].hp,
                    match.player_tables[player_idx1].charactors[2].hp,
                    match.player_tables[player_idx1].charactors[0].charge,
                    match.player_tables[player_idx1].charactors[1].charge,
                    match.player_tables[player_idx1].charactors[2].charge,
                    match.player_tables[player_idx1].charactors[0].is_alive,
                    match.player_tables[player_idx1].charactors[1].is_alive,
                    match.player_tables[player_idx1].charactors[2].is_alive,
                    len(match.player_tables[player_idx1].charactors[0].element_application),
                    len(match.player_tables[player_idx1].charactors[1].element_application),
                    len(match.player_tables[player_idx1].charactors[2].element_application),
                    len(match.player_tables[player_idx1].hands),
                    len(match.player_tables[player_idx1].dice.colors),
                    len(match.player_tables[1 - player_idx1].hands),
                    len(match.player_tables[1 - player_idx1].dice.colors),
                    match.player_tables[player_idx1].active_charactor_idx,
                    match.player_tables[1 - player_idx1].active_charactor_idx,
                    match.player_tables[player_idx1].has_round_ended,
                    match.player_tables[1 - player_idx1].has_round_ended,
                    match.player_tables[player_idx1].arcane_legend,
                    match.player_tables[1 - player_idx1].arcane_legend,
                    match.player_tables[player_idx1].charge_satisfied,
                    match.player_tables[1 - player_idx1].charge_satisfied,
                    match.player_tables[player_idx1].plunge_satisfied,
                    match.player_tables[1 - player_idx1].plunge_satisfied
                ]).to(device)

                # action = response.number
                reward1 += EACH_STEP_REWARD # 每次行动的奖励

                # if new_round_number > old_round_number:
                if response1.name == 'SwitchCharactorResponse':
                    reward1 += SWITCH_CHARACTOR_REWARD
                if response1.name == 'DeclareRoundEndResponse':
                    reward1 += EACH_ROUND_REWARD # 回合结束奖励
                if match.winner == 1:
                    reward1 += WIN_REWARD # 胜利奖励

                old_round_number1 = new_round_number1
                done = match.is_game_end()

                # 存入经验库
                rlDQN.store_transition(Data(state_tensor_player1, action_index1, reward1, next_state_tensor1, done)) # 存储经验

                ep_reward1 += reward1
                
                if rlDQN.memory_counter > MIN_CAPACITY and turns % Steps_Till_Backprop == 0 and not TEST:
                    rlDQN.learn()

                if done:
                    print("episode: {} , the episode reward0 is {}".format(i, round(ep_reward, 8)))
                    print("episode: {} , the episode reward1 is {}".format(i, round(ep_reward1, 8)))
                    break
                state_tensor_player1 = next_state_tensor1 # 更新状态
            
            if match.is_game_end():
                ep_reward += reward
                ep_reward1 += reward1
                print("episode: {} , the episode reward0 is {}".format(i, round(ep_reward, 8)))
                print("episode: {} , the episode reward1 is {}".format(i, round(ep_reward1, 8)))
                break
        
        writer.add_scalar('reward', ep_reward, global_step=i)
        writer.add_scalar('reward1', ep_reward1, global_step=i)
        writer.add_scalar('mean_action_value', mean_action_value, global_step=i)
        writer.add_scalar('max_action_value', max_action_value, global_step=i)
            
        print(f'winner is {match.winner}')

        
        if TEST:
            server = HTTPServer()
            server.match = match
            server.run()
            break

if __name__ == '__main__':
    main()
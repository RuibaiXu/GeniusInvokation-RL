import torch

from lpsim import Match, Deck
from lpsim.agents import RandomAgent
from lpsim.server.interaction import (
    Responses,
    SwitchCardRequest, SwitchCardResponse,
    ChooseCharactorRequest, ChooseCharactorResponse,
    RerollDiceRequest, RerollDiceResponse,
    DeclareRoundEndResponse,
    ElementalTuningRequest, ElementalTuningResponse,
    SwitchCharactorRequest, SwitchCharactorResponse,
    UseSkillRequest, UseSkillResponse,
    UseCardRequest, UseCardResponse,
)

class RLAgent(RandomAgent):
    # def __init__(self, player_idx):
    #     super().__init__(player_idx)

    def action_to_req_index(self, reqs, action_index):
        res = 0
        # res = len(reqs) -1

        req_dict = {
            'DeclareRoundEndRequest': [i for i, req in enumerate(reqs) if req.name == 'DeclareRoundEndRequest'],
            'ElementalTuningRequest': [i for i, req in enumerate(reqs) if req.name == 'ElementalTuningRequest'],
            'SwitchCharactorRequest': [i for i, req in enumerate(reqs) if req.name == 'SwitchCharactorRequest'],
            'UseCardRequest': [i for i, req in enumerate(reqs) if req.name == 'UseCardRequest'],
            'UseSkillRequest': [i for i, req in enumerate(reqs) if req.name == 'UseSkillRequest']
        }
        
        if action_index == 0:   # If action_index is 0, we need DeclareRoundEndRequest
            # return req_dict['DeclareRoundEndRequest'][0]   # We guarantee there's always one DeclareRoundEndRequest
            return 0
        elif action_index == 1:  # For ElementalTuningRequest
            return req_dict['ElementalTuningRequest'][0] if req_dict['ElementalTuningRequest'] else res
        elif 2 <= action_index <= 3: # For SwitchCharactorRequest
            # return req_dict['SwitchCharactorRequest'][-1] if req_dict['SwitchCharactorRequest'] else 0
            if len(req_dict['SwitchCharactorRequest']) > action_index-2:
                return req_dict['SwitchCharactorRequest'][action_index-2]
            elif req_dict['SwitchCharactorRequest']:
                return req_dict['SwitchCharactorRequest'][-1]   # return the largest index of UseCardRequest if exists
            else:
                return res
        elif 4 <= action_index <= 13:  # For UseCardRequest
            if len(req_dict['UseCardRequest']) > action_index-4:
                return req_dict['UseCardRequest'][action_index-4]
            elif req_dict['UseCardRequest']:
                return req_dict['UseCardRequest'][-1]   # return the largest index of UseCardRequest if exists
            else:
                return res
        elif 14 <= action_index <= 19:  # For UseSkillRequest
            if len(req_dict['UseSkillRequest']) > action_index-14:
                return req_dict['UseSkillRequest'][action_index-14]
            elif req_dict['UseSkillRequest']:
                return req_dict['UseSkillRequest'][-1]   # return the largest index of UseSkillRequest if exists
            else:
                return res
        else:
            print("Invalid action index!")
            return 0

# ways to call this function could include:
# index = action_to_req_index(reqs, action_index)

    def generate_response(self, match, action_tensor, epsilon = 0.01):
        # reqs = match.requests
        # reqs = reqs = [x for x in match.requests if 1]
        # reqs.sort(key=lambda x: x.name)

        # print(f'match.requests:  {match.requests}')

        reqs = list(([x for x in match.requests if x.player_idx == self.player_idx]))

        if len(reqs) == 0:
            return None

        reqs.sort(key = lambda x: x.name)

        # ## DEBUG PRINT
        # print(len(reqs))
        # # print(f'reqs:  {reqs}')
        # # 循环print reqs
        # for req in reqs:
        #     print(req.name)
        
        # print()

        # 生成一个随机数epsilon
        if torch.rand(1).item() > epsilon:
            # 取action_tensor中最大值对应的index,好像应该再接受一个epsilon参数
            index = torch.argmax(action_tensor).item() 
        ### 改成新的处理方法了
        #     if index >= len(reqs):
        #         index = len(reqs) -1
            action_to_req_index = self.action_to_req_index(reqs, index)
        else:
            index = int(torch.rand(1).item() * len(reqs))
            action_to_req_index = index
            # index = torch.randint(len(req_names), (1,)).item()

        # ## DEBUG PRINT
        # print(f'index:  {index}')
        # # action_to_req_index = self.action_to_req_index(reqs, index)
        # print(f'action_to_req_index:  {action_to_req_index}')
        # print(f'reqs[action_to_req_index].name:  {reqs[action_to_req_index].name}')
        # print()
        
        
        # req = reqs[index]
        req = reqs[action_to_req_index]

        # 
        if req.name == 'SwitchCardRequest':
            resp = self.resp_switch_card(req)
        elif req.name == 'ChooseCharactorRequest':
            resp = self.resp_choose_charactor(req)
        elif req.name == 'RerollDiceRequest':
            resp = self.resp_reroll_dice(req)
        elif req.name == 'DeclareRoundEndRequest':
            resp = DeclareRoundEndResponse(request = req)
        elif req.name == 'ElementalTuningRequest':
            resp = self.resp_elemental_tuning(req)
        elif req.name == 'SwitchCharactorRequest':
            resp = self.resp_switch_charactor(req)
        elif req.name == 'UseSkillRequest':
            resp = self.resp_use_skill(req)
            # print(f'resp:  {resp}')


        elif req.name == 'UseCardRequest':
            resp = self.resp_use_card(req)
        else:
            raise ValueError(f'Unknown request name: {req.name}')

        ### DEBUG PRINT         
        # print(f'resp:  {resp}')
        # print()

        return resp, index
        # return resp, action_to_req_index
    
    def resp_reroll_dice(self, req: RerollDiceRequest) -> RerollDiceResponse:
        """
        Randomly choose a subset of dice.
        """
        # print(f'req:  {req}')
        reroll_dice_idxs = []
        for i in range(len(req.colors)):
            if req.colors[i] == 'CRYO' and self.random() < 0.9:
                reroll_dice_idxs.append(i)
            elif req.colors[i] == 'PYRO' and self.random() < 0.9:
                reroll_dice_idxs.append(i)
            elif req.colors[i] == 'GEO' and self.random() < 0.9:
                reroll_dice_idxs.append(i)
            elif req.colors[i] == 'ANEMO' and self.random() < 0.9:
                reroll_dice_idxs.append(i)
        return RerollDiceResponse(
            request = req, reroll_dice_idxs = reroll_dice_idxs
        )
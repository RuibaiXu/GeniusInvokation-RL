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

    def generate_response(self, match, action_tensor, epsilon = 0.01):
        # reqs = match.requests
        # reqs = reqs = [x for x in match.requests if 1]
        # reqs.sort(key=lambda x: x.name)

        # print(f'match.requests:  {match.requests}')

        reqs = list(([x for x in match.requests if x.player_idx == self.player_idx]))
        # print(f'reqs:  {reqs}')
        reqs.sort(key = lambda x: x.name)

        # print(f'reqs:  {reqs}')
        
        # req_names = list(set([x.name for x in match.requests 
        #                       if x.player_idx == self.player_idx]))
        # req_names.sort()

        if len(reqs) == 0:
            return None
        
        # 生成一个随机数epsilon
        if torch.rand(1).item() > epsilon:
            # 取action_tensor中最大值对应的index,好像应该再接受一个epsilon参数
            index = torch.argmax(action_tensor).item() 
            if index >= len(reqs):
                index = len(reqs) -1
        else:
            index = int(torch.rand(1).item() * len(reqs))
            # index = torch.randint(len(req_names), (1,)).item()        
        
        req = reqs[index]

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
        elif req.name == 'UseCardRequest':
            resp = self.resp_use_card(req)
        else:
            raise ValueError(f'Unknown request name: {req.name}')

        return resp, index
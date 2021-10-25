import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, options):
        super(Policy, self).__init__()

        # actor's layer
        self.action_head = nn.Sequential(
            nn.Linear(options['out_dim']+1, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 2)
        )

        # critic's layer
        self.value_head = nn.Sequential(
            nn.Linear(options['out_dim']+1, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # actor: chooses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values

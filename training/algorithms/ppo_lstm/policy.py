import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOPolicy(nn.Module):
    def __init__(self, observation_shape, action_dim, conv_channels, lstm_hidden=256, lstm_layers=1):
        super().__init__()
        C, H, W = observation_shape
        convs = []
        in_c = C
        for out_c in conv_channels:
            convs.append(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1))
            convs.append(nn.ReLU())
            convs.append(nn.LayerNorm([out_c, H, W]))
            in_c = out_c
        self.conv_net = nn.Sequential(*convs) if convs else nn.Identity()
        conv_out = in_c * H * W
        self.lstm = nn.LSTM(conv_out, lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.policy_head = nn.Linear(lstm_hidden, action_dim)
        self.value_head = nn.Linear(lstm_hidden, 1)

    def init_hidden(self, batch_size: int = 1):
        num_layers = self.lstm.num_layers; hidden = self.lstm.hidden_size
        h = torch.zeros(num_layers, batch_size, hidden, device=next(self.parameters()).device)
        c = torch.zeros(num_layers, batch_size, hidden, device=next(self.parameters()).device)
        return (h, c)

    def forward(self, obs: torch.Tensor, hidden_state=None):
        B = obs.size(0)
        conv_out = self.conv_net(obs).view(B, -1).unsqueeze(1)
        if hidden_state is None:
            hidden_state = self.init_hidden(B)
        lstm_out, lstm_hidden = self.lstm(conv_out, hidden_state)
        x = lstm_out[:, -1, :]
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value, lstm_hidden

    def act(self, obs: torch.Tensor, action_mask: torch.Tensor = None, hidden_state=None):
        logits, value, new_hidden = self(obs, hidden_state)
        if action_mask is not None:
            logits = logits + (action_mask + 1e-8).log()
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach(), log_prob.detach(), value.detach(), new_hidden

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor = None, hidden_state=None):
        logits, value, _ = self(obs, hidden_state)
        if action_mask is not None:
            logits = logits + (action_mask + 1e-8).log()
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, value

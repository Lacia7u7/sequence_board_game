import json, argparse, torch, torch.optim as optim
from ..envs.sequence_env import SequenceEnv
from ..envs.vectorized_env import VectorizedEnv
from ..algorithms.ppo_lstm import policy as ppo_policy
from ..algorithms.ppo_lstm import storage as ppo_storage
from ..algorithms.ppo_lstm import learner as ppo_learner
from ..algorithms.ppo_lstm import utils as ppo_utils
from ..utils import logging as log_utils, jsonio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--override", type=str, default=None)
    args = parser.parse_args()
    config = json.load(open(args.config))
    if args.override:
        config = jsonio.override_config(config, json.loads(args.override))

    num_envs = config["training"].get("num_envs", 1)
    envs = VectorizedEnv(num_envs, lambda: SequenceEnv(config)) if num_envs > 1 else SequenceEnv(config)
    seed = config["training"].get("seed", None)
    if seed is not None:
        if isinstance(envs, VectorizedEnv):
            envs.reset(seeds=[seed+i for i in range(num_envs)])
        else:
            envs.reset(seed=seed)

    dummy_env = SequenceEnv(config) if isinstance(envs, VectorizedEnv) else envs
    obs_shape = dummy_env.observation_space.shape
    action_dim = dummy_env.action_space.n
    conv_channels = config["model"].get("conv_channels", [64,64,128,128])
    lstm_hidden = config["model"].get("lstm_hidden", 256)
    lstm_layers = config["model"].get("lstm_layers", 1)
    policy = ppo_policy.PPOPolicy(obs_shape, action_dim, conv_channels, lstm_hidden, lstm_layers)
    if torch.cuda.is_available(): policy = policy.cuda()
    optimizer = optim.Adam(policy.parameters(), lr=config["training"]["lr"])

    rollout_length = config["training"]["rollout_length"]
    nenv = num_envs if isinstance(envs, VectorizedEnv) else 1
    storage = ppo_storage.RolloutStorage(rollout_length, nenv, obs_shape, lstm_hidden)

    total_updates = config["training"]["total_updates"]
    log = log_utils.Logger(config)

    for update in range(1, total_updates+1):
        lr = ppo_utils.linear_lr_decay(config["training"]["lr"], update-1, total_updates)
        for g in optimizer.param_groups: g["lr"] = lr
        if isinstance(envs, VectorizedEnv):
            obs_list, _info = envs.reset()
        else:
            ob, _info = envs.reset(); obs_list = [ob]

        obs_tensor = torch.tensor(obs_list, dtype=torch.float32)
        storage.obs[0].copy_(obs_tensor)
        hidden = policy.init_hidden(batch_size=nenv)

        for step in range(rollout_length):
            obs_batch = obs_tensor.cuda() if torch.cuda.is_available() else obs_tensor
            action, log_prob, value, new_hidden = policy.act(obs_batch, action_mask=None, hidden_state=hidden)
            action_cpu = action.cpu().numpy()
            if isinstance(envs, VectorizedEnv):
                obs_list, reward_list, done_list, trunc_list, info_list = envs.step(action_cpu.tolist())
            else:
                ob2, rw, dn, tr, info = envs.step(int(action_cpu[0]))
                obs_list = [ob2]; reward_list = [rw]; done_list = [dn]; trunc_list = [tr]
            obs_tensor = torch.tensor(obs_list, dtype=torch.float32)
            rewards = torch.tensor(reward_list, dtype=torch.float32)
            dones = torch.tensor([1.0 if d or t else 0.0 for d,t in zip(done_list, trunc_list)], dtype=torch.float32)
            storage.insert(obs_tensor, action.cpu(), log_prob.cpu(), value.cpu(), rewards.cpu(), dones.cpu(), hidden[0].cpu(), hidden[1].cpu())
            hidden = new_hidden
        with torch.no_grad():
            last_obs = storage.obs[-1]
            last_value = policy(last_obs.cuda() if torch.cuda.is_available() else last_obs)[1].cpu()
        advantages, returns = storage.compute_returns_and_advantages(last_value, config["training"]["gamma"], config["training"]["gae_lambda"])
        storage.returns = returns

        learner = ppo_learner.PPOLearner(policy, optimizer, config["training"]["clip_eps"], config["training"]["value_coef"], config["training"]["entropy_coef"], config["training"]["max_grad_norm"])
        pl, vl, ent = learner.update(storage)
        log.log_metrics(update, {"policy_loss": pl, "value_loss": vl, "entropy": ent})
        storage.step = 0

    torch.save(policy.state_dict(), f"{config['logging'].get('logdir','runs')}/policy_final.pth")
    print("Training finished. Model saved.")

if __name__ == "__main__":
    main()

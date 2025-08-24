import random
class OpponentPool:
    def __init__(self):
        self.snapshots = []
        self.heuristics = []
    def add_snapshot(self, policy):
        self.snapshots.append(policy)
        return len(self.snapshots)-1
    def sample_opponent(self, current_policy, probabilities=None):
        probabilities = probabilities or {"current":0.5,"snapshots":0.3,"heuristics":0.2}
        x = random.random()
        if x < probabilities["current"] or (not self.snapshots and not self.heuristics):
            return current_policy
        x2 = random.random()
        if x2 < probabilities["snapshots"] and self.snapshots:
            return random.choice(self.snapshots)
        if self.heuristics:
            return random.choice(self.heuristics)
        return current_policy
    def add_heuristic(self, agent):
        self.heuristics.append(agent)

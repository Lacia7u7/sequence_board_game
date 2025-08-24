import random
class RandomPolicy:
    def select_action(self, obs, legal_mask):
        if legal_mask is None:
            return random.randrange(0, 100)
        legal = [i for i, m in enumerate(legal_mask.flatten()) if m > 0.5]
        return random.choice(legal) if legal else 0

import math
def calculate_elo(rating_a: float, rating_b: float, result_a: float, k: float = 32.0):
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 / (1.0 + 10 ** ((rating_a - rating_b) / 400.0))
    new_a = rating_a + k * (result_a - expected_a)
    new_b = rating_b + k * ((1 - result_a) - expected_b)
    return new_a, new_b

class TrueSkillTeamRating:
    def __init__(self, mu: float = 25.0, sigma: float = 25.0/3.0):
        self.mu = mu; self.sigma = sigma

def update_trueskill(team1: TrueSkillTeamRating, team2: TrueSkillTeamRating, result: float):
    beta = team1.sigma
    if result == 1: team1.mu += beta/2; team2.mu -= beta/2
    elif result == 0: team1.mu -= beta/2; team2.mu += beta/2
    team1.sigma = max(team1.sigma*0.99, 1.0)
    team2.sigma = max(team2.sigma*0.99, 1.0)
    return team1, team2

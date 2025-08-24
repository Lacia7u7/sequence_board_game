from training.encoders.probability_layers import probability_cell_playable_next, probability_cell_playable_k_steps

def test_hypergeom_single():
    deck_counts = {"AS":1, "JC":1, "JD":0}
    total = 10
    p = probability_cell_playable_next(deck_counts, total, "AS")
    expected = 0.1 + 0.1 - 0.01
    assert abs(p - expected) < 1e-6

def test_multi_step_independence():
    p = 0.1
    p2 = probability_cell_playable_k_steps(p, 2)
    p3 = probability_cell_playable_k_steps(p, 3)
    assert p < p2 < p3 < 1.0

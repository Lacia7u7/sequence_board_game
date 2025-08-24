from training.envs.action_mapping import flat_to_components, components_to_flat

def test_round_trip_mapping():
    max_hand = 7; include_pass = True
    flats = [0, 45, 99, 100, 103, 107, 108]
    for f in flats:
        typ, det = flat_to_components(f, max_hand, include_pass)
        if typ == "unknown": continue
        f2 = components_to_flat(typ, det, max_hand, include_pass)
        assert f == f2

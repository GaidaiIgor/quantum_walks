from quantum_walks import PathFinderMHSNonlinear

pathf=PathFinderMHSNonlinear()

def test_get_remaining_states():
    states=["00", "01", "10", "11"]
    idx=2
    remaining=pathf._get_remaining_states(states, idx)
    assert remaining==["00", "01", "11"]

def test_get_diffs():
    z1="11111"
    z2="01100"
    diffs=pathf._get_diffs(z1, z2)
    assert diffs==[0,3,4]

def test_get_mhs():
    diffs=[[0, 1, 2, 3], [1, 2], [2], [3]]
    mhs=pathf._get_mhs(diffs)
    assert mhs==[2,3]

def test_get_most_infrequent_elem():
    mhs=[2,5,7,8]
    size_ones=[(0, [7]), (1, [7]), (2, [2]), (3, [8]), (4, [7]), (5, [2]), (6, [5]), (7, [5])]
    target=pathf._get_most_infrequent_elem(mhs, size_ones)
    assert target==8

    mhs=[0,2,5,7,8]
    size_ones=[(0, [7]), (1, [7]), (2, [2]), (3, [8]), (4, [7]), (5, [2]), (6, [5]), (7, [5]), (8, [8]), (9, [0])]
    target=pathf._get_most_infrequent_elem(mhs, size_ones)
    assert target==0
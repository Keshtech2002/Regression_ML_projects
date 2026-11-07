from downscale import make_data,run
def test_reproducible(tmp_path):
    assert run(tmp_path/'a')==run(tmp_path/'b')
def test_shapes():
    y,b,X,positions=make_data()
    assert y.shape==b.shape==positions.shape==(48,48)
    assert X.shape==(48*48,6)

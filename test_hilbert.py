from quanition_studio_v1.0 import HilbertModule

def test_hilbert_run_ok():
    res = HilbertModule.run({"pA":0.55, "pB_given_A":0.58, "pA_given_B":0.60})
    assert res.ok
    assert 0.0 <= res.details.get("phi", 0.0) <= 6.4  # radians

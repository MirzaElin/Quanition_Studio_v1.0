from quanition_studio_v1.0 import QDTModule

def test_qdt_probs_sum_to_one():
    res = QDTModule.run({"prospects":[{"name":"A","utility":0.2},{"name":"B","utility":0.1}] , "tau":1.0})
    assert res.ok
    ps = res.details["final_probabilities"]
    assert abs(sum(ps) - 1.0) < 1e-8

from quanition_studio_v1.0 import QGameModule

def test_qgame_runs():
    res = QGameModule.run({"R":3,"S":0,"T":5,"P":1,"gamma":0.6})
    assert res.ok
    best = res.details.get("best")
    assert best and isinstance(best, list)

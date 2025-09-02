from quanition_studio_v1.0 import QMoneyModule

def test_qmoney_sim():
    res = QMoneyModule.run({"T":10,"c0":1.0,"c1":0.05,"H0":100,"D0":50,"L0":20})
    assert res.ok
    hs = res.details.get("H", [])
    assert len(hs) == 10

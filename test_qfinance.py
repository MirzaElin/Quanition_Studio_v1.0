from quanition_studio_v1.0 import QFinanceModule

def test_qfinance_binomial():
    res = QFinanceModule.run({"S0":100,"K":100,"r":0.01,"sigma":0.2,"T":1.0,"steps":50})
    assert res.ok
    price = res.details["binomial"]
    assert price > 0.0

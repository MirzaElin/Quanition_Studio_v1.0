from quanition_studio_v1.0 import CbDModule

def test_cbd_threshold_flag():
    vals = {"E11":0.7,"E21":0.6,"E22":0.6,"E12":0.7,"mA1":0.2,"mA2":0.1,"mB1":0.05,"mB2":0.02}
    res = CbDModule.run(vals)
    assert res.ok
    assert "contextual" in res.details

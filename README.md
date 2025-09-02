Quanition Studio v1.0

Quanition Studio v1.0 is a desktop app (PySide6) for quantum‑inspired and behavioral modeling with six built‑in methods:
Hilbert, QDT, CbD, Quantum Game (Eisert), QFinance (binomial option pricing), and QMoney (toy macro money).



Features
- Import CSV/XLSX datasets
- Auto mode to run multiple analyses end‑to‑end
- HTML/DOCX reporting
- Built‑in docs (EULA, privacy, manual) and Learning Center tabs
- Example datasets generator (accessible from the app)

Install (from source)
```bash
in the repo root
python -m pip install -U pip
pip install -e .
```

> Requirements: Python 3.9+, [PySide6](https://pypi.org/project/PySide6/). Optional: `python-docx` for DOCX export.

Run the GUI
```bash
quanition-studio
```
or from Python:
```python
from quanition_studio_v1.0 import main
main()
```

Minimal Programmatic Examples
```python
from quanition_studio_v1.0 import HilbertModule, QDTModule, CbDModule, QGameModule, QFinanceModule, QMoneyModule

assert HilbertModule.run({"pA":0.55,"pB_given_A":0.58,"pA_given_B":0.60}).ok
assert QDTModule.run({"prospects":[{"name":"A","utility":0.1},{"name":"B","utility":0.2}], "tau":1.0}).ok
assert CbDModule.run({"E11":0.7,"E21":0.6,"E22":0.6,"E12":0.7,"mA1":0.2,"mA2":0.1,"mB1":0.05,"mB2":0.02}).ok
assert QGameModule.run({"R":3,"S":0,"T":5,"P":1,"gamma":0.6}).ok
assert QFinanceModule.run({"S0":100,"K":100,"r":0.01,"sigma":0.2,"T":1.0}).ok
assert QMoneyModule.run({"T":20,"c0":1.0,"c1":0.05,"H0":100,"D0":50,"L0":20}).ok
```

Tests
Run all tests with:
```bash
pytest -q
```

Packaging Notes
- This repo uses `pyproject.toml` + `setuptools` and exposes a console script entry point `quanition-studio` that launches the GUI (Qt).
- If tests run in headless CI, they only exercise non‑GUI logic.

License
MIT License © Mirza Niaz Zaman Elin



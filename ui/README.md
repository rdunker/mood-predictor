### Quick start

1. Install dependencies (prefer a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ui/requirements.txt
```

2. Prepare artifacts (run these cells in `AI_try.ipynb` after training the network):

```python
import joblib
import numpy as np

# after training in the notebook, save the preprocessor and weights
import os
os.makedirs('artifacts', exist_ok=True)
joblib.dump(preprocessor, 'artifacts/preprocessor.joblib')
np.savez(
    'artifacts/model_weights.npz',
    W1=network.W1, b1=network.b1,
    W2=network.W2, b2=network.b2,
    W3=network.W3, b3=network.b3,
)
```

3. Run the UI:

```bash
python ui/app.py
```

4. Open http://localhost:5000 in your browser.

Notes
- The app will automatically load `artifacts/preprocessor.joblib` and `artifacts/model_weights.npz` if they exist.
- `region` options are populated from the fitted OneHotEncoder if available; otherwise a free-text field is shown.

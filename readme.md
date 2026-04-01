# High Expectation vs UCB



This folder contains a small set of notebooks exploring the relationship between a high-expectation strategy and GP-UCB-style acquisition functions. The main example combines a synthetic binary outcome model, Gaussian-process-based surrogates, and visual comparisons between exploration strategies.

## Contents

- [highexp-vs-ucb.ipynb](highexp-vs-ucb.ipynb): background notes and a 1D Gaussian-process regression example.
- [highexp-vs-ucb-example.ipynb](highexp-vs-ucb-example.ipynb): the main 2D notebook with the binary outcome toy problem, sequential sampling, and visualizations.
- [requirements.txt](requirements.txt): Python packages used by the notebooks.

## Run the notebooks

### Binder

Click the Binder badge below to launch the main notebook in a hosted environment.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PaulEibensteiner/highexp-vs-ucb/HEAD?urlpath=%2Fdoc%2Ftree%2Fhighexp-vs-ucb-example.ipynb)

### Local setup

1. Create a virtual environment.
2. Install the dependencies from [requirements.txt](requirements.txt).
3. Open the notebook in Jupyter or VS Code and run the cells top to bottom.

Example:

```bash
python3 -m venv .env
source .env/bin/activate
python -m pip install -r requirements.txt
```

## Notes

- The notebooks are meant as explanatory demos, not production-ready Bayesian optimization code.
- The visualizations emphasize intuition: the synthetic ground-truth surface, the sampled observations, and the difference between exploratory strategies.

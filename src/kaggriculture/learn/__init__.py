"""Training-time code. Never imported by the submitted agent.

The competition sandbox has two cores, 6.8 GB and no network, and importing
torch there costs 10.7 seconds of a 60-second overage pool. `package.py`
excludes this package from the submission archive for that reason, and nothing
under `kaggriculture.policy` or `main.py` may import from it.
"""

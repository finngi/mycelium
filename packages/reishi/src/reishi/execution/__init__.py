"""Execution: the Producer contract (contract.py), discovery of installed
producers (registry.py), and the in-process local executor (local.py).

reishi itself never imports Ray, MLX, or any runtime-specific dependency --
these modules define the seam executors plug into, not an executor.
"""

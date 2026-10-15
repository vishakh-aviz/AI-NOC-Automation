"""Grading: checks.py (one check per prompt type), compare.py (read tables / names / numbers
out of an answer), judge.py (optional second-opinion note, never changes a result).

gpu.py holds the checks of the GPU-metric prompts (Prometheus, DCGM; 2026-10-08), bcm_dynamo.py those of the BCM
and Dynamo prompts (2026-10-14); they are added to checks.CHECKS / METRIC_CHECKS here, so evaluate() and the runner
see one table.
"""
from ncp_suite.grading import bcm_dynamo, checks, gpu

for _module in (gpu, bcm_dynamo):
    checks.CHECKS.update(_module.CHECKS)
    checks.METRIC_CHECKS.update(_module.METRIC_CHECKS)

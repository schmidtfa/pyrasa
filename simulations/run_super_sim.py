# %%
import numpy as np
from cluster_jobs.super_simulation import super_simulation
from plus_slurm import JobCluster, PermuteArgument


# % get jobcluster
job_cluster = JobCluster(
    required_ram='2G', request_cpus=1, request_time=30, python_bin='/home/schmidtfa/miniconda3/envs/pyrasa/bin/python'
)

# % put in jobs...
sim_kinds = ['fixed', 'knee', 'burst', 'broad_fixed', 'broad_knee']
EXPS = np.round(np.arange(-3, 0.5, 0.5), 1).tolist()
KNEES = [25, 100, 400, 900, 1600]

FREQS = np.arange(5, 36, 1).tolist()
POWERS = np.round(np.arange(0, 2.0, 0.2), 1).tolist()
BWS = np.round(np.arange(0.5, 3.5, 0.5), 1).tolist()

# Burst related parameters
BPROBS = np.round(np.arange(0.2, 0.8, 0.2), 1).tolist()
# len(BPROBS) * len(BWS) * len(POWERS) * len(FREQS) * len(KNEES) * len(EXPS)
# %%
job_cluster.add_job(
    super_simulation,
    subject_id='broad_knee',  # PermuteArgument(sim_kinds),
    exp=PermuteArgument(EXPS),
    knee=PermuteArgument(KNEES),
    freq=PermuteArgument(FREQS),
    bw=PermuteArgument(BWS),
    height=PermuteArgument(POWERS),
    # b_proba = PermuteArgument(BPROBS),
)

# % submit...
job_cluster.submit(do_submit=True)
# %%

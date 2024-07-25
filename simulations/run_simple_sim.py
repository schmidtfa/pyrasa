# %%
import numpy as np
from cluster_jobs.simple_simulation import simple_simulation
from plus_slurm import JobCluster, PermuteArgument

# % get jobcluster
job_kwargs = {
    'required_ram': '2G',
    'request_cpus': 1,
    'request_time': 30,
    'python_bin': '/home/schmidtfa/miniconda3/envs/pyrasa/bin/python',
}

# % put in jobs...
# aperiodic parameters
EXPS = np.round(np.arange(-3, 0.5, 0.5), 1).tolist()
EXPS_KNEE = np.round(np.arange(-4, -1, 0.2), 1).tolist()  # need to start above 1
KNEE_FREQS = np.arange(5, 40, 5).tolist()

# Freq related parameters
FREQS = np.arange(5, 36, 1).tolist()
POWERS = np.round(np.arange(0, 2.0, 0.2), 1).tolist()
BWS = np.round(np.arange(0.5, 3.5, 0.5), 1).tolist()

# Burst related parameters
BPROBS = np.round(np.arange(0.2, 0.8, 0.1), 1).tolist()

# %% exponent_delta
job_cluster = JobCluster(**job_kwargs)
job_cluster.add_job(
    simple_simulation,
    subject_id='exponent_delta',  # PermuteArgument(sim_kinds),
    exp=PermuteArgument(EXPS),
    fit_type='fixed',
)
job_cluster.submit(do_submit=True)

# %% knee_delta
job_cluster = JobCluster(**job_kwargs)
job_cluster.add_job(
    simple_simulation,
    subject_id='knee_delta',  # PermuteArgument(sim_kinds),
    knee_freq=PermuteArgument(KNEE_FREQS),
    fit_type='knee',
)
job_cluster.submit(do_submit=True)

# %% exponent_knee_delta
job_cluster = JobCluster(**job_kwargs)
job_cluster.add_job(
    simple_simulation,
    subject_id='exponent_knee_delta',  # PermuteArgument(sim_kinds),
    exp=PermuteArgument(EXPS_KNEE),
    fit_type='knee',
)
job_cluster.submit(do_submit=True)

# %%burst_delta
job_cluster = JobCluster(**job_kwargs)
job_cluster.add_job(
    simple_simulation,
    subject_id='burst_delta',  # PermuteArgument(sim_kinds),
    b_proba=PermuteArgument(BPROBS),
    fit_type='fixed',
)
job_cluster.submit(do_submit=True)

# %% freq_width_delta
job_cluster = JobCluster(**job_kwargs)
job_cluster.add_job(
    simple_simulation,
    subject_id='freq_width_delta',  # PermuteArgument(sim_kinds),
    bw=PermuteArgument(BWS),
    fit_type='fixed',
)
job_cluster.submit(do_submit=True)

# %% freq_height_delta
job_cluster = JobCluster(**job_kwargs)
job_cluster.add_job(
    simple_simulation,
    subject_id='freq_height_delta',  # PermuteArgument(sim_kinds),
    height=PermuteArgument(POWERS),
    fit_type='fixed',
)
job_cluster.submit(do_submit=True)

# %% freq_delta
job_cluster = JobCluster(**job_kwargs)
job_cluster.add_job(
    simple_simulation,
    subject_id='freq_delta',  # PermuteArgument(sim_kinds),
    freq=PermuteArgument(FREQS),
    fit_type='fixed',
)
job_cluster.submit(do_submit=True)

# %%

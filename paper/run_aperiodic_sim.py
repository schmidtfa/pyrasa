#%%
import numpy as np
from cluster_jobs.knee_simulations import knee_simulations
from plus_slurm import JobCluster, PermuteArgument
import os


#% get jobcluster
job_cluster = JobCluster(required_ram='2G',
                         request_cpus=2,
                         request_time=60*5,
                         python_bin='/home/schmidtfa/miniconda3/envs/pyrasa/bin/python')

#% put in jobs...
job_cluster.add_job(knee_simulations,
                    fit_func='knee',
                    exponent_1 = PermuteArgument([0, .5, 1]),
                    n_seconds=PermuteArgument([4, 8, 30, 60, 60*3, 60*5]),
                    knee_freq=PermuteArgument(np.arange(1, 30).tolist()),
                    
                    )

#% submit...
job_cluster.submit(do_submit=True)
# %%

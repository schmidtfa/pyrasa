# %%
import numpy as np
from cluster_jobs.analyze_eo_ec import eo_ec_analysis
from plus_slurm import JobCluster, PermuteArgument


# % get jobcluster
job_cluster = JobCluster(
    required_ram='2G', request_cpus=2, request_time=30, python_bin='/home/schmidtfa/miniconda3/envs/pyrasa/bin/python'
)


sids = [str(i) for i in np.arange(2, 12).tolist()]

# %%
job_cluster.add_job(eo_ec_analysis, subject_id=PermuteArgument(sids), condition=PermuteArgument(['EO', 'EC']))

# % submit...
job_cluster.submit(do_submit=True)
# %%

#!/bin/bash
{% set log_dir = jobs_dir / "log" %}
#SBATCH --cpus-per-task={{ request_cpus }}
#SBATCH --mem={{ required_mem }}
#SBATCH --array=1-{{ n_jobs }}
#SBATCH --time={{ request_time }}
#SBATCH --output={{ log_dir }}/out_%a.log
{% if exclude_nodes %}
#SBATCH --exclude="{{ exclude_nodes }}"
{% endif %}

echo "Executing Job $SLURM_JOB_ID on $SLURMD_NODENAME"
{{ executable }} {{ arguments }}
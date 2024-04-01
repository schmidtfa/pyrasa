# -*- coding: UTF-8 -*-
# Copyright (c) 2018, Thomas Hartmann
#
# This file is part of the plus_slurm Project,
# see: https://gitlab.com/thht/plus-slurm
#
#    plus_slurm is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    plus_slurm is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with obob_subjectdb. If not, see <http://www.gnu.org/licenses/>.
import datetime
import sys
import os
import socket
from pathlib import Path
import psutil
import resource

os.chdir('{{ working_directory }}')

sys.path.append(os.getcwd())

requested_ram = {{ required_mem }} / 1024  # noqa

from plus_slurm.job import JobItem  # noqa

if __name__ == '__main__':
    job_info = {
        'ClusterId': 0,
        'ProcId': 0,
        'requested_ram': requested_ram,
    }

    slurm_task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    slurm_job_id = int(os.getenv('SLURM_ARRAY_JOB_ID'))

    additional_path = '{{ append_to_path }}'
    if additional_path:
        sys.path.append(additional_path)

    job_fname = Path('{{ jobs_dir }}',
                     'slurm',
                     f'job{slurm_task_id:03d}.json.gzip')

    job_item = JobItem(job_fname)

    job_object = job_item.make_object()

    job_started = datetime.datetime.now()

    psutil_process = psutil.Process()
    psutil_info = psutil_process.as_dict()

    print('Running on: %s' % (socket.gethostname(), ))
    print(f'Running on CPUs: {psutil_info["cpu_affinity"]}')
    print('Now running %s' % (job_item, ))
    print('Parameters: %s' % (job_item.args, ))
    print('Keyword Parameters: %s' % (job_item.kwargs, ))
    print(f'Job ID: {slurm_job_id}, Task ID: {slurm_task_id}')

    print(f'Starting Job at {job_started}\n##########')
    job_object.run_private()
    job_stopped = datetime.datetime.now()
    print(f'##########\nJob stopped at {job_stopped}')
    print(f'Execution took {job_stopped - job_started}')

    initial_cpu_time = sum(psutil_info['cpu_times'])
    psutil_info = psutil_process.as_dict()
    final_cpu_time = sum(psutil_info['cpu_times'])

    avg_amount_of_cpus_used = (final_cpu_time - initial_cpu_time) / (job_stopped - job_started).seconds  # noqa

    mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

    mem_toomuch = 100 * (requested_ram - mem_used) / mem_used

    print(f'Your job used an average of {avg_amount_of_cpus_used:.2f} CPUs')
    print('Your job asked for %.2fGB of RAM' % (requested_ram, ))
    print('Your job used a maximum of %.2fGB of RAM' % (mem_used, ))
    print('You overestimated you memory usage by %.2f%%.' % (mem_toomuch, ))

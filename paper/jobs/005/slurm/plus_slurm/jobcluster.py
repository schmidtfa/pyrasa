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

import getpass
import humanfriendly
import six
import sys
import os
import plus_slurm.job
import subprocess
import logging
import jinja2
import uuid
import numpy
import itertools
import copy
import collections
import inspect
import shutil
from pathlib import Path

from plus_slurm.job import JobItem


class JobCluster(object):
    """
    This is the main class, the *controller* of plus_slurm. It collects all
    the jobs and takes care of submitting them
    to the cluster. It also contains information about how much RAM the jobs
    need, how many CPUs are requested etc.

    Parameters
    ----------
    required_ram : :class:`str`, :class:`float`, :class:`int`, optional
        The amount of RAM required to run one Job in megabytes. A string like
        "2G" or "200M" will be converted accordingly.
    request_cpus : :class:`int`, optional
        The number of CPUs requested
    request_time : :class:`int`, optional
        Maximum duration of a job in minutes.
    jobs_dir ::class:`str`, optional
        Folder to put all the jobs in. This one needs to be on the shared
        filesystem (so somewhere under /mnt/obob)
    inc_jobsdir ::class:`str`, optional
        If this is set to True (default), jobs_dir is the parent folder for all
        the jobs folders. Each time a job is
        submitted, a new folder is created in the jobs_dir folder that contains
        all the necessary files and a folder
        called "log" containing the log files. If jobs_dir is set to False, the
        respective files are put directly
        under jobs_dir. In this case, jobs_dir must either be empty or not
        exist at all to avoid any side effects.
    python_bin ::class:`str`, optional
        The path to the python interpreter that should run the jobs. If you do
        not set it, it gets chosen automatically.
        If the python interpreter you are using when submitting the jobs is on
        /mnt/obob/ that one will be used.
        If the interpreter you are using is **not** on /mnt/obob/
        the default one at /mnt/obob/obob_mne will be used.
    working_directory ::class:`str`, optional
        The working directory when the jobs run.
    exclude_nodes : :class:`str`, optional
        Comma separated list of nodes to exclude.
    max_jobs_per_jobcluster : :class:`int`, optional
        Slurm only allows a certain number of jobs per array job
        (1000 by default).
        If the number of jobs to be submitted is higher that this number, jobs
        will be split in more than one array job.
    append_to_path : :class:`str`, optional
        Path to append to the python module search path.
    """

    _slurm_submit = '/usr/bin/sbatch'
    _runner_template = 'runner.py'
    _submit_template = 'submit.sh'

    def __init__(self, required_ram='2G', request_cpus=1, jobs_dir='jobs',
                 request_time=10,
                 inc_jobsdir=True,
                 python_bin=None, working_directory=None,
                 exclude_nodes=None, max_jobs_per_jobcluster=1000,
                 append_to_path=None):
        self.required_ram = required_ram
        self.request_time = request_time
        self.request_cpus = request_cpus
        self.jobs_dir = jobs_dir
        self.inc_jobsdir = inc_jobsdir
        self.python_bin = python_bin
        self.working_directory = working_directory
        self._output_folder = None
        self._exclude_nodes = exclude_nodes
        self._max_jobs_per_jobcluster = max_jobs_per_jobcluster
        self._append_to_path = append_to_path

        self._jobs = list()

        self._jinja_env = jinja2.Environment(
            loader=jinja2.PackageLoader('plus_slurm', 'jinja2_templates'),
            trim_blocks=True,
            lstrip_blocks=True
        )

        self._jobs_submitted = list()
        self._current_jobs_submitting = None

    def add_job(self, job, *args, **kwargs):
        """
        Add one job to the JobCluster.
        All further arguments will be passed on to the Job.

        Parameters
        ----------
        job : child of :class:`plus_slurm.Job`
            The job class to be added.

        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """

        if plus_slurm.job.Job not in inspect.getmro(job):
            raise TypeError('Job must be a subclass of plus_slurm.Job')

        kwargs = collections.OrderedDict(kwargs)
        args_permuted = [idx for (idx, cur_arg) in enumerate(args) if isinstance(cur_arg, PermuteArgument)]  # noqa
        kwargs_permuted = [idx for (idx, cur_key) in enumerate(kwargs.keys()) if isinstance(kwargs[cur_key], PermuteArgument)]  # noqa

        if not args_permuted and not kwargs_permuted:
            self._jobs.append(JobItem(job, *args, **kwargs))
        else:
            all_kwargs_as_array = numpy.array([kwargs[cur_key] for (idx, cur_key) in enumerate(kwargs.keys()) if idx in kwargs_permuted])  # noqa
            all_args_as_array = numpy.hstack((numpy.array(args)[args_permuted], all_kwargs_as_array))  # noqa
            all_args_permutations = itertools.product(*[x.args for x in all_args_as_array])  # noqa
            for cur_perm_args in all_args_permutations:
                new_args = numpy.array(args)
                new_kwargs = copy.deepcopy(kwargs)
                cur_perm_args_list = list(cur_perm_args)
                if args_permuted:
                    new_args[args_permuted] = cur_perm_args_list[0:len(args_permuted)]  # noqa
                    del cur_perm_args_list[0:len(args_permuted)]

                if kwargs_permuted:
                    for idx, cur_kwarg_idx in enumerate(kwargs_permuted):
                        new_kwargs[list(new_kwargs.keys())[cur_kwarg_idx]] = cur_perm_args_list[idx]  # noqa

                self.add_job(job, *new_args, **new_kwargs)

    def run_local(self):
        """
        Runs the added jobs locally.

        """

        self._remove_notrunning_jobs()
        while self.n_jobs > 0:
            self._current_jobs_submitting = self._prepare_jobs_directory()
            self._generate_runner_file()
            self._generate_submit_file()

            for idx, _ in enumerate(
                    sorted(
                        Path(self.output_folder, 'slurm').glob('*.json.gzip')
                    )
            ):
                with (
                    open(Path(
                        self._output_folder, 'log', f'out_{idx + 1}.log'
                    ), 'wt') as out,
                ):

                    env = os.environ
                    env['SLURM_ARRAY_TASK_ID'] = str(idx + 1)
                    env['SLURM_ARRAY_JOB_ID'] = '0'

                    subprocess.run(
                        [self.executable] + str(self.arguments).split(' '),
                        cwd=self.working_directory,
                        stderr=subprocess.STDOUT,
                        stdout=out,
                        env=env
                    )

            self._jobs_submitted.extend(self._current_jobs_submitting)
            self._current_jobs_submitting = None

    def submit(self, do_submit=True):
        """
        Runs the added jobs on the cluster.

        Parameters
        ----------
        do_submit : :class:`bool`, optional
            Set this to false to not actually submit but prepare all files.

        """

        old_njobs = self.n_jobs
        self._remove_notrunning_jobs()
        while self.n_jobs > 0:
            self._current_jobs_submitting = self._prepare_jobs_directory()
            self._generate_runner_file()
            submit_fname = self._generate_submit_file()

            if do_submit:
                output = subprocess.run(
                    [self._slurm_submit, submit_fname],
                    stderr=subprocess.STDOUT,
                    cwd=self.working_directory,
                    stdout=subprocess.PIPE, text=True)
                print(f'Submit output:\n{output.stdout}\n')
                print(f'Submitted {len(self._current_jobs_submitting)}/{old_njobs} jobs.')  # noqa
            else:
                logging.info('Not actually submitting')

            self._jobs_submitted.extend(self._current_jobs_submitting)
            self._current_jobs_submitting = None

    def _remove_notrunning_jobs(self):
        new_job_list = []
        for cur_job in self._jobs:
            cur_job_object = cur_job.make_object()
            if cur_job_object.shall_run_private():
                new_job_list.append(cur_job)

        self._jobs = new_job_list

    @property
    def _jinja_context(self):
        return {
            'executable': self.executable,
            'arguments': self.arguments,
            'jobs_dir': self.output_folder,
            'required_mem': int(self.required_ram / 1024 / 1024),
            'request_cpus': self.request_cpus,
            'request_time': self.request_time,
            'uuid': str(uuid.uuid4()),
            'python_home': self.python_home,
            'n_jobs': len(self._current_jobs_submitting),
            'working_directory': self.working_directory,
            'exclude_nodes': self._exclude_nodes,
            'append_to_path': self._append_to_path
        }

    @property
    def jobs_submitted(self):
        return tuple(self._jobs_submitted)

    @property
    def n_jobs_submitted(self):
        return len(self._jobs_submitted)

    def _generate_submit_file(self):

        template = self._jinja_env.get_template(self._submit_template)

        submit_rendered = template.render(self._jinja_context)
        submit_fname = Path(self.output_folder, 'slurm', 'submit.sh')
        with open(submit_fname, 'wt') as submit_file:
            submit_file.write(submit_rendered)

        return submit_fname

    def _generate_runner_file(self):
        template = self._jinja_env.get_template(self._runner_template)

        runner_rendered = template.render(self._jinja_context)
        with open(self.runner_filename, 'wt') as runner_file:
            runner_file.write(runner_rendered)

    def _prepare_jobs_directory(self):
        if self.inc_jobsdir:
            Path(self.jobs_dir).mkdir(exist_ok=True)

            jobs_idx = 1
            while Path(self.jobs_dir, '%03d' % (jobs_idx, )).exists():
                jobs_idx = jobs_idx + 1

            final_jobs_dir = Path(self.jobs_dir, '%03d' % (jobs_idx, ))
        else:
            final_jobs_dir = self.jobs_dir
            if final_jobs_dir.exists():
                raise ValueError('The jobs folder already exists. '
                                 'Please choose another one')

        Path(final_jobs_dir).mkdir(exist_ok=True)

        json_folder = Path(final_jobs_dir, 'slurm')
        Path(json_folder).mkdir(exist_ok=True)
        log_folder = Path(final_jobs_dir, 'log')
        Path(log_folder).mkdir(exist_ok=True)

        current_jobs = self._jobs[:self._max_jobs_per_jobcluster]
        del self._jobs[:self._max_jobs_per_jobcluster]

        for idx, cur_job in enumerate(current_jobs):
            cur_job.to_json(
                Path(json_folder, 'job%03d.json.gzip' % (idx + 1, ))
            )

        self._output_folder = final_jobs_dir.resolve()

        shutil.copytree(
            Path(
                inspect.getsourcefile(inspect.getmodule(JobItem))).parent,
            Path(json_folder, 'plus_slurm')
        )

        return current_jobs

    @property
    def owner(self):
        return getpass.getuser()

    @property
    def required_ram(self):
        return self._required_ram

    @required_ram.setter
    def required_ram(self, required_ram):
        if isinstance(required_ram, six.string_types):
            required_ram = humanfriendly.parse_size(required_ram)

        if not isinstance(required_ram, (int, float)):
            raise TypeError(
                'required_ram must be either a string or a number.'
            )

        self._required_ram = required_ram

    @property
    def python_bin(self):
        return self._python_bin

    @python_bin.setter
    def python_bin(self, python_bin):
        if not python_bin:
            python_bin = sys.executable
        if not Path(python_bin).is_file():
            raise ValueError('The python interpreter does not exist.')

        if 'Python' not in str(
                subprocess.check_output([python_bin, '-V'],
                                        stderr=subprocess.STDOUT)
        ):
            raise ValueError('Cannot execute the python interpreter or '
                             'it is not a python interpreter')

        self._python_bin = python_bin

    @property
    def executable(self):
        return self.python_bin

    @property
    def arguments(self):
        return self.runner_filename

    @property
    def python_home(self):
        return Path(self.python_bin).parent.parent

    @property
    def working_directory(self):
        return self._working_directory

    @working_directory.setter
    def working_directory(self, working_directory):
        if not working_directory:
            working_directory = os.getcwd()

        if not Path(working_directory).is_dir():
            raise ValueError('working_directory is not a valid directory.')

        self._working_directory = working_directory

    @property
    def output_folder(self):
        return self._output_folder

    @property
    def n_jobs(self):
        return len(self._jobs)

    @property
    def runner_filename(self):
        return Path(self.output_folder, 'slurm', 'runner.py')


class ApptainerJobCluster(JobCluster):
    """
    If you use this class instead of :class:`JobCluster`, the jobs will
    run in a Apptainer Container.

    You need to supply the same parameters as when instantiating
    :class:`JobCluster`. Additionally, you need to at least set the
    Apptainer Image you want to use.

    Parameters
    ----------
    apptainer_image ::class:`str`
        Set this to a apptainer image to have the jobs execute in it.
        Can be a link to a local file or to some online
        repository.
    mounts : :class:`tuple`, :class:`list`, optional
        What to mount in the container.
    mount_slurm_folders : :class:`bool`
        If `true`, mount `/etc/slurm` and `/var/run/munge` so you can
        issue slurm commands from within the job.
    apptainer_args : :class:`str`, optional
        Additional arguments to supply to apptainer
    """

    _slurm_folders = ('/etc/slurm', '/var/run/munge')

    def __init__(self, *args, apptainer_image=None,
                 mounts=('/mnt', ),
                 mount_slurm_folders=True,
                 apptainer_args='', **kwargs):
        super().__init__(*args, **kwargs)
        if apptainer_image is None:
            raise RuntimeError('You must specify a Apptainer Image')

        self._apptainer_image = apptainer_image
        self._mounts = list(mounts)
        self._apptainer_args = apptainer_args

        if mount_slurm_folders:
            for mnt in self._slurm_folders:
                if mnt not in self._mounts:
                    self._mounts.append(mnt)

    @property
    def apptainer_command(self):
        return '/usr/bin/singularity'

    @property
    def apptainer_arguments(self):
        cmd_list = ['exec']
        for mnt in self._mounts:
            cmd_list.append(f'-B {mnt}')

        if self._apptainer_args:
            cmd_list.append(self._apptainer_args.strip())

        return ' '.join(cmd_list)

    @property
    def executable(self):
        return self.apptainer_command

    @property
    def arguments(self):
        return f'{self.apptainer_arguments} {self._apptainer_image} {self.python_bin} {self.runner_filename}'  # noqa


class PermuteArgument(object):
    """
    This is a container for to-be-permuted arguments.
    See the example in the introductions for details.
    """
    def __init__(self, args):
        self.args = args

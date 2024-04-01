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
import gzip
import importlib
import inspect
import json
import sys
from abc import ABC

import six
from copy import deepcopy
import hashlib
from pathlib import Path


class JobBase(object):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    @property
    def args(self):
        return deepcopy(self._args)

    @property
    def kwargs(self):
        return deepcopy(self._kwargs)

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def shall_run(self, *args, **kwargs):
        return True

    def run_private(self):
        return self.run(*self.args, **self.kwargs)

    def shall_run_private(self):
        return self.shall_run(*self.args, **self.kwargs)

    def make_hash_from_args(self):
        args_dict = {
            'args': self.args,
            'kwargs': self.kwargs
        }

        args_json = json.dumps(args_dict, sort_keys=True)

        hash = hashlib.blake2b(args_json.encode(), digest_size=5).hexdigest()
        return hash


class Job(JobBase):
    """
    Abstract class for Jobs. This means, in order to define you own jobs, they
    need to be a subclass of this one.

    You **must** implement (i.e. define in your subclass) the :func:`run`
    method. The run method can take as many
    arguments as you like. Only the types of arguments are restricted because
    they need to be saved to disk. In general,
    strings, numbers, lists and dictionaries are fine.

    You **can** implement :func:`shall_run`. This can be used to see whether
    some output file already exists and restrict
    job submission to missing files.
    """

    def run(self, *args, **kwargs):
        """
        Implement this method to do the job.
        """
        raise NotImplementedError

    def shall_run(self, *args, **kwargs):
        """
        This is an optional method. It gets called with the same arguments as
        the :func:`run` method, **before** the job
        is submitted. If it returns True, the job is submitted,
        if it returns False, it is not.
        """
        return True


class JobItem(object):
    """
    Internal class for items in the job queue
    """

    def __init__(self, job_class_or_file, *args, **kwargs):
        if isinstance(job_class_or_file, (six.string_types, Path)):
            self._init_from_json(job_class_or_file)
        elif Job in inspect.getmro(job_class_or_file):
            self._init_from_class(job_class_or_file, *args, **kwargs)
        else:
            raise TypeError('JobItem needs either a filename or a job class')

    def _init_from_class(self, job_class, *args, **kwargs):
        self.job_module = inspect.getmodule(job_class).__name__
        if self.job_module == '__main__':
            self.job_module = Path(sys.argv[0]).name.stem

        self.job_class = job_class.__name__

        self.args = args
        self.kwargs = kwargs

    def _init_from_json(self, f_name):
        with gzip.open(f_name, 'rt') as gzip_file:
            raw_dict = json.load(gzip_file)
            self.job_class = raw_dict['job_class']
            self.job_module = raw_dict['job_module']
            self.args = raw_dict['args']
            self.kwargs = raw_dict['kwargs']

    def make_object(self):
        mod = importlib.import_module(self.job_module)
        this_class = getattr(mod, self.job_class)

        return this_class(*self.args, **self.kwargs)

    def to_json(self, f_name):
        with gzip.open(f_name, 'wt') as gzip_file:
            json.dump({
                'job_class': self.job_class,
                'job_module': self.job_module,
                'args': self.args,
                'kwargs': self.kwargs
            }, gzip_file)

    def __str__(self):
        return '.'.join((self.job_module, self.job_class))


class AutomaticFilenameJob(Job, ABC):
    """
    Abstract class for Jobs providing automatic filename generation.

    In order for this to work, you need to:

    1. Set :attr:`base_data_folder` and :attr:`job_data_folder` as a
        class attribute.
    2. If you use :meth:`shall_run`, you need to do the super call.

    This class then automatically creates the filename for each job using all
    the keyword arguments supplied.

    Please take a look at :doc:`autofilename` for detailed examples.

    Attributes
    ----------
    base_data_folder : str or pathlib.Path
        The base folder for the data.
        Is normally set once for all jobs of a project.

    job_data_folder : str or pathlib.Path
        The folder where the data for this job should be saved.

    exclude_kwargs_from_filename : list
        Normally, all keyword arguments are used to build the filename. if you
        want to exclude some of them, put the key in the list here.

    include_hash_in_fname : bool
        Include a hash of all arguments in the filename. This is helpful if
        you excluded some keyword arguments from filename creation but still
        need to get distinct filename.

    run_only_when_not_existing : bool
        If true, this job will only run if the file does not already exist.

    create_folder : bool
        If true, calling folders are created automatically

    data_file_suffix : str, optional
        The extension of the file. Defaults to `.dat`


    """
    base_data_folder = ''
    job_data_folder = ''
    exclude_kwargs_from_filename = []
    include_hash_in_fname = False
    run_only_when_not_existing = True
    create_folders = True
    data_file_suffix = '.dat'

    def __init__(self, *args, **kwargs):
        if 'subject_id' not in kwargs:
            self.subject_id = 'dummy'
        else:
            self.subject_id = kwargs['subject_id']

        super().__init__(*args, **kwargs)

    @classmethod
    def get_full_data_folder(cls):
        """
        Return the data folder for this job (i.e. :attr:`base_data_folder`
        plus :attr:`job_data_folder`).
        """
        if not cls.base_data_folder:
            raise ValueError('base_data_folder must be set')

        if not cls.job_data_folder:
            raise ValueError('job_data_folder must be set')

        folder = Path(cls.base_data_folder, cls.job_data_folder)

        if cls.create_folders:
            folder.mkdir(exist_ok=True, parents=True)

        return folder

    @property
    def output_folder(self):
        """
        pathlib.Path: The output folder for this subject.
        """

        folder = Path(self.get_full_data_folder(), self.subject_id)

        if self.create_folders:
            folder.mkdir(exist_ok=True, parents=True)

        return folder

    @property
    def output_filename(self):
        """
        str: The filename for this subject.
        """
        f_name_list = [self.subject_id]

        for key, val in self._kwargs.items():
            if key == 'subject_id' or key in self.exclude_kwargs_from_filename:
                continue

            f_name_list.append('%s_%s' % (key, str(val)))

        if self.include_hash_in_fname:
            f_name_list.append(self.make_hash_from_args())

        f_name = '__'.join(f_name_list) + self.data_file_suffix

        return f_name

    @property
    def full_output_path(self):
        """
        pathlib.Path: The full path to the output file.
        """
        return Path(self.output_folder, self.output_filename)

    def shall_run(self, *args, **kwargs):
        this_shall_run = True
        if self.run_only_when_not_existing:
            this_shall_run = not self.full_output_path.exists()

        return this_shall_run and super().shall_run(*args, **kwargs)

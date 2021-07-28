#!/usr/bin/env python3

# Some utilities to handle tiled_ndmaps. Assumes tiles are along the zeroth-axis, and only supports basic
# communication to send groups of tiles to various processes (parallelizing for loops over tiles).

# Heavily inspired by the MPIBase class here: https://github.com/AdriJD/beamconv/blob/master/beamconv/instrument.py
# as well as the functions in pixell.utils (https://github.com/simonsobs/pixell/blob/master/pixell/utils.py) and
# orphics.mpi (https://github.com/msyriac/orphics/blob/master/orphics/mpi.py)

from pixell import enmap, utils
from mnms import utils as tnu
from mnms.tiled_ndmap import tiled_ndmap

import numpy as np
import warnings
import os

import time

try:
    from mpi4py import MPI
except ImportError:
    warnings.warn('Failed to import mpi4py, continuing without MPI', RuntimeWarning)

MPI_MAX_MSG_LENGTH = int(1e9)

# adapted from orphics.mpi.mpi_distribute but doesn't return full list of tasks,
# just the "start" indices
def mpi_distribute(num_tasks, num_cores, allow_empty=True):
    if not allow_empty:
        assert num_cores <= num_tasks
    div, mod = divmod(num_tasks, num_cores)
    counts = np.full(num_cores, div)
    counts[counts.size-mod:] += 1
    displs = np.cumsum(np.append([0,], counts)) # len(displs) = num_cores+1
    return tuple(displs)

def mpi_distribute_array_shapes(arr, num_cores, allow_empty=True):
    num_tasks = len(arr)
    task_displs = mpi_distribute(num_tasks, num_cores, allow_empty=allow_empty)
    task_slices = tuple(slice(task_displs[i], task_displs[i+1]) for i in range(len(task_displs)-1)) # len(task_displs) = num_cores+1
    shapes = (arr[s].shape for s in task_slices)
    return tuple(shapes)

def mpi_distribute_buffer(arr, num_cores, allow_empty=True, max_msg_length=MPI_MAX_MSG_LENGTH, msg_num=0):
    num_tasks = len(arr)
    task_displs = mpi_distribute(num_tasks, num_cores, allow_empty=allow_empty)
    task_slices = tuple(slice(task_displs[i], task_displs[i+1]) for i in range(len(task_displs)-1)) # len(task_displs) = num_cores+1
    unchunked_sendbuf_counts = np.array([arr[s].size for s in task_slices])
    unchunked_sendbuf_displs = np.cumsum(np.append([0,], unchunked_sendbuf_counts))
    
    msg_start = msg_num*max_msg_length
    msg_stop = (msg_num+1)*max_msg_length
    chunked_recbuf_slices = []
    chunked_sendbuf_counts =[]
    for i in range(num_cores):
        slice_start = np.clip(msg_start - unchunked_sendbuf_displs[i], 0, unchunked_sendbuf_counts[i])
        slice_stop = np.clip(msg_stop - unchunked_sendbuf_displs[i], 0, unchunked_sendbuf_counts[i])
        chunked_recbuf_slices.append(slice(slice_start, slice_stop))
        chunked_sendbuf_counts.append(slice_stop - slice_start)

    chunked_sendbuf_displs = np.cumsum(np.append([0,], chunked_sendbuf_counts))[:-1] # len(chunked_sendbuf_displs) = num_cores+1
    return tuple(chunked_sendbuf_counts), tuple(chunked_sendbuf_displs), tuple(chunked_recbuf_slices)

def get_MPI_datatype(dtype):
    char_dtype = np.dtype(dtype).char
    return MPI._typedict[char_dtype]

class MPIManager:
    """Every MPI process will get a MPIManager instance which helps wrap the low-level calls to MPI
    """

    def __init__(self, mpi=True, comm=None, root=0, max_msg_length=MPI_MAX_MSG_LENGTH, **kwargs):
        if mpi:
            try:
                from mpi4py import MPI
                self.mpi = True
            except ImportError:
                warnings.warn('Failed to import mpi4py, continuing without MPI', RuntimeWarning)
                self.mpi = False
            
            if self.mpi:
                if comm is None:
                    self.comm = MPI.COMM_WORLD
                else:
                    self.comm = comm

                self.size = self.comm.Get_size()
                self.rank = self.comm.Get_rank()
                self.root = root
                self.is_root = self.rank == root
                self.max_msg_length = max_msg_length

                if self.size == 1:
                    self.mpi = False
                    warnings.warn('Only one process, continuing without MPI', RuntimeWarning)
                elif self.size > 1:
                    self.mpi = True
                else:
                    raise ValueError('MPI Size is less than 1; please check your SLURM script setup')
        
        if not mpi or not self.mpi:
            assert root == 0, 'If MPI disabled, only one rank; root must be 0'
            self.mpi = False
            self.comm = 'No communicator; MPI Disabled'
            self.size = 1
            self.rank = root
            self.root = root
            self.is_root = True
            self.max_msg_length = max_msg_length

    def __repr__(self):
        s = f'MPI Enabled: {self.mpi}\n'
        s += f'Communicator: {self.comm}\n'
        s += f'Communicator Root: {self.root}\n'
        s += f'Communicator Size: {self.size}\n'
        s += f'Process Rank: {self.rank}\n'
        return s

    def __str__(self):
        return repr(self)

    def Scatterv(self, arr, max_msg_length=None):
        """Performs a Scatterv on the arr buffer along its first axis
        """
        if not self.mpi:
            return np.asarray(arr)
        else:
            if max_msg_length is None:
                max_msg_length = self.max_msg_length

            if self.is_root:
                arr = np.asarray(arr)
                shapes = mpi_distribute_array_shapes(arr, self.size)
                
                shape = self.comm.scatter(shapes, root=self.root)
                dtype, size = self.comm.bcast((arr.dtype, arr.size), root=self.root)
            else:
                shape = self.comm.scatter(None, root=self.root)
                dtype, size = self.comm.bcast(None, root=self.root)
                arr = np.empty(0, dtype=dtype)
            
            recbuf = np.empty(shape, dtype=dtype)
            datatype = get_MPI_datatype(dtype)

            num_msgs = np.ceil(size / max_msg_length).astype(int)
            for i in range(num_msgs):
                if self.is_root:
                    counts, displs, recbuf_slices = mpi_distribute_buffer(arr, self.size, max_msg_length=max_msg_length, msg_num=i)
                    counts, displs = self.comm.bcast((counts, displs), root=self.root)
                    recbuf_slice = self.comm.scatter(recbuf_slices, root=self.root)
                else:
                    counts, displs = self.comm.bcast(None, root=self.root)
                    recbuf_slice = self.comm.scatter(None, root=self.root)

                msg_start = i*max_msg_length
                msg_stop = (i+1)*max_msg_length
                self.comm.Scatterv([arr.reshape(-1)[msg_start:msg_stop], counts, displs, datatype], [recbuf.reshape(-1)[recbuf_slice], datatype], root=self.root)
            
            arr = None
            return recbuf

    def Gatherv(self, arr, max_msg_length=None):
        """Performs a Gatherv on the various arr buffers along their first axes (like append)
        """
        if not self.mpi:
            return np.asarray(arr)
        else:
            if max_msg_length is None:
                max_msg_length = self.max_msg_length

            arr = np.asarray(arr)
            size = np.sum(self.comm.allgather(arr.size), dtype=int)
            dtype = self.comm.allgather(arr.dtype)
            assert np.unique(dtype).size == 1 # make sure all ranks have same dtype
            dtype = dtype[0]
            datatype = get_MPI_datatype(dtype)

            if self.is_root:
                recbuf = np.empty(size, dtype=dtype).reshape((-1,) + arr.shape[1:])
            else:
                recbuf = np.empty(0, dtype=dtype)

            num_msgs = np.ceil(size / max_msg_length).astype(int)
            for i in range(num_msgs):
                if self.is_root:
                    counts, displs, arr_slices = mpi_distribute_buffer(recbuf, self.size, max_msg_length=max_msg_length, msg_num=i)
                    counts, displs = self.comm.bcast((counts, displs), root=self.root)
                    arr_slice = self.comm.scatter(arr_slices, root=self.root)
                else:
                    counts, displs = self.comm.bcast(None, root=self.root)
                    arr_slice = self.comm.scatter(None, root=self.root)
                
                msg_start = i*max_msg_length
                msg_stop = (i+1)*max_msg_length
                self.comm.Gatherv([arr.reshape(-1)[arr_slice], datatype], [recbuf.reshape(-1)[msg_start:msg_stop], counts, displs, datatype], root=self.root)

            arr = None
            return recbuf

    def bcast(self, items):
        if not self.mpi:
            return items
        else:
            if self.is_root:
                return self.comm.bcast(items, root=self.root)
            else:
                return self.comm.bcast(None, root=self.root)

    def Bcast(self, arr):
        if not self.mpi:
            return np.asarray(arr)
        else:
            if self.is_root:
                arr = np.asarray(arr)
                shape, dtype = self.comm.bcast((arr.shape, arr.dtype), root=self.root)
            else:
                shape, dtype = self.comm.bcast(None, root=self.root)
                arr = np.empty(shape, dtype=dtype)

            datatype = get_MPI_datatype(dtype)
            self.comm.Bcast([arr, datatype], root=self.root)
            return arr


class TiledMPIManager(MPIManager):
    """Every MPI process will get a TiledMPIManager instance which helps wrap the low-level calls to MPI, including tiled_ndmap constructor args
    """

    def __init__(self, mpi=True, comm=None, root=0, max_msg_length=MPI_MAX_MSG_LENGTH, **kwargs):
        super().__init__(mpi=mpi, comm=comm, root=root, max_msg_length=max_msg_length, **kwargs)

    def Scatterv_tiled_ndmap(self, tiled_imap, max_msg_length=None):
        if not self.mpi:
            return tiled_imap
        else:
            if max_msg_length is None:
                max_msg_length = self.max_msg_length

            if self.is_root:
                wcs = self.comm.bcast(tiled_imap.wcs)
                tiled_ndmap_kwargs = self.comm.bcast(tiled_imap.tiled_info())
                if tiled_ndmap_kwargs['tiled']:
                    unmasked_tiles = self.Scatterv(tiled_imap.unmasked_tiles, max_msg_length=max_msg_length)
                    tiled_ndmap_kwargs.update({'unmasked_tiles': unmasked_tiles})
            else:
                wcs = self.comm.bcast(None)
                tiled_ndmap_kwargs = self.comm.bcast(None)
                if tiled_ndmap_kwargs['tiled']:
                    unmasked_tiles = self.Scatterv(None, max_msg_length=max_msg_length)
                    tiled_ndmap_kwargs.update({'unmasked_tiles': unmasked_tiles})
                tiled_imap = None

            tiled_arr = self.Scatterv(tiled_imap, max_msg_length=max_msg_length)
            return tiled_ndmap(enmap.ndmap(tiled_arr, wcs), **tiled_ndmap_kwargs)

    def Gatherv_tiled_ndmap(self, tiled_imap, max_msg_length=None):
        if not self.mpi:
            return tiled_imap
        else:
            if max_msg_length is None:
                max_msg_length = self.max_msg_length

            if tiled_imap.tiled:
                unmasked_tiles = self.Gatherv(tiled_imap.unmasked_tiles, max_msg_length=max_msg_length)
            else:
                unmasked_tiles = tiled_imap.unmasked_tiles

            tiled_arr = self.Gatherv(tiled_imap, max_msg_length=max_msg_length)
            return tiled_imap.sametiles(tiled_arr, unmasked_tiles=unmasked_tiles)



from typing import Dict, List

import jax
import mpi4jax
import numpy as np
from mpi4py import MPI


def get_metric_stats(comm, _key: str, _val: List, metric_dict: Dict) -> Dict:
    metric_dict[_key + "_mean"] = comm.allreduce(np.mean(_val), op=MPI.SUM)
    metric_dict[_key + "_std"] = comm.allreduce(np.std(_val), op=MPI.SUM)
    metric_dict[_key + "_max"] = comm.allreduce(np.max(_val), op=MPI.SUM)
    metric_dict[_key + "_min"] = comm.allreduce(np.min(_val), op=MPI.SUM)
    return metric_dict


def tree_all_reduce(tree):
    """Applies an allreduce to a PyTree with mpi4jax"""
    comm = MPI.COMM_WORLD
    token = jax.lax.create_token()

    def reduce_leaf_func(leaf):
        nonlocal token
        res, token = mpi4jax.allreduce(leaf, token=token, op=MPI.SUM, comm=comm)
        return res / comm.Get_size()

    return jax.tree_map(reduce_leaf_func, tree)


def tree_bcast(tree):
    """Broadcasts a PyTree with mpi4jax"""
    comm = MPI.COMM_WORLD
    token = jax.lax.create_token()

    def reduce_leaf_func(leaf):
        nonlocal token
        res, token = mpi4jax.bcast(leaf, root=0, token=token, comm=comm)
        return res

    return jax.tree_map(reduce_leaf_func, tree)

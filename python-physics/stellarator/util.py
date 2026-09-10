import jax
import jax.numpy as jnp
import equinox as eqx
from desc.backend import tree_unstack


# None of the other options for chunked vmap work (too slow, need chunks to be too small) so we literally split the arguments and do a for loop
def _wrap_vmap_maybe_chunk(func, vmap_axes, chunk_size, nchunks, sharding):
    def wrapper(*args):

        def _wrap_shard(*args):
            if sharding is not None:
                args_s = tuple(
                    eqx.filter_shard(arg, sharding) if ax is not None else arg
                    for arg, ax in zip(args, vmap_axes)
                )
                return args_s
            else:
                return args

        if chunk_size > 0:
            if len(args) != len(vmap_axes):
                raise RuntimeError(
                    f"args and vmap_axes must be the same length; got len(args)={len(args)}, len(vmap_axes)={len(vmap_axes)}"
                )

            # split by putting the axis we want to split over in the 0th position then unstacking the tree
            def split_leaf(arg, ax):
                if ax is not None:
                    chunked_leaf = jax.tree.map(
                        lambda x: x.reshape((nchunks, chunk_size) + x.shape[1:]),
                        arg,
                    )
                    return tree_unstack(chunked_leaf)
                else:
                    return arg

            split_args = []
            for arg, ax in zip(args, vmap_axes):
                split_args.append(split_leaf(arg, ax))

            # create argument list
            arg_tuples = []
            for i in range(0, nchunks):
                arg_tuples.append(
                    tuple(
                        arg[i] if ax is not None else arg
                        for arg, ax in zip(split_args, vmap_axes)
                    )
                )

            outputs = []
            # call vmap with sharded inputs
            for arg in arg_tuples:
                outputs.append(
                    eqx.filter_jit(eqx.filter_vmap(func, in_axes=vmap_axes))(
                        *_wrap_shard(*arg)
                    )
                )
            return jax.tree.map(
                lambda *chunks: jnp.concatenate(chunks, axis=0), *outputs
            )
        else:
            return eqx.filter_jit(eqx.filter_vmap(func, in_axes=vmap_axes))(
                *_wrap_shard(*args)
            )

    return wrapper


def VmapWrapper(func, vmap_axes, chunk_size=0, nchunks=0, sharding=None):
    """
    Wrapper for vmap that allows for splitting the input pytrees into chunks

    Parameters
    ----------

    func: callable
        function to be mapped over
    vmap_axes: tuple
        tuple of axes to vmap over
    chunk_size: int
        size of chunks to split arrays into
    nchunks: int
        number of chunks to split arrays into
    sharding: jax.sharding
        sharding to apply to inputs

    Returns
    -------

    VmapWrapper: callable
        callable vmap wrapper
    """
    return _wrap_vmap_maybe_chunk(
        func, vmap_axes, chunk_size=chunk_size, nchunks=nchunks, sharding=sharding
    )

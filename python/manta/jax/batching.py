import jax
import jax.numpy as jnp
import equinox as eqx
import jax.tree_util as jtu


@jax.jit
def tree_unstack(tree):
    # taken from the DESC project, reproduced here to avoid adding it as a dependency
    # see: https://github.com/PlasmaControl/DESC
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.

    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.
    """
    # from https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75

    leaves, treedef = jtu.tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves)]


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
                    dynamic, static = eqx.partition(arg, eqx.is_array)
                    chunked_leaves = tree_unstack(
                        jax.tree.map(
                            lambda x: x.reshape((nchunks, chunk_size) + x.shape[1:]),
                            dynamic,
                        )
                    )
                    return [eqx.combine(chunk, static) for chunk in chunked_leaves]

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

            # reassemble arrays
            return jax.tree.map(
                lambda *chunks: jnp.concatenate(chunks, axis=0), *outputs
            )
        else:
            return eqx.filter_jit(eqx.filter_vmap(func, in_axes=vmap_axes))(
                *_wrap_shard(*args)
            )

    return wrapper


def vmap_batched(func, vmap_axes, chunk_size=0, nchunks=0, sharding=None):
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

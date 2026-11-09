import jax
import jax.numpy as jnp
import equinox as eqx
import jax.tree_util as jtu
from jax.sharding import NamedSharding, PartitionSpec

from .multihost import gather_to_host


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


def _leaf_specs(args, vmap_axes, axis):
    """One PartitionSpec per leaf of `args`, in `jax.tree.leaves` order:
    split over `axis` where `vmap_axes` maps the leaf, replicated where it
    does not. `vmap_axes` is a prefix of `args`, as `filter_vmap` takes it."""
    specs = []
    for arg, ax in zip(args, vmap_axes):

        def visit(a, sub):
            if a is not None and a != 0:
                raise ValueError(
                    f"use_shard_map splits the leading axis only; got in_axes {a}"
                )
            spec = PartitionSpec() if a is None else PartitionSpec(axis)
            specs.extend([spec] * len(jax.tree.leaves(sub)))

        jax.tree.map(visit, ax, arg, is_leaf=lambda x: x is None)
    return specs


def _place(args, vmap_axes, mesh, axis):
    """`args` with every array leaf on the mesh: mapped leaves split, the rest
    replicated -- what `_shard_mapped_vmap`'s in_specs say they are."""
    leaves, treedef = jax.tree.flatten(args)
    specs = _leaf_specs(args, vmap_axes, axis)
    return jax.tree.unflatten(
        treedef,
        [
            jax.device_put(leaf, NamedSharding(mesh, spec))
            if eqx.is_array(leaf)
            else leaf
            for leaf, spec in zip(leaves, specs)
        ],
    )


@eqx.filter_jit
def _shard_mapped_vmap(func, vmap_axes, mesh, axis, args):
    """`filter_vmap(func)(*args)`, with each device vmapping its own slice.

    Under a plain jit of a vmap over a sharded batch, a `while_loop` in `func`
    -- an iterative solve -- runs until every point in the *whole* batch has
    converged, and XLA decides that with an all-reduce on every iteration:
    across nodes, a network round trip per Krylov or multigrid sweep. Inside
    shard_map each device runs its own loop on its own points and stops when
    they have converged, so the program has no collective until the results
    are gathered. `func` goes in as an argument, so a module holding arrays
    has them traced (replicated) rather than baked in as constants.
    """
    full = (func,) + tuple(args)
    axes = (None,) + tuple(vmap_axes)
    leaves, treedef = jax.tree.flatten(full)
    specs = _leaf_specs(full, axes, axis)
    mask = [eqx.is_array(leaf) for leaf in leaves]
    dynamic = tuple(leaf for leaf, m in zip(leaves, mask) if m)
    in_specs = tuple(spec for spec, m in zip(specs, mask) if m)

    def body(*local):
        it = iter(local)
        f, *a = jax.tree.unflatten(
            treedef, [next(it) if m else leaf for leaf, m in zip(leaves, mask)]
        )
        return eqx.filter_vmap(f, in_axes=vmap_axes)(*a)

    return jax.shard_map(
        body,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=PartitionSpec(axis),
        check_vma=False,
    )(*dynamic)


# None of the other options for chunked vmap work (too slow, need chunks to be too small) so we literally split the arguments and do a for loop
def _wrap_vmap_maybe_chunk(
    func, vmap_axes, chunk_size, nchunks, sharding, has_carry, use_shard_map=False
):
    if use_shard_map:
        if not isinstance(sharding, NamedSharding) or len(sharding.spec) != 1:
            raise ValueError(
                "use_shard_map needs a NamedSharding splitting the leading axis "
                "over one mesh axis, e.g. NamedSharding(mesh, P('axis'))"
            )
        mesh, axis = sharding.mesh, sharding.spec[0]

    def _wrap_shard(*args):
        if sharding is not None:
            args_s = tuple(
                eqx.filter_shard(arg, sharding) if ax is not None else arg
                for arg, ax in zip(args, vmap_axes)
            )
            return args_s
        else:
            return args

    def call(*args):
        if use_shard_map:
            return _shard_mapped_vmap(
                func, vmap_axes, mesh, axis, _place(args, vmap_axes, mesh, axis)
            )
        return eqx.filter_jit(eqx.filter_vmap(func, in_axes=vmap_axes))(
            *_wrap_shard(*args)
        )

    def wrapper(*args):
        if nchunks > 1:
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
                outputs.append(call(*arg))

            # reassemble arrays
            out = jax.tree.map(
                lambda *chunks: jnp.concatenate(chunks, axis=0), *outputs
            )
        else:
            out = call(*args)
        # whole and local on every process when the mesh spans several, so the
        # caller can unpad and hand it to MaNTA as in a single process
        if has_carry:
            # the carry goes back in on the next call, sharded as it came out:
            # gathering it would move every point's copy to every process
            out, carry = out
            return gather_to_host(out), carry
        return gather_to_host(out)

    return wrapper


def vmap_batched(
    func,
    vmap_axes,
    chunk_size=0,
    nchunks=0,
    sharding=None,
    has_carry=False,
    use_shard_map=False,
):
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
    has_carry: bool
        `func` returns `(out, carry)`, carry being per point state the caller
        hands back in on a later call (a solver's warm start, say). The wrapper
        then returns `(out, carry)` too, with `out` gathered as usual and
        `carry` left as the batched calls produced it: padded, and sharded
        like the inputs, so it can be passed straight back.
    use_shard_map: bool
        run each round under `jax.shard_map` over `sharding`'s mesh, so each
        device vmaps its own slice of the batch with no collective inside
        `func`. Without it the round is one program over the sharded batch,
        and a `while_loop` in `func` synchronises every device on each
        iteration (see `_shard_mapped_vmap`). Needs `sharding` to be a
        NamedSharding over one mesh axis, and maps the leading axis only.

    Returns
    -------

    VmapWrapper: callable
        callable vmap wrapper
    """
    return _wrap_vmap_maybe_chunk(
        func,
        vmap_axes,
        chunk_size=chunk_size,
        nchunks=nchunks,
        sharding=sharding,
        has_carry=has_carry,
        use_shard_map=use_shard_map,
    )

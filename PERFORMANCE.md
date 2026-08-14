# Performance criteria

MaNTA is designed to be used with expensive flux and source functions.

MaNTA's performance should be determined by the number of calls to the flux/source
functions required to achieve a given accuracy.

MaNTA should be tuned to work at low-to-moderate (4-10 cells, polynomial order 2-5)
spatial resolution. Tradeoffs that increase the accuracy at high resolution but
increase the number of calls into a `TransportSystem` at low spatial resolution
(4 cells, polynomial order 2-3) should be viewed skeptically and generally exist
only as options, not defaults.

MaNTA's performance should be compared to the algorithms described in
[`refs/`](refs/Refs.md). The three benchmarks under
[`python-examples/`](python-examples/) do that, each reporting evaluations of the
physics per point.

Where only the steady state is wanted, `SteadyStateSolver` chooses how it is
reached, and the choice is worth an order of magnitude. Measured, for answers
identical in every digit:

| benchmark | `TimeMarch` | `PseudoTransient` | `Newton` |
|---|---|---|---|
| `park-convergence` | 113 | **19** | **11** |
| `jardin-critical-gradient` | 212 | **127** | 152 |
| `shestakov-nonlinear` | **283** | 705 | 731 |

`TimeMarch` sizes every step from a local error estimate on a transient that is
then discarded, which is where the factor goes; Park's own solver reaches the
same state in 9-15 iterations. The third row is the counter-example -- a
degenerate flux for which the pseudo-time term was doing useful damping -- and is
why `TimeMarch` remains available and remains that example's setting. See
`docs/running.rst`.

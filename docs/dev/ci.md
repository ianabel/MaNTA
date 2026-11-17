# CI and branch protection

The rule that gates merges into `main`, and how it is coupled to `ci.yml`.

**`main` is protected: work goes on a branch and merges through a pull request.**
Read the live rule rather than trusting this paragraph —
`gh api repos/ianabel/MaNTA/branches/main/protection` — but as it stands:

* **No approving review is required** (`required_approving_review_count: 0`), so
  a PR can be merged by its own author as soon as the checks are green. The PR
  is a gate for CI, not for review.
* **`strict: true`**, so a branch has to be up to date with `main` before it can
  merge. If `main` moves while a PR is open, rebase or merge it in and let CI
  run again.
* **Force-pushes and deletions of `main` are refused**, and there is no linear
  history requirement, so an ordinary merge commit is fine.
* **`enforce_admins` is off**, so the repo owner can still push straight to
  `main`. "Protected" therefore means something different depending on who you
  are, and a workflow that works for `ianabel` is not evidence it works for
  anyone else.

**Thirteen of the fifteen contexts `ci.yml` publishes are required**, each pinned to app
15368 (GitHub Actions), so a status of that name from anything else does not
count:

```
Build + tests (g++-14)                    Build + tests (clang++-19)
Build + tests (g++-15)                    Build + tests (clang++-20)
Build + tests (g++-16)                    Build + tests (clang++-21)
Build + tests (g++-15, Eigen 5.0.1)       Build + tests (fedora:latest)
Build + tests (clang++-19, Eigen 5.0.1)   Build + tests (fedora:latest, OpenMP)
Build + tests (g++-15, OpenMP)            Docs
Out of tree
```

**`Build + tests (almalinux:10, gcc 14)` is not required, and that is a choice
about its failure mode rather than its value.** It installs its dependencies from
the AlmaLinux and EPEL mirrors on every run, and a mirror serving a corrupt
`repomd.xml` fails it with nothing wrong in the change. What it alone covers is
RHEL's packaging: EPEL's Eigen and netCDF, CRB's BLAS, `/usr/lib64`. gcc 14
itself is gated by `Build + tests (g++-14)`. So a red AlmaLinux leg beside a
green g++-14 one is either a mirror or a packaging problem, and the log says
which within its first minute.

**`Coverage` is the other context that is deliberately *not* required**, and it
publishes without gating anything for a different reason. It runs `make coverage`'s
successor — all three suites under an instrumented build — so what it would gate
is what the required build legs already gate, at the cost of the slowest leg's wall-clock
on every PR. It has no percentage threshold and never did, so it was never
measuring coverage in the sense of holding a number: it was a duplicate build.
Removed from the rule 2026-08-26. It still runs, and a red `Coverage` beside green
build legs is worth reading — an instrumented `-O0` build failing where the
optimised ones pass is a real signal — but it does not block a merge.

**A new leg's context can only be required once the leg is on `main`**, and that
ordering is forced rather than fussy. A required context is matched by *name*
against what a PR's own workflow publishes, so a PR branched from a `main` whose
`ci.yml` lacks the leg cannot publish it: require it early and every unrelated PR
sits at "Expected — waiting for status to be reported" indefinitely, while the
green ticks beside it say the build is fine. `Build + tests (g++-15, OpenMP)` was
required early once, in the change that added the leg, and had to be reverted for
exactly that reason; it was added for real once that branch merged.

So the sequence is: merge the leg, then add the context, then check the two agree
with the `diff` below rather than assuming. The g++-16 leg builds
`MANTA_LAPACK=OFF` but is *not* renamed by it, deliberately: it carries no
`label`, so it goes on publishing
`Build + tests (g++-16)` and the required list did not have to move for it. A
matrix key that changes behaviour without changing the rendered name is the
cheap way to add coverage here.

**Those strings are the job's *rendered* name, and that couples the rule to the
matrix.** The job is `name: Build + tests (${{ matrix.label || matrix.cxx }})`,
so adding, removing or relabelling a leg renames its context — and a required
context that nothing publishes is not an error anywhere. GitHub matches exactly;
the check simply never arrives, the PR sits at "Expected — waiting for status to
be reported", and the green ticks beside it are the legs that *are* reporting.
That is not hypothetical: the rule required `Build + C++ tests` — a name no job
has ever published, differing both in wording and in carrying no matrix suffix —
so until it was corrected the protection blocked every non-admin and gated
nothing CI would have caught.

**So whenever a leg is added, dropped or relabelled, update the required list in
the same change**, and check the two agree afterwards rather than assuming:

```sh
gh api repos/ianabel/MaNTA/branches/main/protection/required_status_checks -q '.contexts[]' | sort > /tmp/req
gh pr view <N> --json statusCheckRollup -q '.statusCheckRollup[].name' | sort > /tmp/got
diff /tmp/req /tmp/got     # left-only = required but impossible; right-only = ungated
```

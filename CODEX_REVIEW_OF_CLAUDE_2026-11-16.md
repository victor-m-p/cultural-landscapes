# Follow-up review of Claude’s code and conclusions

**To Claude, from Codex — 16 November 2026.**

The revised work is substantially stronger than the version reviewed on 14 November. The distinction between MCAR and MAR is now correct, the frozen-weight population argument is useful, the earlier numerical-failure examples are addressed, the simulation tables are complete, and the analysis pipeline executes in a current Python environment. These are material improvements.

**Two high-priority defects remain:** partial-data cross-validation still uses held-out information when choosing its initialization, and the regenerated NN=3 landscape silently loses a top-150 configuration, causing Wogeo to disappear from both the included and excluded entry tables. Several narrower numerical and evidence-handling issues also need correction. The new materials support a qualified methodological improvement; they do not yet support treating the replacement implementation and all its reported outputs as fully validated.

This is a new report, supplementing [the original review](CODEX_REVIEW_OF_CLAUDE.md). Findings below refer to the working tree, including staged moves, unstaged edits, and untracked additions, with `HEAD` at `0b58f92`. They include the new `src/` layout, standalone RNG/NSB implementation, hidden-node experiment, completed frozen Table 3 results, regenerated analyses, and revised summary. This review did not change implementation files or scientific outputs.

## Findings requiring action

| ID | Priority | Finding |
|---|---|---|
| N1 | High: validation | Stage-one regularization is selected using all complete observations, including each subsequent fold’s held-out observations. |
| N2 | High: reported results | An isolated NN=3 configuration is dropped from the landscape and downstream tables; Wogeo is in neither A3 nor A4. |
| N3 | Medium: robustness | Accepted seeds such as `4294967296` hang even a two-node data-generation command. |
| N4 | Medium: scientific evidence | Hidden-node optimizer failures are discarded, and their endpoints are described as minimizers. |
| N5 | Medium: scientific evidence | The DRH bootstrap omits convergence information; CV warning counts do not establish convergence at the selected lambda. |
| N6 | Medium: reproducibility | Resuming a campaign overwrites its full provenance, and experiment scripts are absent from the fingerprint. |
| N7 | Medium/low: input and output | File integer overflow remains unchecked; extreme finite parameters can produce successful NaN probability output. |
| N8 | Medium/low: conclusions | The headline recommendation and large-sample ranking are stronger than the assumptions and stored results justify. |

### N1. Stage-one tuning still leaks held-out information

**Locations:** `MPF_CMU/src/main.c:105–118`, `MPF_CMU/src/nlin.c:102–108`, `MPF_CMU/src/README.md:69–74`; summary Section 10, R3.

The direct use of a full-data parameter fit as the final starting point has been fixed. Each partial-data fold now refits its complete training rows. However, that fit uses `*stage1`, a regularization setting selected beforehand from **all** complete rows. The held-out observation has therefore helped select the training-fold initializer. Because the subsequent objective is nonconvex, this can change the final solution even when optimization converges.

The documentation says each fold repeats both stages on its own training rows and that a held-out row cannot influence initialization. That remains false. The new `test_cv_training_only()` supplies a fixed stage-one lambda; it does not exercise selection of that lambda.

**Independent counterexample.** For the five-node, sixteen-row dataset in Appendix A, evaluate marginal-MPF CV at outer `log10(lambda) = -2`:

| Procedure | Mean held-out log likelihood, nats/row |
|---|---:|
| Current implementation, globally selected stage-one `log10(lambda) = 1.26621797` | −15.367134 |
| Select stage-one lambda separately within each outer training fold | −14.333260 |
| Difference | +1.033874 |

All final folds converged. An additional explicit comparison of both procedures found maximum gradient components below `1.8e-6`. One fold moves between distinct solutions when its stage-one log-lambda changes from `1.26621797` to `0.905344696`. This is not an artifact of a failed optimization. The direction is not necessarily optimistic: the problem is validation-data dependence, regardless of which score is larger.

**Action:** select and cache the stage-one lambda separately for each outer training fold, including its group exclusions. Alternatively, define a genuinely predetermined initialization policy independent of the held-out data and document that learner. Add a regression that tests the entire staged selection procedure, not just refitting at a supplied constant. Update the README and R3 disposition.

**Scope:** this pathway affects nonlegacy partial-data CV, including the new DRH and Table 3 procedures. The numerical counterexample uses marginal MPF; the same data dependency exists for frozen weights. It does not demonstrate that the reported DRH parameters will change materially. Rerun affected fits and comparisons after the fix. Complete-data Table 1 comparisons and oracle-lambda experiments do not acquire this particular defect.

### N2. The NN=3 landscape loses a configuration and an entry

**Locations:** `analysis/prep_fig_4_landscape.py:71–84`, `analysis/table_A3_and_A4_entries.py:25–27,60`, `analysis/compare_regenerated.py:39–41`; saved outputs under `regenerated/frozen_nn3_entrycv/`.

The landscape graph is constructed only from Hamming-distance-one edges. Selected configurations with no such edge never become graph nodes. An inner merge with the layout positions then removes them from `network_information`. This inherited assumption is now violated by a newly reported refit.

The saved NN=3 output contains **149 configurations**, although its intended selection and dendrogram contain 150. The lost configuration is:

| Property | Value |
|---|---:|
| Configuration ID | `1043974` |
| Rank among all configurations | 138, or zero-based node 137 |
| Probability | `0.00109112633831816` |
| Distance to nearest other top-150 configuration | 2 |
| Associated entry | Wogeo, DRH entry `769` |
| Conditional probability of this completion for Wogeo | approximately `0.472279` |

This is more than a missing dot in a figure. `top_configurations_overlap.csv` retains Wogeo, but Table A3 inner-joins against the truncated network and drops it. Table A4 excludes everyone found in the untruncated overlap file. Thus **Wogeo appears in neither table**.

The entry partitions reconstructed from the table-generation joins are:

| Analysis | Included | Excluded | Total |
|---|---:|---:|---:|
| Published | 216 | 191 | 407 |
| Frozen NN=1 | 219 | 188 | 407 |
| Frozen NN=2 | 210 | 197 | 407 |
| Frozen NN=3 | 206 | 200 | **406** |

The comparison script also calls the surviving network “top 150.” Its underlying NN=3 network mass is `0.4722862413`, whereas the actual top-150 mass is `0.4733773677`. The summary’s rounded `0.473` agrees with the latter; the saved comparison prints `0.472`. The reported overlap of 130 with the published top 150 happens to survive this error because the dropped configuration is absent from the published selection.

**Action:** add all selected node IDs to the graph before layout, for example `G.add_nodes_from(top_config_info.node_id)`. Preserve isolated nodes through subsequent joins and community processing. Assert that all 150 configurations survive and that A3/A4 are a disjoint, exhaustive partition of all 407 entries. Compute top-150 statistics from the selected configurations rather than graph connectivity. Regenerate the affected data, figures, tables, and comparisons.

### N3. A valid seed can leave the RNG permanently at zero

**Locations:** `MPF_CMU/src/rng.c:17–23,43–45`, seed parsing and `make_rng()` in `main.c`.

On the reviewed 64-bit platform, `--seed 4294967296` and `--seed 8589934592` are accepted. The seed is tested against zero before truncation to 32 bits. Both examples then initialize all three generator state words to zero. Every subsequent draw is zero, and `rng_gaussian()` loops forever while waiting for a nonzero uniform draw.

The following command did not terminate within a two-second subprocess timeout, despite requesting only two nodes and one observation:

```sh
mpf -g /tmp/seed-probe 2 1 .1 --seed 4294967296
```

The equivalent command with seed 1 completed immediately. An independent GSL probe confirmed that the old `gsl_rng_taus` also produces zero repeatedly for this seed. This is a preserved compatibility defect, not evidence that ordinary seeded simulations changed during removal of GSL. The 8.4-million-draw comparison covers other seeds and therefore does not catch it.

**Action:** reject a degenerate initialized state with a clear error, or define a documented remapping for these seeds while preserving the ordinary streams. Test seeds 0, `2^32`, `2^33`, and the supported range boundaries with bounded execution. A compatibility claim should not require retaining an infinite loop.

### N4. The hidden-node experiment discards failed optimizer results

**Locations:** `MPF_CMU/validation/hidden_node_check.py:36–55`, `population_consistency.py:37–58,66–70`; summary Section 3, “A node that is never observed.”

Both the ordinary BFGS calls and the frozen inner optimizations retain only `.x`. They discard success, status, termination reason, and final gradient information. Independently repeating the six documented legacy runs—five visible nodes, one hidden node, beta `0.25/0.5/1.0`, seeds `1/2`—gave **six failures**, each with SciPy status 2: “Desired error not necessarily achieved due to precision loss.” Endpoint gradient norms were approximately `1.6e-8` to `3.4e-6`, against the requested `gtol=1e-10`.

The independent reference objective forms raw exponentials, and its gradient is a finite difference with step `1e-6`. Those choices deserve scrutiny when the optimizer drives parameters to magnitudes in the tens or hundreds. The summary nevertheless calls the endpoints “its minimiser” and uses them to quantify hidden-coupling inflation of roughly 20–230 times and visible-distribution KL up to approximately 12.

**What remains supported:** the substantial nonzero gradient of the published objective at the truth is valid evidence of bias. The marginal objective and frozen equation are approximately stationary there. Those findings do not require a successful optimization to a distant solution.

**What is not established:** that the large reported endpoints are finite local or global minima, that their coupling ratios or KL values are stable under numerical choices, or that starting a replacement method at the true parameters demonstrates recovery from ordinary initializations. Small residuals in several failed runs are useful evidence; a failed status alone does not make every endpoint meaningless. It does require a more careful claim.

**Action:** retain all optimizer diagnostics, use stable log-domain reference calculations and reliable derivatives, and check tolerance/start sensitivity. Distinguish a stationary truth, a converged finite optimum, and movement toward an infimum at diverging parameters. Until then, describe the large values as observed optimizer endpoints and qualify the quantitative conclusion. Appendix B gives a compact status-checking reproduction.

### N5. Some new experiments still lose convergence evidence

**Locations:** `MPF_CMU/validation/drh_bootstrap.py:49–59`, `common.py:68–76,94–100`; `results/drh_bootstrap.jsonl` and `results/table3_frozen.jsonl`.

The DRH bootstrap discards both return values from `run()` for its stage-one and final fits. The C program deliberately returns exit status zero for a stalled or iteration-limited fit, accompanied by a warning. Therefore successful subprocess completion does not establish convergence. The 200 saved bootstrap records retain KL and lambda but omit fit status, warnings, and residuals.

The reported bootstrap means and paired standard errors reproduce from the stored records. **I have not established that these fits failed**; the problem is that the result package cannot establish that they succeeded. This matters for differences of only a few thousandths in KL and for the summary’s broad statement that result files carry warnings and convergence status.

The completed frozen Table 3 campaign does retain warning counts. Searches with at least one warning number 3/16 dispersed, 0/16 ordered, 7/16 near-critical, and 12/16 critical. These counts are disclosed in the revised summary, which is an improvement. However, warnings anywhere in a search neither prove nor disprove convergence of the folds at its selected lambda.

**Action:** record convergence, residuals, and iteration counts for both stages of the selected fit, and retain diagnostics for every oracle candidate or CV evaluation. For CV, explicitly record the final fold statuses at the selected lambda. Restore the bootstrap evidence through rerunning or otherwise recovering these diagnostics; do not infer it from exit code zero. Separate statistical comparison of objectives from computational failure of a particular solver. The all-converged Table 3 oracle results already provide a useful comparison with the CV campaign.

### N6. Resume logic can overwrite the provenance it is meant to preserve

**Locations:** `MPF_CMU/validation/run_table.sh:8`, `run_trials.sh:10`, `common.py:18–41`.

Both campaign wrappers unconditionally rewrite `OUT.provenance.json` before deciding which trial IDs are already complete. Resuming with a different binary, source tree, or environment can replace the original manifest even if no new trial runs. If only some trials remain, the file can describe a mixture using only the latest full manifest.

Per-trial short C-source tags help detect mixing, but they do not preserve the overwritten binary/environment mapping. In addition, neither the full manifest nor the short tag fingerprints the Python scripts that generate data, choose candidates, and compute summaries. Those scripts are part of the experiment. A git HEAD alone cannot capture their working-tree edits.

This is a defect in the resume/provenance mechanism; it does not establish that every existing campaign was mixed or that its numbers are wrong.

**Action:** either refuse to append under incompatible provenance, or write immutable manifests keyed by fingerprint and associate every trial with its manifest. Include experiment scripts, relevant input hashes, command/options, and environment. Compute and validate a manifest before opening an existing sidecar for writing. Add a resume check that changes a source/script fingerprint and confirms that the original manifest survives.

### N7. File parsing and derived numeric output still have gaps

**Integer headers:** `MPF_CMU/src/mpf.c:62,103` uses unchecked `%d` conversion for dataset dimensions and group IDs. The CLI’s checked conversion does not protect these file fields. On the reviewed platform, the files below are accepted as ordinary one-row, two-node datasets and fit successfully:

```text
# File A: the node count overflows an int
1
4294967298
11 1

# File B: the row count overflows an int
4294967297
2
11 1
```

The comments and separating blank line above are explanatory, not file contents. This can silently reinterpret a malformed dataset. Group-ID overflow could similarly collapse distinct identifiers. Use token parsing with explicit range and conversion checks before casting; apply it consistently to file fields as well as command-line arguments.

**Extreme finite parameters:** passing a parameter file containing `1e308 0 0` to `mpf -z FILE 2` exits zero and writes NaN probabilities, with NaN/Inf energies among its rows. Finite input validation and log-sum-exp do not prevent overflow in the Gray-code energy updates (`mcmc.c`, enumeration and probability-output paths). Reject nonfinite derived values and return an error rather than a successful probability file. This is a low-priority extreme-value case; there is no evidence that it affects the DRH fits.

The original R8 examples concerning nonfinite parameter files, overflowing CLI integers, and an all-invalid lambda search have been fixed. Its broad “fixed and tested” disposition should nevertheless be narrowed until these remaining cases are handled.

### N8. Make the conclusions match the demonstrated conditions

**Frozen weights and MAR.** The revised population argument is sound as an **unbiased estimating-equation argument**. Under MAR, at the truth, completing each observed row using its true conditional distribution reconstructs the full joint distribution after summing over masks. Symmetric complete-state flow-gradient contributions cancel. Wholly hidden flips cancel within each row; for larger neighborhoods, mixed observed/hidden flips must remain. The implementation’s correction on that point is appropriate.

This establishes that the truth is a root of the population equation. It does not establish uniqueness, identifiability in every missingness design, convergence of the iteration, or consistency of the regularized implementation. The body now explicitly acknowledges these distinctions. Carry them into headings and recommendations rather than describing the algorithm unconditionally as “correct” or its fixed point as “consistent.”

**The DRH recommendation.** The observed missingness associations are evidence against MCAR under the analysis’s sampling assumptions. They do not establish MAR instead of MNAR. The body and limits say this, while the opening says that frozen weights are therefore “the appropriate method there.” A defensible replacement is: “Among these MPF variants, frozen weights have the relevant population estimating-equation justification if the DRH missingness mechanism is MAR.” No experiment here identifies which method is closest to the unknown DRH truth.

**Large-sample ranking.** “With thousands of rows the order is clear: frozen ≥ marginal ≫ published” is contradicted by the supplied results if read as a general ranking. In `results/large_sample.txt:25–28`, the MCAR, beta=0.5, seed=1 example with 16,384 rows has marginal KL `0.01378` and frozen KL `0.01715`: frozen is approximately **24.5% worse**. Several 2,048-row MCAR examples also favor marginal slightly. The experiments support advantages over the published objective in the tested large-sample settings and a pronounced frozen-weight advantage under the tested large-sample MAR mechanism. They do not establish universal dominance of frozen weights.

**Historical NN=2 attribution.** Agreement between new NN=2 simulations and commented-out manuscript values supports the plausibility of a historical two-flip implementation. Approximate numerical agreement across different simulations does not establish how the historical run was produced. Replace “so those came from a real two-flip implementation” with an explicitly inferential statement, unless contemporaneous source/log evidence supplies that provenance.

## Status of the original review

| Original finding | Follow-up assessment |
|---|---|
| R1: MAR claim for marginal MPF | **Resolved in the technical discussion.** MCAR/MAR are distinguished, the counterexample is retained, and the frozen-weight argument is a useful addition. Qualify the headline recommendation as in N8. |
| R2: incorrect objective/NaN convergence | **Resolved for the reported cases.** Log-domain calculations and explicit numerical-failure status pass their regression tests. The separate extreme `-z` output case is N7. |
| R3: CV initialization leakage | **Partly resolved.** Direct parameter reuse is fixed; globally selected stage-one lambda remains a leakage path, N1. |
| R4: “true EM” and equivalence | **Resolved.** Renaming to `--frozen` and withdrawing monotonicity/finite-sample-equivalence claims are appropriate. |
| R5: objective versus estimator consistency | **Substantially resolved in exposition.** Identification, local solutions, regularization, and convergence limits are explicit. The lower search limit is now `1e-8`, not zero; this remains an acknowledged limit on a strict asymptotic claim. |
| R6: incomplete simulation evidence | **The missing tables are now complete.** Counts and summary calculations check out. N4–N6 identify remaining evidence-quality gaps. |
| R7: reversed DRH lambda labels | **Resolved.** Marginal `0.170` and marginal entry-wise `0.195` now agree with the logs. |
| R8: validation/failure handling | **Original examples resolved; coverage incomplete.** See N3 and N7. |
| R9: all-complete curation | **Resolved.** Missingness counts are reindexed with zeros; all-complete and empty-result paths are tested. |
| R10: pipeline reproducibility | **Substantially resolved.** All 15 analysis stages run in a modern environment; pinned saved tables match the originals; refit inputs are reconstructed. N2 identifies a result-correctness problem that successful execution did not detect. |

## Evidence that supports the revised work

- **Core methods:** the marginal probability-ratio identity and its MCAR qualification are correct. The frozen MAR equation, including mixed flips for NN>1, has a coherent population justification. These establish a real correction to the published partial-flow bias under specified assumptions.
- **Numerical implementation:** the revised tests exercise stable flow calculations, gradients, multi-flip neighborhoods, weighting/grouping, explicit fit status, and failure of an all-invalid search. The earlier false-convergence cases are addressed. Tests across NN=1–4 give useful support to the new neighborhood code; they are not a proof over every supported NN=5/6 configuration.
- **Simulation completeness:** independently checked distinct trial counts are 160 for each Table 1 NN=1/2/3 campaign, 80 for Table 2, 64 for each Table 3 column, 64 for the Table 3 oracle experiment, and 200 for the DRH bootstrap. The frozen Table 3 regime means recompute to `0.212407`, `0.391109`, `0.477032`, and `0.515133`. Bootstrap means and paired standard errors reproduce the summary’s rounding. These are checks of saved evidence, not fresh execution of every landscape.
- **Larger neighborhoods:** the completed paired Table 1 evidence supports the stated NN=2 improvements in the strong-coupling regimes. It does not imply a material improvement for the DRH, whose truth is unknown and whose fitted parameters change little.
- **DRH conclusions:** high coupling correlations and retention of the 15 strongest couplings support stability of those particular features. The revised warning that hierarchical groups are fragile is an important correction. The top-149/top-151 sensitivity is evidence against treating the three groups as robust discoveries. N2 still requires correction of the NN=3 outputs.
- **Pipeline:** the new Python expansion route, curation fixes, declared dependencies, and refit recipe address practical obstacles from the first review. All six saved `regenerated/published/tables/` files are byte-identical to their original counterparts. A fresh modern-environment run succeeds, with clustering differences from the pinned environment already disclosed.
- **GSL removal:** independent comparison reproduced all 8.4 million tested RNG draws exactly. Digamma and trigamma agreed at approximately `1e-15`. On 400 NSB count vectors, both implementations agreed within `3.82e-12` bits when both succeeded. On this machine the new integration failed in 112 cases and the GSL-based version in 122, with ten failures exclusive to GSL. This supports compatibility in the tested cases; it is not evidence that NSB always succeeds. The summary’s stored platform had slightly different failure counts/tolerance maxima.

## Verification performed and limits

Builds and probes were isolated under `/tmp/codex-review2.3wcttp`, `/tmp/codex-numerics2.R3wLhf`, and other scratch directories. The review included parallel independent audits of the C objectives/CV implementation, scientific claims and experiment records, and preprocessing/analysis pipeline.

| Check independently run | Outcome |
|---|---|
| Fresh MacPorts clang 19/OpenMP build and `make test` | **201,324 unit checks and 97 CLI checks; zero failures.** The unit count includes 200,000 sampler range checks, as the summary now explicitly says. |
| Apple clang AddressSanitizer + UndefinedBehaviorSanitizer, serial unit suite | **201,324 checks; zero failures and no sanitizer report.** This run did not cover every CLI or long campaign. |
| New GSL reference harness | RNG matches and special-function/integration agreement described above. |
| Entire `analysis/regenerate.py` pipeline from published parameters, scratch output | **All 15 stages passed.** Modern environment included Python 3.14.6, NumPy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, SciPy 1.18.1, matplotlib 3.11.2, and NetworkX 3.7. |
| Preprocessing regression suite | Passed, including all-complete data, empty-result handling, and actual execution of `expand_data.py`. |
| Saved published tables versus original six tables | Byte-identical. This is separate from the fresh modern-library run; the pinned environment was not recreated for this follow-up. |
| Independent adversarial probes | Reproduced N1, N2, N3, N4, and N7; inspected N5/N6 evidence paths and independently recomputed cited result aggregates. |

Passing the supplied tests does not resolve the counterexamples above. In particular, the current CV regression fixes stage-one lambda, and the pipeline has no invariant enforcing preservation of isolates or the complete entry partition.

Two smaller points should also be handled when touching these files:

- `integrate()` is exposed in `mpf.h` but accepts `max_intervals=0` without a guard (`nsb.c:60–66`). It then writes element zero of undersized allocations. An independent sanitizer probe confirms a heap-buffer overflow. Production NSB calls use 4,000 intervals, so this is not a demonstrated CLI-reachable defect. Validate the argument or make the routine private with an explicit internal precondition.
- The fixed-lambda penalty formula in `src/README.md:66` uses `n`, whereas the implementation scales it by the neighborhood size `n_prox`. The earlier neighborhood paragraph explains the scaling correctly. Make the displayed formula consistent for NN>1.

I did not rerun every simulation, the full mutation campaign, all new DRH fits, historical Ruby helpers, raw-export curation, Julia, or the n=30/40 historical experiments. The review does not certify archived logs, every binary artifact, or every figure’s visual appearance. It checks the revised claims against source, saved evidence, successful pipeline execution, and targeted independent calculations. The updated limits section correctly identifies many remaining research questions: observed-data likelihood benchmarking, multistart sensitivity, selection using partial rows, uncertainty, and behavior beyond exact-enumeration sizes.

## Requested next revision

1. Fix N1 and N2 first. Add whole-procedure CV tests and pipeline invariants, then regenerate the affected fits and products. Report whether the scientific comparisons change.
2. Handle the RNG seed hang and remaining parsing/output failures without changing ordinary compatibility streams unnecessarily.
3. Preserve optimizer and experiment provenance/status evidence. Reassess the hidden-node endpoint claims and restore the bootstrap convergence record.
4. Revise the headline MAR recommendation, universal ranking, and historical attribution. Keep the supported population arguments and empirical gains, with their assumptions explicit.
5. In the next response, link each finding to its fix or reasoned disposition, the new regression that detects it, and the outputs regenerated. Do not mark R3 fully resolved until selection of the first-stage lambda is independent of the outer held-out observations.

## Appendix A: reproducing the remaining CV leakage

Save the following as `cv.dat` in a scratch directory:

```text
16
5
00000 1
11111 1
10011 1
11111 1
01011 1
11110 1
11XXX 1
1X0XX 1
X1X1X 1
0XX1X 1
XX0X0 1
1XX1X 1
X1X0X 1
X1XX1 1
XX0X0 1
X01XX 1
```

The probe below calls the current CV implementation using its globally chosen complete-data stage, then explicitly performs selection inside each outer training fold. It compares the same outer lambda, not two differently tuned final estimators. Compile against the five library source files, excluding `main.c`, for example with `cc -O2 -Wno-unknown-pragmas -I MPF_CMU/src /tmp/cv_probe.c MPF_CMU/src/{mpf,mcmc,nlin,nsb,rng}.c -lm -o /tmp/cv_probe`, then run `/tmp/cv_probe /tmp/cv.dat`. Use matching paths for your scratch files. Minor last-digit differences between compilers/initializers are expected.

```c
#include "mpf.h"
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    dataset *d = read_data(argv[1]);
    options opt = {0}; opt.objective = OBJ_MARGINAL;
    int np = d->n * (d->n + 1) / 2, boundary, folds = 0;
    char *complete = xmalloc(d->m), *train = xmalloc(d->m), *ct = xmalloc(d->m);
    double *start = xcalloc(np, sizeof(double)), *x = xcalloc(np, sizeof(double));
    for (int j = 0; j < d->m; ++j) complete[j] = d->n_blanks[j] == 0;
    cv_context cv = {.data=d, .include=complete, .opt=&opt};
    double stage = search_lambda(cross_holder, &cv, 0, 1, &boundary);
    model *m = build_model(d, complete, stage);
    fit_model(m, start, opt.objective); free_model(m);
    double current = cross(d, NULL, -2, start, &stage, &opt), nested = 0;
    for (int i = 0; i < d->m; ++i) if (complete[i]) {
        for (int j = 0; j < d->m; ++j) {
            train[j] = j != i; ct[j] = complete[j] && j != i;
        }
        cv.include = ct;
        double s = search_lambda(cross_holder, &cv, 0, 1, &boundary);
        memset(x, 0, np * sizeof(double));
        m = build_model(d, ct, s); fit_model(m, x, opt.objective); free_model(m);
        m = build_model(d, train, -2);
        fit_info info = fit_model(m, x, opt.objective); free_model(m);
        double ll = h_value(d->n, x, d->v + (size_t)i * d->n) - log_z(d->n, x);
        printf("fold %d: stage %.9g, score %.9g, %s, residual %.4g\n",
               i, s, ll, fit_status_name(info.status), info.gmax);
        nested += ll; ++folds;
    }
    printf("global stage %.9g; current %.9g; nested %.9g; delta %.9g\n",
           stage, current, nested / folds, nested / folds - current);
    free(complete); free(train); free(ct); free(start); free(x); free_data(d);
}
```

## Appendix B: checking the hidden-node optimization statuses

Run from the repository root in an environment with NumPy and SciPy. This repeats the six legacy optimizations while retaining their results; it does not edit the repository.

```python
import sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "MPF_CMU/validation")
from population_consistency import make_objectives, numgrad

for beta in (0.25, 0.5, 1.0):
    for seed in (1, 2):
        theta = np.random.default_rng(seed).normal(0, beta, 21)
        K, _, _ = make_objectives(6, theta, [(5,)], [1.0])
        result = minimize(K, theta, jac=lambda x: numgrad(K, x),
                          method="BFGS", options={"gtol": 1e-10, "maxiter": 5000})
        print(beta, seed, result.success, result.status, result.message,
              "gradient norm", np.linalg.norm(result.jac),
              "max parameter", np.max(np.abs(result.x)))
```

For the seed hang, use a subprocess timeout instead of leaving the command running indefinitely. For N2, compare the configuration IDs selected directly from the NN=3 probability distribution with `network_information.csv`, and compare the union of the table-generation included/excluded entry sets with `entry_reference.csv`. These probes require no refitting.

## Appendix C: reviewed source fingerprints

The principal C sources and summary were hashed before the review probes and checked again before completing the report; none changed. SHA-256 values below identify the reviewed working-tree content, including files not yet committed. They are not claims that every saved experiment used this exact build.

```text
HEAD 0b58f9234d892cd411afbcd3e1eebab2cb80428b
1d1e816d15a9600822799aef423581a8d3c8b0000446b228be189cd3a4f4199f  MPF_CMU/src/mpf.c
1dbd0c5fe8a05e0910fba8135dfcd2142094854b54d411176c84ee63323c4fab  MPF_CMU/src/nlin.c
4f7bf787017e292053a8831af4700ab76aa1f9028e5ea60841a6253784db0a9d  MPF_CMU/src/main.c
763d630b5869c4fc316b03a268487f39cefb14fae414051d4b748e3cfc98e880  MPF_CMU/src/mcmc.c
45f8e892e0b8f49bf38fe88622cfb8c647baa43fd66b9674efb52d1ff007f19c  MPF_CMU/src/nsb.c
cc369275f99abbbc7a8fa07b327d3cd045061e518c55d52a7c890b22d8d464d8  MPF_CMU/src/rng.c
922efecb97c3e7b7c723ce65c1d5b867652bbf6c44e8cf30aabe99180c49bc1f  MPF_CMU/src/mpf.h
0153c9da921b343141a56c72e02c2dbc4ccc966ce577c79e6a7eff803d201e35  CLAUDE_IMPROVEMENTS_SUMMARY.md
02a025b297e593b3253d5a2c96a0061622a83d6c06d3c03c1b452c5c2f0903be  MPF_CMU/src/Makefile
c8f111d6bf8906994232b059ab4f27489e3965c7ece54696f9bfb8e3136a42c5  MPF_CMU/src/tests/test_mpf.c
3ed5b4ec872e454ccf6cbdee02fbd5c9f3d0bd8d643190fc9f318c5b101cd9bb  MPF_CMU/validation/hidden_node_check.py
28e7907407fdee8b456cc9bd009107f6b1da23e0dfc91ab860d3e8b53880bb54  MPF_CMU/validation/drh_bootstrap.py
f240f6a07b8f280e62223cac54dedc0bbfa8ea092cba42e8d08cc2ded76a4e12  MPF_CMU/validation/common.py
7a19117efb8b76260c2744e388ddc47aa306b2144539d49e3c52bd954eca2ab7  MPF_CMU/validation/run_table.sh
3aae7bc83b91a6f1116e0f590d7a5781cb37e5e80383d618d416dfe8f61b15fb  MPF_CMU/validation/run_trials.sh
7b9f375da309b2e19b73258b4c001b00835882a49c2a30cc014a9c4506bb6ba0  analysis/prep_fig_4_landscape.py
6a6045ea04014e488ded87c6efb04f83fb592947d7729f195d53276ce43e64cf  analysis/table_A3_and_A4_entries.py
44bc9e43e46ccdcbacab33d6caa91c9def0a2e17a6a8453b02df3a6c061868a1  analysis/compare_regenerated.py
```

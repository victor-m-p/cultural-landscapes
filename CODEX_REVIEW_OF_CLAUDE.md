# Review of Claude’s changes to cultural-landscapes

Reviewed by Codex, 14 November 2026.

**Verdict: substantial, useful work, but not ready to accept as an unqualified methodological correction or a fully validated replacement.** The faster legacy implementation reproduces the published parameter file closely, the central marginal-flow identity is correct, and the new tests provide meaningful coverage. However, I independently reproduced a failure of the stated missing-at-random consistency claim, silent numerical failure reported as convergence, and dependence of cross-validation on initialization that includes the held-out observation. The improvements summary also overstates completed validation and reverses two reported DRH regularization settings.

This report concerns the supplied working-tree changes relative to commit `23dce376815141fb2e2679af80292dd5973b1393`. I did not modify the implementation, published data, or figures. Builds, experiments, and regenerated products were isolated under `/tmp/codex-review.1ggmM1`. Reproduction instructions for the most important findings are included below so the report does not depend on that temporary directory surviving.

## Scope and evidence standard

I inspected all changed source files, the new C and Python tests, the validation scripts and saved results, the original versions of the rewritten C files, the preprocessing/analysis code relevant to the paper, and the published paper’s relevant methods and results. I rebuilt both the original and revised C implementations, ran the supplied tests and independent counterexamples, regenerated preprocessing products in a scratch copy, recomputed result summaries, and attempted the downstream analysis scripts in that copy.

“Confirmed” below means supported by code inspection and/or a reproduced calculation, not merely repeated from Claude’s summary. Statements about experiments I did not rerun are identified as checks of saved evidence. This is not a certification of every historical simulation, dormant helper script, binary, or archived output in the repository.

The repository was receiving new validation output during this review. In particular, the Table 1 exact run grew from 159 to 160 records, its Monte Carlo comparison reached 160 records, and an initially absent Table 2 result file appeared but still contained zero records at final inspection. I used an isolated source copy for execution and checked that the principal source hashes remained unchanged. Findings about missing experiment evidence refer to the reviewed snapshot and those final checks; later outputs should be assessed separately.

## Findings requiring action

| ID | Severity | Finding |
|---|---|---|
| R1 | High, scientific | The new marginal objective is not generally consistent under MAR; the demonstrated assumption is MCAR/value-independent masking. |
| R2 | High, correctness | Finite inputs can produce incorrect marginal objectives and NaN gradients, and the optimizer can silently call that convergence. |
| R3 | High, validation | Cross-validation initializes from held-out data; nonconvex fits can retain that information through the selected local optimum. |
| R4 | Medium, scientific | The implemented `--em` procedure is not justified as likelihood EM and does not generally give the same finite-data answer as marginal MPF. |
| R5 | Medium, scientific | The asymptotic claim omits identifiability, global optimization, and vanishing regularization; the default search has a strictly positive penalty floor. |
| R6 | Medium, evidence | The “all three tables rerun” claim is ahead of the supplied evidence; prototype results and incomplete provenance need clearer treatment. |
| R7 | Medium, reporting | The DRH marginal and entry-wise CV lambda values are reversed in the summary. |
| R8 | Medium, robustness | Nonfinite parameters and out-of-range integers are accepted; failed fits/searches can still produce apparently successful output. |
| R9 | Medium, pipeline | The curation fix still fails when the selected questions have no missing answers anywhere. |
| R10 | Medium, reproducibility | The complete analysis remains environment-dependent, and the refit recipe relies on ignored input copies. |

### R1. “Missing at random” is too weak for the claimed consistency

**Locations:** `CLAUDE_IMPROVEMENTS_SUMMARY.md:104–109`; `MPF_CMU/mpf.c:228–239`; `MPF_CMU/validation/population_consistency.py`; `MPF_CMU/tests/test_mpf.c`, `test_consistency`.

The identity

\[
\sum_c p_\theta(c\mid o)\exp[H_\theta(o^k,c)-H_\theta(o,c)]
=\frac{p_\theta(o^k)}{p_\theta(o)}
\]

is correct. Consequently the new objective is a sum of single-flip marginal probability-ratio losses. The problem is the distribution over which those losses are averaged.

Under missing completely at random (MCAR), observations with a particular mask have the corresponding population marginal distribution. Under general missing at random (MAR), selection into that mask can depend on the observed answers. The distribution within the mask is then proportional to `p*(o) P(mask | o)`, not simply `p*(o)`. Applying a proper loss for the latter distribution to the former does not establish consistency. MAR and MCAR are distinct assumptions; see the primary discussion in [Seaman et al., What Is Meant by “Missing at Random”?](https://arxiv.org/abs/1306.2812).

**Independent counterexample.** Take two spins with true `(J, h1, h2) = (0.7, 0.2, -0.4)`. Always observe spin 1. Hide spin 2 with probability 0.8 when spin 1 is +1, and 0.2 when spin 1 is −1. This is MAR: missingness depends only on an always-observed variable. Every full configuration still has positive probability of being completely observed, so this is not an identifiability failure.

Use exact population probabilities as row weights, eliminating sampling error. At the truth, the new objective’s gradient is approximately

```text
dK/d(J,h1,h2) = (-0.016917885, -0.054395131, -0.163678253)
```

The revised C implementation, with negligible regularization, converges to

```text
(J,h1,h2) = (0.922908895, 0.133966665, 0.088996545)
```

Thus even the sign of the second field is wrong in this example. I checked the gradient independently using explicit enumeration and finite differences in Python, not just the C implementation. Replacing the mask probabilities by a constant 0.5 gives a gradient at the truth at floating-point roundoff and recovers the truth. This supports the MCAR result while falsifying the broader MAR claim.

**Required correction:** state the missingness assumptions precisely. The current simulations use value-independent random masking and support that setting. Do not generalize this to arbitrary MAR or to the DRH missingness mechanism. For general MAR, investigate an appropriate selection-adjusted loss, restricted valid transitions, or an observed-data likelihood approach, and prove the required conditions. This is a substantive limitation of the new method, not merely a terminology change.

### R2. Marginal-flow numerics can fail silently

**Locations:** `MPF_CMU/mpf.c:276`, `:322–339`, `:376–379`, `:429–452`.

The partition-function implementation uses log-sum-exp, but the fitting objective does not maintain comparable stability. It forms `exp(-x)`, squares flow terms, and divides by their square roots. This can underflow or overflow even when the final marginal ratio is moderate and exactly representable.

Two reproduced cases:

1. A single complete observation `11`, weight 1, initial parameters `(0,400,400)`, and log-lambda 0. The true flow is tiny, but the regularizer has a large, finite gradient. Squaring `exp(-400)` underflows to zero; the gradient computation evaluates `0/0`. `fabs(NaN) > gmax` is false, leaving the computed gradient maximum at zero. The program reports zero iterations, returns the unchanged parameters, writes the output file, exits 0, and prints no warning. Its objective is about **106666.67**, while the all-zero parameters have objective **2**. This is unequivocally false convergence.
2. A single partial observation `1X`, with `(J,h1,h2)=(400,0,0)` and negligible penalty. Symmetry makes the observed marginal uniform, so the exact row objective is **1**. The implementation returns essentially **0** (only the tiny penalty remains) and NaN gradients because a tiny completion probability is lost before it is multiplied by a huge flow.

These are finite, valid parameter values within the advertised interface. They are stress cases, not evidence that the DRH fit itself encountered this failure. The important regression is that a numerical failure can masquerade as a successful fit.

**Required correction:** compute each marginal flow from the difference of two stable log-partition sums over completions, and compute its gradient from separately normalized completion distributions for the original and flipped observation. Special-case complete rows instead of unnecessarily squaring and taking square roots. Explicitly reject nonfinite objective/gradient components. Distinguish gradient convergence, numerical failure, line-search failure, stagnation, and iteration exhaustion. A failed line search or three small objective changes is not sufficient evidence of stationarity.

The same status discipline is needed in the outer `--em` loop: small parameter change can also result from a failed inner optimization.

### R3. Warm starts make the new cross-validation procedure depend on held-out observations

**Locations:** `MPF_CMU/nlin.c:54–78`; `MPF_CMU/main.c:81–106`.

At each candidate lambda, `cross()` first fits all included rows, including the rows to be held out. Every fold then starts at that full-data solution. This is harmless for an accurately solved strictly convex problem with a unique optimum. Missing-data objectives are nonconvex, however, and the starting point can determine which solution is returned. The staged complete-data initialization also uses observations subsequently held out during the second stage.

**Reproduced example:** on the twelve-row, five-node dataset in Appendix B, at log-lambda −2, the mean complete-row held-out log score is:

```text
current full-data warm-start CV:    -12.8500 approximately
separate training-only zero starts: -11.7518 approximately
difference:                         -1.09826 nats per held-out row
```

The direction happens to be unfavorable to the warm start here. The defect is dependence on the validation observation, not a claim that it always makes the score optimistic. Neither initialization is established to find the global optimum. A follow-up check found all six fold fits reporting convergence with maximum gradient components below `4e-7`. In the second and third folds the warm starts found substantially lower training objectives but different held-out scores, confirming that this is a local-solution issue rather than merely an unfinished run.

The supplied CV test uses one modestly regularized random problem (`log-lambda=0.3`) where the two starts agree. That is useful but does not cover this failure mode. The mutation-test discussion that a warm-start defect cannot change convex fits does not establish safety for the partial-data path.

**Required correction:** construct fold initialization and any staged preprocessing from training rows alone. Use reproducible multistart checks on difficult partial-data fits. Compare actual objective values and convergence diagnostics across starts. Then rerun the partial-data CV comparisons and DRH lambda selection. The numerical agreement of the current DRH refits remains useful, but it does not validate this CV design in general.

Also, `cross()` discards the `fit_info` returned by both full and fold fits. An unconverged fold is silently scored as though it were valid. Propagate failures into the lambda search and record them in experiment output.

### R4. The `--em` description is too strong

**Locations:** `MPF_CMU/mpf.c:459–485`; summary lines 104–106; `MPF_CMU/README.md`, options table.

The procedure estimates conditional completion weights, then minimizes a frozen-weight **MPF flow loss** for a few steps. Ordinary likelihood EM instead optimizes the expected complete-data log likelihood. No likelihood lower bound, majorization argument, or monotonicity proof is provided for this MPF alternation. Calling it “true EM” or “generalised EM” imports guarantees that the implementation and tests have not established.

It also does not generally reach the same answer as marginal MPF. On Appendix B’s dataset with log-lambda 0, the two fitted parameter vectors differ by **0.1685075** in the largest component. Their different estimating equations make such a difference expected. Agreement at a correctly specified population fixed point does not imply equality for finite samples or regularized fits.

In the MAR population example above, the frozen-weight procedure has a fixed point at the truth whereas the marginal objective does not. This further demonstrates that they are different procedures. That observation alone is not a general convergence theorem for the frozen-weight procedure.

**Required correction:** call this an iterative frozen-posterior-weight MPF method unless a suitable EM/majorization proof is supplied. Separate fixed-point consistency, convergence from arbitrary starts, and finite-sample equivalence. The existing population unit test invokes `fit_model()` sixty additional times for EM; it therefore does not show that one ordinary CLI fit attains the reported `2e-8` error.

### R5. Objective consistency is not consistency of the shipped default estimator

**Locations:** `MPF_CMU/mpf.c:203–204`; `MPF_CMU/nlin.c:128–130`; summary Section 2.

There is a clean population argument for the **unpenalized marginal objective under value-independent masking**. For a fixed mask, pair the two directions of every observed-state edge. With true probabilities `p_a,p_b` and model ratio `r=q_b/q_a`, their contribution is

\[
p_a\sqrt r+p_b/\sqrt r\;\geq\;2\sqrt{p_a p_b},
\]

with equality at `r=p_b/p_a`. Connectivity identifies each observed marginal; sufficiently informative masks then identify the joint model. This directly supports the central correction and avoids conflating it with every version of MPF’s historical consistency argument. Applying MPF after marginalizing hidden units also has precedent in [Sohl-Dickstein et al., Section 4.2](https://icml.cc/2011/papers/480_icmlpaper.pdf).

But three qualifications matter:

- **Identifiability:** observing only separate single-spin marginals generally cannot identify all pairwise couplings. The tests include complete observations with substantial population mass.
- **Optimization:** a local stationary point need not be the global population minimum.
- **Regularization:** the penalty is proportional to total data weight, just like the flow objective. Duplicating a dataset does not weaken the relative penalty. The CV search fixes `log10(lambda)` to `[−2,4]`, so lambda cannot tend to zero below 0.01 as sample size grows. At a nonzero true parameter vector the penalty contributes a nonzero population gradient even when the unpenalized gradient vanishes.

The population tests avoid this issue by using essentially zero regularization. The large-sample script even tests log-lambda −3, outside the production CV search range. Neither validates asymptotic consistency of the shipped default procedure.

**Required correction:** state a theorem with assumptions, distinguish it from the finite-sample algorithm, and permit an adaptive/wider regularization range when studying large-sample behavior. Boundary warnings help but do not remove the penalty floor.

### R6. The simulation evidence does not support every statement in the summary

**Locations:** summary lines 19, 234, 270–276; `MPF_CMU/validation/README.md`; validation result files and runners.

At the reviewed snapshot, `TBD_TIME`, `TBD_TABLE1`, and `TBD_TABLE2` remain in the summary. There are no completed saved Table 2 records: `table2_exact.jsonl` was initially absent and was empty at final inspection. Therefore the claim that all three tables have been rerun with exact sampling and leave every conclusion unchanged is not established by the supplied evidence.

Table 1 exact output reached 160 records during the review. Recomputing it gives:

| Regime | Trials | Oracle KL, mean ± SE | CV KL, mean ± SE | Unregularized KL, mean |
|---|---:|---:|---:|---:|
| Dispersed | 40 | 0.207 ± 0.022 | 0.212 ± 0.022 | 1.18 |
| Ordered | 40 | 0.560 ± 0.012 | 0.564 ± 0.012 | 4.36 |
| Near-critical | 40 | 0.621 ± 0.025 | 0.631 ± 0.026 | 408.60 |
| Critical | 40 | 0.522 ± 0.057 | 0.538 ± 0.057 | 655.18 |

These saved results support the regularization/CV conclusion for those runs. They do not turn optimizer-dependent unregularized fits into well-defined converged estimates. Experiment JSON omits convergence status; `common.run()` discards stderr for exit-zero runs, including warnings. In `-o`, some nonconvergence warnings are printed only with `--verbose`, which the table runner does not request.

For Table 3, the 64-row file is explicitly from a prototype; only 14 trials were rerun with final code. I verified overlap by trial ID, with a largest KL difference of about **3.50e-4** across the saved comparisons. This is good evidence of close numerical agreement on the tested subset, not a final-code rerun of all 64 trials. The summary should carry this qualification next to the table.

Recomputing Table 3 reproduces the displayed rounded means. Both legacy and marginal partial fits beat the 128-complete baseline in 60/64 trials. However, “indistinguishable” and “equivalent” are too strong: the near-critical paired difference is **+0.0120 ± 0.0037** for marginal minus legacy KL, favoring legacy in that experiment. A nonsignificant difference in other regimes is not an equivalence test. Naive imputation is also not uniformly demonstrated to hurt by a resolved amount in every regime.

The large-sample saved results do show a real advantage for marginal MPF in the six 8192+8192 cases examined. They use truth-selected lambdas on a coarse grid, with some minima at grid endpoints. They do not establish a universal crossover sample size, uniformly better practical CV behavior, or superiority to maximum likelihood. The limitation is partly disclosed; the headline language should be equally restrained.

**Required correction:** finish and archive the missing experiments, retain stderr and fit diagnostics, record source/build hashes and environment, detect duplicate/missing trial IDs, and distinguish inspected saved results from completed final-code reruns. The append-only shell runners can append duplicate trials when rerun, and the summary scripts do not reject duplicates.

### R7. The DRH summary swaps the two nonlegacy lambda settings

**Locations:** summary “DRH refits” table; `MPF_CMU/validation/results/drh_refits/*.log`.

The parameter-comparison statistics in `drh_compare.txt` are reproducible, but the lambda labels in the summary do not match the named logs:

| Fit | Lambda reported by its log | Lambda in Claude’s table | Maximum parameter difference from committed fit |
|---|---:|---:|---:|
| Legacy | 0.206738 | 0.207 | 0.0005547 |
| Marginal, weighted row-wise CV | **0.170203** | **0.195** | 0.0479886 |
| Marginal, entry-wise CV | **0.195048** | **0.170** | 0.0576064 |

These are log10(lambda) values. I independently rebuilt the revised executable and ran the grouped DRH fit using a groups file checked against the reference CSV. It selected **0.195048** and reproduced the saved grouped parameter vector exactly at file precision. Thus the table’s direction of change on adding entry-wise CV is wrong, even though its parameter-difference and correlation columns correspond to the intended files.

**Required correction:** generate the entire table from named logs and parameter files in one script, rather than transcribing lambda values separately.

### R8. Input validation and failure handling are incomplete

**Locations:** `MPF_CMU/mpf.c:98–110`; `MPF_CMU/main.c:50–55`, option parsing; `MPF_CMU/nlin.c:139–168`.

Reproduced examples:

- A parameter file containing `nan 0 0` is accepted by `-z ... 2`, produces four rows of NaN energies/probabilities, and exits 0. `read_params()` checks parsing success but not finiteness.
- `-g prefix 4294967298 1 0.1 --seed 1` succeeds as a two-node request on this build. `parse_int()` narrows `long` to `int` before checking the node range, without checking overflow.
- Passing a callback that returns NaN everywhere to `search_lambda()` returns −2 as the “best” lambda. Grid values are converted to infinity, but the function never checks that a finite candidate exists.

Other inspected gaps: `--seed` does not properly reject negative/empty/overflowing values; large finite log-lambda values can overflow the penalty; several allocations and file-write/close operations are unchecked. `-o` accepts `--groups` in the global parser but never reads the group file, silently ignoring the option in that mode.

**Required correction:** validate finite parameters and representable numeric ranges before conversion; fail explicitly if no valid search candidate exists; check writes; and propagate numerical fit failures. Treat unsupported option/mode combinations explicitly. The claim “exit 2 with a message on any error” is not yet true.

The grid-plus-golden-section search is an improvement over ignoring a GSL bracket error. It still searches only around the best coarse-grid point and is not a guarantee of the global minimum for an arbitrary non-unimodal CV curve. The two-well unit test covers one favorable example, not that general property.

### R9. The fully-answered-question fix does not work through the entire curation path

**Locations:** `preprocessing/curation_functions.py:74–97`, `:118–126`.

`subset_nbest()` now correctly retains questions with zero missing answers. But if all selected questions are completely answered, `max_constraints()` builds its indicator grid from values that actually occur. Only `has_answer=1` occurs; it never creates the `has_answer=0` rows representing zero missingness. Filtering to zero then discards every entry.

I reproduced this with three entries and two questions, all answered: `subset_nbest()` selects both questions, then `max_constraints(..., number_nan=0, minimize_value=0, ...)` returns **zero answer rows** instead of six. This is an inherited downstream defect newly relevant to the advertised fix. It does not show a problem with the committed DRH subset.

**Required correction:** count missingness per entry and reindex all entries with zero fill, rather than inferring possible indicator values from observed categories. Add the all-complete case and an empty-result case to the curation tests; `to_dataframe()` also assumes a nonempty list.

### R10. Reproducibility remains narrower than “the pipeline now runs”

**Locations:** `requirements.txt`; `MPF_CMU/validation/results/drh_refits/refit.sh`; `analysis/prep_fig_4_landscape.py:42–51`; preprocessing/analysis documentation.

The modified preprocessing functions and the tested committed products agree. That should be preserved as a positive result. It does not establish end-to-end execution of the raw curation and figure pipeline.

- `refit.sh` assumes three `.mpf` input copies already exist beside it. Those files are explicitly ignored by the new `.gitignore`; the script does not create them from the committed input. They happen to be present locally, but the documented recipe is incomplete for a clean checkout. It also lacks fail-fast handling.
- In the current isolated Python environment, `prep_fig_4_landscape.py` fails with `KeyError: 'node_id'`: its `groupby(...).apply(...)` result no longer retains the grouping column as expected. This is inherited code and an environment-compatibility problem, not a defect introduced by the vectorization. It means the modern-environment preprocessing check cannot be advertised as a complete figure-pipeline check.
- With the available intermediates in the scratch copy, `fig_4_methodist.py` fails at the removed `plt.cm.get_cmap` API. After regenerating the dendrogram and landscape outputs, `table_A2_clades.py` also fails because a requested clade comparison has fewer than two labels. These are additional reasons to pin and verify the complete dependency chain; I have not attributed them to Claude’s source changes or asserted that they occur in the historical pinned environment.
- Graph layout requires Graphviz/`pygraphviz`; table rendering requires Jinja2. Neither direct requirement is declared in `requirements.txt`. I installed them only into the isolated review environment to continue checking scripts.
- The pinned environment and the modern environment used for review are different; neither Claude’s tests nor mine establish execution under all the advertised pins. Julia and the raw DRH export were not available for an actual end-to-end curation/expansion run.

**Required correction:** provide an executable, fail-fast reproduction entry point with explicit inputs, a tested environment, complete direct dependencies, and distinct stages for published versus corrected parameter products. Run it in a clean checkout. Keep compatibility problems in untouched analysis code separate from the narrower fixes Claude actually made.

## Improvements that are supported

### Legacy reproduction and speed

I rebuilt the committed C implementation separately from the revised code and fitted the same scratch DRH input at log-lambda `0.205745`.

| Comparison | Maximum absolute parameter difference |
|---|---:|
| Rebuilt original vs committed parameter file | 1.97e-6 |
| Revised `--legacy` vs committed parameter file | 1.89e-6 |
| Rebuilt original vs revised `--legacy` | 4.49e-7 |

The observed fixed-lambda runtimes were about 7.56 seconds for the original and 0.040 seconds for revised legacy, supporting a substantial speed improvement. These are individual, uncontrolled timings, not a reproducible benchmark suite. Other simulation processes were active, so my approximately 196-second grouped CV run should not be compared directly with the summary’s 25-second laptop estimate. The supplied refit logs themselves contain roughly 192–304-second timings; the summary’s full-CV benchmark remains unfinished.

The local-field formulation, sparse parameter updates per flip, duplicate merging, deterministic summation order, and Gray-code enumeration are sensible improvements. The supplied reference calculations and my rebuilt tests support their correctness in the tested range. Correcting the 32-bit state shifts is essential for larger systems. Replacing noisy rare-state frequency estimates with a documented pseudo-likelihood score above the exact-enumeration limit is a practical improvement, although that score is a different model-selection criterion and its large-system performance needs its own validation.

### Partial-MPF bias and the marginal identity

The decomposition of the legacy gradient into a frozen-weight flow term and a completion-weight covariance term is correct. The population examples demonstrate genuine legacy bias under value-independent masking. My two-spin MCAR case moved the legacy coupling from 0.7 to approximately 1.01166, while marginal MPF recovered the truth. The square-root-outside-the-average identity, gradient expression, and Jensen comparison are sound mathematically, subject to R1/R2/R5.

The new method enumerates the same number of completions and has the same asymptotic computational order. “Identical cost” is too literal: it performs different reductions/divisions and may need different optimizer iterations. This is a wording correction, not a reason to reject the algorithm.

### DRH stability: useful, but limited in scope

I recomputed the saved parameter/probability comparisons. They reproduce the coupling correlations, top-15 coupling overlap, top-150 configuration overlaps of 143 and 141, and the Spartan conditional probabilities reported by `drh_compare.py`. The published Spartan “ritual and no sacrifice” probability is about 0.696528; the corrected saved fits move it to about 0.720–0.722.

This is evidence of **local numerical stability of these summaries**. It does not establish that every landscape interpretation, clade assignment, pathway, or scientific conclusion is unchanged. Even the top-150 graph changes: I counted 318 single-flip edges for the published fit and 327 for the saved grouped marginal fit, although both graphs remain connected. Clustering and other discrete outputs should be regenerated and compared explicitly before making a blanket statement about the landscape.

The missingness mechanism, dependence between religious groups, model misspecification, uncertainty in rare practices, and sensitivity to curation are not resolved by reproducing a parameter vector. There are no new bootstrap uncertainty estimates, independent predictive evaluation, or exact-likelihood benchmark in this work. Those are limitations of the inference claim, not proof that the published substantive conclusions are false.

### Preprocessing fixes

The vectorized `p_dist()` and `bin_states()` preserve the Python configuration ordering. That ordering differs from the C packed-state index convention, and the validation scripts explicitly account for it. The column rename and Julia output-filename correction match the downstream consumers. The new `_nrows_` naming matches the committed files.

I ran `preprocessing/test_pipeline.py` successfully. I also ran `prep_early.py` and `prep_late.py` in a scratch copy. Of the nine files written, six remained byte-identical; the probability file differed by at most **3.12e-17**, and two CSV files agreed numerically at the comparison tolerance. The Python reimplementation of Julia’s expansion reproduced the committed expanded states and conditional weights. This verifies those products, not execution of Julia itself or raw-data curation.

## Audit of the numbered claims in Claude’s summary

| Claude claim numbers | Assessment |
|---|---|
| 1: partial objective bias | Supported for the tested value-independent missingness setting. The proposed fix needs R1/R2/R5 qualifications. “No effect” on finite results is too absolute. |
| 2: simulation sampler | The inspected algorithm and stored equilibrium diagnostics support failures at strong coupling; exact sampling is an appropriate correction. Full table-level impact is not settled by the incomplete rerun evidence. |
| 3: unregularized optimization | The original status/cap handling is problematic; new saved results also remain optimization-dependent. Reporting must retain actual convergence diagnostics. |
| 4: mixed Table 3 runs | The 519 committed log records reproduce the summary’s reconstructed averages. The asserted historical combination of manuscript runs remains a hypothesis; I did not authenticate the claimed arXiv-source comment. |
| 5: penalty sign/scaling | The executable penalty is positive quadratic with the documented normalization and log10 convention. The published Eq. 8 sign discrepancy is real. |
| 6: posterior/likelihood terminology | The wording needs correction, but replacing “posterior” by “likelihood” alone is incomplete: MPF minimizes a surrogate loss, not generally the exact negative log likelihood. |
| 7: equal flow counts | Correct that partial rows have fewer observed-bit transitions. This does not by itself prove optimal information weighting. |
| 8: inconsistent-entry percentage | The curated subset can establish its own expansion rate, not the rate in an earlier/raw subset. Treat the historical 17% explanation as unresolved. |
| 9–10: N2 and sampler naming | The inspected code supports single-bit neighborhoods and a heat-bath update. Rejecting unsupported N2 is preferable to silently accepting it, but the N2 result is still unreproduced. |
| 11: CV weights/groups | Weighted scoring and grouped exclusion address real concerns. R3 remains; only fully observed groups are scored, and the DRH lambda table is mislabeled. |
| 12–13: larger-n issues | The original shifts and rare-event likelihood estimator are real defects. Quarantine the old larger-n fits pending refitting. A small n=20 comparison does not validate pseudo-likelihood CV for all n>24. |
| 14–17: threading, freeing, optimizer, lambda search | The changes address identifiable defects. Reproduction of the DRH result supports limited impact there; it cannot prove a race was harmless in every historical run. Optimizer and search robustness still need R2/R3/R8 fixes. |
| 18–20: modes and CLI checks | Previously broken paths have improved and the CLI suite passes. “Any error exits 2” is disproved by R8. |
| 21: NSB | The cleanup/documentation is useful. The observed-alphabet estimator remains inappropriate as a full configuration-space entropy estimate. |
| 22: stale Ruby scripts | Labeling them is honest but does not make them executable. The paper-related runners and the rewritten defaults also need explicit provenance/mode selection when reused. |
| 23–25: Python/Julia fixes | Narrow fixes are supported; R9/R10 limit the end-to-end claim. |
| 26: committed products | The supplied regression tests and independent scratch preprocessing run support agreement for the checked products. |

The relevant published discussion is in [Poulsen and DeDeo, Inferring Cultural Landscapes with the Inverse Ising Model](https://www.mdpi.com/1099-4300/25/2/264). My independent counterexamples and repository comparisons, rather than the paper’s assertions, are the evidence for the new findings above.

## Tests and limitations of this review

| Check | Result |
|---|---|
| Fresh revised build, MacPorts clang 19 + OpenMP + GSL | Passed |
| Revised C unit suite | 200,933 checks, zero failures |
| Revised CLI suite, including fixed-lambda DRH reproduction | 65 checks, zero failures |
| Separate Apple-clang AddressSanitizer + UndefinedBehaviorSanitizer unit build | 200,933 checks, zero failures; no sanitizer report |
| Rebuilt original vs revised legacy fixed-lambda DRH fit | Agreement reported above |
| Independent grouped DRH CV fit | Saved grouped parameters reproduced; log-lambda 0.195048 |
| Python regression suite | Passed |
| Scratch early/late preprocessing | Nine outputs agree as described above |
| Spartan analysis and dendrogram script | Executed successfully |
| Remaining downstream analysis, after installing Graphviz/Jinja2 interfaces | Parameter, landscape, and Roman scripts and A1/A3/A4/annotation/community table scripts executed; landscape-preparation, Methodist, and clade-table failures described in R10 |
| Independent MAR population/MCAR control | MAR inconsistency reproduced; MCAR control succeeds |
| Extreme finite-parameter objective/optimizer probes | Silent numerical failures reproduced |
| Warm-start vs training-only CV probe | Approximately 1.09826-nat difference reproduced |
| EM vs marginal finite-sample comparison | Maximum parameter difference 0.1685075 |
| Malformed numeric inputs and all-invalid lambda callback | Incorrect success/fallback behavior reproduced |
| All-complete synthetic curation | Entries incorrectly dropped |

The unit-check count is not 200,933 independent scenarios: 200,000 checks are bounds checks on individual exact-sampler draws. The remaining tests still have real value, especially reference objectives, finite-difference gradients, exact probabilities, and CLI regressions. Their limitations are distributional coverage, modest objective parameter magnitudes, and reliance on the same optimizer in some reference refits. The large-parameter tests cover enumeration, not the unstable fitting objective. Sanitizers do not generally diagnose IEEE NaNs or incorrect statistical assumptions.

I did not independently rerun Claude’s claimed 60-dataset legacy comparison, 24 mutations, macOS leak-tool run, all large-sample simulations, or the entire sampler/table campaign. The stored outputs are useful evidence but several claims lack a saved standalone runner/log sufficient to audit their exact provenance. I did not reproduce every final figure’s layout, run Julia, obtain missing raw DRH data, or execute PSC-dependent historical jobs. The review environment used Python 3.14, NumPy 2.5.4, pandas 3.1.1, and SciPy 1.18.1 in an isolated virtual environment; this is not a test of the pinned historical environment.

## Recommended disposition

1. Preserve the published data products and retain an explicit legacy reproduction path. The performance and cleanup improvements are worth keeping.
2. Fix R2/R8 before treating successful program exit or `converged` as trustworthy. Add the reproduced adversarial cases to the tests.
3. Correct the missingness theorem and EM terminology. Add MAR counterexamples, identifiability tests, and explicit penalty/optimization assumptions to the methodological documentation.
4. Remove held-out-dependent initialization for nonconvex CV; record fold diagnostics and test multiple training-only starts. Rerun the affected comparisons after this change.
5. Complete the evidence package, correct the DRH lambda table, and generate summary tables from immutable experiment outputs with source hashes.
6. Repair the remaining curation/reproduction issues and compare downstream scientific outputs, with uncertainty, before saying all paper conclusions are unchanged.

An appropriately bounded replacement for the headline would be: **“The revision substantially speeds legacy reproduction, fixes several software defects, and introduces a marginal-flow objective that removes the demonstrated legacy population bias under value-independent missingness. The checked DRH summaries remain close. General missingness guarantees, robust nonconvex cross-validation, numerical failure handling, and complete simulation/analysis validation remain unfinished.”**

## Appendix A: minimal reproducible inputs

Run these using a freshly built revised `MPF_CMU/mpf`, in a temporary directory so output parameter files cannot overwrite repository data.

### A1. MAR population counterexample

Write `mar.dat` as follows; weights are the exact two-spin population probabilities multiplied by the observation-pattern probabilities described in R1:

```text
8
2
00 0.37226571202662528
0X 0.093066428006656321
01 0.041248216642553154
0X 0.010312054160638288
10 0.034237225526910997
1X 0.13694890210764402
11 0.062384292305794363
1X 0.24953716922317751
```

Execute `mpf -l mar.dat -100 1`. Compare the output against the true parameters `0.7 0.2 -0.4`. The output is approximately `0.922908895 0.133966665 0.088996545`.

An independent loss calculation is to enumerate the four complete states, compute `H=J*s1*s2+h1*s1+h2*s2`, sum `exp(H)` over completions of each row and its observed-bit flip, take the square root of their ratio, and average using these weights. Central finite differences of size `1e-6` reproduce the gradient in R1.

### A2. False numerical convergence

`one.dat`:

```text
1
2
11 1
```

`strong.par`:

```text
0 400 400
```

Execute `mpf -l one.dat 0 1 --start strong.par`. The reviewed implementation exits 0, writes the unchanged parameters, and prints `K=106666.6666666667 after 0 iterations (1 evaluations)` without a warning. Starting from zero instead finds a much better fit.

For the partial-row objective failure, call `compute_k()` with a model built from one `1X 1` row, log-lambda −100, and parameters `400 0 0`. The correct flow contribution is 1; the reviewed implementation loses it and returns NaN gradients.

### A3. Numeric validation

Write `bad.par` containing `nan 0 0`; run `mpf -z bad.par 2`. Inspect the output and exit status. Separately run `mpf -g wrapped 4294967298 1 0.1 --seed 1`; on the reviewed build this produces two-node data instead of rejecting the node count.

## Appendix B: CV and finite-sample EM counterexample

```text
12
5
00000 1
11110 1
10001 1
X0XX0 1
XX0X1 1
1XX0X 1
X1X0X 1
XXX10 1
X0XX1 1
XX0X1 1
X0X1X 1
XX1X1 1
```

For CV, call `cross(d, NULL, -2, NULL, &opt)` with zero-initialized options (marginal objective, weighted CV). Compare it with three explicit folds, each deleting one of the first three rows, building its training model at −2, fitting from zeros with `fit_model()`, and scoring the deleted row with `h_value() - log_z()`. Average the three scores. The respective results are approximately −12.8500 and −11.7518.

For the EM comparison, run `mpf -l datafile 0 1 --marginal`, retain its parameter vector, then run `mpf -l datafile 0 1 --em` and compare. Maximum component difference is approximately 0.1685075. Do not compare the printed K values as though they were the same objective.

## Appendix C: reviewed source fingerprints

SHA-256 values, useful because the reviewed work was uncommitted:

```text
CLAUDE_IMPROVEMENTS_SUMMARY.md
5d996248584d7481b0a6c85cb93b1f2bebf4c06e3e5e176b0a1950533b9c0fc5
MPF_CMU/mpf.c
76449c297d9f6b7b7f9fd0db09a85c99817ea9db5afa61dc5511c66963ca2c92
MPF_CMU/nlin.c
2520b97127de3914ac00d25231223d214d929fe6df85442804f7d98645dbff1a
MPF_CMU/main.c
1b70a439a7390e0e6993d24201a857d6e329fe297ccaa7fa76c9a4508e57b13e
MPF_CMU/mcmc.c
922dd4ce64f5bb05683fd844842c8fde4ed83618eb4f361d86f79c4e7f09c973
MPF_CMU/nsb.c
ba5ed18a6d5dfde28ab8958969d9008df8073c80089f3e3df7dd3aadcd970540
MPF_CMU/mpf.h
a010bf7dd20e4f1503ec12793257f7ce55ee36dee41790ed458e3f2484ee6e69
```

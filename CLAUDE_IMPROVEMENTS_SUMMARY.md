# 2026 Claude/Codex analysis of the code and methods

This is a re-examination, in November 2026, of the code and methods behind "Inferring Cultural Landscapes
with the Inverse Ising Model" (Poulsen & DeDeo, *Entropy* 25(2), 264, 2023). It was carried out by Claude
(Anthropic) working with Simon DeDeo. It was reviewed twice, independently, by Codex (OpenAI): a first
version ([`CODEX_REVIEW_OF_CLAUDE.md`](CODEX_REVIEW_OF_CLAUDE.md)) and a later one
([`CODEX_REVIEW_OF_CLAUDE_2026-11-16.md`](CODEX_REVIEW_OF_CLAUDE_2026-11-16.md)). Every finding of both
reviews that was tested was confirmed; Section 10 records how each was handled.

The repository now contains the revised code. The published code is in the git history (commit `23dce37`
and earlier), and the published results can still be reproduced with the `--legacy` option.

Contents: [1 Summary](#1-summary) · [2 What to use](#2-what-to-use) ·
[3 Missing data](#3-missing-data-the-slight-tension-and-two-replacements) ·
[4 Neighbourhood size](#4-neighbourhood-size-nn--1-2-3) ·
[5 The DRH landscape refitted](#5-the-drh-landscape-refitted) ·
[6 The paper's simulation tables re-run](#6-the-papers-simulation-tables-re-run) ·
[7 Errors found](#7-errors-found-and-whether-they-reach-the-paper) · [8 What changed](#8-what-changed-in-the-repository) ·
[9 Testing](#9-testing) · [10 The external review](#10-the-external-review) · [11 Limits](#11-limits) ·
[12 Where everything is](#12-where-everything-is)

## 1. Summary

**The published results reproduce.** The revised code in `--legacy` mode matches the published parameter
file to 2e-6. In the pinned software environment the whole analysis pipeline regenerates the paper's six
tables byte for byte, on two machines (macOS and Linux).

**Partial-MPF, the paper's method for missing answers, is biased.** The "slight tension" mentioned in
Section 2.3.5 of the paper is a bias that does not go away with more data. Two replacements are
implemented at the same computational order:

* `--marginal` (the default): MPF on the marginal distribution of the answers that were observed — the
  published formula with a square root moved outside an average. With unlimited data its minimum is at the
  true parameters when blanks are unrelated to any answer.
* `--frozen`: the weights on the possible completions are held fixed while minimising, then re-estimated,
  to a fixed point. With unlimited data the true parameters solve its equation under the weaker condition
  that blanks may depend on *other, observed* answers. It is slower and is not the minimum of a single
  function, and nothing shows that the solution is unique or that the iteration finds it.

**In the DRH data, blanks do depend on other answers** (for example, cultures with a blank somewhere have
scriptures 53% of the time, against 89% for fully answered ones), so fully answered entries are not a
representative sample and the condition that `--marginal` needs fails. Of the three variants only
`--frozen` has a justification that could apply, and it applies only if blanks depend on observed answers
alone, which the data cannot show. No experiment here says which variant is closest to the unknown truth
for the DRH.

**At the paper's sample sizes the choice makes little practical difference.** The three methods give KL
divergences within a few percent of each other on simulated data of the DRH's size, and the published
objective is slightly *better* than both replacements when couplings are strong and data are few. The bias
matters with thousands of rows, where the published objective can be worse than discarding incomplete rows.

**Larger neighbourhoods (N₂, N₃) now work and help when couplings are strong.** With two-flip neighbours
the KL divergence is 8% lower in the "near-critical" regime and 20% lower in the "critical" regime than
with single flips; there is no difference in the weakly coupled regimes, where the DRH sits. This corrects
the paper's statement that N₂ "does not improve significantly".

**The DRH landscape is stable; the grouping of it is not.** Refitting with `--frozen` and one-, two- or
three-flip neighbours leaves the 15 strongest couplings unchanged (coupling correlations 0.989–0.997 with
the published fit), keeps 130–136 of the 150 most probable configurations, and leaves the local
neighbourhoods of the Free Methodist Church and the Roman Imperial Cult as the paper describes them. But
the hierarchical clustering into three groups (Figure 4A, Tables 5 and A2–A4) is fragile: with the
*published* parameters, clustering the top 149 or 151 configurations instead of 150 already changes the
groups substantially. The refits change the groups by a similar amount.

**Other findings.** The sampler used for the paper's simulations was not at equilibrium for strong
couplings (re-running the three simulation tables with exact sampling gives the same qualitative results).
Cross-validation started each fold from a fit that had seen the held-out row, and counted the fractional
rows of inconsistently coded entries as whole observations. Several software defects were fixed (wrong
results above 31 questions, a thread race, leaks, crashes). None changes a number in the paper that could
be checked.

**The code is roughly 50–150 times faster** (a cross-validated fit of the DRH data takes under a minute
instead of an estimated 35–40), has a test suite, and the whole analysis can be regenerated from a
parameter file with one command.

## 2. What to use

```
cd MPF_CMU/src && make && make test
```

| purpose | command |
|---|---|
| Fit data whose blanks may depend on other, observed answers (as the DRH's do) | `./mpf -c DATA 1 --frozen --groups GROUPS` |
| Reproduce the published fit | `./mpf -c DATA 1 --legacy` |
| Complete data only, or blanks known to be unrelated to answers | `./mpf -c DATA 1` |
| Two- or three-flip neighbourhoods | replace `1` by `2` or `3` |
| Regenerate every figure and table from a parameter file | `python analysis/regenerate.py PARAMS OUTDIR` |

`GROUPS` has one integer per data row; rows from the same DRH entry (the expansions of an inconsistently
coded entry) share an id and are held out together in cross-validation. `MPF_CMU/src/README.md` documents all
options, and a table there states what each missing-data option assumes.

## 3. Missing data: the "slight tension" and two replacements

### What the paper says

Section 2.3.5 expands an observation with *m* missing answers into its 2^m completions, weights each
completion by its probability under the current parameters, and minimises the weighted flow. It notes that
because "the weights depend on θ, this changes the form of the derivative", and that "minimizing K(θ) will
be in slight tension with the new (exact) inference step".

### What the tension is

Write *o* for the observed answers of a row, *c* for a completion of its missing answers,
w_c(θ) = p_θ(c | o) for the completion weights, and

    T_ck = exp( ½ [ H(o^k, c) − H(o, c) ] )

for the flow out of the completed state along observed question *k* (o^k is *o* with answer *k* flipped;
H is the log-probability up to a constant, as in Eq. 2 of the paper). The published objective for the row is

    K_o(θ) = Σ_c w_c(θ) Σ_k T_ck(θ).                                   (published)

Its gradient has two parts:

    ∇K_o = Σ_c w_c Σ_k ∇T_ck   +   Σ_c (∇w_c) Σ_k T_ck
         =  [flow gradient, weights held fixed]  +  Cov_{c|o}( f , g ),

where f is the vector of sufficient statistics (σ_a σ_b and σ_a), g_c = Σ_k T_ck is the total flow out of
completion *c*, and the covariance is over completions (using ∇w_c = w_c (f_c − ⟨f⟩)).

With unlimited data, blanks unrelated to the answers, and θ at its true value, the first part vanishes.
The second does not. It is the gradient of a different incentive: *shift weight onto completions out of
which little probability flows* — choose parameters under which the missing answers make the observed
answers look like a local peak. The optimiser can lower K that way without fitting anything.

A two-question example shows the direction. With question 1 observed and question 2 missing, the second
part of ∂K/∂J at the true parameters is

    − sinh(J) · Σ_{σ1} p(σ1) e^{−σ1 h1} sech²(J σ1 + h2),

which always has the opposite sign to J: K can always be lowered by making the coupling between an observed
and a missing question stronger than it is. So the "tension" is a bias, and more data does not remove it.

### Two ways to remove it

**(a) Marginal MPF (`--marginal`).** Take the square root outside the average over completions:

    K_o(θ) = Σ_k sqrt( Σ_c w_c(θ) T_ck(θ)² ) = Σ_k sqrt( p_θ(o^k) / p_θ(o) ),

where p_θ(o) is the model's marginal probability of the observed answers (the identity holds because
Σ_c w_c exp[H(o^k,c) − H(o,c)] = Σ_c e^{H(o^k,c)} / Σ_c e^{H(o,c)}). This is the MPF objective for the
marginal distribution over the questions that were answered, with flows along those questions — the way
the original MPF paper treats hidden units. By Jensen's inequality the published objective is ≤ this one;
they coincide when nothing is missing. Its gradient is ∇ sqrt(r) = ½ sqrt(r) (⟨f⟩_{c|o^k} − ⟨f⟩_{c|o}),
with r = p_θ(o^k)/p_θ(o).

**(b) Frozen-weight iteration (`--frozen`).** Drop the second part of the gradient: solve

    Σ_o Σ_c p_θ(c | o) ∇_θ [ Σ_A T_cA(θ) ] + penalty = 0

by holding the weights fixed, taking a few minimisation steps on the (convex) frozen-weight flow objective,
re-estimating the weights, and repeating until the residual of this equation is small. This is an
estimating equation, not the gradient of an objective. It is not likelihood EM — there is no likelihood
bound behind it and no monotonicity guarantee — and it does not give the same finite-sample answer as (a).

Here A ranges over the sets of answers that are flipped together (single answers for NN=1). For NN=1, A is
an observed answer. For NN>1, A may be any set that contains at least one observed answer, *including sets
that also flip missing answers*: see below for why.

### What each one assumes

Call the blanks *MCAR* (missing completely at random) if which answers are blank in a row is unrelated to
any of its answers, and *MAR* (missing at random) if it may depend on the answers observed in that row but
not on the missing ones.

**Marginal MPF: the truth minimises the population objective under MCAR, not under MAR.** For a fixed pattern of blanks, pair the two
directions of each edge (o, o^k) of the observed sub-cube. With row probabilities p_a, p_b and model ratio
r = p_θ(o^k)/p_θ(o), their contribution is p_a √r + p_b/√r ≥ 2 √(p_a p_b), with equality exactly at
r = p_b/p_a. So the unpenalised population objective is minimised, globally, exactly by the parameters
whose observed-answer marginals match those of the rows with that pattern. Under MCAR those rows are a
random sample of the population, so the truth is a global minimiser, and the only one if the patterns
together identify the model (for example, if complete rows occur). Under MAR the rows with a given pattern
are a *selected* sample — their distribution is proportional to p(o) P(pattern | o) — and the minimiser
matches that instead.

**Frozen weights: the truth solves the population equation under MAR, for any neighbourhood.** Under MAR, a row's
observed part combined with the model's distribution over its completions reproduces, summed over all
patterns of blanks, the true joint distribution: Σ_patterns P(pattern | x) p(x) = p(x). So if every
completed state sends flows to *all* its neighbours, the left side of the equation above, at the true
parameters, is the complete-data MPF equation evaluated on the true distribution, which vanishes for any
symmetric neighbourhood. (This is the "expected estimating equation" construction for missing data.) Sets A
that lie wholly inside the missing answers cancel identically within a row and can be dropped; for NN=1
that leaves exactly the observed single flips. If instead flips are restricted to observed answers when
NN>1, the weight carried from x to x^A is p(x) P(all of A observed | x), which is not symmetric — whether
k is observed may depend on the value of l — and the method is biased; the first implementation here made
that mistake and was off by 0.25 in the parameters on an exact test case.

This shows the truth solves the equation; it does not show that the solution is unique or that the
iteration converges to it.

**The published objective** has neither property. **None of the three** is right if whether an
answer is missing depends on that answer. And fitting only the complete rows — stage one of every fit, and
the paper's "128 full" baseline — is itself biased under MAR, because which rows are complete then depends
on their content.

What these statements do not cover: all three are non-convex with missing data, so the optimiser returns a
local solution (found from the complete-rows fit); the penalty is proportional to the amount of data, so
the regularised estimate converges on the truth only if λ → 0 (the search for λ no longer stops at 0.01,
but nothing proves cross-validation drives λ to zero); and identifiability needs patterns of blanks that
between them inform every coupling.

### Evidence

*Exact population data (no sampling)* — largest error in any parameter (`MPF_CMU/src/tests/test_mpf.c`):

| setting | published | marginal | frozen |
|---|---|---|---|
| MCAR, 5 questions, NN=1 | 0.14 | 4e-9 | 7e-7 |
| MAR, 2 questions (the Codex review's example), NN=1 | 0.49 | 0.49 | 2e-7 |
| MAR, 4 questions, NN=1 | 0.44 | 0.19 | 1e-6 |
| MCAR, 4 questions, NN=2 | 0.17 | 3e-9 | 3e-7 |
| MAR, 4 questions, NN=2 | 0.55 | 0.32 | 2e-6 |

Independently of the C code, `validation/population_consistency.py` finds couplings inflated by 11–21% for
the published objective at 6 and 8 questions under MCAR, and exact recovery for the other two.

*Large samples, 20 questions* — KL from the truth at each method's best λ on a grid (an oracle choice, so
that objectives are compared rather than cross-validation; `validation/large_sample.py`):

| blanks | NN | rows | landscapes | complete rows only | published | marginal | frozen |
|---|---|---|---|---|---|---|---|
| MCAR | 1 | 16,384 | 6 | 0.015–0.024 | 0.015–0.032 (1–50% *worse* than complete rows only) | 0.009–0.014 | 0.009–0.017 |
| MAR | 1 | 16,384 | 4 | 0.23–0.32 | 0.08–0.82 | 0.025–0.041 | 0.011–0.015 |
| MCAR | 1 | 2,048 | 4 | 0.09–0.13 | 0.08–0.10 | 0.07–0.08 | 0.07–0.08 |
| MAR | 2 | 4,096 | 3 | 0.27–0.33 | 0.10–0.13 | 0.051–0.062 | 0.042–0.058 |
| MCAR | 2 | 4,096 | 1 | 0.073 | 0.051 | 0.041 | 0.040 |

(MCAR here: half the rows have 5 of 20 answers blanked at random. MAR: questions 16–20 are blanked with
probability 0.85 or 0.15 depending on questions 1–5.)

*The paper's Table 3 setting* — 128 complete + 128 partial rows, MCAR, 64 landscapes, each method at its
best λ (`validation/partial_oracle_trial.py`); the cross-validated version is in Section 6:

| regime | 128 full | published | marginal | frozen | 256 full | marginal − published | frozen − published |
|---|---|---|---|---|---|---|---|
| dispersed | 0.277 | 0.215 | 0.215 | 0.213 | 0.191 | −0.000 ± 0.001 | −0.002 ± 0.002 |
| ordered | 0.567 | 0.393 | 0.391 | 0.390 | 0.341 | −0.002 ± 0.003 | −0.003 ± 0.003 |
| near-critical | 0.662 | 0.454 | 0.462 | 0.469 | 0.422 | +0.008 ± 0.003 | +0.015 ± 0.004 |
| critical | 0.680 | 0.430 | 0.440 | 0.477 | 0.466 | +0.010 ± 0.010 | +0.047 ± 0.013 |

*Data like the DRH* — 200 datasets of 407 entries drawn from a fitted DRH model, given the real patterns of
blanks either at random (MCAR) or according to the simulated answer to "Scriptures" (MAR, mimicking the
strongest dependence in the real data), each method at its best λ (`validation/drh_bootstrap.py`):

| blanks | complete rows only | published | marginal | frozen | frozen − published | frozen − marginal |
|---|---|---|---|---|---|---|
| MCAR | 0.340 | 0.2117 | 0.2086 | 0.2089 | −0.0029 ± 0.0004 | +0.0003 ± 0.0003 |
| MAR | 0.512 | 0.2117 | 0.2083 | 0.2088 | −0.0029 ± 0.0004 | +0.0005 ± 0.0003 |

Reading these together: with a few hundred rows sampling noise dominates. The published objective's bias
towards stronger couplings then does little harm, and when couplings are strong it helps slightly, while
the frozen-weight iteration is the worst of the three by up to 0.05. On data shaped like the DRH the two
replacements are 1.5% better than the published objective and indistinguishable from each other. With
thousands of rows both replacements beat the published objective in every setting tested, and under the
MAR mechanism tested the frozen-weight iteration is clearly the best while the published objective and
complete-rows-only fitting are far off. Between the two replacements there is no general order: under MCAR
the marginal objective is sometimes ahead (by 24% in one 16,384-row case).

*A node that is never observed* — the extreme case of missing data, and the way hidden units are added to
these models. On exact population data from six models with 5 visible nodes and 1 hidden node, without
regularisation (`validation/hidden_node_check.py`, independent of the C code): the marginal objective and
the frozen-weight equation are stationary at the truth (gradient below 2e-9), and an optimiser started
there stays there. The published objective is not stationary at the truth (gradient 0.06–0.32). That is
the evidence of bias, and it does not depend on any optimisation. To see its size, an optimiser started at
the truth was followed: it ends with the couplings to the hidden node 20–230 times their true size, the
objective 13–51% lower than at the truth, and the distribution of the visible nodes wrong by a KL of 0.009
to 12. Those end points have gradients below 4e-6 and the objective rises again beyond them, but the
optimiser reported convergence in only two of the six cases, so they are approximate, not established
minima. Regularisation limits this in practice, and none of it shows what the corrected methods do from an
ordinary starting point; but fits with hidden nodes made with the published objective should be treated
with caution.

### Are the DRH blanks related to the answers?

Yes (`validation/drh_missingness.py`). Of 407 entries, 226 have at least one blank. Comparing entries
that are complete elsewhere with entries that have a blank elsewhere, the "yes" rate differs for 4 of the
20 questions after Bonferroni correction: Scriptures 0.89 vs 0.53 (p ≈ 1e-15), Grave goods 0.34 vs 0.60,
Co-sacrifices 0.09 vs 0.23, Official political support 0.53 vs 0.68. Testing "question k is missing"
against the answer to question q, 51 of 341 pairs are significant at 0.05 (17 expected by chance) and 7
survive Bonferroni correction; the strongest is that when "Distinct written language" is missing,
"Scriptures" is yes 12% of the time, otherwise 77%.

So MCAR is rejected for the DRH. Whether the blanks are MAR — dependent only on observed answers — cannot
be tested from the data; it is the weaker assumption, and `--frozen` is the method that needs only it.

### What this means for the paper

* Table 3 and the DRH parameters are not materially affected.
* Section 2.3.5 should say that the weighted-flow form is biased, give one of the replacements, and state
  the assumption about missingness. The comparison with EM should not suggest that differentiating through
  the weights is the better choice.
* The title of Section 4.3 ("...can be consistently integrated into inference") holds for the replacements
  under their assumptions, not for the published objective.

## 4. Neighbourhood size (NN = 1, 2, 3)

`NN=k` sends flows from each data state to every state within Hamming distance k: n neighbours per state
for NN=1, n + n(n−1)/2 for NN=2 (210 at n=20), and 1,350 for NN=3. The published code had the two- and
three-flip blocks commented out, so `NN=2` silently ran as `NN=1`; the paper's LaTeX source has
commented-out N₂ rows for Table 1, so N₂ had been run with an earlier version.

It is implemented again, for all three treatments of missing data, with a multi-flip flow computed from
the same local fields as single flips (the flow for flipping a set A is
exp(−Σ_{k∈A} x_k + 2 Σ_{k<l∈A} J_kl s_k s_l), with x_k = s_k × the local field). On complete data the new
code agrees to 2e-7 with the published code after its commented-out blocks are re-enabled, for NN=1, 2, 3.

Table 1 of the paper re-run on the same 160 landscapes at each NN (20 questions, 128 observations, exact
sampling; KL from the truth with cross-validated λ):

| regime | NN=1 | NN=2 | NN=3 | NN=2 − NN=1 | NN=3 − NN=1 | paper, N₁ | paper's commented-out N₂ row |
|---|---|---|---|---|---|---|---|
| dispersed | 0.212 | 0.212 | 0.212 | −0.000 ± 0.000 | +0.000 ± 0.001 | 0.23 | 0.24 |
| ordered | 0.564 | 0.565 | 0.576 | +0.001 ± 0.002 | +0.012 ± 0.003 | 0.56 | 0.57 |
| near-critical | 0.631 | 0.578 | 0.568 | −0.053 ± 0.008 | −0.063 ± 0.012 | 0.63 | 0.59 |
| critical | 0.538 | 0.430 | 0.398 | −0.109 ± 0.021 | −0.140 ± 0.033 | 0.54 | 0.44 |

The NN=2 column agrees with the paper's commented-out N₂ numbers, which suggests, but does not establish,
that those came from a working two-flip implementation. The caption of the paper's Table 1 says N₂ "does not improve significantly over the simpler
N₁"; that holds in the two weakly coupled regimes and not in the two strongly coupled ones.

## 5. The DRH landscape refitted

All fits use the 455-row, 20-question data file of the paper (`validation/results/drh_refits/refit.sh`).

| fit | log₁₀λ | coupling correlation with published | 15 strongest couplings shared | top 150 configurations shared | mass of top 150 | Spartans: ritual, no sacrifice | time |
|---|---|---|---|---|---|---|---|
| published parameter file | — | — | — | — | 0.422 | 69.7% | — |
| `--legacy` (published procedure) | 0.207 | 1.0000 | 15 | 150 | 0.422 | 69.7% | 17 s |
| marginal | 0.170 | 0.9990 | 15 | 143 | 0.427 | 72.0% | 19 s |
| marginal, entry-wise CV | 0.195 | 0.9990 | 15 | 141 | 0.424 | 72.2% | 19 s |
| **frozen, entry-wise CV, NN=1** | 0.128 | 0.9971 | 15 | 136 | 0.443 | 74.3% | 49 s |
| **frozen, entry-wise CV, NN=2** | 0.216 | 0.9951 | 15 | 135 | 0.456 | 72.0% | 6 min |
| **frozen, entry-wise CV, NN=3** | 0.214 | 0.9890 | 15 | 130 | 0.473 | 72.1% | 41 min |
| `--legacy`, NN=2 | 0.258 | 0.9955 | 15 | 141 | 0.442 | 69.0% | 69 s |

(Times on a shared 256-core machine with 40 threads; on an 18-core laptop the NN=1 fits take 45 s to
3 minutes. No fit produced a convergence warning. All seven fits are from the final program.)

### Frozen-weight fits at NN = 1, 2, 3 compared

| | NN=1 vs NN=2 | NN=1 vs NN=3 | NN=2 vs NN=3 |
|---|---|---|---|
| correlation of couplings | 0.996 | 0.989 | 0.998 |
| largest change in any parameter | 0.060 | 0.101 | 0.048 |
| 15 strongest couplings shared | 15 | 15 | 15 |
| top 150 configurations shared | 139 | 135 | 145 |
| agreement of the three groups (adjusted Rand index) | 0.22 | 0.25 | 0.43 |

Local structure is the same at every NN and in the published fit:

| | published | frozen NN=1 | frozen NN=2 | frozen NN=3 |
|---|---|---|---|---|
| Free Methodist Church: probability, rank | 0.55%, 17 | 0.52%, 18 | 0.43%, 27 | 0.36%, 35 |
| ... neighbours more probable than it | 0 of 20 | 0 | 0 | 0 |
| Roman Imperial Cult: probability, rank | 0.41%, 25 | 0.42%, 28 | 0.42%, 29 | 0.42%, 30 |
| ... neighbours more probable than it | 2 of 20 | 2 | 2 | 2 |
| ... the two: lose distinct written language; gain scriptures | 31%, 23% | 30%, 23% | 30%, 25% | 30%, 27% |
| Archaic Spartan Cults: ritual & no child sacrifice | 69.7% | 74.3% | 72.0% | 72.1% |

So the Free Methodist Church is a local peak and the Roman Imperial Cult sits below the same two
neighbours in every fit, as the paper says. The differences between NN=1, 2 and 3 are small and of the same
size as the differences between the treatments of missing data; the larger neighbourhoods put slightly more
mass on the most probable configurations (0.44 → 0.47 for the top 150). On the simulations of Section 4,
larger neighbourhoods help only when couplings are strong, which is not the DRH's regime, so there is no
evidence here for preferring NN>1 for these data, and no sign that the choice matters for the paper's
conclusions about parameters and local landscapes.

### The grouping is fragile

The paper groups the 150 most probable configurations by Ward hierarchical clustering (Figure A1) and
describes three groups (Figure 4A; Tables 5, A2–A4). The tree depends only on *which* configurations are in
the top 150, not on their probabilities. Agreement with the published three groups, as an adjusted Rand
index on shared configurations (1 = identical, 0 = chance; `validation/clustering_sensitivity.py`):

| analysis | adjusted Rand index | clusters at the paper's cut height |
|---|---|---|
| published parameters, top 149 instead of 150 | 0.60 | 3 |
| published parameters, top 151 | 0.59 | 3 |
| published parameters, top 140 / 145 / 155 / 160 | 0.26 / 0.44 / 0.26 / 0.41 | 5 / 4 / 4 / 4 |
| published parameters, top 100 / 120 / 130 / 170 / 200 | 0.50 / 0.43 / 0.73 / 0.62 / 0.46 | 3 / 6 / 4 / 5 / 4 |
| `--legacy` refit | 1.00 | 5 |
| marginal; marginal with entry-wise CV | 0.38; 0.41 | 5; 4 |
| frozen, NN=1 / 2 / 3 | 0.39 / 0.31 / 0.29 | 4 / 5 / 5 |

Adding or removing a single configuration changes the groups about as much as refitting does. What is
stable is the largest group (Cistercians, Ancient Egypt, Jesuits, Islam in Aceh, ...): under the frozen NN=1
refit 53 of its 56 configurations stay together. The other two groups reorganise; for example, the
Jehovah's Witnesses and Free Methodists remain together but separate from the Church of England, and the
Roman Imperial Cult and Mesopotamia join the largest group. The distinctive features reported for the
groups (Table A2) change accordingly: "small-scale rituals" and "large-scale rituals" still separate the
largest group from the rest in every fit, while the features listed for the second and third groups do not
carry over.

The group structure of the landscape should therefore be read as one of several comparably good ways of
cutting it, not as a property that the data pin down. The comparisons are in `regenerated/`.

### The regenerated analyses

`analysis/regenerate.py PARAMS OUTDIR` runs the whole pipeline (preprocessing, then every figure and table)
from a parameter file in a temporary copy, without touching the published `data/`, `fig/` and `tables/`.
`regenerated/` holds the result for the published parameters and for the three frozen-weight fits, with
pairwise comparison files.

Two things had to change for this to be possible. The pipeline no longer needs Julia
(`preprocessing/expand_data.py` reproduces `expand_data.jl`'s committed output). And the figure and table
scripts hard-coded choices that belonged to the published fit — which clusters merge into which group,
label offsets by rank, rows by position. They now take the three groups from the dendrogram's top-level
branches and place labels by configuration or entry. For the published parameters this reproduces the
committed tables byte for byte and leaves the figures unchanged; for other fits it is what makes the
scripts run. Label positions were tuned by hand for the published layout, so labels in the refit figures
may overlap.

## 6. The paper's simulation tables re-run

All with 20 questions; KL from the truth, mean ± standard error.

**Table 1** (regularisation and cross-validation; 160 landscapes per sampler; 128 observations):

| regime | paper: optimal / CV / none | published sampler: optimal | CV | exact sampling: optimal | CV | no regularisation (median) |
|---|---|---|---|---|---|---|
| dispersed | 0.22 / 0.23 / 1.2 | 0.209 ± 0.023 | 0.215 ± 0.023 | 0.207 ± 0.022 | 0.212 ± 0.022 | 1.2 |
| ordered | 0.55 / 0.56 / 2.3 | 0.557 ± 0.011 | 0.564 ± 0.011 | 0.560 ± 0.012 | 0.564 ± 0.012 | 1.7 |
| near-critical | 0.62 / 0.63 / 19.4 | 0.609 ± 0.021 | 0.619 ± 0.021 | 0.621 ± 0.025 | 0.631 ± 0.026 | about 100 |
| critical | 0.50 / 0.54 / 9.5 | 0.476 ± 0.059 | 0.504 ± 0.059 | 0.522 ± 0.057 | 0.538 ± 0.057 | about 200 |

Cross-validation is within 0.005–0.03 of the best possible λ in every regime, with either sampler; the
sampler makes no visible difference here. Without regularisation the fits have no finite optimum when the
data are separable: the values in the last column, like the paper's 19.4 and 9.5, reflect where the
optimiser stopped and should be read as "very large".

**Table 2** (re-weighting to correct sampling bias; 80 landscapes; exact sampling):

| regime | ideal KL | corrected | naive | bias after correction | bias, naive |
|---|---|---|---|---|---|
| dispersed | 0.13 | 0.16 | 0.16 | −0.1% | −14% |
| ordered | 0.34 | 0.45 | 0.46 | +2% | −39% |
| near-critical | 0.46 | 0.60 | 0.63 | +15% | −49% |
| critical | 0.50 | 0.64 | 0.73 | +8% | −59% |

(Paper: KL 0.13/0.16/0.16, 0.34/0.45/0.46, 0.43/0.55/0.60, 0.48/0.57/0.71; bias −0.2%/−14%, −0.1%/−40%,
9.6%/−51%, 0.1%/−65%.) Re-weighting removes most of the bias in every regime; in the two strongest regimes
a positive bias remains after correction (mean log bias +0.14 ± 0.06 and +0.07 ± 0.11).

**Table 3** (missing data; exact sampling; λ by cross-validation; 16 landscapes per regime):

| regime | 128 full | published Partial-MPF | marginal | frozen | naive imputation | 256 full |
|---|---|---|---|---|---|---|
| dispersed | 0.274 ± 0.050 | 0.214 ± 0.037 | 0.213 ± 0.037 | 0.212 ± 0.037 | 0.289 ± 0.041 | 0.191 ± 0.033 |
| ordered | 0.570 ± 0.019 | 0.394 ± 0.011 | 0.392 ± 0.010 | 0.391 ± 0.010 | 0.574 ± 0.022 | 0.341 ± 0.009 |
| near-critical | 0.670 ± 0.047 | 0.460 ± 0.025 | 0.471 ± 0.026 | 0.477 ± 0.027 | 0.837 ± 0.074 | 0.426 ± 0.024 |
| critical | 0.732 ± 0.133 | 0.461 ± 0.097 | 0.470 ± 0.083 | 0.498 ± 0.082 | 0.984 ± 0.191 | 0.491 ± 0.090 |

Paired differences on the same landscapes:

| | dispersed | ordered | near-critical | critical |
|---|---|---|---|---|
| marginal − published | −0.001 ± 0.001 | −0.001 ± 0.002 | +0.010 ± 0.004 | +0.009 ± 0.030 |
| frozen − published | −0.001 ± 0.001 | −0.002 ± 0.002 | +0.017 ± 0.004 | +0.038 ± 0.030 |
| published − 128 full | −0.060 ± 0.013 | −0.176 ± 0.013 | −0.210 ± 0.025 | −0.272 ± 0.073 |
| published − 256 full | +0.023 ± 0.006 | +0.052 ± 0.007 | +0.035 ± 0.009 | −0.030 ± 0.043 |
| naive − 128 full | +0.015 ± 0.011 | +0.004 ± 0.023 | +0.167 ± 0.050 | +0.252 ± 0.112 |

(The marginal and frozen-weight columns use the cross-validation described in Section 10, N1. Two of the
critical-regime frozen-weight searches took four and six and a half hours, against a median of about
twenty minutes: on those landscapes almost every cross-validation fit at small λ fails to converge.)

Partial data beat 128 complete rows in 60 of 64 landscapes with the published objective. Naive imputation
is worse than Partial-MPF in 62 of 64 landscapes, and worse than not using the partial rows at all in the
two strongest regimes. The three ways of treating the partial rows are indistinguishable in the two weakly
coupled regimes; when couplings are strong the published objective is best and the frozen-weight iteration
worst: by 0.01–0.02 in the near-critical regime, and by amounts within the noise in the critical one. At least
one cross-validation fold failed to converge at some λ tried in about a third of the searches (mostly in the
two strongest regimes), but at the λ finally chosen every fold converged in all 64 marginal searches and in
62 of the 64 frozen-weight ones; both counts are in the result files.

The paper's Table 3 appears to combine two simulation runs: a commented-out row in the arXiv source,
labelled "slightly updated methods gives slightly better results", has the published Partial-MPF and naive
columns but different "full data" columns (256-full 0.42 and 0.44 in the two strongest regimes, where the
table prints 0.40 and 0.38). The simulation logs in `MPF_CMU/paper_runs/NAN_TESTS` (519 trials) agree with the
commented-out row. This is an inference from the source and logs.

**Speed** (DRH data; published code against the revised code, 18-core laptop):

| | published | revised |
|---|---|---|
| one objective-and-gradient evaluation | 9.9 ms | 0.5–1 ms |
| fit at fixed λ | 5.8 s (about 900 evaluations) | 0.03–0.04 s (about 70) |
| log Z for 20 questions | 0.104 s | 0.005 s |
| cross-validated fit, 201 complete rows | 247 s | 4–6 s |
| cross-validated fit, all 455 rows, published procedure | 35–40 min (estimated from one evaluation taking 85 s; the search makes 46) | 45 s |

These are single timings.

## 7. Errors found, and whether they reach the paper

### 7a. Errors that touch a statement or number in the paper

| # | error | effect on the paper | status |
|---|---|---|---|
| 1 | Partial-MPF is biased (Section 3) | No material effect on Table 3 or the DRH parameters at these sample sizes; the description and the word "consistently" need correcting | two replacements; `--legacy` keeps the old form |
| 2 | The claim that N₂ "does not improve significantly" (Table 1 caption) | Not true for strong couplings: N₂ lowers KL by 8% and 20% in the two strongest regimes. True for the weak ones, including the DRH's (Section 4) | NN=1–6 implemented |
| 3 | The three groups of Figure 4A / Tables 5, A2–A4 are not stable to the number of configurations clustered (Section 5) | The group descriptions rest on one of several comparably good partitions. The largest group is the stable part | documented; `validation/clustering_sensitivity.py` |
| 4 | The simulation sampler (`mpf -g`: 500 annealing + 500 equilibrium sweeps per sample) does not reach equilibrium for strong couplings. It depends on the landscape: at β ≥ 0.5 some are fine and others differ from the exact distribution by 4 to 88 standard errors; at β ≤ 0.35 every landscape tested was fine | Affects the "critical" (and marginally "near-critical") rows of Tables 1–3. Re-running them with exact sampling gives the same qualitative results (Section 6) | exact sampling for n ≤ 24; `--mcmc` keeps the old sampler |
| 5 | "Standard MPF" (no regularisation) was not converged: the minimiser stopped at its 3,000-iteration cap. There is often no finite optimum | The third column of Table 1 reflects where the optimiser stopped. The conclusion that regularisation is essential only gets stronger | the outcome of the unregularised fit is printed by `-o` |
| 6 | Table 3 appears to combine two simulation runs (Section 6) | In the two strongest regimes the gap between partial and complete data is within noise; the qualitative claim holds | documented |
| 7 | Equation 8 has the wrong sign: it should read K′ = K **+** λ\|D\|\|N\| Σθ²/2. The code also divides the penalty by the number of parameters and takes λ as a base-10 logarithm, which the paper does not say | Text only; the code was right | documented in the README |
| 8 | Section 2.3.2 says minimising K maximises "the posterior log-probability of the data". K is a surrogate for the negative log-likelihood, not a posterior and not the exact likelihood. Table 1's caption calls the L2 penalty "sparsity" | Text only | — |
| 9 | "Since \|N(j)\| is the same for all j, this provides equal weighting to all data states" is not true once rows have missing answers | Text only | — |
| 10 | "In about 17% of religious groups ... we have inconsistent coding". In the curated file, 24 of 407 entries (5.9%) are expanded into more than one row. The 17% may refer to an earlier or larger subset | Unresolved | — |
| 11 | Section 4 says data were simulated with "the Metropolis–Hastings algorithm"; the sampler is a heat-bath (Glauber) single-spin-flip sampler with annealing | Text only | — |
| 12 | Cross-validation on the DRH data is leave-one-*row*-out, but 30 of the 201 complete rows are fractional-weight expansions of an inconsistently coded entry: their sibling rows stay in the training set, and each counts as a full observation | Small. With the marginal objective, weighted row-wise scoring selects log₁₀λ = 0.170 and holding out whole entries 0.195 | weighted scoring by default; `--groups` for entry-wise folds |
| 13 | Every cross-validation fold with missing data started from one fit to all the complete rows, which had seen the held-out row, made at a λ that had also been chosen with its help. The problem is not convex, so the start can select the local solution | Small. On the simulations it matters only in the critical regime; on the DRH the λ chosen moves in the fourth decimal | each fold now makes both stages of the fit from its own training rows, at the λ being scored; `--legacy` keeps the published behaviour |
| 14 | Complete rows are not a representative sample of the DRH (Section 3), yet they are what cross-validation scores and what stage one of the fit uses | Not quantified. The frozen-weight fit is consistent whatever its starting point if it finds the right solution; the choice of λ is still scored on complete entries only | not fixed (Section 11) |

### 7b. Bugs that do not reach the paper

| # | where (published code) | bug | why the paper is unaffected | status |
|---|---|---|---|---|
| 15 | `mpf.c`, `mcmc.c`: `1 << pos` | States were built with 32-bit shifts, so with more than 31 questions the program silently used wrong configurations (two rows one flip apart at n=40 had 64 distinct neighbours instead of 80) | The paper uses n=20. **The n=40 fits in `data/mdl_experiments` are invalid** | fixed (64-bit); tested to n=62 |
| 16 | `mcmc.c: log_l` | For n > 20 the held-out likelihood was estimated by counting how often 10⁶ six-sweep chains landed on the held-out state, which cannot resolve probabilities of 10⁻⁹; the retry loop never updated its counter | n=20 uses exact enumeration. The n=30 fits in `data/mdl_experiments` (two with λ exactly 1.0, the search's starting value) should be treated as unreliable | exact up to n=24; held-out pseudo-likelihood above. That is a different criterion; it agreed with exact cross-validation on six n=20 landscapes (KL within 3.5%) and is otherwise unvalidated |
| 17 | `mpf.c: cross` | The loop variable `i` was shared between threads while building the held-out configuration | A race in principle. The published fit reproduces, which bounds its effect there; it cannot be excluded for every historical run | fixed |
| 18 | `mpf.c: cross`, `nlin.c` | Every cross-validation fold allocated a full copy of the data and never freed it | Memory only | fixed |
| 19 | `mpf.c: simple_minimizer` | The status returned by GSL's `iterate()` was overwritten, so "no progress" was ignored; each K-and-gradient call computed K twice | Regularised fits did converge. See #5 for the unregularised case | L-BFGS with an explicit status: converged / stalled / iteration limit / numerical failure |
| 20 | `nlin.c: minimize_kl` | The return value of `gsl_min_fminimizer_set` was ignored. GSL rejects the bracket (−2, 1, 4) whenever the score at 1.0 is not better than at both ends — roughly a quarter of the searches in the simulations | The search still behaved sensibly when the curve has one minimum, and the DRH bracket was valid | grid, then golden section in a real bracket; warns at the edge of the range |
| 21 | `main.c -t` | Read the data from an uninitialised filename; segfault | Mode not used for the paper | fixed |
| 22 | `main.c -o`, `-t` | `cv->best_fit` passed uninitialised (it happened to be NULL) | Worked by luck | fixed |
| 23 | `main.c` | `exit(1)` on success; no checks on missing files, malformed rows or wrong argument counts (`NN=0` segfaulted) | — | exit 0 on success; exit 2 with a message for every error tested |
| 24 | `nsb.c` | The NSB entropy takes the number of *observed* states as the alphabet size, so it is not an estimate of the entropy over all 2^n configurations | Printed by `-o` only; not used in the paper | documented, not changed |
| 25 | three helper scripts (`best_fit_DRH_2.rb`, `cv_scan.rb`, `param_scan.rb`) | Call `-c` / `-t` with the argument order of an earlier version of the program | The scripts behind the tables match the current interface. Run today, every script uses the new defaults unless given `--legacy` and `--mcmc` (and NN=1) | every script that calls `mpf` carries a note; none was repaired |
| 26 | `preprocessing/prep_early.py` | Expected column names that the committed reference file does not have; the script failed on the committed data | The committed outputs predate the refactoring and are correct (#30) | accepts both |
| 27 | `preprocessing/expand_data.jl` | Wrote `data_expanded_2.csv`; `prep_late.py` reads `data_expanded.csv` | The committed `data_expanded.csv` is what the script computes | fixed (not executed: no Julia); `expand_data.py` added and tested |
| 28 | `preprocessing/curation_functions.py` | Wrote `..._rows_...` file names where everything downstream expects `..._nrows_...`; dropped any question that every entry answered, and then every entry if all selected questions were fully answered | The committed files have the right names; no DRH question is answered by every entry | fixed |
| 29 | `analysis/` figure and table scripts | Hard-coded groupings, label offsets by rank and rows by position (Section 5); two scripts failed under current pandas / matplotlib | Correct for the published fit under the pinned versions | generalised; published tables still reproduced byte for byte |
| 30 | — | (check, not a bug) The committed data products, the Spartan table and all six tables agree with the committed parameter file under the pinned versions | — | now regression-tested |
| 31 | `analysis/fig_A1_dendrogram.py` | With current scikit-learn/scipy the published parameters give a different dendrogram (four clusters, a different partition): the clustering depends on tie-breaking inside the library | The paper used the pinned versions, under which it reproduces | documented; see #3 |
| 32 | `analysis/table_5_annotations.py` | The third column of the generated `tables/table_5_annotations.txt` is the DRH name of the configuration's most heavily weighted entry, not of the labelled entry (it pairs "Cistercians" with "Greek Chalcedonian Christians, Nicaea", which share a configuration); the column headers are not renamed | Table 5 in the paper has the right names, so it was corrected by hand | documented, not changed (the committed table is still what the script produces) |

## 8. What changed in the repository

**C code (`MPF_CMU/src`)** — `mpf.h`, `mpf.c`, `mcmc.c`, `nlin.c`, `nsb.c`, `main.c`, `Makefile` rewritten; `rng.c` new. The
seven command-line modes, the file formats, and the output lines the helper scripts parse are unchanged.

* Objective and gradient from local fields: all n neighbours of a state cost O(n²) rather than O(n³).
  In `--legacy` mode it agrees with the published function to 3e-16.
* Numerics in the log domain; a flow that overflows a double is an error, not a "fit".
* Three treatments of missing data (`--marginal`, `--frozen`, `--legacy`) and neighbourhoods NN = 1–6.
* L-BFGS minimiser with an explicit status for every fit; deterministic (starts from zero).
* Partition function and KL by Gray-code enumeration with a running log-sum-exp.
* Cross-validation: weighted scoring; `--groups`; folds built from their own training rows; unconverged
  folds reported; bit-identical for any number of threads; no leaks.
* λ search: coarse grid on [−2, 4], extended downwards while the score improves, then golden section.
* Exact sampling for n ≤ 24; `--seed`.
* Input checking with exit status 0 / 2; limits explicit (62 questions, 20 blanks per row).
* No external libraries. The GNU Scientific Library was used only for random numbers and for the NSB
  entropy printed by `-o`; both are now done by code in `rng.c` and `nsb.c`. The random streams are
  identical to GSL's, so `mpf -g --seed N` writes the same files as before, and fits are unchanged.
  The Makefile uses OpenMP if the compiler has it and builds a single-threaded program otherwise.

**Layout of `MPF_CMU`** — `src/` is the program with its tests (`make test`), self-contained so that it can
be copied and used elsewhere; `validation/` is new and holds every experiment in this document, with
outputs, source hashes and environment; `paper_runs/` holds the scripts, logs and scratch files of the
original work, which used to sit beside the C files.

**Python** — `fun.py` (`p_dist`, `bin_states` vectorised: identical output, seconds instead of a minute and
several GB); `prep_early.py`, `curation_functions.py` fixes; `expand_data.py` (Julia-free);
`analysis/regenerate.py` and `analysis/compare_regenerated.py`; the figure and table scripts generalised
(#29); `preprocessing/test_pipeline.py`; `requirements.txt` completed. `data/analysis/dendrogram_clades.csv`
added (the three top-level groups for the published fit, which the scripts now read).

**New folder `regenerated/`** — the analysis regenerated from the published parameters and from the three
frozen-weight fits, with comparisons.

**Not changed** — the published data products, figures and tables; the parameter file behind them; the
Ruby helper scripts apart from a note at the top of each one that calls `mpf`.

**Removed** — files that should not have been tracked (`analysis/__pycache__/`, `MPF_CMU/mpf.dSYM/`; now in
`.gitignore`) and `tables/annotation_table.txt`, an early draft of Table 5 with a five-group labelling that
no script produces. All remain in the git history.

## 9. Testing

* **Unit tests** (`MPF_CMU/src/tests/test_mpf.c`): 201,361 checks, of which 200,000 are range checks on
  individual draws of the exact sampler. Every numerical routine is compared with a slow implementation
  written from the definitions: the objective for all three treatments of missing data (n from 2 to 62,
  NN from 1 to 4, blanks, weights, duplicate rows); gradients by finite differences; extreme parameters;
  partition function, KL and pseudo-likelihood; the `-z` output across block boundaries; both samplers;
  the minimiser and its status on overflowing starts; cross-validation against explicit refits (row-wise,
  weighted, grouped, masked, training-rows-only, independent of the starting vector, NN=2); the λ search;
  and the consistency results of Section 3 under MCAR and MAR at NN = 1, 2, 3 on exact population data.
* **Command-line tests** (`MPF_CMU/src/tests/test_cli.sh`): 111 checks — exit codes, about thirty-five kinds of bad
  input, all seven modes, NN = 2 and 3, bit-identical results for 1 or many threads, a fit that runs out
  of iterations being reported as unconverged, more than 31 questions, reproduction of the published DRH
  parameters.
* **Mutation testing** (`validation/mutation_test.py`): deliberate bugs are injected one at a time into a
  copy of the sources, and `make test` must fail for each. 42 of 44 are caught. The two survivors remove
  checks that are redundant in practice: one for a non-finite gradient (the objective is non-finite in the
  same cases, and that is checked too), one for a non-finite number in the input (the fit or the
  parameter check then fails anyway, with the same exit status). An earlier survivor exposed a real gap —
  nothing tested that a fit which fails to converge is reported as such — and a test was added for it.
* **Sanitizers**: an address + undefined-behaviour + leak sanitizer build of the final sources (gcc, with
  OpenMP, `-Wall -Wextra` clean) passes both suites and cross-validated DRH fits under all three options
  at NN=1 and under the frozen-weight option at NN=2, with no reports and the same λ as the optimised
  build.
* **Two platforms**: macOS/clang and Linux/gcc give the same test results; DRH fits agree to 1e-11 and
  simulation results to 2e-4 in KL. The final sources pass both suites on Linux (gcc, with OpenMP) and on macOS
  (Apple's compiler, without OpenMP or any external library).
* **Removing GSL** (`validation/gsl_reference/check_against_gsl.c`): 8.4 million random draws (uniform,
  integer, Gaussian; 8 seeds) are identical to GSL's; the digamma and trigamma functions agree with GSL's
  to 1e-15; the NSB entropy agrees to 2e-12 bits on 400 random cases wherever both integrations succeed
  (GSL's fails in 11 cases where the new one does not, never the reverse). At the level of the program, the
  GSL-free binary writes byte-identical data and parameter files for `-g --seed` in 12 settings (exact and
  Monte Carlo sampling, 5 to 62 nodes) and byte-identical parameter files for the DRH fits.
* **Against the published code** (`validation/compare_with_published_code.py`): fixed-λ fits on 60 random
  datasets, with and without missing data, agree to 2e-7; cross-validated fits on 8 small problems to
  0.0004 in log₁₀λ; NN = 1, 2, 3 against the published code with its multi-flip blocks re-enabled to 2e-7;
  the published DRH parameters to 2e-6. (The published code starts from a random point, so these figures
  vary a little from run to run; an earlier run gave 8e-6 and 7e-7.)
* **Pipeline**: in the pinned environment (Python 3.10, numpy 1.21.3, pandas 1.5.2, scikit-learn 1.2.0,
  scipy 1.10.1, matplotlib 3.6.2), `analysis/regenerate.py` on the published parameters reproduces the six
  committed tables byte for byte, the dendrogram clusters, annotations and group labels, on macOS and on
  Linux. As a last check, the files that make up this release were copied to a clean directory and
  everything in `regenerated/` was produced again from them, with identical tables.
* **Code review**: two separate review passes over all changes before release. The first found ten issues
  (an output-directory deletion hazard in `regenerate.py`, a data file the scripts needed but that was not
  committed, a difference between `--legacy` cross-validation and the published code in how folds start,
  a stale mutation, two Makefile defects, stale status reporting in one loop, memory use of `-z`, and two
  inefficiencies). The second, on the corrected tree, found ten more, none affecting a fit: `-o` did not
  warn when its cross-validated fit failed to converge (Table 1 was re-run to confirm no trial was
  affected); `regenerate.py` replaced an earlier output even when a script had failed; the notes on the
  historical Ruby scripts were on the wrong scripts; a group label in Tables A3/A4 assumed nested
  clusters; an environment `CFLAGS` could silently drop OpenMP; scipy was not pinned; one test bound was
  tighter than the behaviour it tested; and three inefficiencies. Everything was fixed and re-tested
  except three inefficiencies, which were left because changing them would alter the iteration path of
  fits already reported (cross-validation refits each fold's first-stage fit at every λ; the frozen-weight
  loop repeats one evaluation per pass and has no early exit when it stalls).

## 10. The external reviews

### The first review

[`CODEX_REVIEW_OF_CLAUDE.md`](CODEX_REVIEW_OF_CLAUDE.md) reviewed the first version of this work. Each of
its findings was reproduced before being acted on. All ten were correct.

| review finding | what was done |
|---|---|
| R1. "Missing at random" was too weak a condition for the consistency claim | Correct: marginal MPF needs MCAR. The review's counterexample is a unit test. Working through why the frozen-weight variant passed it led to the MAR result of Section 3 |
| R2. Finite inputs could give wrong objectives and NaN gradients, reported as convergence | Objective recomputed in the log domain; explicit fit status; numerical failure is an error. Both of the review's cases are tests |
| R3. Cross-validation folds were initialised from a fit that included the held-out row | Each fold now repeats the staged fit on its training rows; `--legacy` retains the published behaviour. Multistart checks were not added. The second review found that this was incomplete (N1 below) |
| R4. "EM" overclaimed | Renamed `--frozen`; described as a fixed-point iteration with no monotonicity guarantee; the claim that it reaches the same answers as marginal MPF withdrawn |
| R5. Consistency of the objective is not consistency of the shipped estimator | Section 3 separates the two and lists what is not covered. The λ search no longer has a floor at 0.01 |
| R6. The summary claimed more than the evidence supported | All three simulation tables re-run in full with the final code; result files carry source hashes, warnings and convergence status |
| R7. Two DRH λ values were swapped | A transcription error. The tables are now generated from the logs |
| R8. Non-finite parameters and out-of-range integers were accepted; a search over invalid values returned a "best" λ | Fixed and tested |
| R9. The curation fix did not survive all-complete data | Fixed and tested against the old code |
| R10. Reproducibility was narrower than stated | One command now regenerates the analysis; it reproduces the published tables in the pinned environment on two machines |

### The second review

[`CODEX_REVIEW_OF_CLAUDE_2026-11-16.md`](CODEX_REVIEW_OF_CLAUDE_2026-11-16.md) reviewed the work as it stood
after the first response, the larger neighbourhoods, the regenerated analyses and the removal of GSL. Each
finding was reproduced before being acted on. All were correct.

| finding | what was done | test that now detects it | what was re-run |
|---|---|---|---|
| N1. Cross-validation with missing data still used held-out information: the first-stage λ was chosen once, from all the complete rows | The first stage of every fit is now the complete rows *at the λ being scored*: no separate λ is chosen for it, so nothing in a fold depends on the rows it holds out, and the final fit is made the same way. Choosing the first-stage λ separately inside every fold (the other remedy) was implemented and measured first: on 16 simulated data sets it gave the same answer as the leaky version in 15, and on the DRH it changed parameters by less than 1e-3, at 15–27 times the cost. `--legacy` keeps the published protocol | unit test comparing `cross()` with staged refits made by hand from each fold's training rows, on two data sets including the review's, at three values of λ; two mutations | every cross-validated fit with missing data: the DRH refits, the regenerated analyses, the marginal and frozen columns of Table 3, the `mpf` side of the bgms comparison |
| N2. In the NN=3 analysis a configuration with no single-flip neighbour was dropped from the landscape, and the Wogeo entry appeared in neither Table A3 nor A4 | Isolated configurations are kept in the layout and in the figure | the pipeline now stops unless all 150 configurations survive and Tables A3 and A4 list every entry exactly once | all four regenerated analyses (the published one is unchanged, byte for byte) |
| N3. A seed whose low 32 bits are zero (2^32, 2^33, ...) left the random number generator at zero and `mpf -g` never finished | Such a seed is treated as 1. GSL has the same flaw; this is the one case in which the generator now differs from GSL's | unit and command-line tests (the latter with a time limit); one mutation | — |
| N4. The hidden-node check discarded the optimiser's status and called its end points minimisers | The script records status, end-point gradient and the objective along the path; the text now separates the gradient at the truth (the evidence of bias) from where an optimiser ends up (Section 3) | — | the check |
| N5. The DRH bootstrap kept no record of convergence, and warnings during a λ search say nothing about the λ chosen | The bootstrap records how both stages of the chosen fit ended and how many candidate fits did not converge. `mpf -c` prints how many fold fits failed to converge at the λ it chose, and the trial scripts can read it | — | the bootstrap |
| N6. Resuming a run overwrote its provenance file, and the experiment scripts were not fingerprinted | A manifest is named by its fingerprint and never overwritten; the fingerprint covers the C sources, the binary and the scripts, and is stored with every trial | — | applies to runs made from now on; earlier result files keep their old manifests |
| N7. Integers in data and group files were not range-checked; `-z` and `-k` could write NaN for parameters too large to represent | Checked parsing for file integers; non-finite energies are an error | six command-line tests; one mutation | — |
| N8. Some conclusions were stronger than the evidence: `--frozen` as "the appropriate method" for the DRH, a general large-sample ranking, the origin of the paper's N₂ numbers, "correct" and "consistent" in headings | Reworded throughout (Sections 1–4, the READMEs): what is shown is that the truth solves the population equation under the stated assumption; no experiment says which variant is closest to the truth for the DRH; marginal is sometimes ahead of frozen under MCAR | — | — |
| Smaller points: `integrate()` accepted a workspace of no intervals; the penalty formula in the README was right only for NN=1 | Fixed | unit test; one mutation | — |

A simplification pass followed, at the same time, with four reviewers reading the code for dead code,
duplicated logic, wasted work, and fixes made at the wrong depth. The two-stage fit, which had been written
out twice with slightly different conditions, is now one function used by the cross-validation folds and by
the final fit. Numbers are parsed by one mechanism on the command line and in every file. Parameters too
large to represent are rejected where they enter, in place of checks at four places where they were used.
Shared pieces of the objective, the enumeration and the entropy code are factored out, and leftover
parameters, fields and branches are gone. The experiment scripts take their simulated data from one place,
are driven by one runner, and stamp their results in one way. The simplified program was compared with the
one before it on 47 outputs (generated data, probability tables, fixed-λ and cross-validated fits under
every option): all were byte-identical. Speed-ups that would have added complexity were left out.

Two further changes were made while everything was being re-run. The frozen-weight iteration now gives
up, and reports the fit as stalled, when its residual has not fallen by 1% in 300 passes; it used to run
its full 5,000 passes, which is what made a few simulated data sets take seven hours. Fits that converge
are unaffected (the DRH frozen fit is identical with and without the rule). And four of the DRH refits as
first committed had been made with a binary older than its sources; all were redone, and the helpers now
refuse to run in that state.

## 11. Limits

* **Missingness in the DRH is assumed, not established, to depend only on observed answers.** If an answer
  being unknown depends on what it would have been, none of these methods is right.
* **Model selection still uses only complete entries**, which are not a representative sample (#14).
  Scoring held-out partial rows by their marginal likelihood would use all entries; it is not implemented.
* **A comparison with another package** (bgms, in R) was made afterwards and is in the README at the top of
  the repository: neither method dominates, and on the real DRH data bgms predicts held-out entries
  slightly better (than `--frozen` with N₁, N₂ or N₃ alike).
* **No comparison with exact maximum likelihood.** At 20 questions the exact penalised likelihood — and
  with missing answers the exact marginal likelihood, which is consistent under MAR — is affordable, and is
  the natural benchmark for what MPF gives up. It was not run.
* **Non-convexity is handled by one deterministic start** (the complete-rows fit). There is no multistart
  check. The frozen-weight iteration has no convergence proof; it converged in every DRH fit, and where a
  simulation fold did not converge it is counted in the result file.
* **An unregularised fit to separable data** has its minimum at infinity; the program reports a small
  gradient as convergence there. Very large parameters are the sign.
* **No uncertainty estimates** for couplings, probabilities or group memberships. The bootstrap in Section 3
  measures KL from a known truth, not the uncertainty of the DRH fit.
* **Simulation evidence is modest in size**: 16 landscapes per regime for Table 3, 15 large-sample
  landscapes, one MAR mechanism at 20 questions. They show the direction of effects, not thresholds.
* **The effect of NN on the DRH fit is small, and which NN is closer to the truth there is not known**;
  the simulations favour larger NN only for stronger couplings than the DRH's.
* **Not run**: `expand_data.jl` (no Julia available; its Python equivalent is tested), the curation scripts
  end to end (the raw DRH export is not in the repository), the Ruby helper scripts, N > 24 beyond smoke
  tests. **Not regenerated**: the n=30 and n=40 fits in `data/mdl_experiments`, which should not be used.

## 12. Where everything is

| what | where |
|---|---|
| Program, options, data format | `MPF_CMU/src/` (README there) |
| Tests | `MPF_CMU/src/tests/` (`make test`) |
| Every experiment above, with scripts and outputs | `MPF_CMU/validation/` (index in its README) |
| DRH refits: parameters, logs, recipe | `MPF_CMU/validation/results/drh_refits/` |
| The analysis regenerated from each fit, and comparisons | `regenerated/` |
| Pipeline regeneration and comparison scripts | `analysis/regenerate.py`, `analysis/compare_regenerated.py` |
| Comparison with the R package bgms | `README.md` (summary); `MPF_CMU/validation/bgms/`, `MPF_CMU/validation/results/bgms/` |
| The independent review of the first version | `CODEX_REVIEW_OF_CLAUDE.md` |
| Scripts and logs of the original runs for the paper | `MPF_CMU/paper_runs/` |
| The published code | git history, commit `23dce37` and earlier |

Simulations were run on an 18-core Apple laptop and on a 256-core Linux server; each result file records
the hashes of the C sources that produced it, the machine and the Python environment.

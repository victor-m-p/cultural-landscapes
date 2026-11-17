#!/bin/bash
# Command-line tests for mpf: exit codes, error handling, file formats, determinism, and reproduction of
# the published DRH fit. Run with `make test` (or `bash tests/test_cli.sh`) from the directory that holds
# the Makefile. Everything it needs is in this directory.
# Nothing is written outside a temporary directory.
cd "$(dirname "$0")/.."
MPF="$PWD/mpf"
# the 455-row, 20-question DRH data of the paper, and the parameters published with it
DRH="$PWD/tests/drh_data.mpf"
DRH_PUBLISHED="$PWD/tests/drh_published_params.dat"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
cd "$T"
pass=0; fail=0
ok()   { pass=$((pass+1)); }
bad()  { fail=$((fail+1)); echo "  FAILED: $1"; }
# expect_exit CODE DESCRIPTION CMD...
expect_exit() { local want=$1 what=$2; shift 2; "$@" > out.txt 2> err.txt; local got=$?; if [ $got -eq $want ]; then ok; else bad "$what: exit $got, expected $want ($(head -c 200 err.txt))"; fi; }
num() { python3 -c "import sys; print(1 if ($1) else 0)"; }
maxdiff() { python3 - "$1" "$2" <<'PY'
import sys
a=[float(x) for x in open(sys.argv[1]).read().split()]; b=[float(x) for x in open(sys.argv[2]).read().split()]
print(max(abs(x-y) for x,y in zip(a,b)) if len(a)==len(b) and a else 1e9)
PY
}

echo "command line: usage and errors"
expect_exit 2 "no arguments" "$MPF"
expect_exit 2 "unknown mode" "$MPF" -q foo
expect_exit 2 "missing arguments" "$MPF" -l foo
expect_exit 2 "unknown option" "$MPF" -g x 5 10 0.2 --nonsense
expect_exit 2 "missing data file" "$MPF" -l does_not_exist.dat 0 1
expect_exit 2 "missing parameter file" "$MPF" -z does_not_exist.dat 5

printf '3\n4\n0101 1.0\n1X10 1.0\n0011 0.5\n' > good.dat
expect_exit 0 "valid file" "$MPF" -l good.dat 0 1
expect_exit 2 "NN=0" "$MPF" -l good.dat 0 0
expect_exit 2 "NN=7 (above the limit)" "$MPF" -l good.dat 0 7
expect_exit 2 "NN larger than the number of nodes" "$MPF" -l good.dat 0 5
expect_exit 2 "NN not a number" "$MPF" -l good.dat 0 two
expect_exit 2 "non-numeric sparsity" "$MPF" -l good.dat abc 1
printf '3\n4\n0101 1.0\n1210 1.0\n0011 0.5\n' > badchar.dat;   expect_exit 2 "bad character" "$MPF" -l badchar.dat 0 1
printf '3\n4\n0101 1.0\n110 1.0\n0011 0.5\n' > badlen.dat;     expect_exit 2 "short row" "$MPF" -l badlen.dat 0 1
printf '5\n4\n0101 1.0\n1110 1.0\n0011 0.5\n' > short.dat;     expect_exit 2 "fewer rows than the header says" "$MPF" -l short.dat 0 1
printf '2\n4\n0101 1.0\n1110\n' > noweight.dat;                expect_exit 2 "missing weight" "$MPF" -l noweight.dat 0 1
printf '2\n4\n0101 1.0\n1110 -1.0\n' > negweight.dat;          expect_exit 2 "negative weight" "$MPF" -l negweight.dat 0 1
printf '2\n4\n0101 1.0\nXXXX 1.0\n' > allblank.dat;            expect_exit 2 "row with nothing observed" "$MPF" -l allblank.dat 0 1
printf '2\n70\n' > toowide.dat;                                expect_exit 2 "more nodes than supported" "$MPF" -l toowide.dat 0 1
printf '2\n4\n0X01 1.0\n1X10 1.0\n' > nocomplete.dat;          expect_exit 2 "CV with no complete rows" "$MPF" -c nocomplete.dat 1
printf '' > empty.dat;                                          expect_exit 2 "empty file" "$MPF" -l empty.dat 0 1
printf '1 2 3' > shortpar.dat;                                  expect_exit 2 "short parameter file" "$MPF" -z shortpar.dat 5
printf 'nan 0 0' > nanpar.dat;                                  expect_exit 2 "NaN in a parameter file" "$MPF" -z nanpar.dat 2
printf 'inf 0 0' > infpar.dat;                                  expect_exit 2 "infinity in a parameter file" "$MPF" -z infpar.dat 2
expect_exit 2 "node count that overflows an int" "$MPF" -g wrap 4294967298 1 0.1 --seed 1
expect_exit 2 "negative seed" "$MPF" -g wrap 4 5 0.1 --seed -3
expect_exit 2 "empty seed" "$MPF" -g wrap 4 5 0.1 --seed ""
expect_exit 2 "log-sparsity that overflows the penalty" "$MPF" -l good.dat 400 1
# false convergence (external review, A2): a start with huge fields must still be optimised properly ...
printf '1\n2\n11 1\n' > one.dat; printf '0 400 400' > strong.par; printf '0 -800 -800' > over.par
"$MPF" -l one.dat 0 1 > /dev/null; cp one.dat_params.dat one_zero.dat
expect_exit 0 "start with huge fields" "$MPF" -l one.dat 0 1 --start strong.par
[ "$(num "$(maxdiff one_zero.dat one.dat_params.dat) < 1e-5")" = 1 ] && ok || bad "start with huge fields does not reach the optimum ($(cat one.dat_params.dat))"
# ... and a start whose flows overflow must be an error, not a "fit"
expect_exit 2 "start whose flows overflow" "$MPF" -l one.dat 0 1 --start over.par
printf '3\r\n4\r\n0101 1.0\r\n1X10 1.0\r\n0011 0.5\r\n' > crlf.dat
expect_exit 0 "Windows line endings" "$MPF" -l crlf.dat 0 1
[ "$(maxdiff good.dat_params.dat crlf.dat_params.dat)" = "0.0" ] && ok || bad "Windows line endings change the fit"
# header may promise fewer rows than the file holds (the simulation scripts rely on this)
printf '2\n4\n0101 1.0\n1X10 1.0\n0011 0.5\n' > head2.dat; printf '2\n4\n0101 1.0\n1X10 1.0\n' > only2.dat
"$MPF" -l head2.dat 0 1 > /dev/null; "$MPF" -l only2.dat 0 1 > /dev/null
[ "$(maxdiff head2.dat_params.dat only2.dat_params.dat)" = "0.0" ] && ok || bad "rows beyond the header count were used"

echo "command line: numbers that do not fit, and parameters too large to represent"
printf '1\n4294967298\n11 1\n' > ovf_n.dat;    expect_exit 2 "node count that overflows an int" "$MPF" -l ovf_n.dat 0 1
printf '4294967297\n2\n11 1\n' > ovf_m.dat;    expect_exit 2 "row count that overflows an int" "$MPF" -l ovf_m.dat 0 1
printf '2\n2.5\n11 1\n01 1\n' > frac_n.dat;   expect_exit 2 "node count that is not an integer" "$MPF" -l frac_n.dat 0 1
printf '4294967297\n1\n7\n' > ovf_groups.txt;  expect_exit 2 "group id that overflows an int" "$MPF" -c good.dat 1 --groups ovf_groups.txt
printf '1e308 0 0' > huge_params.dat;           expect_exit 2 "-z with parameters too large to represent" "$MPF" -z huge_params.dat 2
[ ! -e huge_params.dat_probs.dat ] && ok || bad "-z wrote a file for parameters it cannot represent"
printf '0.1 0.2 0.3 0.4' > long_params.dat;     expect_exit 2 "parameter file with more values than the model has" "$MPF" -z long_params.dat 2
printf '2\n2\n11 1.0abc\n00 1\n' > junk_w.dat; expect_exit 2 "weight followed by junk" "$MPF" -l junk_w.dat 0 1
printf '2\n2\n11 inf\n00 1\n' > inf_w.dat;     expect_exit 2 "infinite weight" "$MPF" -l inf_w.dat 0 1
printf '2\n2\n11 nan\n00 1\n' > nan_w.dat;     expect_exit 2 "weight that is not a number" "$MPF" -l nan_w.dat 0 1
expect_exit 2 "log-sparsity that is not a number" "$MPF" -l good.dat nan 1
expect_exit 2 "infinite beta" "$MPF" -g infbeta 5 10 inf
printf '2\n2\n11 1\n00 1\n' > two.dat;        expect_exit 2 "-k with parameters too large to represent" "$MPF" -k two.dat huge_params.dat huge_params.dat
# seeds whose low 32 bits are zero used to hang the generator: run with a time limit
python3 - "$MPF" <<'PY' && ok || bad "-g with --seed 4294967296 did not finish"
import subprocess, sys
for seed in ('4294967296', '8589934592', '0'):
    r = subprocess.run([sys.argv[1], '-g', 'seedprobe', '2', '1', '.1', '--seed', seed], capture_output=True, timeout=20)
    assert r.returncode == 0
PY

echo "command line: generate, fit, compare"
expect_exit 0 "-g" "$MPF" -g sim 8 300 0.3 --seed 7
"$MPF" -g sim2 8 300 0.3 --seed 7 > /dev/null
cmp -s sim_data.dat sim2_data.dat && cmp -s sim_params.dat sim2_params.dat && ok || bad "-g with the same seed is not reproducible"
"$MPF" -g sim3 8 300 0.3 --seed 8 > /dev/null
cmp -s sim_data.dat sim3_data.dat && bad "-g with different seeds gives the same data" || ok
[ "$(head -1 sim_data.dat)" = "300" ] && [ "$(sed -n 2p sim_data.dat)" = "8" ] && [ "$(wc -l < sim_data.dat | tr -d ' ')" = "302" ] && ok || bad "-g data file layout"
[ "$(awk 'NR>2 && ($1 !~ /^[01]{8}$/ || $2 != "1.0")' sim_data.dat | wc -l | tr -d ' ')" = "0" ] && ok || bad "-g data rows malformed"
[ "$(wc -w < sim_params.dat | tr -d ' ')" = "36" ] && ok || bad "-g parameter count"
expect_exit 0 "-g with the Monte Carlo sampler" "$MPF" -g simmc 8 50 0.3 --seed 7 --mcmc
expect_exit 0 "-l" "$MPF" -l sim_data.dat 0.5 1
grep -q '^params=\[' out.txt && ok || bad "-l does not print params=[...]"
python3 - <<'PY' && ok || bad "-l: printed parameters differ from the parameter file"
import re
p=[float(x) for x in re.search(r'params=\[(.*)\]', open('out.txt').read()).group(1).split(',')]
q=[float(x) for x in open('sim_data.dat_params.dat').read().split()]
assert len(p)==len(q)==36 and max(abs(a-b) for a,b in zip(p,q))<1e-12
PY
cp sim_data.dat_params.dat fit_l.dat
"$MPF" -l sim_data.dat 0.5 1 --start sim_params.dat > /dev/null
[ "$(num "$(maxdiff fit_l.dat sim_data.dat_params.dat) < 1e-6")" = 1 ] && ok || bad "-l: complete-data fit depends on the starting point"
"$MPF" -l sim_data.dat 0.5 1 --legacy > /dev/null
[ "$(num "$(maxdiff fit_l.dat sim_data.dat_params.dat) < 1e-6")" = 1 ] && ok || bad "-l: objectives differ on complete data"
expect_exit 0 "-k" "$MPF" -k sim_data.dat sim_params.dat sim_params.dat
[ "$(num "abs($(awk '/^KL:/{print $2}' out.txt)) < 1e-9")" = 1 ] && ok || bad "-k of a parameter file against itself is not zero"
"$MPF" -k sim_data.dat sim_params.dat fit_l.dat > out.txt
[ "$(num "0 < $(awk '/^KL:/{print $2}' out.txt) < 0.5")" = 1 ] && ok || bad "-k of the fit from the truth is implausible: $(cat out.txt | head -1)"
expect_exit 0 "-z" "$MPF" -z sim_params.dat 8
[ "$(wc -l < sim_params.dat_probs.dat | tr -d ' ')" = "256" ] && [ "$(num "abs($(awk '{s+=$3} END{printf "%.12f", s}' sim_params.dat_probs.dat) - 1) < 1e-8")" = 1 ] && ok || bad "-z probabilities do not sum to one"
expect_exit 0 "-c" "$MPF" -c sim_data.dat 1
grep -q '^Best log_sparsity: ' out.txt && ok || bad "-c does not print 'Best log_sparsity:'"
cp sim_data.dat_params.dat fit_c.dat
"$MPF" -c sim_data.dat 1 > /dev/null
[ "$(maxdiff fit_c.dat sim_data.dat_params.dat)" = "0.0" ] && ok || bad "-c is not deterministic"
OMP_NUM_THREADS=1 "$MPF" -c sim_data.dat 1 > /dev/null
[ "$(maxdiff fit_c.dat sim_data.dat_params.dat)" = "0.0" ] && ok || bad "-c depends on the number of threads"
expect_exit 0 "-t" "$MPF" -t sim_data.dat sim_params.dat 1
grep -q '^KL divergence from truth: ' out.txt && ok || bad "-t does not print the KL divergence"
"$MPF" -k sim_data.dat sim_params.dat fit_c.dat > k.txt
[ "$(num "abs($(awk '/^KL divergence/{print $5}' out.txt) - $(awk '/^KL:/{print $2}' k.txt)) < 1e-8")" = 1 ] && ok || bad "-t and -c followed by -k disagree"
expect_exit 0 "-o" "$MPF" -o sim 1
grep -q '^Unregularised fit: ' out.txt && ok || bad "-o does not report how the unregularised fit ended"
python3 - <<'PY' && ok || bad "-o: val=[...] line is missing or inconsistent"
import re
v=[float(x) for x in re.search(r'val=\[(.*)\]', open('out.txt').read()).group(1).split(',')]
# the default search covers [-2, 4] and may extend down to -8; on this (seeded) data the oracle lambda is no worse than the CV one
assert len(v)==8 and v[2] <= v[4]+1e-9 and v[2] <= v[6]+1e-9 and -8 <= v[3] <= 4 and -8 <= v[5] <= 4 and v[7]==-1000
PY

echo "command line: a fit that does not converge is reported as such"
# 128 observations of 20 nodes (mpf -g x 20 128 0.4 --seed 4): without regularisation the minimiser runs out
# of iterations with a gradient about 100 times the convergence threshold
cp "$(dirname "$MPF")/tests/unconverged_data.dat" hard.dat
expect_exit 0 "unregularised fit that runs out of iterations" "$MPF" -l hard.dat -100 1
grep -Eq '^K=.*; (reached the iteration limit|stalled)' out.txt && grep -q 'warning: the fit' err.txt && ok || bad "an unconverged fit was not reported as unconverged"
expect_exit 0 "regularised fit of the same data" "$MPF" -l hard.dat 0.5 1
grep -q '^K=.*; converged' out.txt && ! grep -q 'warning' err.txt && ok || bad "a converged fit was not reported as converged"

echo "command line: missing data, weights, groups"
python3 - <<'PY'
import random
random.seed(3)
rows=open('sim_data.dat').read().split('\n')[2:302]
out=[]; grp=[]
for i,r in enumerate(rows):
    s=list(r.split()[0])
    if i>=150:
        for k in random.sample(range(8),2): s[k]='X'
    out.append("".join(s)+" 1.0"); grp.append(str(i))
# one inconsistently coded entry: two half-weight rows that differ in one answer
out.append("01010101 0.5"); out.append("01010111 0.5"); grp += ["9000","9000"]
open('part.dat','w').write(f"{len(out)}\n8\n"+"\n".join(out)+"\n"); open('part.grp','w').write("\n".join(grp)+"\n")
PY
for flag in "" --legacy --frozen; do
  expect_exit 0 "-c with missing data $flag" "$MPF" -c part.dat 1 $flag
  "$MPF" -k part.dat sim_params.dat part.dat_params.dat > k.txt
  [ "$(num "0 < $(awk '/^KL:/{print $2}' k.txt) < 0.3")" = 1 ] && ok || bad "missing data $flag: KL from truth is implausible ($(head -1 k.txt))"
done
expect_exit 0 "-c with groups" "$MPF" -c part.dat 1 --groups part.grp
expect_exit 2 "groups file too short" "$MPF" -c part.dat 1 --groups sim_params.dat
expect_exit 2 "missing groups file" "$MPF" -c part.dat 1 --groups nope.grp
expect_exit 2 "removed option --em" "$MPF" -c part.dat 1 --em
# the default CV with missing data must not depend on thread count either
"$MPF" -c part.dat 1 > /dev/null; cp part.dat_params.dat part_mt.dat; OMP_NUM_THREADS=1 "$MPF" -c part.dat 1 > /dev/null
[ "$(maxdiff part_mt.dat part.dat_params.dat)" = "0.0" ] && ok || bad "-c with missing data depends on the number of threads"

echo "command line: larger neighbourhoods (NN > 1)"
for nn in 2 3; do
  expect_exit 0 "-l NN=$nn" "$MPF" -l sim_data.dat 0.5 $nn
  grep -q "neighbours per complete state" out.txt && ok || bad "-l NN=$nn does not report the neighbourhood size"
  "$MPF" -k sim_data.dat sim_params.dat sim_data.dat_params.dat > k.txt
  [ "$(num "0 < $(awk '/^KL:/{print $2}' k.txt) < 0.5")" = 1 ] && ok || bad "NN=$nn fit: KL from truth is implausible ($(head -1 k.txt))"
done
[ "$("$MPF" -l sim_data.dat 0.5 2 | awk '/neighbours per complete state/{print $(NF-4)}')" = "36" ] && ok || bad "NN=2 with 8 nodes should have 8 + 28 = 36 neighbours per state"
"$MPF" -l sim_data.dat 0.5 1 > /dev/null; cp sim_data.dat_params.dat nn1.dat; "$MPF" -l sim_data.dat 0.5 2 > /dev/null
[ "$(num "$(maxdiff nn1.dat sim_data.dat_params.dat) > 1e-4")" = 1 ] && ok || bad "NN=2 gives the same fit as NN=1 (the argument is being ignored)"
expect_exit 0 "-c NN=2" "$MPF" -c sim_data.dat 2
for flag in "" --legacy --frozen; do expect_exit 0 "-c NN=2 with missing data $flag" "$MPF" -c part.dat 2 $flag; done
expect_exit 0 "-o NN=2" "$MPF" -o sim 2
expect_exit 0 "-t NN=2" "$MPF" -t sim_data.dat sim_params.dat 2

echo "command line: more than 31 nodes"
expect_exit 0 "-g n=40" "$MPF" -g big 40 60 0.05 --seed 5
expect_exit 0 "-l n=40" "$MPF" -l big_data.dat 0.5 1
[ "$(wc -w < big_data.dat_params.dat | tr -d ' ')" = "820" ] && ok || bad "n=40 parameter count"
expect_exit 0 "-c n=40 (pseudo-likelihood CV)" "$MPF" -c big_data.dat 1
expect_exit 2 "-k n=40 refuses to enumerate" "$MPF" -k big_data.dat big_params.dat big_params.dat

echo "reproduction of the published DRH fit (legacy objective at the published lambda)"
cp "$DRH" drh.mpf
expect_exit 0 "DRH fit" "$MPF" -l drh.mpf 0.205745 1 --legacy
[ "$(num "$(maxdiff drh.mpf_params.dat "$DRH_PUBLISHED") < 1e-5")" = 1 ] && ok || bad "DRH fit differs from the published parameters by $(maxdiff drh.mpf_params.dat "$DRH_PUBLISHED")"

echo "$((pass+fail)) command-line checks, $fail failed"
[ $fail -eq 0 ]

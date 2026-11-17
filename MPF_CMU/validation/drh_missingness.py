"""Are the missing answers in the curated DRH data related to the answers that are observed?
If not (missing completely at random), the pattern of blanks should be independent of every answer.

  1. per question: is its "yes" rate different between complete entries and entries with a blank elsewhere?
  2. per pair (k, q): is "question k is missing" associated with the answer to question q?  (Fisher exact tests)

Needs pandas and scipy.   Usage: python3 drh_missingness.py     (output: results/drh_missingness.txt)
"""
import os
import pandas as pd
from scipy.stats import fisher_exact
here = os.path.dirname(os.path.abspath(__file__)); repo = os.path.join(here, '..', '..')
ref = pd.read_csv(os.path.join(repo, 'data', 'reference', 'direct_reference_questions_20_maxna_5_nrows_455_entries_407.csv'))
qref = pd.read_csv(os.path.join(repo, 'data', 'preprocessing', 'question_reference.csv'))
names = dict(zip(qref['question_id_drh'].astype(str), qref['question']))
q = [c for c in ref.columns if c not in ('entry_id', 'weight')]
# one record per entry: +1 / -1 if answered consistently, NaN if answered both ways, and a separate missing flag
g = ref.groupby('entry_id')[q]
missing = (g.first() == 0)
value = g.mean().where((g.nunique() == 1) & ~missing)
out = []
def say(s=''):
    print(s); out.append(s)
n = len(value); has_blank = missing.any(axis=1)
say(f"{n} entries; {int(has_blank.sum())} have at least one missing answer; {int(missing.values.sum())} missing answers in all")
say("missing answers per question: " + ", ".join(f"{names[c][:18]}={int(missing[c].sum())}" for c in q))
say()
say("1. 'yes' rate of each question: complete entries vs entries with a blank in some OTHER question")
tests = []
for c in q:
    others = missing.drop(columns=c).any(axis=1)
    obs = value[c].notna()
    a = value.loc[obs & ~others, c]; b = value.loc[obs & others, c]
    table = [[int((a > 0).sum()), int((a < 0).sum())], [int((b > 0).sum()), int((b < 0).sum())]]
    p = fisher_exact(table)[1]
    tests.append((p, c, (a > 0).mean(), (b > 0).mean(), len(a), len(b)))
for p, c, ra, rb, na, nb in sorted(tests):
    say(f"   {names[c][:34]:34s} complete-elsewhere {ra:5.2f} (n={na:3d})   blank-elsewhere {rb:5.2f} (n={nb:3d})   p = {p:.1e}")
sig = sum(p < 0.05 / len(q) for p, *_ in tests)
say(f"   {sig} of {len(q)} questions differ at p < 0.05/{len(q)} (Bonferroni)")
say()
say("2. 'question k missing' against the answer to question q, all pairs with at least 10 missing k")
pairs = []
for k in q:
    if missing[k].sum() < 10: continue
    for c in q:
        if c == k: continue
        obs = value[c].notna()
        m1 = value.loc[obs & missing[k], c]; m0 = value.loc[obs & ~missing[k], c]
        if len(m1) < 5: continue
        table = [[int((m1 > 0).sum()), int((m1 < 0).sum())], [int((m0 > 0).sum()), int((m0 < 0).sum())]]
        pairs.append((fisher_exact(table)[1], k, c, (m1 > 0).mean(), (m0 > 0).mean(), len(m1)))
pairs.sort()
bonf = 0.05 / len(pairs)
say(f"   {len(pairs)} pairs tested; {sum(p < bonf for p, *_ in pairs)} significant at p < 0.05/{len(pairs)}; {sum(p < 0.05 for p, *_ in pairs)} at p < 0.05 (expected by chance: {0.05 * len(pairs):.0f})")
say("   strongest:")
for p, k, c, r1, r0, n1 in pairs[:10]:
    say(f"     when '{names[k][:28]}' is missing (n={n1}), '{names[c][:28]}' is yes {r1:.2f} of the time, otherwise {r0:.2f}   p = {p:.1e}")
open(os.path.join(here, 'results', 'drh_missingness.txt'), 'w').write("\n".join(out) + "\n")

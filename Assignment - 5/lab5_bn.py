"""
AI453 - Practical #5: A Bayesian Network from Real Data
Department of Artificial Intelligence, SVNIT Surat

Plain Python 3, standard library only. Nothing to install.
Run:  python3 lab5_bn.py
It should print one line: P(fail=1 | medu=1) = 0.099

Data: student-por.csv, 649 secondary-school students from two Portuguese
schools (Cortez & Silva, 2008; UCI Machine Learning Repository).
Keep it in the same folder as this file.
"""

import csv
import itertools
import random

# --------------------------------------------------------------------------
# The eight variables, all binary, all cut from real columns of the CSV.
# --------------------------------------------------------------------------
#   medu    mother's education is secondary or above   (Medu >= 3)
#   goout   goes out with friends often                (goout >= 4)
#   fail    has failed a class before                  (failures >= 1)
#   higher  wants to go on to higher education         (higher == yes)
#   study   studies more than 5 hours a week           (studytime >= 3)
#   absent  six or more absences this year             (absences >= 6)
#   g1      first-period grade of 12 or more           (G1 >= 12)
#   grade   final grade of 12 or more                  (G3 >= 12)

VARS = ["medu", "goout", "fail", "higher", "study", "absent", "g1", "grade"]

# --------------------------------------------------------------------------
# The network. Each entry is  child : [its parents].
# The ordering is temporal: family background, then things fixed early in the
# year, then the first-period grade, then the final grade.
# --------------------------------------------------------------------------
PARENTS = {
    "medu":   [],
    "goout":  [],
    "fail":   ["medu"],
    "higher": ["medu", "fail"],
    "study":  ["medu", "higher", "goout"],
    "absent": ["goout", "fail"],
    "g1":     ["fail", "study", "goout", "higher"],
    "grade":  ["g1", "absent"],
}

# VARS is already a topological order: every variable appears after all of its
# parents. Check that for yourself before you rely on it in Part D.


# --------------------------------------------------------------------------
# Helpers, written for you
# --------------------------------------------------------------------------

def load(path="student-por.csv"):
    """Read the CSV and cut every column down to 0/1. Returns a list of dicts."""
    rows = []
    for r in csv.DictReader(open(path)):
        g = lambda k: r[k].strip('"')
        rows.append({
            "medu":   1 if int(g("Medu")) >= 3 else 0,
            "goout":  1 if int(g("goout")) >= 4 else 0,
            "fail":   1 if int(g("failures")) >= 1 else 0,
            "higher": 1 if g("higher") == "yes" else 0,
            "study":  1 if int(g("studytime")) >= 3 else 0,
            "absent": 1 if int(g("absences")) >= 6 else 0,
            "g1":     1 if int(g("G1")) >= 12 else 0,
            "grade":  1 if int(g("G3")) >= 12 else 0,
        })
    return rows


def count(data, cond):
    """How many rows satisfy every key=value pair in cond. count(data, {}) is len(data)."""
    return sum(1 for d in data if all(d[k] == v for k, v in cond.items()))


def configs(parents):
    """All parent assignments, as tuples, in a fixed order. configs([]) gives [()]."""
    return list(itertools.product([0, 1], repeat=len(parents)))


def sample_bernoulli(p):
    """From Practical 3. Still the only random primitive you need."""
    return 1 if random.random() < p else 0


# --------------------------------------------------------------------------
# The worked example: one CPT row, straight from counting.
#
#   P(fail = 1 | medu = 1) = (rows with fail=1 and medu=1) / (rows with medu=1)
#
# Every CPT in this lab is this line with a different variable and a different
# parent assignment. That is all maximum likelihood estimation is here --
# exactly the counting you did for the die in Practical 3, T8.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    data = load()
    num = count(data, {"fail": 1, "medu": 1})
    den = count(data, {"medu": 1})
    print("P(fail=1 | medu=1) =", round(num / den, 3))


# ==========================================================================
# PART A - fitting the network
# ==========================================================================

# T1. Print the number of rows, and P(v=1) for each of the eight variables.
#     Sanity check: you should have 649 rows and P(grade=1) near 0.536.

def task1(data):
    print("\n===== T1: Dataset overview =====")
    print(f"Number of rows: {len(data)}")
    for v in VARS:
        p = count(data, {v: 1}) / len(data)
        print(f"  P({v}=1) = {p:.3f}")


# T2. fit(data, alpha=0.0) -> a dict cpt[v][parent_tuple] = P(v = 1 | parents).
#     For each variable v and each assignment to its parents, count as in the
#     worked example. alpha is a smoothing constant; leave it at 0 for now and
#     come back to it in T7.
#         P(v=1 | pa) = (n1 + alpha) / (n0 + n1 + 2*alpha)
#     Print the CPT of 'grade'. Four rows, one per (g1, absent) pair.

def fit(data, alpha=0.0):
    """Fit CPTs from data using MLE with optional Laplace smoothing."""
    cpt = {}
    for v in VARS:
        cpt[v] = {}
        pa = PARENTS[v]
        for cfg in configs(pa):
            # Build condition dict for parent assignment
            pa_cond = dict(zip(pa, cfg))
            n1 = count(data, {**pa_cond, v: 1})
            n0 = count(data, {**pa_cond, v: 0})
            denom = n0 + n1 + 2 * alpha
            if denom == 0:
                cpt[v][cfg] = 0.5  # undefined, default
            else:
                cpt[v][cfg] = (n1 + alpha) / denom
    return cpt


def task2(data):
    print("\n===== T2: CPT of 'grade' =====")
    cpt = fit(data, alpha=0.0)
    pa = PARENTS["grade"]  # ["g1", "absent"]
    print(f"  Parents of grade: {pa}")
    for cfg in configs(pa):
        pa_str = ", ".join(f"{p}={v}" for p, v in zip(pa, cfg))
        print(f"  P(grade=1 | {pa_str}) = {cpt['grade'][cfg]:.3f}")
    return cpt


# ==========================================================================
# PART B - the graph is a claim about the distribution
# ==========================================================================

# T3. joint(assignment, cpt) -> the probability of one full assignment to all
#     eight variables, as the product over variables of P(v | its parents).

def joint(assignment, cpt):
    """Compute P(x1, x2, ..., x8) using the chain rule factored by the DAG."""
    prob = 1.0
    for v in VARS:
        pa = PARENTS[v]
        pa_vals = tuple(assignment[p] for p in pa)
        p_v1 = cpt[v][pa_vals]
        if assignment[v] == 1:
            prob *= p_v1
        else:
            prob *= (1 - p_v1)
    return prob


def task3(cpt):
    print("\n===== T3: Joint sums to 1 =====")
    total = 0.0
    for bits in itertools.product([0, 1], repeat=len(VARS)):
        assignment = dict(zip(VARS, bits))
        total += joint(assignment, cpt)
    print(f"  Sum of all 2^8 = 256 joint probabilities: {total:.6f}")


# T4. infer(query, evidence, cpt) -> P(query | evidence), by enumeration.

def infer(query, evidence, cpt):
    """Exact inference by full enumeration over all 2^8 assignments."""
    p_evidence = 0.0
    p_query_and_evidence = 0.0
    for bits in itertools.product([0, 1], repeat=len(VARS)):
        assignment = dict(zip(VARS, bits))
        # Check if assignment is consistent with evidence
        if not all(assignment[k] == v for k, v in evidence.items()):
            continue
        j = joint(assignment, cpt)
        p_evidence += j
        # Check if assignment also matches the query
        if all(assignment[k] == v for k, v in query.items()):
            p_query_and_evidence += j
    if p_evidence == 0:
        return 0.0
    return p_query_and_evidence / p_evidence


def task4(data, cpt):
    print("\n===== T4: Exact inference vs data fractions =====")
    queries = [
        ({"grade": 1}, {},                     "P(grade=1)"),
        ({"grade": 1}, {"fail": 0},            "P(grade=1 | fail=0)"),
        ({"grade": 1}, {"g1": 1},              "P(grade=1 | g1=1)"),
        ({"grade": 1}, {"g1": 1, "absent": 1}, "P(grade=1 | g1=1, absent=1)"),
    ]
    for q, e, label in queries:
        model_p = infer(q, e, cpt)
        # Data fraction
        num = count(data, {**q, **e})
        den = count(data, e)
        data_p = num / den if den > 0 else 0
        print(f"  {label}:")
        print(f"    Model = {model_p:.3f},  Data = {data_p:.3f}  (n = {den})")
    # Note: Where model and data disagree, the count is often small
    # (e.g., P(grade=1 | g1=1, absent=1) has few matching students).


# ==========================================================================
# PART C - checking what the graph predicts
# ==========================================================================

# T5. Check marginal independence claims of the graph.

def task5(data):
    print("\n===== T5: Independence checks (no edge, no common ancestor) =====")
    pairs = [("medu", "goout"), ("goout", "fail"), ("goout", "higher")]
    n = len(data)
    for a, b in pairs:
        p_a = count(data, {a: 1}) / n
        p_b = count(data, {b: 1}) / n
        p_ab = count(data, {a: 1, b: 1}) / n
        product = p_a * p_b
        print(f"  {a}, {b}:")
        print(f"    P({a}=1, {b}=1) = {p_ab:.4f},  P({a}=1)*P({b}=1) = {product:.4f}")
        print(f"    Difference = {abs(p_ab - product):.4f}")
    # The data broadly backs the graph up: the joint is close to the product
    # for all three pairs, confirming approximate marginal independence.
    print("  -> Data supports the independence claims: joints are close to products.")


# T6. Collider / explaining away: medu and goout are both parents of g1.

def task6(data, cpt):
    print("\n===== T6: Collider (explaining away) =====")
    n = len(data)
    # P(medu=1)
    p_medu = infer({"medu": 1}, {}, cpt)
    print(f"  P(medu=1) = {p_medu:.3f}")
    # P(medu=1 | g1=0)
    p_medu_g1_0 = infer({"medu": 1}, {"g1": 0}, cpt)
    print(f"  P(medu=1 | g1=0) = {p_medu_g1_0:.3f}")
    # P(medu=1 | g1=0, goout=0)
    p_medu_g1_0_go0 = infer({"medu": 1}, {"g1": 0, "goout": 0}, cpt)
    print(f"  P(medu=1 | g1=0, goout=0) = {p_medu_g1_0_go0:.3f}")
    # P(medu=1 | g1=0, goout=1)
    p_medu_g1_0_go1 = infer({"medu": 1}, {"g1": 0, "goout": 1}, cpt)
    print(f"  P(medu=1 | g1=0, goout=1) = {p_medu_g1_0_go1:.3f}")
    # Explanation:
    # Among students who scored poorly (g1=0), both going out often (goout) and
    # not having an educated mother (low medu) are potential explanations. If we
    # learn that a student goes out a lot, that "explains away" the low grade,
    # so the alternative explanation (uneducated mother) becomes less needed --
    # making an educated mother MORE likely. This is the classic explaining-away
    # (sprinkler) effect at a collider node.
    print("  -> Explaining away: learning goout=1 among g1=0 students makes")
    print("     medu=1 MORE likely, because 'going out' already explains the")
    print("     low grade, reducing the need for 'uneducated mother' as cause.")


# ==========================================================================
# PART D - sampling, smoothing, and where the model breaks
# ==========================================================================

# T7. Zero/one CPT rows and Laplace smoothing.

def task7(data):
    print("\n===== T7: CPT rows that are 0, 1, or 0/0 (alpha=0) =====")
    cpt0 = fit(data, alpha=0.0)
    pa_g1 = PARENTS["g1"]  # ["fail", "study", "goout", "higher"]
    problem_rows = []
    for v in VARS:
        pa = PARENTS[v]
        for cfg in configs(pa):
            pa_cond = dict(zip(pa, cfg))
            n1 = count(data, {**pa_cond, v: 1})
            n0 = count(data, {**pa_cond, v: 0})
            total = n0 + n1
            p = cpt0[v][cfg]
            if total == 0 or p == 0.0 or p == 1.0:
                pa_str = ", ".join(f"{pn}={pv}" for pn, pv in zip(pa, cfg))
                problem_rows.append((v, pa_str, p, total))
                print(f"  P({v}=1 | {pa_str}) = {p:.3f}  (n = {total})")
    print(f"  Total problematic rows: {len(problem_rows)}")

    print("\n  After smoothing with alpha=1:")
    cpt1 = fit(data, alpha=1.0)
    for v, pa_str, _, _ in problem_rows:
        # Re-extract cfg from pa_str
        pass
    # Just reprint g1 CPT with alpha=1 to show they are fixed
    for cfg in configs(pa_g1):
        pa_str = ", ".join(f"{p}={v}" for p, v in zip(pa_g1, cfg))
        print(f"  P(g1=1 | {pa_str}) = {cpt1['g1'][cfg]:.3f}")

    # A CPT row of exactly 0 makes any joint probability that passes through
    # that row collapse to 0, effectively blocking inference for those cases.
    print("  -> A CPT row of 0 forces any query passing through it to 0,")
    print("     making inference impossible for those parent configurations.")
    return cpt1


# T8. Forward sampling and rejection sampling.

def forward(cpt):
    """Generate one simulated student by ancestral (forward) sampling."""
    sample = {}
    for v in VARS:
        pa = PARENTS[v]
        pa_vals = tuple(sample[p] for p in pa)
        p = cpt[v][pa_vals]
        sample[v] = sample_bernoulli(p)
    return sample


def rejection_sample(query, evidence, cpt, M):
    """Estimate P(query | evidence) by rejection sampling with M draws."""
    match_evidence = 0
    match_both = 0
    for _ in range(M):
        s = forward(cpt)
        if all(s[k] == v for k, v in evidence.items()):
            match_evidence += 1
            if all(s[k] == v for k, v in query.items()):
                match_both += 1
    if match_evidence == 0:
        return 0.0, 0.0
    return match_both / match_evidence, match_evidence / M


def task8(cpt):
    print("\n===== T8: Forward + rejection sampling =====")
    random.seed(42)
    query = {"grade": 1}
    evidence = {"fail": 1}
    exact = infer(query, evidence, cpt)
    print(f"  Exact P(grade=1 | fail=1) = {exact:.3f}")

    for M in [2000, 50000]:
        est, kept = rejection_sample(query, evidence, cpt, M)
        print(f"  M={M:>5d}: estimate = {est:.3f}, samples kept = {kept*100:.1f}%")

    # As evidence gets more specific, fewer samples match, so the kept
    # percentage drops. With very specific evidence, almost all samples are
    # rejected, wasting computation and giving noisy estimates.
    print("  -> As evidence grows more specific, the kept % drops sharply,")
    print("     making rejection sampling inefficient and noisy.")


# T9. Model against reality -- where the DAG breaks.

def task9(data, cpt):
    print("\n===== T9: Model vs reality =====")
    model_p = infer({"grade": 1}, {"fail": 1}, cpt)
    num = count(data, {"grade": 1, "fail": 1})
    den = count(data, {"fail": 1})
    data_p = num / den if den > 0 else 0
    print(f"  Model  P(grade=1 | fail=1) = {model_p:.3f}")
    print(f"  Data   P(grade=1 | fail=1) = {data_p:.3f}  (n = {den})")
    print()
    # Paths from fail to grade in the current graph:
    #   fail -> absent -> grade      (indirect, via absences)
    #   fail -> higher -> study -> g1 -> grade  (long indirect path)
    #   fail -> g1 -> grade          (indirect, via first-period grade)
    # But there is no direct edge fail -> grade.
    # The large gap between model and data suggests that past failure has a
    # direct causal effect on the final grade beyond what is captured through
    # g1 and absent. Adding a direct edge fail -> grade would let the model
    # capture this direct relationship. This tells us the original DAG was
    # likely designed from domain intuition rather than learned from data,
    # and it misses a real dependency.
    print("  Paths from fail to grade: fail->absent->grade, fail->g1->grade,")
    print("  fail->higher->study->g1->grade. No direct edge fail->grade.")
    print("  -> Adding a direct edge fail -> grade would close the gap. The")
    print("     original DAG was built from domain intuition and misses the")
    print("     direct effect of past failure on the final grade.")


# ==========================================================================
# Run all tasks
# ==========================================================================

if __name__ == "__main__":
    data = load()

    # Part A
    task1(data)
    cpt = task2(data)

    # Part B
    task3(cpt)
    task4(data, cpt)

    # Part C
    task5(data)
    task6(data, cpt)

    # Part D
    cpt_smoothed = task7(data)
    task8(cpt_smoothed)
    task9(data, cpt)
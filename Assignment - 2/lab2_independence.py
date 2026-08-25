"""
AI453  --  Practical #2: Conditional Independence
Two joint distributions, printed on the handout. Plain Python 3, no imports.
Run:  python3 lab2_independence.py
"""

# TABLE 1 -- P(A, B, C).  Key (a, b, c) means A=a, B=b, C=c.
P1 = {
    #  A  B  C
    (0, 0, 0): 0.36,
    (0, 0, 1): 0.04,
    (0, 1, 0): 0.01,
    (0, 1, 1): 0.09,
    (1, 0, 0): 0.09,
    (1, 0, 1): 0.01,
    (1, 1, 0): 0.04,
    (1, 1, 1): 0.36,
}

# TABLE 2 -- P(R, S, W).  R = it rained, S = sprinkler was on, W = grass is wet.
P2 = {
    #  R  S  W
    (0, 0, 0): 0.27,
    (0, 0, 1): 0.03,
    (0, 1, 0): 0.12,
    (0, 1, 1): 0.18,
    (1, 0, 0): 0.08,
    (1, 0, 1): 0.12,
    (1, 1, 0): 0.02,
    (1, 1, 1): 0.18,
}


# ----------------------------------------------------------------------
# Last week's loop, wrapped up. Position 0 is the first variable, 1 the
# second, 2 the third.
#
#   prob(P1, {0: 1})           P(A=1)
#   prob(P1, {0: 1, 2: 1})     P(A=1, C=1)
#   cond(P1, {2: 1}, {0: 1})   P(C=1 | A=1)     query first, then given
# ----------------------------------------------------------------------
def prob(table, conditions):
    total = 0.0
    for row, p in table.items():
        if all(row[i] == v for i, v in conditions.items()):
            total += p
    return total


def cond(table, query, given):
    both = dict(given)
    both.update(query)
    return prob(table, both) / prob(table, given)


print("P(A=1) =", prob(P1, {0: 1}))


# ======================================================================
# TABLE 1
# ======================================================================

# T1.  Print the total of each table. Both should be 1.
total = 0.0
for (a, b, c), p in P1.items():
    total += p
print("Total(P1) = ", round(total, 1))

total = 0.0
for (a, b, c), p in P2.items():
    total += p
print("Total(P2) = ", round(total, 1))


# T2.  Print P(A=1, C=1) and P(A=1) * P(C=1).
#      Equal? If not, A and C are dependent.
print("P(A = 1, C = 1) = ", round(prob(P1, {0: 1, 2: 1}), 1))
print("P(A = 1) * P(C = 1) = ", round(prob(P1, {0: 1}) * prob(P1, {2: 1}), 1))


# T3.  Print P(C=1 | A=1) and P(C=1 | A=0). How far apart?
print("P(C = 1 | A = 1) = ", round(cond(P1, {2: 1}, {0: 1}), 1))
print("P(C = 1 | A = 0) = ", round(cond(P1, {2: 1}, {0: 0}), 1))


# T4.  Print these three:
#          P(C=1 | B=1)      P(C=1 | B=1, A=1)      P(C=1 | B=1, A=0)
#      Once B is known, does A still change anything?
print("P(C = 1 | B = 1) = ", round(cond(P1, {2: 1}, {1: 1}), 1))
print("P(C = 1 | B = 1, A = 1) = ", round(cond(P1, {2: 1}, {0: 1, 1: 1}), 1))
print("P(C = 1 | B = 1, A = 0) = ", round(cond(P1, {2: 1}, {0: 0, 1: 1}), 1))


# T5.  Same three for B=0. Then finish this line:
#          "A and C are ____________, but ____________ given B."
print("P(C = 1 | B = 0) = ", round(cond(P1, {2: 1}, {1: 0}), 1))
print("P(C = 1 | B = 0, A = 1) = ", round(cond(P1, {2: 1}, {0: 1, 1: 0}), 1))
print("P(C = 1 | B = 0, A = 0) = ", round(cond(P1, {2: 1}, {0: 0, 1: 0}), 1))

# ANSWER: "A and C are dependent, but independent given B."


# ======================================================================
# TABLE 2
# ======================================================================

# T6.  Print P(R=1, S=1) and P(R=1) * P(S=1). These should agree --
#      rain and sprinklers are unrelated.
print("P(R = 1, S = 1) = ", round(prob(P2, {0: 1, 1: 1}), 1))
print("P(R = 1) * P(S = 1) = ", round(prob(P2, {0: 1}) * prob(P2, {1: 1}), 1))


# T7.  Print these four:
#          P(R=1)              P(R=1 | W=1)
#          P(R=1 | W=1, S=1)   P(R=1 | W=1, S=0)
print("P(R = 1) = ", round(prob(P2, {0: 1}), 1))
print("P(R = 1 | W = 1) = ", round(cond(P2, {0: 1}, {2: 1}), 1))
print("P(R = 1 | W = 1, S = 1) = ", round(cond(P2, {0: 1}, {1: 1, 2: 1}), 1))
print("P(R = 1 | W = 1, S = 0) = ", round(cond(P2, {0: 1}, {1: 0, 2: 1}), 1))


# T8.  In T5, conditioning REMOVED a dependence. In T7 it CREATED one.
#      Two or three lines: why does learning the sprinkler was on make
#      rain less likely, when the grass is just as wet either way?
# ANSWER: When we know the grass is wet, both rain and the sprinkler are possible causes.
#         If we then learn that the sprinkler was on, there is less need to explain the wet grass by rain,
#         so rain becomes less likely. This is called explaining away: two independent causes become dependent once their common effect is observed.
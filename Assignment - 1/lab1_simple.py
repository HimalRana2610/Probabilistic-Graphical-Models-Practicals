"""
AI453 Probabilistic Graphical Models  --  Practical #1: Introduction to Probability Theory
SVNIT Surat, Department of Artificial Intelligence

You are given ONE joint distribution over three binary variables A, B, C.
Everything you compute today comes out of that one table. Nothing is loaded,
downloaded, or estimated from data.

Run:  python3 lab1_simple.py
Dependencies: NONE. Plain Python 3.
"""

# ----------------------------------------------------------------------
# THE JOINT DISTRIBUTION P(A, B, C)
#
# Three binary variables A, B, C, each 0 or 1.  Eight combinations, eight
# numbers.  The key (a, b, c) means "A=a and B=b and C=c".
#
#       P[(1, 0, 1)]  is  P(A=1, B=0, C=1)  =  0.06
# ----------------------------------------------------------------------
P = {
    #  A  B  C        probability
    (0, 0, 0): 0.06,
    (0, 0, 1): 0.24,
    (0, 1, 0): 0.04,
    (0, 1, 1): 0.16,
    (1, 0, 0): 0.09,
    (1, 0, 1): 0.06,
    (1, 1, 0): 0.21,
    (1, 1, 1): 0.14,
}


# ----------------------------------------------------------------------
# WORKED EXAMPLE  --  read this carefully, every task below is this loop again
#
#   P(A=1)  =  sum of P(A=1, B=b, C=c)  over every b and every c
#
# In words: walk through all eight rows, and add up the ones where A is 1.
# ----------------------------------------------------------------------
total = 0.0
for (a, b, c), p in P.items():
    if a == 1:
        total += p
print("P(A = 1) = ", round(total, 1))

# That is the whole idea. A marginal is a sum over the rows that match.
# A conditional is one such sum divided by another.


# ----------------------------------------------------------------------
# T1.  Check that the table is a valid distribution: all eight numbers
#      must add up to 1.  Print the total.
# ----------------------------------------------------------------------
total = 0.0
for (a, b, c), p in P.items():
    total += p
print("Total = ", round(total, 1))


# ----------------------------------------------------------------------
# T2.  Compute and print P(B=1).
#      Same loop as the worked example, different condition.
# ----------------------------------------------------------------------
total = 0.0
for (a, b, c), p in P.items():
    if b == 1:
        total += p
print("P(B = 1) = ", round(total, 1))


# ----------------------------------------------------------------------
# T3.  Compute and print P(C=1).
# ----------------------------------------------------------------------
total = 0.0
for (a, b, c), p in P.items():
    if c == 1:
        total += p
print("P(C = 1) = ", round(total, 1))


# ----------------------------------------------------------------------
# T4.  Compute and print the joint P(A=1, B=1).
#      Now the condition has two parts.
# ----------------------------------------------------------------------
total = 0.0
for (a, b, c), p in P.items():
    if a == 1 and b == 1:
        total += p
print("P(A = 1, B = 1) = ", round(total, 1))


# ----------------------------------------------------------------------
# T5.  Compute and print the conditional P(C=1 | A=1).
#
#                        P(A=1, C=1)
#      P(C=1 | A=1)  =  --------------
#                          P(A=1)
#
#      Two sums, one divided by the other. Compute the top and the bottom
#      in the same loop if you like.
# ----------------------------------------------------------------------
num = 0.0
den = 0.0
for (a, b, c), p in P.items():
    if a == 1 and c == 1:
        num += p
    if a == 1:
        den += p
print("P(C = 1 | A = 1) = ", round(num / den, 1))


# ----------------------------------------------------------------------
# T6.  Compute and print P(B=1 | A=0, C=1).
#      Two things known, one thing asked. Same pattern.
# ----------------------------------------------------------------------
num = 0.0
den = 0.0
for (a, b, c), p in P.items():
    if a == 0 and b == 1 and c == 1:
        num += p
    if a == 0 and c == 1:
        den += p
print("P(B = 1 | A = 0, C = 1) = ", round(num / den, 1))


# ----------------------------------------------------------------------
# T7.  THE CHAIN RULE.  In class we showed that for any three variables
#
#          P(A,B,C)  =  P(A) * P(B|A) * P(C|A,B)
#
#      Check it numerically. For every one of the eight rows (a,b,c):
#        - look up P(A=a, B=b, C=c) straight from the table
#        - separately compute P(A=a), then P(B=b|A=a), then P(C=c|A=a,B=b)
#          and multiply the three together
#        - print both numbers side by side and say whether they match
#          (allow a tiny difference, e.g. 1e-9, for floating point)
#
#      Then answer in a comment: does the chain rule hold only for THIS
#      table, or for every joint distribution? Why?
# ----------------------------------------------------------------------
pa = 0.0
panb = 0.0
panbnc = 0.0
for (a, b, c), p in P.items():
    if a == 1:
        pa += p
    if a == 1 and b == 1:
        panb += p
    if a == 1 and b == 1 and c == 1:
        panbnc += p
print("P(A, B, C) = ", round(pa * (panb / pa) * (panbnc / panb), 1))

# Chain rule holds for every joint distribution because P(A) and P(A, B)
# cancel out each other in numerator and denominator and we are left with
# only P(A, B, C) every time.

# ----------------------------------------------------------------------
# T8.  BAYES' RULE.  You know P(A=1) already -- that was the worked
#      example. Now suppose you are told that C = 1. Compute
#
#          P(A=1 | C=1)
#
#      and compare it with P(A=1). Did learning C=1 make A=1 more likely
#      or less likely? Write ONE line saying by how much, and in which
#      direction.
# ----------------------------------------------------------------------
num = 0.0
den = 0.0
for (a, b, c), p in P.items():
    if a == 1 and c == 1:
        num += p
    if c == 1:
        den += p
print("P(A = 1 | C = 1) = ", round(num / den, 1))

# P(A = 1) = 0.5 and P(A = 1 | C = 1) = 0.3
# So learning C = 1 make A = 1 less likely by 40% in negative direction
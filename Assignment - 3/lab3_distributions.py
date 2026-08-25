"""
AI453  --  Practical #3: Distributions, Sampling and Maximum Likelihood

Deck 3 and the first half of Deck 4, in code. You write eight small
functions; each one is a few lines.

Plain Python 3, standard library only. Nothing to install.

Run:  python3 lab3_distributions.py
It prints the worked example, then stops at T1.
"""

import math
import random

# A fixed seed, so your own runs repeat. It does NOT make your numbers match
# your neighbour's -- as soon as two people draw a different number of samples,
# the streams diverge. Anything you estimate from samples will differ a little
# from everyone else's, and that is the subject of T3. Compare your sampled
# numbers against the formulas, never against each other.
random.seed(453)


# ======================================================================
# THE DATA  --  both of these are printed on your handout.
# ======================================================================

# Twenty flips of a coin, in order. 1 = heads.
COIN = "11000100100000101000"

# A loaded six-sided die: P(1), P(2), ..., P(6). Note the last one.
DIE = [0.25, 0.20, 0.20, 0.15, 0.18, 0.02]


# ======================================================================
# GIVEN TO YOU  --  use these, do not edit them.
# ======================================================================

def mean_of(xs):
    """Average of a list."""
    return sum(xs) / len(xs)


def var_of(xs):
    """Variance of a list, as E[X^2] - (E[X])^2."""
    m = mean_of(xs)
    return sum(x * x for x in xs) / len(xs) - m * m


def ascii_hist(values):
    """Draw a histogram of a list of whole numbers."""
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.values())
    for v in range(min(values), max(values) + 1):
        c = counts.get(v, 0)
        print(f"    {v:>4} | {'#' * round(54 * c / top):<54} {c}")


def check(label, got, want, tol=1e-9):
    """Compare a number with what it should be, and say so."""
    ok = abs(got - want) <= tol
    print(f"[{'  ok  ' if ok else ' FAIL '}] {label:<34} "
          f"got {got:>10.4f}   want {want:>10.4f}")


# ----------------------------------------------------------------------
# WORKED EXAMPLE -- nothing to write.
#
# A Bernoulli draw: 1 with probability p, else 0. This is the one-line
# pattern that every sampler below is built out of.
# ----------------------------------------------------------------------

def sample_bernoulli(p):
    return 1 if random.random() < p else 0


print("=" * 74)
print("WORKED EXAMPLE  --  a Bernoulli draw")
print("=" * 74)

draws = [sample_bernoulli(0.3) for _ in range(10000)]
check("fraction of 1s in 10000 draws", mean_of(draws), 0.3, tol=0.02)
print()


# ======================================================================
# PART A  --  SAMPLING
# ======================================================================

# T1. Write sample_die(). It should return one roll of the loaded die
#     above, as an index 0..5 (so face 1 is 0, face 6 is 5).
#
#     The method is the INVERSE CDF, and it is four lines:
#
#         draw u = random.random(), a number in [0, 1)
#         keep a running total, adding DIE[0], then DIE[1], and so on
#         the moment the running total goes past u, return that index
#
#     Picture the line from 0 to 1 cut into six pieces, of lengths 0.25,
#     0.20, 0.20, 0.15, 0.18, 0.02. The question is: which piece did u
#     land in? This is how every discrete distribution in this course is
#     sampled.
#
#     Then draw 10000 rolls and print the fraction of each face. They
#     should match DIE.
# ----------------------------------------------------------------------

def sample_die():
    u = random.random()
    total = 0
    for i, prob in enumerate(DIE):
        total += prob
        if u < total:
            return i


print("=" * 74)
print("T1  sampling the die")
print("=" * 74)

draws = [sample_die() for _ in range(10000)]
print("Fractions of each face in 10000 rolls:")
for i, prob in enumerate(DIE):
    print(f"    Face {i + 1}: {draws.count(i) / len(draws):.3f}")

# T2. Write sample_binomial(n, p): the number of 1s in n Bernoulli draws.
#     Do NOT use a formula with factorials. Call sample_bernoulli(p) n
#     times and add up what you get -- that is what "the Binomial is a
#     sum of Bernoullis" means, and T5 depends on your having seen it
#     that way.
#
#     Draw 10000 values of sample_binomial(10, 0.3) and check that their
#     mean is about n*p = 3 and their variance about n*p*(1-p) = 2.1.
# ----------------------------------------------------------------------

def sample_binomial(n, p):
    return sum(sample_bernoulli(p) for _ in range(n))


print("=" * 74)
print("T2  the Binomial is a sum of Bernoullis")
print("=" * 74)

draws = [sample_binomial(10, 0.3) for _ in range(10000)]
print(f"Mean of 10000 binomial(10, 0.3) values: {mean_of(draws):.3f}")
print(f"Variance of 10000 binomial(10, 0.3) values: {var_of(draws):.3f}")

# ======================================================================
# PART B  --  EXPECTATION AND VARIANCE
# ======================================================================

# T3. Deck 3 gives the mean and variance of the die by formula:
#
#         E[X]   = sum over k of  k * DIE[k]
#         Var(X) = sum over k of  k*k * DIE[k]  -  E[X]^2
#
#     (k is the 0-based index, matching what sample_die returns.)
#
#     Compute both from the formula and print them. Then draw n rolls
#     and estimate both with mean_of() and var_of(), for
#     n = 100, 1000, 10000, 100000. Print a small table of the errors.
#
#     Do the two formula values on paper first -- it is twelve
#     multiplications -- then check the computer agrees.
#
#     One line in a comment: each time n goes up by a factor of 100,
#     what happens to the error? Deck 3 predicted this.
# ----------------------------------------------------------------------

print("=" * 74)
print("T3  mean and variance: formula against data")
print("=" * 74)

mean_formula = sum(k * DIE[k] for k in range(6))
var_formula = sum(k * k * DIE[k] for k in range(6)) - mean_formula * mean_formula
print(f"By formula:  E[X] = {mean_formula:.4f}   Var(X) = {var_formula:.4f}")
print()
# A single run of n rolls is noisy, so each n is repeated 10 times and the
# errors averaged -- otherwise luck hides the trend the answer is about.
REPEATS = 10
print(f"{'n':>8}  {'mean':>9}  {'mean error':>11}  {'variance':>9}  {'var error':>11}")
for n in [100, 1000, 10000, 100000]:
    mean_errors = []
    var_errors = []
    for _ in range(REPEATS):
        rolls = [sample_die() for _ in range(n)]
        m = mean_of(rolls)
        v = var_of(rolls)
        mean_errors.append(abs(m - mean_formula))
        var_errors.append(abs(v - var_formula))
    print(f"{n:>8}  {m:>9.4f}  {mean_of(mean_errors):>11.4f}  "
          f"{v:>9.4f}  {mean_of(var_errors):>11.4f}")
print("(estimates are from the last of the 10 runs; errors are averaged over all 10)")

# ANSWER: Every factor of 100 in n cuts the error by roughly 10, because the
#         error of an estimate made from n samples shrinks like 1 / sqrt(n) --
#         a hundred times the data buys only ten times the accuracy.


# T4. Variance of a sum. Draw 20000 pairs (x, y), twice over:
#
#         independent:  x and y each a separate sample_bernoulli(0.3)
#         coupled:      x a sample_bernoulli(0.3), and then y = x
#
#     Both times, estimate Var(X + Y) with var_of(). Compare each against
#     Var(X) + Var(Y) = 0.21 + 0.21 = 0.42.
#
#     One agrees and one does not. In a comment: what is the extra term
#     in the general formula, what is its value in the coupled case, and
#     when does it vanish?
# ----------------------------------------------------------------------

print("=" * 74)
print("T4  variance of a sum")
print("=" * 74)

independent = [sample_bernoulli(0.3) + sample_bernoulli(0.3) for _ in range(20000)]

coupled = []
for _ in range(20000):
    x = sample_bernoulli(0.3)
    y = x
    coupled.append(x + y)

print("Var(X) + Var(Y) = 0.21 + 0.21 = 0.42")
check("independent:  Var(X + Y)", var_of(independent), 0.42, tol=0.02)
print(f"[ note ] coupled (Y = X):  Var(X + Y)      got {var_of(coupled):>10.4f}   "
      f"want {0.42:>10.4f}  <- disagrees")

# ANSWER: The general formula is Var(X + Y) = Var(X) + Var(Y) + 2 * Cov(X, Y).
#         With Y = X the extra term is 2 * Cov(X, X) = 2 * Var(X) = 0.42, so
#         the coupled variance is 0.84 -- twice what adding the variances says.
#         The term vanishes when Cov(X, Y) = 0, i.e. when X and Y are
#         uncorrelated, which independent variables always are; that is why the
#         independent run lands on 0.42.


# ======================================================================
# PART C  --  WHY GAUSSIANS ARE EVERYWHERE
# ======================================================================

# T5. Deck 3 claimed that adding up many independent things gives you a
#     bell curve, whatever you started from. Check it.
#
#     For k = 1, 5, 30, 100: build a list of 5000 values, each one the
#     sum of k draws of sample_bernoulli(0.1), and ascii_hist() it.
#
#     Bernoulli(0.1) is about as far from a bell as a distribution gets
#     -- at k = 1 your histogram is two bars. Look at k = 100.
#
#     Hint: sample_binomial(k, 0.1) already IS the sum of k draws of
#     sample_bernoulli(0.1). You wrote it in T2.
#
#     For each k, also print the mean and variance of your 5000 sums
#     beside k*0.1 and k*0.09, which is what Deck 3 predicts for a sum of
#     k independent copies.
#
#     Two lines in a comment. The shape changed completely between k = 1
#     and k = 100, but those two predictions held at every k, including
#     k = 1 where nothing is remotely bell-shaped. So what did the
#     central limit theorem actually give you -- the mean, the variance,
#     or the shape?
# ----------------------------------------------------------------------

print("=" * 74)
print("T5  sums of many things")
print("=" * 74)

for k in [1, 5, 30, 100]:
    sums = [sample_binomial(k, 0.1) for _ in range(5000)]
    print(f"k = {k}:  5000 sums of {k} draws of Bernoulli(0.1)")
    ascii_hist(sums)
    print(f"    mean     {mean_of(sums):>8.4f}   predicted k*0.1  = {k * 0.1:>8.4f}")
    print(f"    variance {var_of(sums):>8.4f}   predicted k*0.09 = {k * 0.09:>8.4f}")
    print()

# ANSWER: The mean k*p and the variance k*p*(1-p) held at every k, k = 1
#         included, because they follow from linearity of expectation and from
#         variances adding over independent terms -- neither argument needs a
#         bell. So the central limit theorem gave me only the SHAPE: it says a
#         large sum is Gaussian-looking, not where its centre is or how wide it
#         is, and those two I already had for free at every k.


# ======================================================================
# PART D  --  MAXIMUM LIKELIHOOD
# ======================================================================

# T6. The twenty coin flips at the top of the file. Read them with
#         flips = [int(c) for c in COIN]
#
#     Write likelihood(theta, flips): the probability of getting exactly
#     these flips if the coin comes up heads with probability theta. It
#     is theta for each head and (1 - theta) for each tail, all
#     multiplied together.
#
#     Then try every theta from 0.00 to 1.00 in steps of 0.01, and print
#     the one that gives the largest value. Compare it with
#     (number of heads) / 20.
#
#     You are not being asked to trust the lecture's formula. You are
#     checking every possible theta by brute force, and finding that the
#     best one is where the lecture said it would be.
# ----------------------------------------------------------------------

def likelihood(theta, flips):
    p = 1.0
    for f in flips:
        p *= theta if f == 1 else (1 - theta)
    return p


print("=" * 74)
print("T6  maximum likelihood, by brute force")
print("=" * 74)

flips = [int(c) for c in COIN]

best_theta = 0.0
best_value = -1.0
for step in range(101):
    theta = step / 100
    value = likelihood(theta, flips)
    if value > best_value:
        best_value = value
        best_theta = theta

print(f"Flips: {COIN}   heads = {sum(flips)} of {len(flips)}")
print(f"Best of the 101 thetas tried: {best_theta:.2f}   "
      f"likelihood = {best_value:.6e}")
print(f"heads / 20                  : {sum(flips) / len(flips):.2f}")


# T7. Print likelihood(0.3, flips * 65). That is the same twenty flips
#     repeated 65 times: 1300 flips, which is a small dataset.
#
#     Look at what comes out. Then write log_likelihood(theta, flips),
#     which adds up math.log(...) instead of multiplying, and print that
#     for the same 1300 flips.
#
#     One line in a comment: what happened to the first number, and why
#     does every library in this course work in logs?
# ----------------------------------------------------------------------

def log_likelihood(theta, flips):
    total = 0.0
    for f in flips:
        total += math.log(theta) if f == 1 else math.log(1 - theta)
    return total


print("=" * 74)
print("T7  why everything is done in logs")
print("=" * 74)

long_flips = flips * 65
print(f"{len(long_flips)} flips, {sum(long_flips)} of them heads")
print(f"likelihood(0.3, flips * 65)     = {likelihood(0.3, long_flips)}")
print(f"log_likelihood(0.3, flips * 65) = {log_likelihood(0.3, long_flips):.4f}")

# ANSWER: The first number came out as 0.0 -- 1300 factors smaller than 1
#         multiply down past the smallest double the machine can hold, so the
#         likelihood underflows and every theta looks equally impossible.
#         Working in logs turns that product into a sum of 1300 moderate terms
#         (about -794), which cannot underflow, and because log is increasing
#         the theta that maximises the log is the same one that maximises the
#         product. That is why every library here works in log space.


# T8. Maximum likelihood for the die is just counting: the estimate of
#     P(face k) is (number of times face k came up) / (total rolls).
#
#     Write fit_die(rolls) returning a list of six such estimates.
#     Check it on 100000 rolls -- you should get DIE back.
#
#     NOW BREAK IT. Face 6 has probability 0.02, so in a short run it
#     often does not appear at all. Fit the die on just 30 rolls, and do
#     that 100 times over. Count how many of the 100 estimates give face
#     6 a probability of exactly 0. Compare with 0.98^30, which you can
#     work out with a pen.
#
#     Then take one of those estimates and use it to score a NEW
#     sequence of ten rolls in which face 6 appears once -- multiply the
#     ten probabilities together. Print the result.
#
#     Two or three lines in a comment. Your estimate says face 6 is
#     impossible. It is not: you were told at the top of this file that
#     its probability is 0.02, and the die does not care what you saw in
#     thirty rolls. Is "I have never seen it, so it cannot happen" a
#     reasonable thing for an estimate to say? If not, what would you
#     rather it did -- and where could that information come from, given
#     that it is definitely not in your thirty rolls?
#
#     You are not expected to know the fix. The next lecture is the
#     answer.
# ----------------------------------------------------------------------

def fit_die(rolls):
    return [rolls.count(k) / len(rolls) for k in range(6)]


print("=" * 74)
print("T8  where maximum likelihood breaks")
print("=" * 74)

fit = fit_die([sample_die() for _ in range(100000)])
print("Fitted on 100000 rolls:")
for k in range(6):
    print(f"    Face {k + 1}: estimate {fit[k]:.3f}   true {DIE[k]:.2f}")

zeros = 0
broken = None
for _ in range(100):
    short_fit = fit_die([sample_die() for _ in range(30)])
    if short_fit[5] == 0.0:
        zeros += 1
        broken = short_fit

print()
print(f"Fits on 30 rolls giving face 6 probability exactly 0: {zeros} of 100")
print(f"0.98^30 = {0.98 ** 30:.4f}, so about {100 * 0.98 ** 30:.0f} of 100 were expected")

if broken is not None:
    print()
    print("One such estimate: " + ", ".join(f"{p:.3f}" for p in broken))
    new_rolls = [0, 2, 1, 0, 5, 3, 1, 4, 2, 0]   # index 5 = face 6, once
    score = 1.0
    for r in new_rolls:
        score *= broken[r]
    print(f"Scoring the new rolls {[r + 1 for r in new_rolls]}: {score}")

# ANSWER: The estimate hands face 6 a probability of 0, so the ten new rolls
#         score exactly 0.0 -- a single unseen face poisons the whole product
#         and nothing the other nine rolls do can rescue it.
#         "I have never seen it, so it cannot happen" is not a reasonable thing
#         for an estimate to say: unseen in thirty rolls should mean small, not
#         impossible, and the die goes on showing a 6 2% of the time whatever I
#         happened to observe. I would rather the estimate kept a little
#         probability on every face. That cannot come from the thirty rolls, so
#         it has to come from what was known beforehand -- a prior saying a
#         six-sided die can land on any of its six faces -- which is smoothing
#         / Bayesian estimation, i.e. the next lecture.


# ======================================================================
# IF YOU FINISH EARLY
# ======================================================================
#
# 1. Uncorrelated is not independent. Let X be equally likely to be -1,
#    0 or 1, and let Y = X * X. Estimate their covariance from samples:
#        cov = mean_of([x*y ...]) - mean_of(xs) * mean_of(ys)
#    You will get 0. But Y is computed from X, so they are about as
#    dependent as two things can be. Deck 3 asserted this; build the
#    counterexample yourself.
#
# 2. Sample a Gaussian. Box-Muller turns two uniforms into one standard
#    normal draw:
#        z = math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)
#    Then mu + sigma * z has mean mu and variance sigma^2. Check it with
#    mean_of() and var_of(). (Why 1 - u1 and not u1?)
#
# 3. In T5, how good is the bell at k = 100 in the middle, and how good
#    is it at the far right edge? The answer to the second half is why
#    Unit 4 needs sampling methods at all.
# ----------------------------------------------------------------------

print("=" * 74)
print("EXTRA 1  uncorrelated is not independent")
print("=" * 74)

xs = [random.choice([-1, 0, 1]) for _ in range(20000)]
ys = [x * x for x in xs]
cov = mean_of([x * y for x, y in zip(xs, ys)]) - mean_of(xs) * mean_of(ys)
print(f"Cov(X, Y) estimated from 20000 samples: {cov:.4f}")
print(f"P(Y = 0) = {ys.count(0) / len(ys):.3f}, "
      f"but P(Y = 0 | X = 0) = 1.000")

# ANSWER: The covariance is 0, yet Y is a function of X -- knowing X tells you
#         Y exactly. Covariance only sees LINEAR association, and here the
#         relation is a parabola, symmetric about 0, so the positive and
#         negative contributions cancel. Zero covariance therefore does not
#         imply independence; independence implies zero covariance, not the
#         other way round.

print()
print("=" * 74)
print("EXTRA 2  sampling a Gaussian with Box-Muller")
print("=" * 74)


def sample_gaussian(mu, sigma):
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)
    return mu + sigma * z


gauss = [sample_gaussian(5.0, 2.0) for _ in range(100000)]
check("mean of 100000 N(5, 4) draws", mean_of(gauss), 5.0, tol=0.05)
check("variance of 100000 N(5, 4) draws", var_of(gauss), 4.0, tol=0.10)

# ANSWER: random.random() returns a number in [0, 1), so u1 can be exactly 0.0
#         and math.log(0) blows up. 1 - u1 lies in (0, 1], which log always
#         accepts. The two are identically distributed, so nothing else changes.

print()
print("=" * 74)
print("EXTRA 3  how good is the bell, in the middle and in the tail")
print("=" * 74)

k, p = 100, 0.1
mu = k * p
sigma = math.sqrt(k * p * (1 - p))


def normal_cdf(x):
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def binomial_tail(c):
    return sum(math.comb(k, i) * p ** i * (1 - p) ** (k - i) for i in range(c, k + 1))


middle_exact = binomial_tail(8) - binomial_tail(13)
middle_bell = normal_cdf(12.5) - normal_cdf(7.5)
print(f"middle,    P(8 <= X <= 12):  exact {middle_exact:.5f}   "
      f"bell {middle_bell:.5f}   ratio {middle_bell / middle_exact:.3f}")

tail_exact = binomial_tail(25)
tail_bell = 1 - normal_cdf(24.5)
print(f"far right, P(X >= 25)     :  exact {tail_exact:.3e}   "
      f"bell {tail_bell:.3e}   ratio {tail_bell / tail_exact:.3f}")

# ANSWER: In the middle the bell is excellent -- within about a percent of the
#         true binomial. Out in the far right tail it is wrong by more than an
#         order of magnitude, and the further out you go the worse the ratio
#         gets: the Gaussian's tail dies off much faster than the real one.
#         Rare events are exactly what we care about in inference, and no
#         closed-form approximation gets them right, which is why Unit 4 has to
#         estimate them by sampling instead.

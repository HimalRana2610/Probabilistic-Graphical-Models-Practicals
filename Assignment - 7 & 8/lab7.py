"""Assignment 7: Hidden Markov Models.

Only the Python standard library is used.  Positions in the printed report are
1-based; lists and loops in the implementation are 0-based.
"""

from itertools import product
from math import inf, isclose, log, exp
import random


STATES = ("Fair", "Loaded")
ROLLS = tuple(int(value) for value in "615241364212641162565626666656")
INITIAL = (0.5, 0.5)
TRANSITION = (
    (0.95, 0.05),  # Fair -> Fair, Loaded
    (0.10, 0.90),  # Loaded -> Fair, Loaded
)
EMISSION = (
    (1 / 6,) * 6,
    (0.10, 0.10, 0.10, 0.10, 0.10, 0.50),
)


def check_rows(matrix):
    return [sum(row) for row in matrix]


def generate(length, rng):
    """Generate observations and the hidden state path from the HMM."""
    state = 0 if rng.random() < INITIAL[0] else 1
    states = []
    observations = []
    for _ in range(length):
        states.append(state)
        probabilities = EMISSION[state]
        draw = rng.random()
        cumulative = 0.0
        for roll, probability in enumerate(probabilities, start=1):
            cumulative += probability
            if draw < cumulative:
                observations.append(roll)
                break
        if state == 0:
            state = 0 if rng.random() < TRANSITION[0][0] else 1
        else:
            state = 0 if rng.random() < TRANSITION[1][0] else 1
    return observations, states


def forward(observations):
    """Return alpha values and P(observations) using ordinary probabilities."""
    alpha = []
    first = [
        INITIAL[state] * EMISSION[state][observations[0] - 1]
        for state in range(2)
    ]
    alpha.append(first)
    for observation in observations[1:]:
        alpha.append([
            EMISSION[state][observation - 1] * sum(
                alpha[-1][previous] * TRANSITION[previous][state]
                for previous in range(2)
            )
            for state in range(2)
        ])
    return alpha, sum(alpha[-1])


def explicit_probability(observations):
    total = 0.0
    for path in product(range(2), repeat=len(observations)):
        probability = INITIAL[path[0]] * EMISSION[path[0]][observations[0] - 1]
        for position in range(1, len(observations)):
            probability *= (
                TRANSITION[path[position - 1]][path[position]]
                * EMISSION[path[position]][observations[position] - 1]
            )
        total += probability
    return total


def logsumexp(values):
    largest = max(values)
    if largest == -inf:
        return -inf
    return largest + log(sum(exp(value - largest) for value in values))


def log_forward(observations):
    log_alpha = [[
        log(INITIAL[state]) + log(EMISSION[state][observations[0] - 1])
        for state in range(2)
    ]]
    for observation in observations[1:]:
        log_alpha.append([
            log(EMISSION[state][observation - 1]) + logsumexp([
                log_alpha[-1][previous] + log(TRANSITION[previous][state])
                for previous in range(2)
            ])
            for state in range(2)
        ])
    return log_alpha, logsumexp(log_alpha[-1])


def scaled_forward(observations):
    """Return normalized forward vectors and log P(observations)."""
    first = [
        INITIAL[state] * EMISSION[state][observations[0] - 1]
        for state in range(2)
    ]
    scale = sum(first)
    alpha = [[value / scale for value in first]]
    log_probability = log(scale)
    for observation in observations[1:]:
        values = [
            EMISSION[state][observation - 1] * sum(
                alpha[-1][previous] * TRANSITION[previous][state]
                for previous in range(2)
            )
            for state in range(2)
        ]
        scale = sum(values)
        alpha.append([value / scale for value in values])
        log_probability += log(scale)
    return alpha, log_probability


def backward(observations):
    beta = [[0.0, 0.0] for _ in observations]
    beta[-1] = [1.0, 1.0]
    for position in range(len(observations) - 2, -1, -1):
        next_observation = observations[position + 1]
        beta[position] = [
            sum(
                TRANSITION[state][next_state]
                * EMISSION[next_state][next_observation - 1]
                * beta[position + 1][next_state]
                for next_state in range(2)
            )
            for state in range(2)
        ]
    probability = sum(
        INITIAL[state] * EMISSION[state][observations[0] - 1] * beta[0][state]
        for state in range(2)
    )
    return beta, probability


def posterior(observations):
    alpha, probability = forward(observations)
    beta, _ = backward(observations)
    smoothed = []
    for position in range(len(observations)):
        values = [
            alpha[position][state] * beta[position][state]
            for state in range(2)
        ]
        normalizer = sum(values)
        smoothed.append([value / normalizer for value in values])
    filtering = [[value / sum(values) for value in values] for values in alpha]
    return filtering, smoothed, probability


def viterbi(observations):
    scores = [[
        log(INITIAL[state]) + log(EMISSION[state][observations[0] - 1])
        for state in range(2)
    ]]
    backpointers = []
    for observation in observations[1:]:
        pointers = []
        row = []
        for state in range(2):
            candidates = [
                scores[-1][previous] + log(TRANSITION[previous][state])
                for previous in range(2)
            ]
            predecessor = max(range(2), key=candidates.__getitem__)
            pointers.append(predecessor)
            row.append(candidates[predecessor] + log(
                EMISSION[state][observation - 1]
            ))
        backpointers.append(pointers)
        scores.append(row)
    path = [max(range(2), key=scores[-1].__getitem__)]
    for pointers in reversed(backpointers):
        path.append(pointers[path[-1]])
    path.reverse()
    return path, scores[-1][path[-1]]


def operation_counts(length, states=2):
    # Direct enumeration evaluates one initial term and (T-1) transitions
    # for every path; the recurrence evaluates each predecessor/state pair.
    direct = states ** length * (1 + max(0, length - 1) * 2)
    forward_count = states + max(0, length - 1) * states * (states + 1)
    return direct, forward_count


def state_names(path):
    return "".join("FL"[state] for state in path)


def report():
    print("ASSIGNMENT 7: HIDDEN MARKOV MODELS")
    print("=" * 38)

    # (a) Each initial, transition, and emission row must sum to one.
    print("\n1(a) Row sums")
    row_sums = [sum(INITIAL), *check_rows(TRANSITION), *check_rows(EMISSION)]
    print("initial:", f"{row_sums[0]:.12f}")
    print("transition:", [f"{value:.12f}" for value in row_sums[1:3]])
    print("emission:", [f"{value:.12f}" for value in row_sums[3:]])
    print("all rows sum to 1:", all(isclose(value, 1.0) for value in row_sums))

    # (b) Graph for T=4: S1 -> S2 -> S3 -> S4, with Oi hanging from Si.
    # The chain asserts Oi is independent of all other variables given Si;
    # S_t separates the past (S_1:t-1, O_1:t-1) from the future
    # (S_t+1:T, O_t+1:T).
    print("\n1(b) Graph: S1 -> S2 -> S3 -> S4; S1->O1, S2->O2,")
    print("     S3->O3, S4->O4. Given S_t, past and future are independent.")

    rng = random.Random(20261015)
    generated_observations, generated_path = generate(300, rng)
    loaded_fraction = sum(generated_path) / len(generated_path)
    stationary_loaded = TRANSITION[0][1] / (
        TRANSITION[0][1] + TRANSITION[1][0]
    )
    print("\n1(c) Generated loaded proportion:", f"{loaded_fraction:.6f}")
    print("     Stationary loaded probability:", f"{stationary_loaded:.6f}")
    sixes = ROLLS.count(6)
    print("\n1(d) Sixes in supplied sequence:", sixes)
    print("     Fair-die expectation:", f"{len(ROLLS) / 6:.1f}")
    print("     The excess alone cannot locate the loaded stretch; positions and")
    print("     transition persistence are also needed.")

    # (e) alpha[t,s] = P(O_1,...,O_t, S_t=s).
    print("\n2(e) alpha(t,s) = P(O_1:t and S_t=s), a joint probability.")
    alpha, probability = forward(ROLLS)
    print("2(f) P(supplied observations):", f"{probability:.12g}")

    short = ROLLS[:5]
    explicit = explicit_probability(short)
    _, recursive = forward(short)
    print("2(g) Length-5 explicit:", f"{explicit:.16g}")
    print("     Length-5 forward:", f"{recursive:.16g}",
          "agree:", isclose(explicit, recursive, rel_tol=1e-14, abs_tol=1e-15))
    print("2(h) Operation counts (direct enumeration, forward recurrence)")
    for length in (5, 20, 300):
        print(f"     T={length}: {operation_counts(length)}")
    print("     Direct: exponential O(K^T*T); forward: O(T*K^2).")

    # (i)-(l) Direct probabilities eventually underflow; log and scaled forms do not.
    print("\n3(i) Underflow experiment")
    underflow_rng = random.Random(7)
    first_underflow = None
    for length in (50, 100, 200, 300, 400, 500, 800):
        observations, _ = generate(length, underflow_rng)
        values, direct_probability = forward(observations)
        if direct_probability == 0.0 and first_underflow is None:
            first_underflow = length
        print(f"     T={length:3d}: min alpha={min(map(min, values)):.3e}, "
              f"P={direct_probability:.3e}")
    print("     First tested length with P(observations) == 0:", first_underflow)
    print("3(j-k) Log-space and scaled log probabilities")
    for length in (50, 100, 200, 300, 400, 500, 800):
        observations, _ = generate(length, random.Random(length))
        _, direct_probability = forward(observations)
        _, log_probability = log_forward(observations)
        _, scaled_log_probability = scaled_forward(observations)
        print(f"     T={length:3d}: direct={direct_probability:.3e}, "
              f"log={log_probability:.6f}, scaled={scaled_log_probability:.6f}")
    print("     Log space preserves tiny values without leaving probability space.")
    print("     Scaling keeps vectors normalized and is convenient for backward passes.")

    # (m-o) beta carries observations after t; gamma is the smoothed posterior.
    beta, backward_probability = backward(ROLLS)
    filtering, smoothing, _ = posterior(ROLLS)
    print("\n4(m-n) Backward P(observations):", f"{backward_probability:.12g}",
          "agrees:", isclose(probability, backward_probability, rel_tol=1e-14))
    print("4(o-p) Position  Filtered Loaded  Smoothed Loaded")
    for position, (filtered, smooth) in enumerate(zip(filtering, smoothing), 1):
        print(f"     {position:2d}       {filtered[1]:.6f}          {smooth[1]:.6f}")
    differences = [abs(filtering[i][1] - smoothing[i][1]) for i in range(len(ROLLS))]
    largest_difference = max(range(len(ROLLS)), key=differences.__getitem__)
    print("4(q) Largest filter/smoother difference: position",
          largest_difference + 1, f"({differences[largest_difference]:.6f})")
    print("     The smoother also knows observations to the right of that position.")

    # (r-t) Viterbi changes the sum in forward's recurrence to a maximum.
    viterbi_path, viterbi_log_probability = viterbi(ROLLS)
    print("\n5(r) Forward uses log-sum-exp; Viterbi uses max.")
    print("5(s) Viterbi path:", state_names(viterbi_path))
    print("     Viterbi log probability:", f"{viterbi_log_probability:.12f}")
    print("5(t) Back-pointers retain the maximizing predecessor. Choosing the")
    print("     best state independently afterward can join incompatible transitions.")

    accuracy = []
    for seed in range(10):
        observations, true_path = generate(300, random.Random(100 + seed))
        decoded, _ = viterbi(observations)
        accuracy.append(sum(a == b for a, b in zip(decoded, true_path)) / 300)
    print("\n5(u) Viterbi accuracy over 10 generated sequences:",
          ", ".join(f"{value:.3f}" for value in accuracy))
    print("     Mean:", f"{sum(accuracy) / len(accuracy):.6f}")

    # (v-x) Posterior decoding independently maximizes each smoothed marginal.
    posterior_path = [max(range(2), key=values.__getitem__) for values in smoothing]
    disagreements = sum(a != b for a, b in zip(viterbi_path, posterior_path))
    print("\n6(v) Posterior path:", state_names(posterior_path))
    print("     Viterbi/posterior disagreements:", disagreements)
    print("6(w) Ten-sequence comparison (Viterbi accuracy, posterior accuracy, disagreements)")
    comparison = []
    for seed in range(10):
        observations, true_path = generate(300, random.Random(200 + seed))
        decoded, _ = viterbi(observations)
        _, smoothed, _ = posterior(observations)
        posterior_decoded = [max(range(2), key=row.__getitem__) for row in smoothed]
        comparison.append((
            sum(a == b for a, b in zip(decoded, true_path)) / 300,
            sum(a == b for a, b in zip(posterior_decoded, true_path)) / 300,
            sum(a != b for a, b in zip(decoded, posterior_decoded)),
        ))
        print("     ", f"{comparison[-1][0]:.3f}", f"{comparison[-1][1]:.3f}",
              comparison[-1][2])
    print("6(x) Viterbi maximizes the probability of the entire sequence.")
    print("     Posterior decoding maximizes each position separately and often")
    print("     gets more positions right. With a transition matrix having zero")
    print("     self-transitions, e.g. [[0,1],[1,0]], posterior choices can")
    print("     alternate incorrectly and produce a sequence of probability zero.")


if __name__ == "__main__":
    report()
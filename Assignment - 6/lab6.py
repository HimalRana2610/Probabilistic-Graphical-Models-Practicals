"""
AI453 - Lab 6: Bayes-Ball and the Markov Blanket
Department of Artificial Intelligence, SVNIT Surat

Standard library only. networkx and pgmpy are not permitted.

Graded modes (stdout carries only the answers; diagnostics go to stderr):

    python3 lab6.py case_p6_main.in              -> one YES/NO line per query
    python3 lab6.py case_p6_main.in --blanket    -> one line per node, its blanket

Analysis mode (not graded; works through every task of the handout and
prints the tables quoted in the ANSWERS block at the bottom of this file):

    python3 lab6.py --report [DIR]

DIR (default: the folder holding this file) must contain the four
case_p6_*.in files. lab4.py, lab5_bn.py and student-por.csv are looked for in
DIR and then in the sibling "Assignment - 4" / "Assignment - 5" folders.
"""

import sys
import os
import time
import itertools
import importlib.util
from collections import deque


# ==========================================================================
# PROBLEM STATEMENT 1 - Graph construction and ancestor computation
# ==========================================================================
#   n m                     n nodes numbered 0..n-1, m edges
#   u v                     m lines, one per directed edge u -> v
#   q                       number of queries
#   x y z1 z2 ... zk        q lines; the pair, then the evidence set Z
# --------------------------------------------------------------------------

def read_graph(path):
    """(a) Returns n, parents, children, queries.

    parents[v] and children[v] are lists; queries is a list of (x, y, Z)
    with Z a set (possibly empty).
    """
    with open(path) as f:
        lines = [ln.split() for ln in f if ln.strip() != ""]
    n, m = int(lines[0][0]), int(lines[0][1])
    parents = [[] for _ in range(n)]
    children = [[] for _ in range(n)]
    for a, b in lines[1:1 + m]:
        u, v = int(a), int(b)
        parents[v].append(u)
        children[u].append(v)
    queries = []
    if len(lines) > 1 + m:
        q = int(lines[1 + m][0])
        for ln in lines[2 + m: 2 + m + q]:
            nums = [int(t) for t in ln]
            queries.append((nums[0], nums[1], set(nums[2:])))
    return n, parents, children, queries


def ancestors_of(parents, Z):
    """(b) Every node with a directed path into Z, Z itself included.

    Iterative upward sweep with an explicit stack, so a long chain cannot
    overflow Python's recursion limit. Each node is pushed once per incoming
    edge and expanded once: O(n + m).

    (c) A node counts as its own ancestor because the collider rule is
    "v is in Z OR some descendant of v is in Z". Putting Z inside its own
    ancestor set folds both halves into one membership test,
    v in ancestors_of(parents, Z).
    """
    seen = set()
    stack = list(Z)
    while stack:
        v = stack.pop()
        if v not in seen:
            seen.add(v)
            stack.extend(parents[v])
    return seen


# ==========================================================================
# PROBLEM STATEMENT 2 - Bayes-Ball reachability
# ==========================================================================
UP, DOWN = 0, 1          # UP: ball arrived from a child; DOWN: from a parent


def reachable(parents, children, x, Z, stats=None):
    """(d) The set of nodes NOT d-separated from x by Z.

    The search state is a (node, direction) pair. Rules:
        (v, UP),   v not in Z       -> every parent UP, every child DOWN
        (v, UP),   v in Z           -> nothing (chain / fork blocked)
        (v, DOWN), v not in Z       -> every child DOWN
        (v, DOWN), v in anc(Z)      -> every parent UP (collider opened)
    A node that is not in Z but is an ancestor of Z satisfies both DOWN rules.

    (e) `visited` holds PAIRS. With a node-level visited set, the first
    arrival at v fixes which way the ball may leave v. If v is first reached
    DOWN (from a parent) and v is not an ancestor of Z, it may only continue
    to its children; the later arrival (v, UP) - which would have continued
    to v's parents and their other descendants - is discarded as "already
    seen". Every trail that passes through v as a chain node going upwards,
    or as a fork, is then lost. buggy_reachable_nodes() and
    counterexample_graph() below build a 6-node graph on which this flips
    an answer from NO to YES.

    (f) A node of Z is never put into the result even though the ball
    touches it (it must, to bounce off an observed collider). d-separation
    is defined between sets disjoint from the evidence: an observed node is
    a constant given Z, and "w is independent of x given Z" is trivially
    true for w in Z. Reporting it as reachable would claim a dependence that
    cannot exist.

    stats, if given, receives 'pops' (pairs popped) and 'visited' (the set
    of pairs) for the complexity analysis of Problem Statement 4.
    """
    anc = ancestors_of(parents, Z)
    visited = {(x, UP)}          # a pair is marked when queued, so it is
    queue = deque([(x, UP)])     # queued - and popped - at most once
    result = set()
    pops = 0

    def push(state):
        if state not in visited:
            visited.add(state)
            queue.append(state)

    while queue:
        v, d = queue.popleft()
        pops += 1
        if v not in Z:
            result.add(v)
        if d == UP and v not in Z:
            for p in parents[v]:
                push((p, UP))
            for c in children[v]:
                push((c, DOWN))
        elif d == DOWN:
            if v not in Z:
                for c in children[v]:
                    push((c, DOWN))
            if v in anc:
                for p in parents[v]:
                    push((p, UP))
    if stats is not None:
        stats["pops"] = pops
        stats["visited"] = visited
    return result


def d_separated(parents, children, x, y, Z, stats=None):
    """(h) x and y are d-separated by Z exactly when y is not reachable."""
    return y not in reachable(parents, children, x, Z, stats)


def buggy_reachable_nodes(parents, children, x, Z):
    """(e) The same search with visited kept over NODES. Do not use."""
    anc = ancestors_of(parents, Z)
    seen = set()
    result = set()
    queue = deque([(x, UP)])
    while queue:
        v, d = queue.popleft()
        if v in seen:
            continue
        seen.add(v)
        if v not in Z:
            result.add(v)
        if d == UP and v not in Z:
            queue.extend((p, UP) for p in parents[v])
            queue.extend((c, DOWN) for c in children[v])
        elif d == DOWN:
            if v not in Z:
                queue.extend((c, DOWN) for c in children[v])
            if v in anc:
                queue.extend((p, UP) for p in parents[v])
    return result


def counterexample_graph():
    """(e) A graph on which node-level visiting gives the wrong answer.

        0:x   1:p   2:a1   3:a2   4:v   5:w
        p -> x,  p -> v,  v -> a2 -> a1 -> x,  w -> v

    Query: x, w given Z = {}. The trail x <- a1 <- a2 <- v <- w is a pure
    chain with nothing observed, so it is active: the answer is NO.
    Breadth-first from (x, UP): depth 1 reaches (p, UP) and (a1, UP);
    depth 2 reaches (v, DOWN) via p -> v, before depth 3 reaches (v, UP)
    via a2. A node-level visited set keeps only (v, DOWN), which may only
    go to children, so w is never reached and the buggy search says YES.
    """
    edges = [(1, 0), (1, 4), (4, 3), (3, 2), (2, 0), (5, 4)]
    n = 6
    parents = [[] for _ in range(n)]
    children = [[] for _ in range(n)]
    for u, v in edges:
        parents[v].append(u)
        children[u].append(v)
    return n, parents, children, (0, 5, set())


# ==========================================================================
# PROBLEM STATEMENT 5 - Markov blanket
# ==========================================================================

def blanket(parents, children, v, coparents=True):
    """(p) Parents, children and co-parents of v; sorted, unique, without v.

    coparents=False drops the third term; it exists only for task (u).
    """
    mb = set(parents[v]) | set(children[v])
    if coparents:
        for c in children[v]:
            mb.update(parents[c])
    mb.discard(v)
    return sorted(mb)


# ==========================================================================
# Graded entry points
# ==========================================================================

def answer_queries(path, out=sys.stdout, err=sys.stderr):
    """(h), (l) YES/NO per query on stdout; pair counts on stderr."""
    n, parents, children, queries = read_graph(path)
    total = worst = 0
    for x, y, Z in queries:
        st = {}
        sep = d_separated(parents, children, x, y, Z, st)
        out.write("YES\n" if sep else "NO\n")
        total += st["pops"]
        worst = max(worst, len(st["visited"]))
    if err is not None:
        m = sum(len(c) for c in children)
        err.write("[pairs] n=%d m=%d 2n=%d queries=%d max-distinct-per-query=%d "
                  "total-popped=%d\n" % (n, m, 2 * n, len(queries), worst, total))


def print_blankets(path, out=sys.stdout):
    """(q) Line i is the blanket of node i; an empty blanket is an empty line."""
    n, parents, children, _ = read_graph(path)
    for v in range(n):
        out.write(" ".join(str(u) for u in blanket(parents, children, v)) + "\n")


# ==========================================================================
# Analysis helpers (used by --report only)
# ==========================================================================

def trail_is_active(trail, parents_sets, Z, anc):
    """Literal Lab-4 definition: check every interior triple of one trail."""
    for i in range(1, len(trail) - 1):
        a, v, b = trail[i - 1], trail[i], trail[i + 1]
        collider = a in parents_sets[v] and b in parents_sets[v]
        if collider:
            if v not in anc:
                return False
        elif v in Z:
            return False
    return True


def enumerate_trails(parents, children, x, y, Z, budget_s):
    """Naive d-separation: enumerate every simple trail x ... y, test each.

    Returns (answer or None if abandoned, trails_checked, seconds).
    """
    anc = ancestors_of(parents, Z)
    pset = [set(p) for p in parents]
    nbrs = [list(parents[v]) + list(children[v]) for v in range(len(parents))]
    t0 = time.perf_counter()
    checked = 0
    path = [x]
    on_path = {x}
    stack = [iter(nbrs[x])]
    while stack:
        nxt = next(stack[-1], None)
        if nxt is None:
            stack.pop()
            on_path.discard(path.pop())
            continue
        if nxt in on_path:
            continue
        if nxt == y:
            checked += 1
            if trail_is_active(path + [y], pset, Z, anc):
                return False, checked, time.perf_counter() - t0
            if checked % 4096 == 0 and time.perf_counter() - t0 > budget_s:
                return None, checked, time.perf_counter() - t0
            continue
        path.append(nxt)
        on_path.add(nxt)
        stack.append(iter(nbrs[nxt]))
    return True, checked, time.perf_counter() - t0


def _find(filename, base, sibling):
    for d in (base, os.path.join(base, "..", sibling)):
        p = os.path.normpath(os.path.join(d, filename))
        if os.path.exists(p):
            return p
    return None


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STUDENT_NAMES = ["medu", "goout", "fail", "higher", "study", "absent", "g1", "grade"]


def report(base):
    cases = ["case_p6_sample.in", "case_p6_student.in",
             "case_p6_main.in", "case_p6_wide.in"]
    paths = {c: os.path.join(base, c) for c in cases}
    G = {c: read_graph(paths[c]) for c in cases}

    def h(title):
        print("\n" + "=" * 72 + "\n" + title + "\n" + "=" * 72)

    # ---- (c) -----------------------------------------------------------
    h("PS1 (c) ancestors on case_p6_sample.in")
    _, par, ch, _ = G["case_p6_sample.in"]
    print("ancestors_of({3}) =", sorted(ancestors_of(par, {3})))

    # ---- (e) -----------------------------------------------------------
    h("PS2 (e) pairs vs nodes in the visited set")
    n, par, ch, (x, y, Z) = counterexample_graph()
    good = y not in reachable(par, ch, x, Z)
    bad = y not in buggy_reachable_nodes(par, ch, x, Z)
    print("edges: 1->0 1->4 4->3 3->2 2->0 5->4   query x=0 y=5 Z={}")
    print("visited over pairs :", "YES" if good else "NO", "(correct)")
    print("visited over nodes :", "YES" if bad else "NO", "(wrong)")
    flips = 0
    for c in cases:
        _, par, ch, qs = G[c]
        for qx, qy, qZ in qs:
            if (qy in reachable(par, ch, qx, qZ)) != (qy in buggy_reachable_nodes(par, ch, qx, qZ)):
                flips += 1
    print("answers changed by the node-level bug on the four case files:", flips)

    # ---- (g) -----------------------------------------------------------
    h("PS2 (g) nodes visited in both directions, case_p6_student.in")
    _, par, ch, qs = G["case_p6_student.in"]
    for i, (qx, qy, qZ) in enumerate(qs, 1):
        st = {}
        reachable(par, ch, qx, qZ, st)
        both = sorted(v for v in range(len(par))
                      if (v, UP) in st["visited"] and (v, DOWN) in st["visited"])
        print("q%-2d x=%d y=%d Z=%-10s both directions: %s"
              % (i, qx, qy, sorted(qZ), ", ".join("%d(%s)" % (v, STUDENT_NAMES[v]) for v in both)))

    # ---- (i), (j) ------------------------------------------------------
    h("PS3 (i)/(j) cross-validation against the Lab 4 program")
    lab4_path = _find("lab4.py", base, "Assignment - 4")
    lab4 = _load_module("lab4", lab4_path) if lab4_path else None
    if lab4 is None:
        print("lab4.py not found; skipped")
    else:
        print("Lab 4 program:", lab4_path)
        for c in cases:
            _, n4c, n4p, desc, qs = lab4.read_input(paths[c])
            _, par, ch, _ = G[c]
            dis = []
            for i, (qx, qy, qZ) in enumerate(qs, 1):
                a4 = "NO" if lab4.has_active_trail(qx, qy, qZ, n4c, n4p, desc) else "YES"
                a6 = "YES" if d_separated(par, ch, qx, qy, qZ) else "NO"
                if a4 != a6:
                    dis.append((i, a4, a6))
            print("%-20s %2d queries, disagreements: %s" % (c, len(qs), dis or "none"))

    # ---- (k) -----------------------------------------------------------
    h("PS3 (k) answers for case_p6_main.in")
    _, par, ch, qs = G["case_p6_main.in"]
    for i, (qx, qy, qZ) in enumerate(qs, 1):
        print("q%-2d %2d %2d | %-14s -> %s" % (i, qx, qy, " ".join(map(str, sorted(qZ))),
              "YES" if d_separated(par, ch, qx, qy, qZ) else "NO"))

    # ---- (m), (n) ------------------------------------------------------
    h("PS4 (m)/(n) pair counts and timing")
    print("%-20s %4s %4s %4s %5s | %12s %12s %10s" %
          ("file", "n", "m", "2n", "q", "max distinct", "max popped", "time (ms)"))
    for c in cases:
        n, par, ch, qs = G[c]
        m = sum(len(x) for x in ch)
        t0 = time.perf_counter()
        md = mp = 0
        for qx, qy, qZ in qs:
            st = {}
            d_separated(par, ch, qx, qy, qZ, st)
            md = max(md, len(st["visited"]))
            mp = max(mp, st["pops"])
        ms = (time.perf_counter() - t0) * 1000
        print("%-20s %4d %4d %4d %5d | %12d %12d %10.3f" % (c, n, m, 2 * n, len(qs), md, mp, ms))
    n, par, ch, qs = G["case_p6_wide.in"]
    print("\ncase_p6_wide.in, per query:")
    for qx, qy, qZ in qs:
        st = {}
        t0 = time.perf_counter()
        sep = d_separated(par, ch, qx, qy, qZ, st)
        ms = (time.perf_counter() - t0) * 1000
        print("  %2d %2d | %-6s -> %-3s  distinct pairs %3d  popped %3d  %.3f ms"
              % (qx, qy, " ".join(map(str, sorted(qZ))), "YES" if sep else "NO",
                 len(st["visited"]), st["pops"], ms))

    # ---- (o) -----------------------------------------------------------
    h("PS4 (o) trail-based programs on case_p6_wide.in")
    budget = 10.0
    for qx, qy, qZ in qs[:3]:
        ans, cnt, sec = enumerate_trails(par, ch, qx, qy, qZ, budget)
        print("literal trail enumeration %2d %2d | %-6s : %s after %d trails, %.2f s"
              % (qx, qy, " ".join(map(str, sorted(qZ))),
                 "ABANDONED" if ans is None else ("YES" if ans else "NO"), cnt, sec))
    if lab4 is not None:
        _, n4c, n4p, desc, qs4 = lab4.read_input(paths["case_p6_wide.in"])
        t0 = time.perf_counter()
        ans = ["NO" if lab4.has_active_trail(a, b, z, n4c, n4p, desc) else "YES" for a, b, z in qs4]
        print("Lab 4 program (has_active_trail), all 7 queries: %s  %.3f ms"
              % (" ".join(ans), (time.perf_counter() - t0) * 1000))

    # ---- (r), (s) ------------------------------------------------------
    h("PS5 (r)/(s) blanket sizes, case_p6_main.in")
    n, par, ch, _ = G["case_p6_main.in"]
    sizes = []
    for v in range(n):
        b = blanket(par, ch, v)
        sizes.append(len(b))
        print("node %2d  |pa|=%d |ch|=%d  size %2d  %s" % (v, len(par[v]), len(ch[v]), len(b), b))
    print("largest :", max(sizes), [v for v in range(n) if sizes[v] == max(sizes)])
    print("smallest:", min(sizes), [v for v in range(n) if sizes[v] == min(sizes)])
    leaves = [v for v in range(n) if not ch[v]]
    print("childless nodes, size == number of parents:",
          all(len(blanket(par, ch, v)) == len(par[v]) for v in leaves), leaves)

    # ---- (t), (u), (v) -------------------------------------------------
    h("PS6 (t)/(u)/(v) the blanket separates")
    for mode, label in ((True, "full blanket"), (False, "co-parents removed")):
        tot = 0
        for c in cases:
            n, par, ch, _ = G[c]
            fails = []
            for v in range(n):
                b = blanket(par, ch, v, coparents=mode)
                reach = reachable(par, ch, v, set(b))
                for w in range(n):
                    if w != v and w not in b and w in reach:
                        fails.append((v, w))
            tot += len(fails)
            print("%-19s %-20s failures: %d %s" % (label, c, len(fails),
                  fails if len(fails) <= 12 else str(fails[:12]) + " ..."))
        print("%-19s total failures: %d\n" % (label, tot))
    n, par, ch, qs = G["case_p6_sample.in"]
    for i, (qx, qy, qZ) in enumerate(qs, 1):
        for a, b in ((qx, qy), (qy, qx)):
            if set(blanket(par, ch, a, coparents=False)) == qZ and \
               not d_separated(par, ch, a, b, qZ):
                print("sample query %d: '%d %d %s' -> Z is the co-parent-free blanket "
                      "of %d, answer NO" % (i, qx, qy, " ".join(map(str, sorted(qZ))), a))
                break

    # ---- (w) -----------------------------------------------------------
    h("PS7 (w) blankets of the Practical 5 network")
    n, par, ch, _ = G["case_p6_student.in"]
    for v in range(n):
        print("%-7s: %s" % (STUDENT_NAMES[v], ", ".join(STUDENT_NAMES[u] for u in blanket(par, ch, v))))

    # ---- (x), (y) ------------------------------------------------------
    h("PS7 (x)/(y) numerical check with the Practical 5 CPTs")
    lab5_path = _find("lab5_bn.py", base, "Assignment - 5")
    csv_path = _find("student-por.csv", base, "Assignment - 5")
    if not (lab5_path and csv_path):
        print("lab5_bn.py / student-por.csv not found; skipped")
        return
    L = _load_module("lab5_bn", lab5_path)
    data = L.load(csv_path)
    picks = [("g1", "medu"), ("study", "absent"), ("absent", "higher")]
    for alpha in (0.0, 1.0):
        cpt = L.fit(data, alpha=alpha)
        print("alpha = %.0f%s" % (alpha, "  (raw MLE, as in Practical 5)" if alpha == 0
                                   else "  (Laplace-smoothed, Practical 5 T7)"))
        print("  %-7s %-37s %-9s %-16s %-16s %s" %
              ("node", "blanket (all = 1)", "extra", "P(v=1|mb)", "P(v=1|mb,extra)", "|diff|"))
        for v, extra in picks:
            vi = STUDENT_NAMES.index(v)
            mb = [STUDENT_NAMES[u] for u in blanket(par, ch, vi)]
            assert STUDENT_NAMES.index(extra) not in blanket(par, ch, vi)
            ev = {u: 1 for u in mb}
            p1 = L.infer({v: 1}, ev, cpt)
            p2 = L.infer({v: 1}, dict(ev, **{extra: 1}), cpt)
            pe = sum(L.joint(dict(zip(STUDENT_NAMES, bits)), cpt)
                     for bits in itertools.product([0, 1], repeat=8)
                     if all(dict(zip(STUDENT_NAMES, bits))[u] == 1 for u in mb))
            print("  %-7s %-37s %-9s %.14f %.14f %.1e%s" %
                  (v, ",".join(mb), extra + "=1", p1, p2, abs(p1 - p2),
                   "  [P(mb)=%.2e]" % pe))


# ==========================================================================
# ANSWERS  (numbers produced by  python3 lab6.py --report)
# ==========================================================================
#
# PS1 (c)  ancestors_of({3}) on case_p6_sample.in = {0, 1, 2, 3}
#          (3 <- 1 <- 0 and 1 <- 2). A node is its own ancestor so that the
#          collider test "v in Z or a descendant of v in Z" is one lookup.
#
# PS2 (e)  Pairs, not nodes: on counterexample_graph() (edges 1->0 1->4
#          4->3 3->2 2->0 5->4, query 0 5 | {}) the pair search answers NO
#          (correct, trail 0<-2<-3<-4<-5 is open), the node search answers
#          YES. The trails lost are those that must leave a node through its
#          parents after the node was first entered from a parent.
#          (None of the four case files happens to trigger it: 0 changes.)
#     (f)  Z nodes are excluded: an observed variable is a constant given Z,
#          so it cannot be dependent on x given Z; the ball visits it only
#          to bounce off an observed collider.
#     (g)  case_p6_student.in, query 5 "0 1 7" (medu, goout | grade): nodes
#          2 fail, 3 higher, 4 study, 5 absent and 6 g1 are each visited
#          both as (v, up) and (v, down). E.g. g1 is entered DOWN from
#          study (medu -> study -> g1) and, since g1 is an ancestor of the
#          observed grade, the ball bounces UP into its parents; g1 is also
#          entered UP from grade, because the observed grade bounces every
#          ball that arrives from above back up into its parents.
#
# PS3 (i)  diff lab4_output.txt lab6_output_sample.txt prints nothing
#          (after normalising Lab 4's CRLF line endings on Windows).
#     (j)  Lab 4 and Lab 6 agree on all 12 + 12 + 40 + 7 queries of the four
#          files, so there was no disagreement to diagnose.
#     (k)  lab6_output.txt = python3 lab6.py case_p6_main.in (40 lines).
#
# PS4 (m)  file                  n    m   2n  max pairs popped per query
#          case_p6_sample.in     5    4   10    7
#          case_p6_student.in    8   14   16   13
#          case_p6_main.in      15   22   30   21
#          case_p6_wide.in      91  120  182   92
#     (n)  case_p6_wide.in: all 7 queries in about 0.16 ms in total; the
#          hardest, "0 90 | 45", pops 90 pairs in about 0.04 ms.
#     (o)  A literal trail enumerator (enumerate_trails) checked about
#          1.7 million trails of "0 90 | 45" in 10 s and was abandoned;
#          at that rate the 2^30 trails need roughly 1.7 hours per query.
#          (The Lab 4 program submitted earlier already memoised
#          (node, previous node, direction) states rather than whole trails,
#          so it finished the file in under 1 ms and does not show the blow-up.)
#          Bound: each (node, direction) pair is queued at most once, so at
#          most 2n pairs are popped, and each popped pair scans its parent
#          and child lists once, so each edge is looked at a constant number
#          of times: O(n + m) per query. Trails never appear because the
#          search state forgets how it got to (v, d); every trail reaching
#          v in direction d is merged into that one state.
#
# PS5 (r)  Blanket sizes in case_p6_main.in, node: size
#          0:5 1:4 2:8 3:5 4:8 5:8 6:4 7:6 8:5 9:7 10:9 11:3 12:3 13:3 14:2
#          largest: node 10 (9 nodes); smallest: node 14 (2 nodes: 2, 10).
#     (s)  A node with no children has no co-parents either, so its blanket
#          is exactly its parent set: size = in-degree (nodes 11-14 here).
#
# PS6 (t)  Full blanket: 0 failures on all four networks.
#     (u)  Without co-parents: 3 + 20 + 86 + 60 = 169 failures.
#     (v)  case_p6_sample.in queries 2 "0 2 | 1" and 5 "0 4 | 1" (answer NO).
#          Without co-parents the blanket of 0 is just {1}. But 1 is a
#          collider 0 -> 1 <- 2, and observing it OPENS the trail 0 -> 1 <- 2
#          (and on to 4 via 2 -> 4). Conditioning on a child always opens
#          the collider formed by its parents, so the other parents of that
#          child must themselves be in the blanket to close those trails.
#
# PS7 (w)  medu  : goout, fail, higher, study
#          goout : medu, fail, higher, study, absent, g1
#          fail  : medu, goout, higher, study, absent, g1
#          higher: medu, goout, fail, study, g1
#          study : medu, goout, fail, higher, g1
#          absent: goout, fail, g1, grade
#          g1    : goout, fail, higher, study, absent, grade
#          grade : absent, g1
#          grade has the smallest blanket (2): it is the sink of the network,
#          so it has no children and no co-parents - only its parents g1 and
#          absent. Once both are known, nothing else tells us more about it.
#     (x)  P(g1=1 | goout,fail,higher,study,absent,grade = 1)
#              alpha=1: 0.59574468085106  with medu=1: 0.59574468085106
#              (|diff| ~ 1e-16, i.e. floating-point round-off only)
#          With the raw alpha=0 CPTs both are exactly 0, because the row
#          P(g1=1 | fail=1, study=1, goout=1, higher=1) is estimated from 2
#          students; they agree, but trivially, hence the smoothed CPTs.
#     (y)  node    added      P(v=1 | mb)        P(v=1 | mb, added)
#          study   absent=1   0.17322834645669   0.17322834645669
#          absent  higher=1   0.32304038004751   0.32304038004751
#          (alpha=1; with alpha=0 absent gives 0.31874159163821 both ways.)
#     (z)  Only the node's blanket: its own CPT and the CPTs of its
#          children, which mention just parents, children and co-parents.
#          That is a constant amount of work, independent of the other
#          ~10,000 nodes. Gibbs sampling (MCMC) relies on exactly this:
#          P(v | everything else) = P(v | blanket(v)).


# ==========================================================================
# Main
# ==========================================================================

def main(argv):
    try:
        sys.stdout.reconfigure(newline="\n")     # LF line endings on Windows too
    except AttributeError:
        pass
    if len(argv) >= 2 and argv[1] == "--report":
        base = argv[2] if len(argv) > 2 else os.path.dirname(os.path.abspath(__file__))
        report(base)
        return 0
    if len(argv) < 2:
        sys.stderr.write("usage: python3 lab6.py FILE [--blanket] | --report [DIR]\n")
        return 1
    if "--blanket" in argv[2:]:
        print_blankets(argv[1])
    else:
        answer_queries(argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

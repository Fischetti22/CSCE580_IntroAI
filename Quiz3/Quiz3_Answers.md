# CSCE 580: Quiz 3 Answers
## Student Name: Pedro H Fischetti

---

## Q1: Search and Heuristics [10 points]

### a) What is an admissible heuristic? [2 points]
- In plain terms, a heuristic is admissible if it never overestimates the true cheapest cost to the goal
- Formally: h(n) ≤ h*(n) for all n (where h*(n) is the actual optimal cost)
- Intuitively, it’s an “optimistic” estimate — at or below the real cost
- With A*, an admissible h keeps the solution optimal

### b) Is h = 0 admissible? [2 points]
- Yes — since actual costs are ≥ 0, h=0 never overestimates
- It’s basically “no guidance,” so A* reduces to uniform-cost search (Dijkstra)
- Admissible, but not informative; I treat it as a sanity-check baseline

### c) Is h = k (where k is any constant, e.g., k=1) admissible? [2 points]
- Not in general — it depends on the problem and the value of k
- Counterexample: if k=1 but you’re already at the goal (true cost 0), then h=1 overestimates
- h=k is admissible only if k ≤ h*(n) for every reachable state (a strong requirement)

### d) Given h1, h2, h3 with at least one admissible, what about min and max? [4 points]

**For h = min(h1, h2, h3):**
- Yes, h = min(h1, h2, h3) is admissible
- Since at least one heuristic (say h1) is admissible: h1 ≤ h*
- The minimum of three values is at most equal to any one of them: min(h1, h2, h3) ≤ h1 ≤ h*
- Therefore, the minimum never overestimates the true cost

**For h = max(h1, h2, h3):**
- No, we cannot guarantee h = max(h1, h2, h3) is admissible
- If h2 or h3 are not admissible, they may overestimate: h2 > h* or h3 > h*
- Then max(h1, h2, h3) could equal the non-admissible heuristic and overestimate
- The maximum is only guaranteed admissible if ALL three heuristics are admissible

---

## Q2: Using Search for a Practical Problem [70 points]

### Q2.1: State Representation and Search Strategy [10 points]

**State Representation [5 points]:**
The provided code uses the following state representation:
- A state consists of 5 components: `(left_missionaries, left_cannibals, right_missionaries, right_cannibals, boat_position)`
- `left_missionaries`: number of missionaries on the left bank
- `left_cannibals`: number of cannibals on the left bank
- `right_missionaries`: number of missionaries on the right bank
- `right_cannibals`: number of cannibals on the right bank
- `boat_position`: either "left" or "right" indicating which bank the boat is on

**Goal State:**
- The goal state is expressed as: `(0, 0, initial_missionaries, initial_cannibals, "right")`
- All missionaries and cannibals have moved to the right bank with the boat on the right side

**Search Strategy [5 points]:**
- The code implements **Breadth-First Search (BFS)**
- Frontier structure: Uses a FIFO queue (`collections.deque`) to explore nodes level by level; in contrast, the DFS variant uses a LIFO stack (`list`).
- Guarantees finding the shortest solution (optimal for unweighted graphs)
- BFS halts as soon as `goal_state()` is true, so the first goal reached is the optimal (least-cost) solution
- The `bfs()` function at line 70 implements this strategy with `append()` (enqueue) and `popleft()` (dequeue)
- Maintains an explored list to avoid revisiting states and prevent cycles

---

## Q2.2 & Q2.3: Implementation and Testing [60 points]

### Q2.2: Different Search Strategy Implementation [30 points]

**Implementation Choice: Depth-First Search (DFS)**

For Q2.2, I chose DFS to contrast with BFS. Main differences I observed:

**Code Changes:**
1. **Data Structure**: Changed from `deque` (queue) to regular `list` (stack)
2. **Node Removal**: Changed from `queue.popleft()` (FIFO) to `stack.pop()` (LIFO)
3. **Search Behavior**: DFS explores deeply before backtracking, BFS explores level-by-level

**Key Code Differences (MCAgent_DFS.py line 61-77):**
```python
def dfs():  # depth-first-search (DFS)
    initial_state = States(initial_missionaries, initial_cannibals, 0, 0, "left")
    if initial_state.goal_state():
        return initial_state
    stack = []  # Use a stack (list) for DFS instead of queue
    explored = []
    stack.append(initial_state)
    while stack:
        node = stack.pop()  # Pop from end (LIFO) instead of popleft (FIFO)
        if node.goal_state():
            return node
        explored.append(node)
        node_children = successors(node)
        for child in node_children:
            if (child not in explored) and (child not in stack):
                stack.append(child)
    return None
```

See complete implementations in:
- `MCAgent_BFS.py` - Original BFS implementation
- `MCAgent_DFS.py` - New DFS implementation
- `MCAgent_Tester.py` - Comprehensive testing script with timing

### Q2.3: Test Results for All 6 Cases [30 points]

**SUMMARY TABLE:**

```text
| Test Case              | Algorithm | Moves | Time (ms) |
|------------------------|-----------|-------|-----------|
| 1M, 1C                 | BFS       | 1     | ~4.30     |
| 1M, 1C                 | DFS       | 1     | ~0.04     |
| 2M, 2C                 | BFS       | 5     | ~0.16     |
| 2M, 2C                 | DFS       | 5     | ~0.11     |
| 3M, 3C                 | BFS       | 11    | ~0.16     |
| 3M, 3C                 | DFS       | 11    | ~0.12     |
| 4M, 3C                 | BFS       | 11    | ~0.18     |
| 4M, 3C                 | DFS       | 11    | ~0.11     |
| 5M, 3C                 | BFS       | 13    | ~0.31     |
| 5M, 3C                 | DFS       | 13    | ~0.14     |
| 2M, 3C (No solution)   | BFS       | N/A   | ~0.09     |
| 2M, 3C (No solution)   | DFS       | N/A   | ~0.07     |
```

All results verified against detailed run logs in `Test_Results.txt`.

Note: timings are approximate and can vary a bit run-to-run depending on the machine load.

**DETAILED SOLUTIONS:**

**Test Case 1: 1M, 1C**
- BFS Solution: [(1, 1)]
- DFS Solution: [(1, 1)]
- Number of moves: 1
- Explanation: Move 1 missionary and 1 cannibal across in one trip

**Test Case 2: 2M, 2C**
- BFS Solution: [(0, 2), (0, 1), (2, 0), (1, 0), (1, 1)]
- DFS Solution: [(0, 2), (0, 1), (2, 0), (1, 0), (1, 1)]
- Number of moves: 5
- Strategy: Send cannibals first, then missionaries, ensuring no outnumbering

**Test Case 3: 3M, 3C (Classic Problem)**
- BFS Solution: [(0, 2), (0, 1), (0, 2), (0, 1), (2, 0), (1, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
- DFS Solution: [(0, 2), (0, 1), (0, 2), (0, 1), (2, 0), (1, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
- Number of moves: 11
- Strategy: Multiple back-and-forth trips to maintain safety constraints

**Test Case 4: 4M, 3C**
- BFS Solution: [(0, 2), (0, 1), (2, 0), (1, 0), (1, 1), (0, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
- DFS Solution: [(0, 2), (0, 1), (2, 0), (1, 0), (1, 1), (0, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
- Number of moves: 11
- Strategy: Similar pattern with extra missionary

**Test Case 5: 5M, 3C**
- BFS Solution: [(2, 0), (1, 0), (1, 1), (1, 0), (2, 0), (1, 0), (1, 1), (1, 0), (2, 0), (1, 0), (2, 0), (1, 0), (1, 1)]
- DFS Solution: [(2, 0), (1, 0), (1, 1), (1, 0), (2, 0), (1, 0), (1, 1), (1, 0), (2, 0), (1, 0), (2, 0), (1, 0), (1, 1)]
- Number of moves: 13
- Strategy: Missionaries can travel in larger groups safely

**Test Case 6: 2M, 3C**
- BFS Solution: [] (No solution)
- DFS Solution: [] (No solution)
- Explanation: Impossible - missionaries would always be outnumbered somewhere

**ANALYSIS:**

**BFS Performance:**
- ✓ Guarantees optimal (shortest) solution
- ✓ Found same or better solution length in all cases
- Uses FIFO queue, explores level-by-level
- Slightly slower in execution due to queue operations

**DFS Performance:**
- May not find optimal solution (but did in these cases)
- ✓ Generally faster execution time
- Uses LIFO stack, explores depth-first
- Less memory usage for deep searches

**Observations:**
- Both BFS and DFS returned the same solution lengths for all solvable cases
- DFS tended to be a bit faster on my runs (likely lower overhead), but times do fluctuate slightly
- Both correctly flagged 2M, 3C as impossible
- I spot-checked the move sequences; they satisfy the constraints at each step

For complete detailed output with verification steps, see `Test_Results.txt`

---

## Q3: Formulating a Search Problem [20 points]

### Q3a: CSP Formulation for TWO + TWO = FOUR [15 points]

**Problem:**
```
  T W O
+ T W O
---------
F O U R
```

**Variables [5 points]:**
- Six variables: T, W, O, F, U, R
- Each letter stands for a distinct digit (0-9)

**Domains [5 points]:**
- T, F: {1,2,3,4,5,6,7,8,9} (no leading zeros)
- W, O, U, R: {0,1,2,3,4,5,6,7,8,9} (all digits okay)

**Constraints [5 points]:**
1. **Uniqueness constraint (AllDifferent)**: All variables must have different values
   - alldiff(T, W, O, F, U, R)
2. **Arithmetic constraints** (column-by-column addition with carries):
   - Units place: O + O = R (mod 10), with carry C1 ∈ {0, 1}
   - Tens place: W + W + C1 = U (mod 10), with carry C2 ∈ {0, 1}
   - Hundreds place: T + T + C2 = O (mod 10), with carry C3 ∈ {0, 1}
   - Thousands place: C3 = F
3. **Global arithmetic equation**: 2 × (100T + 10W + O) = 1000F + 100O + 10U + R
4. **Unary constraints**: T ≠ 0, F ≠ 0 (leading digits cannot be zero)

### Q3b: Non-search Simplification Methods [5 points]

I’ll use arc consistency (AC-3) to simplify this before any backtracking search.

**Pseudo-code for Arc Consistency (AC-3 algorithm):**

```
**function** APPLY_ARC_CONSISTENCY():
    // Initialize work queue with all arcs
    queue = []
    **for each** variable X:
        **for each** constraint involving X and Y:
            queue.add((X, Y))
    
    **while** queue is not empty:
        (Xi, Xj) = queue.remove_first()
        
        **if** REVISE(Xi, Xj):
            **if** domain(Xi) is empty:
                **return** "No solution"
            
            // Add all neighbors of Xi back to queue
            **for each** Xk that has constraint with Xi (except Xj):
                queue.add((Xk, Xi))
    
    **return** "Domains reduced"

**function** REVISE(Xi, Xj):
    revised = false
    
    **for each** value vi in domain(Xi):
        // Check if there exists a value in Xj's domain that satisfies constraint
        **if** no value vj in domain(Xj) satisfies constraint(Xi=vi, Xj=vj):
            remove vi from domain(Xi)
            revised = true
    
    **return** revised
```

**Application to TWO + TWO = FOUR:**

1. **Node Consistency** (enforce unary constraints):
   - Remove 0 from domains of T and F (leading digits)
   - Initial: domain(T) = domain(F) = {1,2,3,4,5,6,7,8,9}

2. **Arc Consistency** (apply AC-3 to reduce domains through constraint propagation):
   - From constraint O + O = R (mod 10), if O = 5, then R = 0 (with C1 = 1)
   - From T + T + C2 = O (mod 10) and knowing C3 = F, we get F = 1 (since max T = 9, C2 = 1, gives 2×9 + 1 = 19, carry = 1)
   - With F = 1, from C3 = 1, we know T + T + C2 ≥ 10
   - This allows **progressive domain reduction** without search

3. **Result**: Arc consistency significantly reduces the search space before any **backtracking search** is needed

**After AC-3 domain reduction:**
- F must be 1 (because max carry from hundreds place is 1)
- From O + O = R (mod 10), trying O=4 gives R=8, carry=0; O=5 gives R=0, carry=1, etc.
- Working through the constraints manually:
  - If O=4, then 2*O=8=R, C1=0; 2*W+0=U(mod10); 2*T+C2=4(mod10), need C3=1=F
  - Turns out T=7, W=3, O=4 works: 734 + 734 = 1468 ✓

---

## Summary

This quiz covered search algorithms (admissible heuristics, BFS vs DFS) and constraint satisfaction (CSP formulation, arc consistency). 

For the missionaries & cannibals problem, BFS gave optimal solutions and DFS matched it in these test cases (though DFS doesn’t always guarantee shortest path). For the TWO+TWO=FOUR cryptarithmetic puzzle, framing it as a CSP and applying AC-3 narrowed down the domains quite a bit before needing a full search.

Overall, the combination of informed/uninformed search plus constraint reasoning seems like a pretty versatile approach for different AI problem types.

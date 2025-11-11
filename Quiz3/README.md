# Quiz 3 - CSCE 580 Fall 2025 by Pedro H Fischetti
## Missionaries and Cannibals Problem & CSP

### 📁 Submission Contents

This folder contains the complete solution for Quiz 3:

#### Main Documents
- **`Quiz3_Answers.md`** - Complete answers to all quiz questions (Q1, Q2, Q3)
- **`Quiz3-CSCE580-Fall2025.pdf`** - Original quiz questions

#### Code Files
- **`MCAgent_BFS.py`** - Breadth-First Search implementation (original algorithm)
- **`MCAgent_DFS.py`** - Depth-First Search implementation (Q2.2 requirement)
- **`MCAgent_Tester.py`** - Comprehensive test script with timing analysis

#### Results
- **`Test_Results.txt`** - Full test output with verification for all 6 test cases

---

### 🎯 Quiz Structure

**Total: 100 points**

#### Q1: Search and Heuristics [10 points]
- Concepts of admissible heuristics
- Analysis of h=0, h=k, min/max of heuristics
- ✓ Completed in `Quiz3_Answers.md`

#### Q2: Using Search for Practical Problem [70 points]
- **Q2.1** [10 pts]: Analyze state representation and search strategy
- **Q2.2** [30 pts]: Implement different search strategy (DFS)
- **Q2.3** [30 pts]: Run all 6 test cases with timing
- ✓ All implementations working and tested

#### Q3: Formulating a CSP [20 points]
- **Q3a** [15 pts]: CSP formulation for TWO + TWO = FOUR
- **Q3b** [5 pts]: Arc consistency pseudo-code
- ✓ Completed with detailed analysis

---

### 🚀 How to Run the Code

#### Run All Tests (Recommended)
```bash
cd /home/droski/Desktop/School/Fall25/AI/CSCE580_IntroAI/Quiz3
python3 MCAgent_Tester.py
```

#### Run Individual Algorithms
```python
# Test BFS
from MCAgent_BFS import MCAgent as MCAgent_BFS
agent = MCAgent_BFS()
solution = agent.solve(3, 3)  # 3 missionaries, 3 cannibals
print(solution)

# Test DFS
from MCAgent_DFS import MCAgent as MCAgent_DFS
agent = MCAgent_DFS()
solution = agent.solve(3, 3)
print(solution)
```

---

### 📊 Test Results Summary

| Test Case | Algorithm | Moves | Time (ms) | Status |
|-----------|-----------|-------|-----------|--------|
| 1M, 1C    | BFS       | 1     | ~4.3      | ✓ Pass |
|           | DFS       | 1     | ~0.04     | ✓ Pass |
| 2M, 2C    | BFS       | 5     | ~0.16     | ✓ Pass |
|           | DFS       | 5     | ~0.11     | ✓ Pass |
| 3M, 3C    | BFS       | 11    | ~0.16     | ✓ Pass |
|           | DFS       | 11    | ~0.12     | ✓ Pass |
| 4M, 3C    | BFS       | 11    | ~0.18     | ✓ Pass |
|           | DFS       | 11    | ~0.11     | ✓ Pass |
| 5M, 3C    | BFS       | 13    | ~0.31     | ✓ Pass |
|           | DFS       | 13    | ~0.14     | ✓ Pass |
| 2M, 3C    | BFS       | N/A   | ~0.09     | ✓ No Solution |
|           | DFS       | N/A   | ~0.07     | ✓ No Solution |

All solutions verified as valid and optimal!

---

### 🔍 Key Insights

**BFS vs DFS Comparison:**
- **BFS**: Guarantees shortest path, uses more memory, level-by-level exploration
- **DFS**: May find longer paths, uses less memory, depth-first exploration
- **Result**: Both found optimal solutions in all test cases

**State Representation:**
- `(left_missionaries, left_cannibals, right_missionaries, right_cannibals, boat_position)`
- Goal: `(0, 0, initial_m, initial_c, "right")`

**CSP Solution (TWO + TWO = FOUR):**
- Variables: T, W, O, F, U, R
- Solution: 734 + 734 = 1468
- Arc consistency reduces search space significantly

---

### 📝 Notes for Submission

1. **All code is functional** and has been tested
2. **All 6 test cases pass** with verification
3. **Complete documentation** provided in Quiz3_Answers.md
4. **Timing data captured** for performance analysis
5. **CSP formulation** includes detailed constraints and arc consistency

---

### 🎓 Student Information

**Student Name:** [Fill in your name in Quiz3_Answers.md]

**Submission Date:** November 11, 2025

**Course:** CSCE 580 - Introduction to AI / Trusted AI  
**Instructor:** Prof. Biplav Srivastava

---

### ✅ Checklist

- [x] Q1: All 4 parts answered with bullet points
- [x] Q2.1: State representation and search strategy identified
- [x] Q2.2: DFS implementation completed
- [x] Q2.3: All 6 test cases run with timing
- [x] Q3a: CSP formulation (variables, domains, constraints)
- [x] Q3b: Arc consistency pseudo-code provided
- [x] Code files included and functional
- [x] Test results documented
- [ ] Student name added to Quiz3_Answers.md


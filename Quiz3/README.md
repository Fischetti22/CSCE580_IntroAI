# Quiz 3 - CSCE 580 Fall 2025
## Missionaries and Cannibals Problem & CSP
**Student:** Pedro H Fischetti

---

## 📁 Project Structure

```
Quiz3/
├── 📄 Quiz3_Answers.md              # Complete written answers (Q1, Q2, Q3)
├── 📓 Quiz3_Complete_Solution.ipynb # Interactive Jupyter notebook
├── 📖 README.md                     # This file
│
├── 📂 code/                         # All Python implementations
│   ├── MCAgent_BFS.py              # Breadth-First Search
│   ├── MCAgent_DFS.py              # Depth-First Search  
│   └── MCAgent_Tester.py           # Test suite with timing
│
├── 📂 data/                         # Test results and outputs
│   └── Test_Results.txt            # Full test logs (6 test cases)
│
└── 📂 docs/                         # Documentation and instructions
    └── Quiz3-CSCE580-Fall2025.pdf  # Original quiz questions
```

---

## 🎯 Quiz Overview

**Total Points: 100**

### Q1: Search and Heuristics [10 points]
- Concepts of admissible heuristics
- Analysis of h=0, h=k, min/max of heuristics
- ✓ Completed with theoretical explanations

### Q2: Using Search for Practical Problem [70 points]
- **Q2.1** [10 pts]: Analyze state representation and search strategy
- **Q2.2** [30 pts]: Implement different search strategy (DFS)
- **Q2.3** [30 pts]: Run all 6 test cases with timing
- ✓ All implementations working and tested

### Q3: Formulating a CSP [20 points]
- **Q3a** [15 pts]: CSP formulation for TWO + TWO = FOUR
- **Q3b** [5 pts]: Arc consistency pseudo-code
- ✓ Completed with solution verification (734 + 734 = 1468)

---

## 🚀 How to Run

### Option 1: Run All Tests (Recommended)
```bash
cd Quiz3
python3 code/MCAgent_Tester.py
```

### Option 2: Interactive Jupyter Notebook
```bash
cd Quiz3
jupyter notebook Quiz3_Complete_Solution.ipynb
# Run all cells to see results, timing, and analysis
```

### Option 3: Run Individual Algorithms
```python
import sys
sys.path.append('code')

# Test BFS
from MCAgent_BFS import MCAgent as MCAgent_BFS
agent = MCAgent_BFS()
solution = agent.solve(3, 3)  # 3 missionaries, 3 cannibals
print(f"BFS Solution: {solution}")

# Test DFS
from MCAgent_DFS import MCAgent as MCAgent_DFS
agent = MCAgent_DFS()
solution = agent.solve(3, 3)
print(f"DFS Solution: {solution}")
```

---

## 📊 Test Results Summary

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

## 🔍 Key Insights

### BFS vs DFS Comparison

**Breadth-First Search (BFS):**
- ✓ Guarantees optimal (shortest) solution
- Uses FIFO queue (`collections.deque`)
- Explores level-by-level
- Halts when goal is first reached
- Slightly slower but finds shortest path

**Depth-First Search (DFS):**
- May not guarantee optimal solution
- Uses LIFO stack (regular `list`)
- Explores depth-first before backtracking
- ✓ Generally faster execution time
- Less memory usage

**Result:** Both found optimal solutions in all solvable test cases!

---

## 📝 File Descriptions

### Main Documents
- **`Quiz3_Answers.md`** - Complete written answers with analysis (302 lines)
  - All questions answered in required format
  - Includes code explanations and test results
  - Professional formatting with markdown
  
- **`Quiz3_Complete_Solution.ipynb`** - Interactive Jupyter notebook (29KB)
  - Executable code cells
  - Live test results
  - Visualization-ready format
  - Can be run in Jupyter or exported to PDF/HTML

### Code Directory (`code/`)
- **`MCAgent_BFS.py`** (4.7KB) - Breadth-First Search implementation
  - FIFO queue using `collections.deque`
  - Guarantees optimal solution
  - Used as baseline algorithm
  
- **`MCAgent_DFS.py`** (4.7KB) - Depth-First Search implementation  
  - LIFO stack using regular Python `list`
  - Explores depth-first
  - Q2.2 requirement (alternative search strategy)
  
- **`MCAgent_Tester.py`** (5.3KB) - Comprehensive test suite
  - Tests both BFS and DFS on 6 cases
  - Measures execution timing
  - Verifies solutions step-by-step
  - Displays formatted results

### Data Directory (`data/`)
- **`Test_Results.txt`** (13KB) - Complete test output
  - All 6 test cases with detailed verification
  - Move-by-move state transitions
  - Constraint validation at each step
  - Performance timing for both algorithms

### Docs Directory (`docs/`)
- **`Quiz3-CSCE580-Fall2025.pdf`** (109KB) - Original quiz questions
  - Problem statements
  - Point breakdown
  - Submission instructions

---

## 🎓 Submission Information

**Student:** Pedro H Fischetti  
**Date:** November 11, 2025  
**Course:** CSCE 580 - Introduction to AI / Trusted AI  
**Instructor:** Prof. Biplav Srivastava

---

## ✅ Completion Checklist

### Q1: Search and Heuristics [10 points]
- [x] a) Admissible heuristics explained [2 pts]
- [x] b) h=0 analysis with justification [2 pts]
- [x] c) h=k analysis with counter-examples [2 pts]
- [x] d) min/max of heuristics proven [4 pts]

### Q2: Missionaries & Cannibals [70 points]
- [x] Q2.1: State representation & BFS strategy identified [10 pts]
- [x] Q2.2: DFS implementation completed [30 pts]
- [x] Q2.3: All 6 test cases run with timing [30 pts]
  - [x] 1M, 1C: 1 move ✓
  - [x] 2M, 2C: 5 moves ✓
  - [x] 3M, 3C: 11 moves ✓
  - [x] 4M, 3C: 11 moves ✓
  - [x] 5M, 3C: 13 moves ✓
  - [x] 2M, 3C: No solution (correctly identified) ✓

### Q3: CSP Formulation [20 points]
- [x] Q3a: Variables, domains, constraints defined [15 pts]
  - Variables: T, W, O, F, U, R
  - Domains: Specified with leading digit constraints
  - Constraints: AllDifferent, arithmetic, unary
- [x] Q3b: AC-3 pseudo-code with application [5 pts]
  - Complete AC-3 algorithm provided
  - Application to TWO + TWO = FOUR
  - Solution verified: 734 + 734 = 1468 ✓

### Deliverables
- [x] All code functional and tested
- [x] Written answers complete and formatted
- [x] Jupyter notebook with interactive results
- [x] Test results documented with verification
- [x] Student name included in all files
- [x] Professional organization and structure
- [x] Clear documentation (README, comments)

---

## 📚 Summary & Key Takeaways

This quiz demonstrates comprehensive understanding of:

**Search Algorithms:**
- Admissible heuristics ensure optimal solutions in informed search
- BFS guarantees shortest path for unweighted graphs
- DFS trades optimality for memory efficiency
- Both strategies successfully solved the Missionaries & Cannibals problem

**Constraint Satisfaction:**
- CSP formulation requires clear variables, domains, and constraints
- Arc consistency (AC-3) reduces search space before backtracking
- Constraint propagation can dramatically simplify complex problems

**Integration:** The combination of search strategies and constraint reasoning forms a complete toolkit for AI problem-solving.

---

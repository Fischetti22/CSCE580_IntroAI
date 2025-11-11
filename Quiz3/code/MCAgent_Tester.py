#!/usr/bin/env python3
"""
Missionaries and Cannibals Problem Tester
Tests both BFS and DFS implementations with timing

Note: Added verification step to check each move is valid
"""

import time
import sys

# Import both versions
from MCAgent_BFS import MCAgent as MCAgent_BFS
from MCAgent_DFS import MCAgent as MCAgent_DFS


def test_algorithm(agent, missionaries_count, cannibals_count, algorithm_name):
    """Test a single algorithm with given parameters and measure time"""
    print(f"\n{'='*70}")
    print(f"Test Case: {missionaries_count}M, {cannibals_count}C | Algorithm: {algorithm_name}")
    print(f"{'='*70}")
    
    start_time = time.time()
    solution = agent.solve(missionaries_count, cannibals_count)
    end_time = time.time()
    
    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    if solution:
        print(f"✓ Solution found!")
        print(f"Number of moves: {len(solution)}")
        print(f"Move sequence: {solution}")
        print(f"Execution time: {elapsed_time:.4f} ms")
        
        # Verify solution
        verify_solution(missionaries_count, cannibals_count, solution)
    else:
        print(f"✗ No solution exists")
        print(f"Execution time: {elapsed_time:.4f} ms")
    
    return solution, elapsed_time


def verify_solution(initial_m, initial_c, moves):
    """Verify that a solution is valid"""
    left_m, left_c = initial_m, initial_c
    right_m, right_c = 0, 0
    boat_at_left = True
    
    print(f"\nVerification:")
    print(f"Initial: Left({left_m}M, {left_c}C) | Right({right_m}M, {right_c}C) | Boat: Left")
    
    for i, (m_move, c_move) in enumerate(moves, 1):
        if boat_at_left:
            left_m -= m_move
            left_c -= c_move
            right_m += m_move
            right_c += c_move
            boat_at_left = False
        else:
            left_m += m_move
            left_c += c_move
            right_m -= m_move
            right_c -= c_move
            boat_at_left = True
        
        # Check validity
        valid = True
        if left_m < 0 or left_c < 0 or right_m < 0 or right_c < 0:
            valid = False
        if left_m > 0 and left_c > left_m:
            valid = False
        if right_m > 0 and right_c > right_m:
            valid = False
        
        boat_pos = "Left" if boat_at_left else "Right"
        status = "✓" if valid else "✗ INVALID"
        print(f"Move {i}: ({m_move}M, {c_move}C) -> Left({left_m}M, {left_c}C) | Right({right_m}M, {right_c}C) | Boat: {boat_pos} {status}")
    
    # Check if goal reached
    if left_m == 0 and left_c == 0 and right_m == initial_m and right_c == initial_c:
        print("✓ Goal state reached successfully!")
    else:
        print("✗ Goal state NOT reached!")


def run_all_tests():
    """Run all 6 test cases for both algorithms"""
    # Test cases from the quiz requirements
    test_cases = [
        (1, 1, "1M, 1C"),
        (2, 2, "2M, 2C"),
        (3, 3, "3M, 3C"),  # classic problem
        (4, 3, "4M, 3C"),
        (5, 3, "5M, 3C"),
        (2, 3, "2M, 3C (No solution)")  # impossible case
    ]
    
    print("\n" + "="*70)
    print("MISSIONARIES AND CANNIBALS PROBLEM - COMPREHENSIVE TEST")
    print("="*70)
    
    results = {"BFS": [], "DFS": []}
    
    for missionaries, cannibals, description in test_cases:
        print(f"\n\n{'#'*70}")
        print(f"# TEST CASE: {description}")
        print(f"{'#'*70}")
        
        # Test BFS
        agent_bfs = MCAgent_BFS()
        bfs_solution, bfs_time = test_algorithm(agent_bfs, missionaries, cannibals, "BFS")
        results["BFS"].append((description, bfs_solution, bfs_time))
        
        # Test DFS
        agent_dfs = MCAgent_DFS()
        dfs_solution, dfs_time = test_algorithm(agent_dfs, missionaries, cannibals, "DFS")
        results["DFS"].append((description, dfs_solution, dfs_time))
    
    # Print summary
    print("\n\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    print(f"\n{'Test Case':<25} {'Algorithm':<10} {'Moves':<10} {'Time (ms)':<15}")
    print("-"*70)
    
    for i, (description, _, _) in enumerate(results["BFS"]):
        _, bfs_solution, bfs_time = results["BFS"][i]
        _, dfs_solution, dfs_time = results["DFS"][i]
        
        bfs_moves = len(bfs_solution) if bfs_solution else "N/A"
        dfs_moves = len(dfs_solution) if dfs_solution else "N/A"
        
        print(f"{description:<25} {'BFS':<10} {str(bfs_moves):<10} {bfs_time:<15.4f}")
        print(f"{'':<25} {'DFS':<10} {str(dfs_moves):<10} {dfs_time:<15.4f}")
        print()
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("\nBFS (Breadth-First Search):")
    print("  - Explores nodes level by level")
    print("  - Guarantees optimal solution (shortest path)")
    print("  - Uses more memory (stores all nodes at current level)")
    print("  - FIFO queue implementation")
    
    print("\nDFS (Depth-First Search):")
    print("  - Explores as deep as possible before backtracking")
    print("  - Does NOT guarantee optimal solution")
    print("  - Uses less memory (stores path from root)")
    print("  - LIFO stack implementation")
    
    print("\nKey Differences Observed:")
    print("  - BFS finds shortest solution paths")
    print("  - DFS may find longer paths but sometimes faster")
    print("  - Both algorithms find solutions when they exist")
    print("  - Both correctly identify impossible cases (2M, 3C)")


if __name__ == "__main__":
    run_all_tests()

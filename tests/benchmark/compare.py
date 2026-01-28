#!/usr/bin/env python3
"""
compare.py - Compare benchmark results between two runs.

Usage:
    python3 compare.py baseline.json current.json [--threshold 10]

Exit codes:
    0 - No significant regressions
    1 - Significant regression detected (>threshold%)
"""

import json
import sys
from typing import Dict, List, Tuple

def load_results(path: str) -> Dict[str, dict]:
    """Load benchmark results from JSON file."""
    with open(path) as f:
        data = json.load(f)
    return {r["name"]: r for r in data["results"]}

def compare_results(
    baseline: Dict[str, dict],
    current: Dict[str, dict],
    threshold: float = 10.0
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Compare benchmark results.
    
    Returns:
        (regressions, improvements, unchanged)
    """
    regressions = []
    improvements = []
    unchanged = []
    
    for name, curr in current.items():
        if name not in baseline:
            continue
            
        base = baseline[name]
        base_mean = base["mean_ns"]
        curr_mean = curr["mean_ns"]
        
        if base_mean == 0:
            continue
            
        change_pct = ((curr_mean - base_mean) / base_mean) * 100
        
        result = {
            "name": name,
            "baseline_ns": base_mean,
            "current_ns": curr_mean,
            "change_pct": change_pct
        }
        
        if change_pct > threshold:
            regressions.append(result)
        elif change_pct < -threshold:
            improvements.append(result)
        else:
            unchanged.append(result)
    
    return regressions, improvements, unchanged

def print_results(
    regressions: List[dict],
    improvements: List[dict],
    unchanged: List[dict],
    threshold: float
) -> None:
    """Print comparison results."""
    print("=" * 70)
    print("BENCHMARK COMPARISON REPORT")
    print("=" * 70)
    print()
    
    if regressions:
        print(f"❌ REGRESSIONS (>{threshold}% slower):")
        print("-" * 70)
        for r in sorted(regressions, key=lambda x: -x["change_pct"]):
            print(f"  {r['name']:<45} +{r['change_pct']:>6.1f}%  "
                  f"({r['baseline_ns']:.0f} → {r['current_ns']:.0f} ns)")
        print()
    
    if improvements:
        print(f"✅ IMPROVEMENTS (>{threshold}% faster):")
        print("-" * 70)
        for r in sorted(improvements, key=lambda x: x["change_pct"]):
            print(f"  {r['name']:<45} {r['change_pct']:>6.1f}%  "
                  f"({r['baseline_ns']:.0f} → {r['current_ns']:.0f} ns)")
        print()
    
    print(f"📊 SUMMARY:")
    print(f"  Regressions:  {len(regressions)}")
    print(f"  Improvements: {len(improvements)}")
    print(f"  Unchanged:    {len(unchanged)}")
    print()

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    baseline_path = sys.argv[1]
    current_path = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    
    try:
        baseline = load_results(baseline_path)
        current = load_results(current_path)
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading results: {e}")
        sys.exit(1)
    
    regressions, improvements, unchanged = compare_results(
        baseline, current, threshold
    )
    
    print_results(regressions, improvements, unchanged, threshold)
    
    if regressions:
        print(f"❌ FAILED: {len(regressions)} benchmark(s) regressed by >{threshold}%")
        sys.exit(1)
    else:
        print("✅ PASSED: No significant regressions detected")
        sys.exit(0)

if __name__ == "__main__":
    main()

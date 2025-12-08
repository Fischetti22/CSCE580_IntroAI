#!/usr/bin/env python3
"""
CSCE 580 - Attendance Audit System
Processes handwritten attendance sheets and generates reports.

Pedro Fischetti
December 2025
"""

import os
import json
from datetime import datetime
import statistics

# Manual extraction results from vision-based analysis of attendance images
# Each entry: (class_number, date_string, attendance_count)
ATTENDANCE_DATA = [
    (1, "Aug 19, 2025", 44),
    (2, "Aug 21, 2025", 49),
    (3, "Aug 26, 2025", 45),
    (4, "Aug 28, 2025", 31),
    (5, "Sep 2, 2025", 41),
    (6, "Sep 4, 2025", 40),
    (7, "Sep 9, 2025", 42),
    (8, "Sep 11, 2025", 37),
    (9, "Sep 16, 2025", 33),
    (10, "Sep 18, 2025", 35),
    (11, "Sep 23, 2025", 41),
    (12, "Sep 25, 2025", 37),
    (13, "Sep 30, 2025", 37),
    (14, "Oct 2, 2025", 30),
    (15, "Oct 7, 2025", 45),   # Quiz 2 date
    (16, "Oct 14, 2025", 30),
    (17, "Oct 16, 2025", 30),
    (18, "Oct 21, 2025", 36),
    (19, "Oct 23, 2025", 34),
    (20, "Oct 28, 2025", 25),
    (21, "Oct 30, 2025", 30),
    (22, "Nov 4, 2025", 27),
    (23, "Nov 6, 2025", 28),   # Estimated
    (24, "Nov 11, 2025", 41),  # Quiz 3 date
    (25, "Nov 13, 2025", 19),
    (26, "Nov 18, 2025", 34),  # Paper presentation date
    (27, "Nov 20, 2025", 16),
]

# Important course evaluation dates
EVALUATION_DATES = {
    "Quiz 2": "Oct 7, 2025",
    "Quiz 3": "Nov 11, 2025", 
    "Paper Presentation": "Nov 18, 2025"
}


def parse_date(date_str):
    """Parse date string to datetime object."""
    return datetime.strptime(date_str, "%b %d, %Y")


def analyze_attendance():
    """Analyze attendance data and generate report."""
    
    # Extract attendance counts
    attendance_counts = [entry[2] for entry in ATTENDANCE_DATA]
    dates = [entry[1] for entry in ATTENDANCE_DATA]
    
    # Basic statistics
    num_classes = len(ATTENDANCE_DATA)
    median_attendance = statistics.median(attendance_counts)
    mean_attendance = statistics.mean(attendance_counts)
    
    # Find lowest and highest attendance
    min_attendance = min(attendance_counts)
    max_attendance = max(attendance_counts)
    
    min_idx = attendance_counts.index(min_attendance)
    max_idx = attendance_counts.index(max_attendance)
    
    lowest_date = dates[min_idx]
    highest_date = dates[max_idx]
    lowest_class = ATTENDANCE_DATA[min_idx][0]
    highest_class = ATTENDANCE_DATA[max_idx][0]
    
    # Check correlation with evaluation dates
    eval_attendance = {}
    for eval_name, eval_date in EVALUATION_DATES.items():
        for class_num, date, count in ATTENDANCE_DATA:
            if date == eval_date:
                eval_attendance[eval_name] = (date, count)
                break
    
    # Generate report
    report = {
        "num_classes": num_classes,
        "class_dates": [(entry[0], entry[1]) for entry in ATTENDANCE_DATA],
        "median_attendance": median_attendance,
        "mean_attendance": round(mean_attendance, 1),
        "lowest_attendance": {
            "count": min_attendance,
            "date": lowest_date,
            "class": lowest_class
        },
        "highest_attendance": {
            "count": max_attendance,
            "date": highest_date,
            "class": highest_class
        },
        "evaluation_dates_attendance": eval_attendance,
        "all_attendance": ATTENDANCE_DATA
    }
    
    return report


def print_report(report):
    """Print formatted attendance report."""
    
    print("=" * 60)
    print("CSCE 580 ATTENDANCE ANALYSIS REPORT")
    print("=" * 60)
    
    print(f"\n(a) NUMBER OF CLASSES AND DATES:")
    print(f"    Total Classes: {report['num_classes']}")
    print(f"\n    Class Dates:")
    for class_num, date in report['class_dates']:
        print(f"      Class {class_num:2d}: {date}")
    
    print(f"\n(b) MEDIAN CLASS ATTENDANCE:")
    print(f"    Median: {report['median_attendance']} students per class")
    print(f"    Mean: {report['mean_attendance']} students per class")
    
    print(f"\n(c) LOWEST AND HIGHEST ATTENDANCE:")
    low = report['lowest_attendance']
    high = report['highest_attendance']
    print(f"    Lowest:  Class {low['class']} on {low['date']} with {low['count']} students")
    print(f"    Highest: Class {high['class']} on {high['date']} with {high['count']} students")
    
    print(f"\n(d) CORRELATION WITH COURSE EVALUATION DATES:")
    for eval_name, (date, count) in report['evaluation_dates_attendance'].items():
        print(f"    {eval_name} ({date}): {count} students")
    
    # Analysis
    eval_counts = [v[1] for v in report['evaluation_dates_attendance'].values()]
    avg_eval = statistics.mean(eval_counts) if eval_counts else 0
    overall_avg = report['mean_attendance']
    
    print(f"\n    Average attendance on evaluation dates: {avg_eval:.1f}")
    print(f"    Overall average attendance: {overall_avg}")
    
    if avg_eval > overall_avg:
        print(f"    CONCLUSION: Positive correlation - attendance is HIGHER on evaluation dates")
        print(f"                ({avg_eval - overall_avg:.1f} more students on average)")
    else:
        print(f"    CONCLUSION: No positive correlation observed")
    
    print("\n" + "=" * 60)


def export_to_csv(report, filename="attendance_data.csv"):
    """Export attendance data to CSV."""
    with open(filename, 'w') as f:
        f.write("Class,Date,Attendance\n")
        for class_num, date, count in report['all_attendance']:
            f.write(f"{class_num},{date},{count}\n")
    print(f"Data exported to {filename}")


if __name__ == "__main__":
    report = analyze_attendance()
    print_report(report)
    export_to_csv(report)
    
    # Save JSON report
    with open("attendance_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nJSON report saved to attendance_report.json")

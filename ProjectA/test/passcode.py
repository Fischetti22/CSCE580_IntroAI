#!/usr/bin/env python3

import random
import csv
from pathlib import Path
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os

# --- CSV config ---
DATA_DIR = Path("guesses")
CSV_FILE = DATA_DIR / "guesses_log.csv"
FIELDS = ["digit", "guess", "is_correct"]

def ensure_csv_exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_FILE.exists() or CSV_FILE.stat().st_size == 0:
        with CSV_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()

def log_guess(digit_label: str, guess: int, target: int):
    """Append guess to CSV file."""
    ensure_csv_exists()
    row = {
        "digit": digit_label,
        "guess": guess,
        "is_correct": int(guess == target)
    }
    with CSV_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerow(row)
        
def load_csv_text() -> str:
    """Return CSV content as text (without index). If empty, return just the header line."""
    ensure_csv_exists()
    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            return ",".join(FIELDS) + "\n"
        return df.to_csv(index=False)
    except Exception:
        return ",".join(FIELDS) + "\n"


        
def load_markdown_text(md_path: str) -> str:
    """Read a markdown file if it exists, otherwise return empty string."""
    path = Path(md_path)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


# --- Load .env ---
# ================= OpenAI setup =================
def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    return OpenAI(api_key=api_key)

# --- Load your guesses CSV ---
csv_path = "guesses/guesses_log.csv"
df = pd.read_csv(csv_path)

# Convert dataframe to CSV string (or summarize if big)
csv_text = df.to_csv(index=False)

# --- Ask ChatGPT about the file ---
def build_user_prompt(csv_text: str, md_text: str) -> str:
    return f"""
You are analyzing two sources:

1. **CSV** (passcode guess history):
{csv_text}

2. **Markdown notes**:
{md_text}

Goal:
- we are trying to unlock this passcode using only knowledge that we have. Attached is a csv file with all previous attempts. A markdown with data stating what is a common 4 digit passcode with url directing where to find more information.
Task:
- Combine insights from both files.
- Propose three candidate 4-digit codes (Plan A, Plan B, Plan C).
- Follow the same scoring rules for the CSV.
- If the markdown has hints about digits, weigh them strongly.
- If no hints, fall back to the CSV logic.

Output format (STRICT):
Plan A: d1 d2 d3 d4
Plan B: d1 d2 d3 d4
Plan C: d1 d2 d3 d4
""".strip()


def print_plans():
    client = get_client()
    csv_text = load_csv_text()
    md_text = load_markdown_text("hints.md")  # change filename as needed

    user_prompt = build_user_prompt(csv_text, md_text)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=128,
        messages=[
            {"role": "system", "content": "You are a precise data analyst. Output ONLY Plan A/B/C."},
            {"role": "user", "content": user_prompt},
        ],
    )
    print(resp.choices[0].message.content.strip())

print_plans()
print()

# Generate 4 random digits (0–9)
digit_1 = 3 #random.randint(0, 9)
digit_2 = 2  #random.randint(0, 9)
digit_3 = 4  #random.randint(0, 9)
digit_4 = 6  #random.randint(0, 9)

guesses = []
correct_guesses = []

#First
try:
	guess = int(input("Guess The first digit of the passcode:"))
except KeyboardInterrrupt:
	print("\nExiting...")
	exit(0)
guesses.append(guess) #saves the guess
log_guess("digit_1", guess, digit_1) 

if guess == digit_1:
		correct_guesses.append(guess)
		print("Correct!")
		
else:
		print("\n Strike 1")
		guess = int(input("Guess The first digit of the passcode:"))
		guesses.append(guess)
		log_guess("digit_1", guess, digit_1) 
if guess == digit_1:
		correct_guesses.append(guess)
		print("Correct!")		
else:
		print("\n Strike 2")
		guess = int(input("Guess The first digit of the passcode:"))
		guesses.append(guess)
		log_guess("digit_1", guess, digit_1) 
if guess == digit_1:
		correct_guesses.append(guess)
		print("Correct!")
else:
		print("💀 You’re locked out, bucko!")
		exit()
		
		


#Second
guess = int(input("Guess The second digit of the passcode:"))
guesses.append(guess)
log_guess("digit_2", guess, digit_2) 

if guess == digit_2:
		correct_guesses.append(guess)
		print("Correct!")
else:
		print("\n Strike 1")
		guess = int(input("Guess The second digit of the passcode:"))
		guesses.append(guess)
		log_guess("digit_2", guess, digit_2)
if guess == digit_2:
		correct_guesses.append(guess)
		log_guess("digit_2", guess, digit_2)
		print("Correct!")		
else:
		print("\n Strike 2")
		guess = int(input("Guess The second digit of the passcode:"))
		guesses.append(guess)
		log_guess("digit_2", guess, digit_2)
if guess == digit_2:
		print("Correct!")
		correct_guesses.append(guess)
else:
		print("💀 You’re locked out, bucko!")
		exit()
		


#Third		
guess = int(input("Guess The third digit of the passcode:"))
guesses.append(guess)
log_guess("digit_3", guess, digit_3)

if guess == digit_3:
		correct_guesses.append(guess)
		print("Correct!")
else:
		print("\n Strike 1")
		guess = int(input("Guess The third digit of the passcode:"))
		guesses.append(guess)
		log_guess("digit_3", guess, digit_3)
if guess == digit_3:
		correct_guesses.append(guess)
		print("Correct!")		
else:
		print("\n Strike 2")
		guess = int(input("Guess The third digit of the passcode:"))
		guesses.append(guess)
		log_guess("digit_3", guess, digit_3)
if guess == digit_3:
		correct_guesses.append(guess)
		log_guess("digit_3", guess, digit_3)
		print("Correct!")
else:
		print("💀 You’re locked out, bucko!")
		exit()
		


#Fourth	
try:	
	guess = int(input("Guess The fourth digit of the passcode:"))
except KeyboardInterrupt:
    print("\nExiting…")
guesses.append(guess)
log_guess("digit_4", guess, digit_4)

if guess == digit_4:
		print("Correct!")
		correct_guesses.append(guess)
else:
		print("\n Strike 1")
		guess = int(input("Guess The fourth digit of the passcode:"))
		guesses.append(guess)
		log_guess("digit_4", guess, digit_4)
if guess == digit_4:
		print("Correct!")
		correct_guesses.append(guess)		
else:
		print("\n Strike 2")
		guess = int(input("Guess The fourth digit of the passcode:"))
		guesses.append(guess)
		log_guess("digit_4", guess, digit_4)
if guess == digit_4:
		print("Correct!")
		correct_guesses.append(guess)
else:
		print("💀 You’re locked out, bucko!")
		# print(guesses)
		# print(correct_guesses)
		exit()


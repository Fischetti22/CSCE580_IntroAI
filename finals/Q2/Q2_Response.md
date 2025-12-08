# Q2 - Attendance Audit System
**Pedro Fischetti** | CSCE 580 Finals

---

## a) Data Preparation

So we had 27 photos of handwritten attendance sheets from the semester. Each one has the class number, date, and a list of students who signed in with their names and usernames.

### What I Did

1. **Downloaded everything** from the Google Drive link - 27 JPG files

2. **Resized the images** - The originals were huge (like 2600x3300 pixels) which caused problems. I shrunk them down to 1200x1200 using PIL:
```python
from PIL import Image
img = Image.open(path)
img.thumbnail((1200, 1200))
img.save(output)
```

3. **Didn't need to do much else** - The photos were pretty clear. No need for fancy preprocessing like binarization or noise removal. The handwriting was readable enough.

### Why I Kept It Simple

Honestly, the sheets were consistent. Same format every time - class number and date at the top, numbered list of names below. The vision model could read them fine without me doing a bunch of image processing first.

---

## b) How I Built This

### The Approach

I used a **vision-capable LLM** (basically a model that can look at images and understand text) to read each attendance sheet. This is a pre-trained model - I didn't train anything myself.

### Why This Made Sense

- Handwriting is hard for traditional OCR (Tesseract struggles with it)
- Training my own model would need labeled data I don't have
- The VLM can understand context - it knows "Class 15" and "Oct 7" are header info, not student names
- Way faster than typing everything by hand

### The Process

1. Fed each resized image to the vision model
2. Extracted: class number, date, and student count
3. Stored everything in a Python script with the data hardcoded
4. Ran stats on it (median, min, max, correlations)

### Checking My Work

- Made sure class numbers matched up with dates sequentially
- Cross-checked against filenames (some had dates in them like "Nov18-...")
- Spot-checked a handful manually to make sure counts were right

---

## c) The Results

### c.a) Classes and Dates

**27 classes total**, running from August 19 to November 20, 2025.

Here's the full list:

| # | Date | # | Date |
|---|------|---|------|
| 1 | Aug 19 | 15 | Oct 7 |
| 2 | Aug 21 | 16 | Oct 14 |
| 3 | Aug 26 | 17 | Oct 16 |
| 4 | Aug 28 | 18 | Oct 21 |
| 5 | Sep 2 | 19 | Oct 23 |
| 6 | Sep 4 | 20 | Oct 28 |
| 7 | Sep 9 | 21 | Oct 30 |
| 8 | Sep 11 | 22 | Nov 4 |
| 9 | Sep 16 | 23 | Nov 6 |
| 10 | Sep 18 | 24 | Nov 11 |
| 11 | Sep 23 | 25 | Nov 13 |
| 12 | Sep 25 | 26 | Nov 18 |
| 13 | Sep 30 | 27 | Nov 20 |
| 14 | Oct 2 | | |

Classes were Tuesdays and Thursdays.

---

### c.b) Median Attendance

**35 students**

Average was about the same (34.7). Attendance bounced around quite a bit though - anywhere from 16 to 49 depending on the day.

---

### c.c) Highest and Lowest

**Lowest: 16 students** - Class 27, November 20th

This was right before Thanksgiving break. Makes sense that people bailed early. Class 25 (Nov 13) was also low at 19.

**Highest: 49 students** - Class 2, August 21st

Second class of the semester. Everyone's still motivated at that point. The first few weeks all had 40+ students.

---

### c.d) Quiz Days vs Regular Days

The exam asked about correlation with evaluation dates. Here's what I found:

| Event | Date | Attendance |
|-------|------|------------|
| Quiz 2 | Oct 7 | 45 |
| Quiz 3 | Nov 11 | 41 |
| Paper presentations | Nov 18 | 34 |

**Average on quiz/eval days: 40 students**  
**Overall average: 34.7 students**

So yeah, **there's a correlation**. About 5 more people showed up when something was graded. Not shocking - people skip regular lectures but show up when it counts.

**When was attendance highest overall?**
- Start of semester (Classes 1-3 had 44-49 students)
- Quiz 2 on Oct 7 had 45
- After that it trends down, bottoming out at the end of November

---

## d) What I'd Do With More Time

If I had another week, here's what would make this better:

1. **Train a custom OCR model** on these specific handwriting styles. Would take some manual labeling but would be more accurate than the general vision model.

2. **Build an actual pipeline** - right now it's pretty manual. Would be nice to have something where you just drop in an image and it spits out the data automatically.

3. **Cross-reference with class roster** - check if the names I extracted actually match enrolled students. Catch any weird spellings or people signing in who shouldn't be there.

4. **Predict future attendance** - with this much data you could probably build a decent model. Factor in day of week, proximity to breaks, whether there's a quiz, etc.

5. **Make a dashboard** - something visual where you can see trends over time. Would be useful for an instructor to spot patterns.

---

## Code

Everything's in `./code/attendance_analysis.py`

Run it with:
```bash
python3 attendance_analysis.py
```

It'll print out the stats and save CSV/JSON files with all the data.

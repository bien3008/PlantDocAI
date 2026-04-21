"""Quick script to analyze class distribution in train split."""
import csv
from collections import Counter

# Load train split
samples = []
with open("data/splits/train.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        samples.append(int(row["labelId"]))

# Load class names
classes = {}
with open("data/splits/classes.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        classes[int(row["labelId"])] = row["className"]

counts = Counter(samples)
print(f"Total train samples: {len(samples)}")
print(f"Number of classes: {len(counts)}")
print()

sorted_counts = sorted(counts.items(), key=lambda x: x[1])
max_count = sorted_counts[-1][1]
min_count = sorted_counts[0][1]

print(f"Max class count: {max_count}")
print(f"Min class count: {min_count}")
print(f"Imbalance ratio (max/min): {max_count/min_count:.1f}x")
print()

for cid, cnt in sorted_counts:
    name = classes.get(cid, "?")
    bar = "#" * int(cnt / max_count * 40)
    print(f"  [{cid:2d}] {name:50s} {cnt:5d}  {bar}")

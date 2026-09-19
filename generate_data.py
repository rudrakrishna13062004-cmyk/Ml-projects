"""
Generates a synthetic but realistic product-review dataset for sentiment
analysis practice. Combines sentence templates + products + adjectives to
create varied positive/negative reviews.

NOTE: For a stronger portfolio project, swap this out later for a real
dataset (e.g. IMDB Movie Reviews or Amazon Reviews from Kaggle) — the
training pipeline in train_model.py works the same way regardless of
where reviews.csv comes from, as long as it has 'review' and 'sentiment'
columns.
"""
import csv
import random

random.seed(42)

products = [
    "phone", "laptop", "headphones", "shoes", "backpack", "watch",
    "camera", "blender", "keyboard", "mouse", "monitor", "charger",
    "speaker", "jacket", "book", "chair", "tablet", "printer",
    "router", "vacuum cleaner"
]

positive_templates = [
    "This {p} is absolutely amazing, exceeded all my expectations!",
    "I love this {p}, works perfectly and looks great.",
    "Best {p} I have ever bought, highly recommend it.",
    "Excellent quality {p}, fast delivery and great packaging.",
    "The {p} works flawlessly, very happy with this purchase.",
    "Great value for money, this {p} is fantastic.",
    "Superb build quality on this {p}, worth every rupee.",
    "Very satisfied with the {p}, will buy again from this brand.",
    "The {p} is durable, stylish, and performs really well.",
    "Impressive {p}, customer service was also very helpful.",
    "This {p} made my life so much easier, thank you!",
    "Outstanding {p}, arrived early and works perfectly.",
]

negative_templates = [
    "This {p} is terrible, completely broke after two days.",
    "Very disappointed with this {p}, waste of money.",
    "The {p} stopped working within a week, awful quality.",
    "Worst {p} I have purchased, do not recommend at all.",
    "Poor build quality, the {p} feels cheap and flimsy.",
    "The {p} arrived damaged and customer support was unhelpful.",
    "I regret buying this {p}, it never worked properly.",
    "Terrible experience, the {p} is defective and slow.",
    "The {p} is overpriced for such low quality.",
    "Extremely unhappy with this {p}, requesting a refund.",
    "This {p} was a huge disappointment, not worth it.",
    "The {p} broke on the first use, very frustrating.",
]

neutral_negations = [
    ("not good at all", 0), ("not what I expected", 0),
    ("not bad, quite decent actually", 1), ("nothing special but works fine", 1),
]

rows = []
for p in products:
    for t in positive_templates:
        rows.append((t.format(p=p), 1))
    for t in negative_templates:
        rows.append((t.format(p=p), 0))

# a few extra varied short reviews to add noise/realism
extra = [
    ("Amazing product, five stars!", 1),
    ("Do not buy, complete garbage.", 0),
    ("Works as described, very pleased.", 1),
    ("Broke immediately, avoid this seller.", 0),
    ("Good product but shipping was slow.", 1),
    ("Cheap material, not durable at all.", 0),
    ("Fantastic purchase, highly satisfied.", 1),
    ("Money wasted, extremely poor quality.", 0),
]
rows.extend(extra * 5)

random.shuffle(rows)

with open("/home/claude/sentiment-analysis-api/data/reviews.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["review", "sentiment"])
    for review, label in rows:
        writer.writerow([review, label])

print(f"Generated {len(rows)} reviews -> data/reviews.csv")

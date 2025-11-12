import csv
def load_subset(csv_path):
    # CSV columns: id,category,original_title
    rows = []
    with open(csv_path, newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows
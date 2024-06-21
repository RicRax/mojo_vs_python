import os
import csv


def update_results(name, base, types, vector, parallel):
    file_exists = os.path.isfile("results.csv")

    with open("results.csv", mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["name", "base", "types", "vector", "parallel"])

        writer.writerow([name, base, types, vector, parallel])

import csv

# Input and output file paths
input_file = "output/labels.txt"
output_file = "output/labels.csv"

# Read the text file and split lines into image filename and text
data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(" ")
        image_filename = parts[0]
        text = " ".join(parts[1:])
        data.append((image_filename, text))

# Write the data to a CSV file
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image Filename", "Text"])
    writer.writerows(data)

print("CSV file has been created successfully.")

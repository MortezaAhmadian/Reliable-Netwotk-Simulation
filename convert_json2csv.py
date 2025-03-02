import json
import csv


with open('logs_year_0.json') as json_file:
    data = json.load(json_file)

# requests_data = data['requests_details']

data_file = open('logs_year_0.csv', 'w')
csv_writer = csv.writer(data_file)
count = 0

for req in data:
    if count == 0:
        header = req.keys()
        csv_writer.writerow(header)
        count += 1
    csv_writer.writerow(req.values())

data_file.close()

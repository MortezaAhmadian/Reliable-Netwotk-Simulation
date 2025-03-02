import csv, json
import random

# Parameters
num_requests = 6000
nodes = [i+1 for i in range(14)]
traffic_range = [(i+2)*50 for i in range(11)]

with open('config/requests.csv', 'w', newline='') as csvfile:
    fieldnames = ['source', 'destination', 'traffic', 'total_traffic']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for _ in range(num_requests):
        source = random.choice(nodes)
        destination = random.choice([node for node in nodes if node != source])
        traffic = random.choice(traffic_range)
        writer.writerow({'source': source, 'destination': destination, 'traffic': traffic,
                         'total_traffic': traffic})

# Save to JSON file
requests = []
with open('config/requests.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        request = {'source': int(row['source']),
                   'destination': int(row['destination']),
                   'traffic': int(row['traffic']),
                   'total_traffic': int(row['total_traffic'])}
        requests.append(request)

with open('config/requests.json', 'w') as jsonfile:
    json.dump(requests, jsonfile, indent=4)

print(f"{num_requests} requests have been generated and saved to requests.json.")

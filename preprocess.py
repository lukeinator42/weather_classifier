import os
import csv
from dateutil.parser import parse
import numpy as np

x_headers = ["day", "month", "year", "tmp_avg", "tmp_max", "tmp_min", "humidity_avg", "wind_speed_avg", "wind_dir_avg"]

x_train = []
x_test = []

y_train = []
y_test = []

for file in os.listdir('./data'):
    with open('./data/'+file, 'rt', encoding="utf-16") as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')

        header = None

        for row in reader:
            if header is None:
                header = row
                continue

            dt = parse(row[1])

            for i in range(len(row)):
                if row[i] is None or row[i] == "":
                    row[i] = 0

            if dt.year != 2018:
                x_train.append(np.array([
                    int(dt.day),
                    int(dt.month),
                    int(dt.year),
                    float(row[8]),
                    float(row[11]),
                    float(row[14]),
                    float(row[17]),
                    float(row[20]),
                    float(row[23])
                ], dtype=np.double))

                y_train.append(np.array(float(row[5]), dtype=np.double))

            else:
                x_test.append(np.array([
                    int(dt.day),
                    int(dt.month),
                    int(dt.year),
                    float(row[8]),
                    float(row[11]),
                    float(row[14]),
                    float(row[17]),
                    float(row[20]),
                    float(row[23])
                ], dtype=np.double))
                y_test.append(np.array(float(row[5]), dtype=np.double))


x_train = np.array(x_train, dtype=np.double)
x_test = np.array(x_test, dtype=np.double)

y_train = np.array(y_train, dtype=np.double)
y_test = np.array(y_test, dtype=np.double)

np.save('./x_train.npy', x_train)
np.save('./y_train.npy', y_train)
np.save('./x_test.npy', x_test)
np.save('./y_test.npy', y_test)


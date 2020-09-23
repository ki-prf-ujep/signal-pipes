import numpy as np
from random import randint
from keras import models, layers, optimizers
import matplotlib.pyplot as plt

def first_member(xs, min, max):
    for x in xs:
        if min <= x <= max:
            return x

def sqeel(t0, dt, s_points, h_points):
    if (any(t0 <= sp <= t0+dt for sp in s_points)
            and (t0 <= hp <= t0 + dt for hp in h_points)):
        return (t0 + dt - first_member(s_points, t0, t0 + dt)) / dt
    else:
        return 0.0

def get_indexes(imin, imax, n):
    assert imax - imin + 1 > n, "Not enough points in interval"
    indexes = []
    dx = (imax - imin) / (n-1)
    for i in range(n-1):
        indexes.append(round(i * dx - 0.5) + imin)
    indexes.append(imax-1)
    return indexes

sources = [
        ("/home/fiser/IKON/ICON/mereni/26_02_2020_11_38/1hp.txt",
         "/home/fiser/IKON/ICON/anotace/1hp.a.csv"),
        ("/home/fiser/IKON/ICON/mereni/26_02_2020_11_38/1hp2.txt",
         "/home/fiser/IKON/ICON/anotace/1hp2.a.csv"),
        ("/home/fiser/IKON/ICON/mereni/26_02_2020_11_38/1hr.txt",
         "/home/fiser/IKON/ICON/anotace/1hr.csv")
    ]

fs = 250
di = int(fs * 0.3)
n = 15

def get_eval_data(fileindex, di, n):
    d = np.loadtxt(sources[fileindex][0])
    xdata = []
    for i in range(0, len(d) - di - 2):
       xdata.append(d[get_indexes(i, i + di, n)])
    return np.array(xdata) / 600.0

def sqeel_array(fileindex, di):
    d = np.loadtxt(sources[fileindex][0])
    a = np.loadtxt(sources[fileindex][1], delimiter=",", converters={1: lambda c: float(b"SHF".index(c.strip()))})
    s_points = a[a[:, 1] == 0.0, 0]
    h_points = a[a[:, 1] == 1.0, 0]
    labels = []
    for i in range(0, len(d) - di - 2):
       labels.append(sqeel(i/fs, di/fs, s_points, h_points))
    return np.array(labels)

def get_rdata(fileindex):
    d = np.loadtxt(sources[fileindex][0])
    return d[:-di-2] / 600.0

def get_data(nsamples, di, n):
    rsamples = []
    rlabels = []

    for data, annot in sources:
        d = np.loadtxt(data)
        a = np.loadtxt(annot, delimiter=",", converters={1: lambda c: float(b"SHF".index(c.strip()))})
        s_points = a[a[:,1] == 0.0, 0]
        h_points = a[a[:, 1] == 1.0, 0]
        for _ in range(nsamples):
            i = randint(0, len(d) - di - 2)
            rsamples.append(d[get_indexes(i, i + di, n)])
            rlabels.append(sqeel(i/fs, di/fs, s_points, h_points))
    return np.array(rsamples) / 600.0, np.array(rlabels)

inputs, labels = get_data(100, di, n)
print(inputs.shape)

model = models.Sequential()
model.add(layers.Dense(32, activation="relu", input_shape=(n,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer=optimizers.RMSprop(),loss="mse", metrics=["accuracy"])
model.fit(inputs, labels, batch_size=100, epochs=5)

predicted_sqeel = model.predict(get_eval_data(0, di, n))
etime = len(predicted_sqeel)/fs
plt.plot(np.arange(0.0,etime,1/fs), sqeel_array(0, di))
plt.plot(np.arange(0.0,etime,1/fs), get_rdata(0))
plt.plot(np.arange(di/fs,etime+di/fs,1/fs), predicted_sqeel, c='k', lw=2)
fig = plt.gcf()
fig.set_size_inches(18, 9)
plt.show()




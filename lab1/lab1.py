from random import randint

a0 = 2
a1 = 3
a2 = 4
a3 = 2

print("A0 = {0}\nA1 = {1}\nA2 = {2}\nA3 = {3}".format(a0, a1, a2, a3))

X1 = [randint(0, 20) for i in range(8)]
X2 = [randint(0, 20) for i in range(8)]
X3 = [randint(0, 20) for i in range(8)]

print("\nX1: ", X1)
print("X2: ", X2)
print("X3: ", X3)

Y = [a0 + a1*X1[i] + a2*X2[i] + a3*X3[i] for i in range(8)]
print("\nY: ", Y)

X01 = (max(X1) + min(X1))/2
X02 = (max(X2) + min(X2))/2
X03 = (max(X3) + min(X3))/2

X0 = [X01, X02, X03]

dx1 = X01 - min(X1)
dx2 = X02 - min(X2)
dx3 = X03 - min(X3)

dx = [dx1, dx2, dx3]

XH1 = [round((i - X01)/dx1, 3) for i in X1]
XH2 = [round((i - X02)/dx2, 3) for i in X2]
XH3 = [round((i - X03)/dx3, 3) for i in X3]

print("\nXH1: ", XH1)
print("XH2: ", XH2)
print("XH3: ", XH3)

Yet = a0 + a1*X01 + a2*X02 + a3*X03

Ymin = min(Y)

print("\nX0: ", X0)
print("dx: ", dx)
print("Yet: ", Yet)

ind = Y.index(Ymin)
print("Точка плану, що задовольняє заданому критерію оптимальності: Y({0},"
      " {1}, {2}) = min(Y) = {3}".format(X1[ind], X2[ind], X3[ind], Ymin))

Yaver = sum(Y)/len(Y)
print("Yaver: ", Yaver)

y = min(Y)
for i in range(len(Y)):
      if Yaver - y > Yaver - Y[i] and Yaver - Y[i] > 0:
            y = Y[i]

print("--> Yaver = ", y)

ind = Y.index(y)
print("Точка плану, що задовольняє критерію оптимальності за варіантом 208: Y({0},"
      " {1}, {2}) = {3}".format(X1[ind], X2[ind], X3[ind], y))

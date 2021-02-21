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

XH1 = [(i - X01)/dx1 for i in X1]
XH2 = [(i - X02)/dx2 for i in X2]
XH3 = [(i - X03)/dx3 for i in X3]

Yet = a0 + a1*X01 + a2*X02 + a3*X03

Ymin = min(Y)

print("X0: ", X0)
print("dx: ", dx)
print("Yet: ", Yet)

ind = Y.index(Ymin)
print("Точка плану, що задовольняє заданому критерію оптимальності: Y({0},"
      " {1}, {2}) = min(Y) = {3}".format(X1[ind], X2[ind], X3[ind], Ymin))

from random import randint
import sklearn.linear_model as lm
from scipy.stats import f, t
from math import sqrt
from pyDOE2 import *

x_range = [(-4, 6), (-1, 2), (-4, 2)]
xcp_min = round(sum([x_range[i][0] for i in range(len(x_range))]) / 3)
xcp_max = round(sum([x_range[i][1] for i in range(len(x_range))]) / 3)
y_min, y_max = 200 + xcp_min, 200 + xcp_max

def matr_y(m, n):
    y = np.zeros(shape=(n, m))
    for i in range(n):
        for j in range(m):
            y[i][j] = randint(y_min, y_max)
    return y

def regression(x, b):
    return sum([x[i] * b[i] for i in range(len(x))])

def matrix_1(m, n):
    x_norm = np.array([[1, -1, -1, -1],
                       [1, -1, -1, 1],
                       [1, -1, 1, -1],
                       [1, -1, 1, 1],
                       [1, 1, -1, -1],
                       [1, 1, -1, 1],
                       [1, 1, 1, -1],
                       [1, 1, 1, 1]])

    x_natur = np.ones(shape=(n, len(x_norm[0])))
    for i in range(len(x_norm)):
        for j in range(1, len(x_norm[i])):
            if x_norm[i][j] == 1:
                x_natur[i][j] = x_range[j-1][1]
            else:
                x_natur[i][j] = x_range[j-1][0]

    y = matr_y(m, n)
    coef_1(x_natur, x_norm, y)


def coef_1(x_natur, x_norm, y):
    y_aver = [sum(y[i]) / m for i in range(n)]
    print("Натуралізована матриця Х\n", x_natur)
    print("\nМатриця Y\n", y)
    print("Cередні значення функції відгуку за рядками:", [round(elem, 3) for elem in y_aver])
    mx1 = sum(x_natur[i][1] for i in range(n)) / n
    mx2 = sum(x_natur[i][2] for i in range(n)) / n
    mx3 = sum(x_natur[i][3] for i in range(n)) / n
    my = sum(y_aver) / n

    a1 = sum(x_natur[i][1] * y_aver[i] for i in range(n)) / n
    a2 = sum(x_natur[i][2] * y_aver[i] for i in range(n)) / n
    a3 = sum(x_natur[i][3] * y_aver[i] for i in range(n)) / n

    a11 = sum(x_natur[i][1] * x_natur[i][1] for i in range(n)) / n
    a22 = sum(x_natur[i][2] * x_natur[i][2] for i in range(n)) / n
    a33 = sum(x_natur[i][3] * x_natur[i][3] for i in range(n)) / n

    a12 = a21 = sum(x_natur[i][1] * x_natur[i][2] for i in range(n)) / n
    a13 = a31 = sum(x_natur[i][1] * x_natur[i][3] for i in range(n)) / n
    a23 = a32 = sum(x_natur[i][2] * x_natur[i][3] for i in range(n)) / n

    matr_X = [[1, mx1, mx2, mx3],
              [mx1, a11, a21, a31],
              [mx2, a12, a22, a32],
              [mx3, a13, a23, a33]]
    matr_Y = [my, a1, a2, a3]
    b_natur = np.linalg.solve(matr_X, matr_Y)

    print("\nНатуралізоване рівняння регресії: y = {0:.3f} {1:+.3f}*x1 {2:+.3f}*x2 {3:+.3f}*x3".format(*b_natur))

    b_norm = [sum(y_aver) / n,
              sum(y_aver[i] * x_norm[i][1] for i in range(n)) / n,
              sum(y_aver[i] * x_norm[i][2] for i in range(n)) / n,
              sum(y_aver[i] * x_norm[i][3] for i in range(n)) / n]
    print("\nНормоване рівняння регресії: y = {0:.3f} {1:+.3f}*x1 {2:+.3f}*x2 {3:+.3f}*x3".format(*b_norm))
    cohren(m, y, y_aver, x_norm, b_norm)


def matrix_2(m, n):
    print("\n---------------------------")
    x_norm = [[1, -1, -1, -1],
              [1, -1, -1, 1],
              [1, -1, 1, -1],
              [1, -1, 1, 1],
              [1, 1, -1, -1],
              [1, 1, -1, 1],
              [1, 1, 1, -1],
              [1, 1, 1, 1]]
    for i in range(n):
        x_norm[i].append(x_norm[i][1] * x_norm[i][2])
        x_norm[i].append(x_norm[i][1] * x_norm[i][3])
        x_norm[i].append(x_norm[i][2] * x_norm[i][3])
        x_norm[i].append(x_norm[i][1] * x_norm[i][2] * x_norm[i][3])

    x_natur = np.ones(shape=(n, len(x_norm[0])))
    for i in range(len(x_norm)):
        for j in range(1, 3):
            if x_norm[i][j] == 1:
                x_natur[i][j] = x_range[j-1][1]
            else:
                x_natur[i][j] = x_range[j-1][0]
    for i in range(n):
        x_natur[i][4] = x_natur[i][1] * x_natur[i][2]
        x_natur[i][5] = x_natur[i][1] * x_natur[i][3]
        x_natur[i][6] = x_natur[i][2] * x_natur[i][3]
        x_natur[i][7] = x_natur[i][1] * x_natur[i][2] * x_natur[i][3]
    print("Натуралізована матриця Х\n", x_natur)
    y = matr_y(m, n)
    coef_2(x_norm, y)

def coef_2(x_norm, y):
    y_aver = [sum(y[i]) / m for i in range(n)]
    print("\nМатриця Y\n", y)
    print("Cередні значення функції відгуку за рядками:", [round(elem, 3) for elem in y_aver])

    b_norm = [sum(y_aver) / n]

    for j in range(1, n):
        b = 0
        for i in range(n):
            b += x_norm[i][j] * y_aver[i]
        b_norm.append(b/n)

    print("\nНормоване рівняння регресії: y = {0:.3f} {1:+.3f}*x1 {2:+.3f}*x2 {3:+.3f}*x3 {4:+.3f}*x12 "
          "{5:+.3f}*x13 {6:+.3f}*x23 {7:+.3f}*x123".format(*b_norm))

    cohren(m, y, y_aver, x_norm, b_norm)

def matrix_3(m, n):
    print("\n---------------------------")
    y = matr_y(m, n)

    no = 1
    x_norm = ccdesign(3, center=(0, no))
    x_norm = np.insert(x_norm, 0, 1, axis=1)

    for i in range(4, 11):
        x_norm = np.insert(x_norm, i, 0, axis=1)

    l = 1.215

    for i in range(len(x_norm)):
        for j in range(len(x_norm[i])):
            if x_norm[i][j] < -1:
                x_norm[i][j] = -l
            elif x_norm[i][j] > 1:
                x_norm[i][j] = l

    def inter_matrix(x):
        for i in range(len(x)):
            x[i][4] = x[i][1] * x[i][2]
            x[i][5] = x[i][1] * x[i][3]
            x[i][6] = x[i][2] * x[i][3]
            x[i][7] = x[i][1] * x[i][2] * x[i][3]
            x[i][8] = x[i][1] * x[i][1]
            x[i][9] = x[i][2] * x[i][2]
            x[i][10] = x[i][3] * x[i][3]

    inter_matrix(x_norm)

    x_natur = np.ones(shape=(n, len(x_norm[0])), dtype=np.float64)
    for i in range(8):
        for j in range(1, 4):
            if x_norm[i][j] == 1:
                x_natur[i][j] = x_range[j-1][1]
            else:
                x_natur[i][j] = x_range[j-1][0]
    x0 = [(x_range[i][1] + x_range[i][0]) / 2 for i in range(3)]
    dx = [x_range[i][1] - x0[i] for i in range(3)]

    for i in range(8, len(x_norm)):
        for j in range(1, 4):
            if x_norm[i][j] == 0:
                x_natur[i][j] = x0[j-1]
            elif x_norm[i][j] == l:
                x_natur[i][j] = l * dx[j-1] + x0[j-1]
            elif x_norm[i][j] == -l:
                x_natur[i][j] = -l * dx[j-1] + x0[j-1]

    inter_matrix(x_natur)
    y_aver = [sum(y[i]) / m for i in range(n)]

    print("Нормована матриця Х\n")
    for i in range(len(x_norm)):
        for j in range(len(x_norm[i])):
            print(round(x_norm[i][j], 3), end=' ')
        print()

    print("\nНатуралізована матриця Х\n")
    for i in range(len(x_natur)):
        for j in range(len(x_natur[i])):
            print(round(x_natur[i][j], 3), end=' ')
        print()

    print("\nМатриця Y\n", y)
    print("\nCередні значення функції відгуку за рядками:\n", [round(elem, 3) for elem in y_aver])
    coef_3(x_natur, y_aver, y, x_norm)

def coef_3(x, y_aver, y, x_norm):
    skm = lm.LinearRegression(fit_intercept=False)
    skm.fit(x, y_aver)
    b = skm.coef_

    print("\nКоефіцієнти рівняння регресії:")
    b = [round(i, 3) for i in b]
    print(b)
    print("\nРезультат рівняння зі знайденими коефіцієнтами:\n", np.dot(x, b))
    cohren(m, y, y_aver, x_norm, b)

# ----------------- Критерій Кохрена -----------------------
def cohren(m, y, y_aver, x_norm, b):
    print("\nКритерій Кохрена")
    dispersion = []
    for i in range(n):
        z = 0
        for j in range(m):
            z += (y[i][j] - y_aver[i]) ** 2
        dispersion.append(z / m)
    print("Дисперсія:", [round(elem, 3) for elem in dispersion])

    Gp = max(dispersion) / sum(dispersion)
    f1 = m - 1
    f2 = n
    q = 0.05
    Gt = f.ppf(q=(1 - q / f1), dfn=f2, dfd=(f1 - 1) * f2)
    Gt = Gt / (Gt + f1 - 1)
    if Gp < Gt:
        print("Gp < Gt\n{0:.4f} < {1} => дисперсія однорідна".format(Gp, Gt))
        student(m, dispersion, y_aver, x_norm, b)
    else:
        print("Gp > Gt\n{0:.4f} > {1} => дисперсія неоднорідна => m+=1".format(Gp, Gt))
        m += 1
        if flag == "1":
            matrix_1(m, n)
        elif flag == "2":
            matrix_2(m, n)
        elif flag == "3":
            matrix_3(m, n)

# ----------------------- Критерій Стюдента --------------------------------
def student(m, dispersion, y_aver, x_norm, b):
    print("\nКритерій Стюдента")
    sb = sum(dispersion) / n
    s_beta = sqrt(sb / (n * m))
    k = len(x_norm[0])
    beta = [sum(y_aver[i] * x_norm[i][j] for i in range(n)) / n for j in range(k)]

    t_t = [abs(beta[i]) / s_beta for i in range(k)]

    f3 = (m - 1) * n
    qq = (1 + 0.95) / 2
    t_table = t.ppf(df=f3, q=qq)

    b_impor = []
    for i in range(k):
        if t_t[i] > t_table:
            b_impor.append(b[i])
        else:
            b_impor.append(0)
    print("Незначні коефіцієнти регресії")
    b_not_imp = []
    for i in range(k):
        if b[i] not in b_impor:
            b_not_imp.append(b[i])
            print("b{0} = {1:.3f}".format(i, b[i]))
        else:
            b_not_imp.append(0)

    print("Рівняння регресії з незначущими коефіцієнтами:\ny = {0:.3f} {1:+.3f}*x1 {2:+.3f}*x2 {3:+.3f}*x3"
          "{4:+.3f}*x1*x2 {5:+.3f}*x1*x3 {6:+.3f}*x2*x3 {7:+.3f}*x1*x2*x3 {8:+.3f}*x1^2 {9:+.3f}*x2^2"
          "{10:+.3f}*x3^2".format(*b_not_imp))

    y_impor = []
    for j in range(n):
        y_impor.append(regression([x_norm[j][i] for i in range(len(t_t))], b_impor))

    print("Значення функції відгуку зі значущими коефіцієнтами\n", [round(elem, 3) for elem in y_impor])
    fisher(m, y_aver, b_impor, y_impor, sb)

# ----------------------- Критерій Фішера --------------------------------
def fisher(m, y_aver, b_impor, y_impor, sb):
    global flag
    print("\nКритерій Фішера")
    d = 0
    for i in b_impor:
        if i:
            d += 1
    f3 = (m - 1) * n
    f4 = n - d
    s_ad = sum((y_impor[i] - y_aver[i]) ** 2 for i in range(n)) * m / f4
    Fp = s_ad / sb
    Ft = f.ppf(dfn=f4, dfd=f3, q=1 - 0.05)

    if Fp < Ft:
        print("Fp < Ft => {0:.2f} < {1}".format(Fp, Ft))
        print("Отримана математична модель при рівні значимості 0.05 адекватна експериментальним даним")
    else:
        print("Fp > Ft => {0:.2f} > {1}".format(Fp, Ft))
        print("Рівняння регресії неадекватно оригіналу при рівні значимості 0.05")
        if flag == "1":
            flag = "2"
            matrix_2(m, n)
        elif flag == "2":
            flag = "3"
            matrix_3(m, 15)


if __name__ == '__main__':
    flag = "1"
    n = 8
    m = 3
    matrix_1(m, n)

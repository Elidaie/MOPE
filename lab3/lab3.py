from random import randint
import numpy as np
from math import sqrt

m = 3

cohren_table = {2: 0.7679,
                3: 0.6841,
                4: 0.6287,
                5: 0.5892,
                6: 0.5598}

student_table = {8: 2.306,
                 12: 2.179,
                 16: 2.120,
                 20: 2.086,
                 24: 2.064}

fisher_table = {8: [5.3, 4.5, 4.1, 3.8],
                12: [4.8, 3.9, 3.5, 3.3],
                16: [4.5, 3.6, 3.2, 3],
                20: [4.4, 3.5, 3.1, 2.9],
                24: [4.3, 3.4, 3, 2.8]}

x1_min, x1_max = -30, 0
x2_min, x2_max = -25, 10
x3_min, x3_max = -25, -5

xcp_min, xcp_max = round((x1_min + x1_min + x1_min) / 3), round((x1_max + x2_max + x3_max) / 3)
y_min, y_max = 200 + xcp_min, 200 + xcp_max

x_norm = [[1, -1, -1, -1],
          [1, -1, 1, 1],
          [1, 1, -1, 1],
          [1, 1, 1, -1]]

x_natur = [[x1_min, x2_min, x3_min],
           [x1_min, x2_max, x3_max],
           [x1_max, x2_min, x3_max],
           [x1_max, x2_max, x3_min]]
print("Натуралізовані значення факторів", x_natur)

def coef(m):
    y = [[randint(y_min, y_max) for _ in range(m)] for _ in range(4)]
    y_aver = [sum(y[i]) / m for i in range(4)]
    print("Y:", y)
    print("Cередні значення функції відгуку за рядками:", y_aver)
    mx1 = sum(x_natur[i][0] for i in range(4)) / 4
    mx2 = sum(x_natur[i][1] for i in range(4)) / 4
    mx3 = sum(x_natur[i][2] for i in range(4)) / 4
    my = sum(y_aver) / 4
    a1 = sum(x_natur[i][0] * y_aver[i] for i in range(4)) / 4
    a2 = sum(x_natur[i][1] * y_aver[i] for i in range(4)) / 4
    a3 = sum(x_natur[i][2] * y_aver[i] for i in range(4)) / 4

    a11 = sum(x_natur[i][0] * x_natur[i][0] for i in range(4)) / 4
    a22 = sum(x_natur[i][1] * x_natur[i][1] for i in range(4)) / 4
    a33 = sum(x_natur[i][2] * x_natur[i][2] for i in range(4)) / 4

    a12 = a21 = sum(x_natur[i][0] * x_natur[i][1] for i in range(4)) / 4
    a13 = a31 = sum(x_natur[i][0] * x_natur[i][2] for i in range(4)) / 4
    a23 = a32 = sum(x_natur[i][1] * x_natur[i][2] for i in range(4)) / 4

    matr_X = [[1, mx1, mx2, mx3],
              [mx1, a11, a21, a31],
              [mx2, a12, a22, a32],
              [mx3, a13, a23, a33]]

    matr_Y = [my, a1, a2, a3]

    b_natur = np.linalg.solve(matr_X, matr_Y)
    print("\nНатуралізоване рівняння регресії: y = {0:.2f} {1:+.2f}*x1 {2:+.2f}*x2 {3:+.2f}*x3".format(*b_natur))

    print("Зробимо перевірку")
    test_y1 = [b_natur[0] + b_natur[1] * x_natur[i][0] + b_natur[2] * x_natur[i][1] + b_natur[3] * x_natur[i][2] for i
               in range(4)]

    print("{0:.2f} {1:+.2f}*{5} {2:+.2f}*{6} {3:+.2f}*{7} = {4:.2f}".format(*b_natur, test_y1[0], x_natur[0][0],
                                                                            x_natur[0][1], x_natur[0][2]))
    print("{0:.2f} {1:+.2f}*{5} {2:+.2f}*{6} {3:+.2f}*{7} = {4:.2f}".format(*b_natur, test_y1[1], x_natur[1][0],
                                                                            x_natur[1][1], x_natur[1][2]))
    print("{0:.2f} {1:+.2f}*{5} {2:+.2f}*{6} {3:+.2f}*{7} = {4:.2f}".format(*b_natur, test_y1[2], x_natur[2][0],
                                                                            x_natur[2][1], x_natur[2][2]))
    print("{0:.2f} {1:+.2f}*{5} {2:+.2f}*{6} {3:+.2f}*{7} = {4:.2f}".format(*b_natur, test_y1[3], x_natur[3][0],
                                                                            x_natur[3][1], x_natur[3][2]))

    b_norm = [sum(y_aver) / 4,
              sum(y_aver[i] * x_norm[i][1] for i in range(4)) / 4,
              sum(y_aver[i] * x_norm[i][2] for i in range(4)) / 4,
              sum(y_aver[i] * x_norm[i][3] for i in range(4)) / 4]

    print("\nНормоване рівняння регресії: y = {0:.2f} {1:+.2f}*x1 {2:+.2f}*x2 {3:+.2f}*x3".format(*b_norm))

    print("Зробимо перевірку")
    test_y2 = [b_norm[0] + b_norm[1] * x_norm[i][1] + b_norm[2] * x_norm[i][2] + b_norm[3] * x_norm[i][3] for i in
               range(4)]

    print("{0:.2f} {1:+.2f}*{5} {2:+.2f}*{6} {3:+.2f}*{7} = {4:.2f}".format(*b_norm, test_y2[0], x_norm[0][1],
                                                                            x_norm[0][2], x_norm[0][3]))
    print("{0:.2f} {1:+.2f}*{5} {2:+.2f}*{6} {3:+.2f}*{7} = {4:.2f}".format(*b_norm, test_y2[1], x_norm[1][1],
                                                                            x_norm[1][2], x_norm[1][3]))
    print("{0:.2f} {1:+.2f}*{5} {2:+.2f}*{6} {3:+.2f}*{7} = {4:.2f}".format(*b_norm, test_y2[2], x_norm[2][1],
                                                                            x_norm[2][2], x_norm[2][3]))
    print("{0:.2f} {1:+.2f}*{5} {2:+.2f}*{6} {3:+.2f}*{7} = {4:.2f}".format(*b_norm, test_y2[3], x_norm[3][1],
                                                                            x_norm[3][2], x_norm[3][3]))
    cohren(m, y, y_aver, x_norm, b_natur)


# ----------------- Критерій Кохрена -----------------------
def cohren(m, y, y_aver, x_norm, b_natur):
    print("\nКритерій Кохрена")
    dispersion = []
    for i in range(4):
        z = 0
        for j in range(m):
            z += (y[i][j] - y_aver[i]) ** 2
        dispersion.append(z / m)
    print("Дисперсія:", dispersion)
    Gp = max(dispersion) / sum(dispersion)
    print("Gp", Gp)
    f1 = m - 1
    Gt = cohren_table[f1]
    if Gp < Gt:
        print("Gp < Gt\n{0:.4f} < {1} => дисперсія однорідна".format(Gp, Gt))
        student(m, dispersion, y_aver, x_norm, b_natur)
    else:
        print("Gp > Gt\n{0:.4f} > {1} => дисперсія неоднорідна => m+=1".format(Gp, Gt))
        m += 1
        coef(m)


# ----------------------- Критерій Стюдента --------------------------------
def student(m, dispersion, y_aver, x_norm, b_natur):
    print("\nКритерій Стюдента")
    sb = sum(dispersion) / 4
    s_beta = sqrt(sb / (4 * m))
    beta = [sum(y_aver[i] * x_norm[i][j] for i in range(4)) / 4 for j in range(4)]

    t = [abs(beta[i]) / s_beta for i in range(4)]

    f3 = (m - 1) * 4
    t_table = student_table[f3]
    b_impor = []
    for i in range(4):
        if t[i] > t_table:
            b_impor.append(b_natur[i])
        else:
            b_impor.append(0)
    print("Незначні коефіцієнти регресії")
    for i in range(4):
        if b_natur[i] not in b_impor:
            print("b{0} = {1:.2f}".format(i, b_natur[i]))

    y_impor = [b_impor[0] + b_impor[1] * x_natur[i][0] + b_impor[2] * x_natur[i][1] + b_impor[3] * x_natur[i][2] for i
               in range(4)]

    print("{0:.2f} {1:+.2f}*{5} {2:+.2f}*{6} {3:+.2f}*{7} = {4:.2f}".format(*b_impor, y_impor[0], x_natur[0][0],
                                                                            x_natur[0][1], x_natur[0][2]))
    print("{0:.2f} {1:+.2f}*{5} {2:+.2f}*{6} {3:+.2f}*{7} = {4:.2f}".format(*b_impor, y_impor[1], x_natur[1][0],
                                                                            x_natur[1][1], x_natur[1][2]))
    print("{0:.2f} {1:+.2f}*{5} {2:+.2f}*{6} {3:+.2f}*{7} = {4:.2f}".format(*b_impor, y_impor[2], x_natur[2][0],
                                                                            x_natur[2][1], x_natur[2][2]))
    print("{0:.2f} {1:+.2f}*{5} {2:+.2f}*{6} {3:+.2f}*{7} = {4:.2f}".format(*b_impor, y_impor[3], x_natur[3][0],
                                                                            x_natur[3][1], x_natur[3][2]))
    fisher(m, y_aver, b_impor, y_impor, sb)


# ----------------------- Критерій Фішера --------------------------------
def fisher(m, y_aver, b_impor, y_impor, sb):
    print("\nКритерій Фішера")
    d = 0
    for i in b_impor:
        if i:
            d += 1
    f3 = (m - 1) * 4
    f4 = 4 - d
    s_ad = sum((y_impor[i] - y_aver[i]) ** 2 for i in range(4)) * m / f4
    Fp = s_ad / sb
    Ft = fisher_table[f3][f4 - 1]
    if Fp < Ft:
      print("Fp < Ft => {0:.2f} < {1}".format(Fp, Ft))
      print("Отримана математична модель при рівні значимості 0.05 адекватна експериментальним даним")
    else:
      print("Fp > Ft => {0:.2f} > {1}".format(Fp, Ft))
      print("Рівняння регресії неадекватно оригіналу при рівні значимості 0.05")


if __name__ == '__main__':
      coef(m)

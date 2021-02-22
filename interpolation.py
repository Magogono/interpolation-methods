import numpy as np
import solve


# Funkcja zwraca wartość wielomianu o współczynnikach @coefs = [a0, a1, .. an] dla argumentu @x
def polynomial(x, coefs):
    result = 0
    for i in range(len(coefs)):
        result += coefs[i] * (x**i)

    return result


# Funkcja do interpolacji metodą funkcji sklejanych (splajnów) 3 stopnia:
# - przyjmuje dane całkowite i żądany odstęp między kolejnymi próbkami
# - zwraca interpolowane wartości danych we wszystkich przekazanych punktach.
def splineInterpolation(data, sample_step):
    # wyznaczamy próbki
    samples = data[::sample_step]
    if data[-1, 0] != samples[-1, 0]:
        samples = np.vstack((samples, data[-1, :]))

    # znajdujemy funkcję interpolacyjną
    coefs = splineInterpolationPolynomials(samples)

    # wyznaczamy interpolowane wartości wysokości
    distance = data[:, 0]
    interpolated_value = []
    range = 0
    for x in distance:
        if (range+1)*sample_step < len(distance) and x > distance[(range+1)*sample_step]:
            range += 1
        h = x - distance[range*sample_step]
        poly_coefs = coefs[4*range:4*(range+1)]
        interpolated_value.append(polynomial(h, poly_coefs))

    return interpolated_value


# Funkcja zwraca w postaci tablicy współczynniki wielomianów 3 stopnia do interpolacji splajnami.
def splineInterpolationPolynomials(samples):
    distance = samples[:, 0]
    height = samples[:, 1]

    # n - liczba próbek
    n = len(distance)

    # wielomianów Si jest n-1, więc szukanych współczynników 4*(n-1)
    numOfCoef = 4*(n-1)
    S_matrix = np.zeros([numOfCoef, numOfCoef])
    y_vectT = np.zeros(numOfCoef)
    numOfEq = 0

    # ========= tworzenie układu równań =========
    # wartości dla i = 0..n-2:
    # 1: Si(xi) = ai = f(xi)
    # 2: Si(xi+1) = ai + bi*h + ci*h^2 + di*h^3 = f(xi+1), h = (xi+1 - xi)
    for i in range(0, n-1):
        coefIndex = 4*i
        # 1:
        S_matrix[numOfEq][coefIndex] = 1.0
        y_vectT[numOfEq] = height[i]
        numOfEq += 1

        # 2:
        h = distance[i+1] - distance[i]
        for power in range(4):
            S_matrix[numOfEq][coefIndex + power] = h**power
        y_vectT[numOfEq] = height[i+1]
        numOfEq += 1

    # pierwsza pochodna dla węzłów wewnętrznych, i = 1..n-2
    # Si-1'(xi) = Si'(xi), h = (xi - xi-1) =>
    # bi-1 + 2ci-1*h + 3di-1*h^2 - bi = 0
    for i in range(1, n-1):
        coefIndex = 4 * i
        h = distance[i] - distance[i-1]

        S_matrix[numOfEq][coefIndex - 3] = 1            # bi-1
        S_matrix[numOfEq][coefIndex - 2] = 2 * h        # ci-1
        S_matrix[numOfEq][coefIndex - 1] = 3 * (h**2)   # di-1
        S_matrix[numOfEq][coefIndex + 1] = -1           # bi
        # y_vectT[numOfEq] = 0

        numOfEq += 1

    # druga pochodna dla węzłów wewnętrznych, i = 1..n-2
    # Si-1''(xi) = Si''(xi), h = (xi - xi-1) =>
    # 2ci-1 + 6d-1*h - 2ci = 0
    for i in range(1, n-1):
        coefIndex = 4 * i
        h = distance[i] - distance[i - 1]

        S_matrix[numOfEq][coefIndex - 2] = 2        # ci-1
        S_matrix[numOfEq][coefIndex - 1] = 6 * h    # di-1
        S_matrix[numOfEq][coefIndex + 2] = -2       # ci
        # y_vectT[numOfEq] = 0

        numOfEq += 1

    # druga pochodna na krawędziach:
    # 1: c0 = 0
    S_matrix[numOfEq][2] = 1
    # y_vectT[numOfEq] = 0
    numOfEq += 1

    # 2: 2cn-2 + 6dn-2*(xn-1 - xn-2)
    S_matrix[numOfEq][4*(n-2) + 2] = 2
    S_matrix[numOfEq][4*(n-2) + 3] = 6 * (distance[n-1] - distance[n-2])
    # y_vectT[numOfEq] = 0
    numOfEq += 1

    # ========= rozwiązanie układu równań =========
    if(n < 1000):
        return solve.factorization_LU(S_matrix, y_vectT)
    else:
        return solve.gauss_seidel(S_matrix, y_vectT)


# Funkcja do interpolacji metodą Lagrange'a:
# - przyjmuje dane całkowite i żądany odstęp między kolejnymi próbkami
# - zwraca interpolowane wartości danych we wszystkich przekazanych punktach.
def lagrangeInterpolation(data, sample_step):
    # wyznaczamy próbki
    samples = data[::sample_step]
    if data[-1, 0] != samples[-1, 0]:
        samples = np.vstack((samples, data[-1, :]))

    # znajdujemy funkcję interpolacyjną
    F = lagrangeInterpolationPolynomial(samples)

    distance = data[:, 0]
    interpolated_value = []
    # wyznaczamy interpolowane wartości wysokości
    for x in distance:
        interpolated_value.append(F(x))

    return interpolated_value


# Funkcja zwraca referencję na funkcję interpolującą Lagrange'a.
def lagrangeInterpolationPolynomial(samples):
    def F(x):
        # długosć x i y to n+1
        n = len(samples)
        fun_F = 0

        # wyliczamy kolejne funkcje fi(x)
        for i in range(n):
            xi, yi = samples[i]
            fun_fi = 1

            for j in range(n):
                if i != j:
                    xj, yj = samples[j]
                    fun_fi *= (x - xj) / (xi - xj)

            fun_F += yi * fun_fi

        return fun_F

    return F
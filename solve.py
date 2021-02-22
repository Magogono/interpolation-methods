import numpy as np
import math

MAX_ITER = 10000
eps = 1e-9


# if equation is not solved, returns -1
# else returns number of iterations and solution vector
def jacobi(A, b):
    # tworzenie macierzy D, L i U
    N = len(A[0])
    D = np.zeros((N, N))
    L = np.zeros((N, N))
    U = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i > j:
                L[i][j] = A[i][j]
            elif i < j:
                U[i][j] = A[i][j]
            else:
                D[i][j] = A[i][j]

    # wektor początkowy rozwiązania
    x = np.ones(N)

    iter = -1
    # iteracje meotdy Jacobiego
    for i in range(MAX_ITER):
        l_u_x_vect = (-L-U) @ x

        comp1_vect = forward_substitution(D, l_u_x_vect)
        comp2_vect = forward_substitution(D, b)

        # obliczanie nowego przybliżenia rozwiązania
        x = comp1_vect + comp2_vect

        res = A @ x - b
        if norm_2(res) < eps:
            iter = i
            break

    return x, iter


# if equation is not solved, returns -1
# else returns number of iterations and solution vector
def gauss_seidel(A, b):
    # tworzenie macierzy D, L i U
    N = len(A[0])
    D = np.zeros((N, N))
    L = np.zeros((N, N))
    U = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i > j:
                L[i][j] = A[i][j]
            elif i < j:
                U[i][j] = A[i][j]
            else:
                D[i][j] = A[i][j]

    # wektor początkowy rozwiązania
    x = np.ones(N)

    iter = -1
    # iteracje meotdy Jacobiego
    for i in range(MAX_ITER):
        u_x_vect = (-U) @ x

        comp1_vect = forward_substitution(D + L, u_x_vect)
        comp2_vect = forward_substitution(D + L, b)

        # obliczanie nowego przybliżenia rozwiązania
        x = comp1_vect + comp2_vect

        res = A @ x - b
        if norm_2(res) < eps:
            iter = i
            break

    return x, iter


def factorization_LU(A, b):
    # rozkład macierzy A na czynniki L i U
    N = len(A[0])
    U = np.matrix.copy(A)
    L = np.identity(N)
    P = np.identity(N)

    # faktoryzacja z pivotingiem
    for k in range(N-1):
        # znajdź pivota
        ind = np.abs(U[k:N, k]).argmax()
        ind = ind + k

        # wymień wiersze
        for i in range(k, N):
            U[k][i], U[ind][i] = U[ind][i], U[k][i]
        for i in range(0, k):
            L[k][i], L[ind][i] = L[ind][i], L[k][i]
        for i in range(0, N):
            P[k][i], P[ind][i] = P[ind][i], P[k][i]

        # faktoryzacja LU
        for j in range(k+1, N):
            L[j][k] = U[j][k]/U[k][k]
            for i in range(k, N):
                U[j][i] = U[j][i] - L[j][k] * U[k][i]

    # rozwiązanie bezpośrednie równania LUx = b (Ax = b)
    y = forward_substitution(L, (P @ b))
    x = backward_substitution(U, y)

    return x


def forward_substitution(L, b):
    N = len(b)
    x = np.zeros(N)
    for i in range(N):
        sum = 0
        for j in range(0, i):
            sum += L[i][j] * x[j]

        x[i] = (b[i] - sum)/L[i][i]

    return x


def backward_substitution(U, b):
    N = len(b)
    x = np.zeros(N)
    for i in reversed(range(N)):
        sum = 0
        for j in range(i+1, N):
            sum += U[i][j] * x[j]

        x[i] = (b[i] - sum)/U[i][i]

    return x


def norm_2(v):
    sum = 0
    for i in range(len(v)):
        sum += v[i] ** 2
    return math.sqrt(sum)


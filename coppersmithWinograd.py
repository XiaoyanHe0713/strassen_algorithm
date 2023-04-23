import random
import numpy as np
import matplotlib.pyplot as plt
# Implementation of the Coppersmith-Winograd algorithm for matrix multiplication
class coppersmithWinograd:

    def __init__(self, n, M):
        self.n = n
        self.n = M

    def coppersmithWinograd(self, M1, M2, M3, n):

        a = [[random.randint(0, 1)] for i in range(n)]

        M2a = [[0] for i in range(n)]
        for i in range(n):
            for j in range(n):
                M2a[i][0] += M2[i][j] * a[j][0]

        M3a = [[0] for i in range(n)]
        for i in range(n):
            for j in range(n):
                M3a[i][0] += M3[i][j] * a[j][0]

        M1M2a = [[0] for i in range(n)]
        for i in range(n):
            for j in range(n):
                M1M2a[i][0] += M1[i][j] * M2a[j][0]

        # Calculate the error in L2 norm
        error = 0
        for i in range(n):
            error += (M1M2a[i][0] - M3a[i][0]) ** 2

        return error
    
    def run(self):
        # Generate a random matrix
        A = np.random.random((n, n))
        I = np.identity(n)

        error = []
        for m in M:
            A = np.dot(A, m)
            B = np.linalg.inv(A)

            error.append(self.coppersmithWinograd(A, B, I, n))

        # plot the error with respect to M
        plt.plot(M, error)
        plt.xlabel("scalar")
        plt.ylabel("error")
        plt.show()

if __name__ == "__main__":
    # test the algorithm
    n = 10
    len = 40

    # Generate a random array of scalars takes from the interval [10, 100]
    M = np.random.randint(10, 200, size=(len))
    M = np.sort(M)
    s = coppersmithWinograd(n, M)
    s.run()
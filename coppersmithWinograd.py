import random
# Implementation of the Coppersmith-Winograd algorithm for matrix multiplication
class coppersmithWinograd:

    def __init__(self, A, B, C, n):
        self.A = A
        self.B = B
        self.C = C
        self.n = n

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
    
if __name__ == "__main__":
    # test the algorithm
    # random matrices
    A = [[1,2],[3,4]]
    B = [[2,0],[1,2]]
    C = [[4,4],[10,8]]
    n = 2
    s = coppersmithWinograd(A, B, C, n)

    print(s.coppersmithWinograd(A, B, C, n))

        
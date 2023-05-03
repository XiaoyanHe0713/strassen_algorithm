import random
from math import log, ceil
import numpy as np
from tqdm import tqdm
class strassen:

    # initialize the class
    def __init__(self, A, B):
        self.A = A
        self.B = B

    # add two matrices together
    def add(self, A, B):
        if len(A) != len(B) or len(A[0]) != len(B[0]):
            raise Exception("Matrices are not the same size!")

        n = len(A)
        C = [[0 for i in range(n)] for j in range(n)]
        for i in range(n):
            for j in range(n):
                C[i][j] = A[i][j] + B[i][j]
        return C

    # subtract two matrices
    def sub(self, A, B):
        if len(A) != len(B) or len(A[0]) != len(B[0]):
            raise Exception("Matrices are not the same size!")

        n = len(A)
        C = [[0 for i in range(n)] for j in range(n)]
        for i in range(n):
            for j in range(n):
                C[i][j] = A[i][j] - B[i][j]
        return C
    
    # naive matrix multiplication
    def multiply(self, A, B):
        if len(A) != len(B) or len(A[0]) != len(B[0]):
            raise Exception("Matrices are not the same size!")

        n = len(A)
        C = [[0 for i in range(n)] for j in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i][j] += A[i][k] * B[k][j]
        return C

    # get the dimensions of a matrix
    def dim(self, A):
        return (len(A), len(A[0]))

    # split a matrix into quarters
    def split(self, A):

        if len(A) % 2 != 0 or len(A[0]) % 2 != 0:
            raise Exception("Odd matrices are not supported!")

        n = len(A)
        m = int(n/2)
        A11 = [[0 for i in range(m)] for j in range(m)]
        A12 = [[0 for i in range(m)] for j in range(m)]
        A21 = [[0 for i in range(m)] for j in range(m)]
        A22 = [[0 for i in range(m)] for j in range(m)]
        for i in range(m):
            for j in range(m):
                A11[i][j] = A[i][j]
                A12[i][j] = A[i][j+m]
                A21[i][j] = A[i+m][j]
                A22[i][j] = A[i+m][j+m]
        return A11, A12, A21, A22

    # pad a matrix with zeros
    def pad(self, A):
        n = len(A)
        nextPowerOfTwo = lambda n: 2 ** int(ceil(log(n, 2)))
        m = nextPowerOfTwo(n)
        C = [[0 for i in range(m)] for j in range(m)]
        for i in range(n):
            for j in range(n):
                C[i][j] = A[i][j]
        return C

    # strassen's algorithm
    def strassen(self, A, B):
        if len(A) != len(B) or len(A[0]) != len(B[0]):
            raise Exception("Matrices are not the same size!")

        if (self.dim(A) == (1,1) or self.dim(B) == (1,1)) or (self.dim(A) == (2,2) or self.dim(B) == (2,2)):
            return self.multiply(A,B)

        # if the matrices are odd, pad a row and column of zeros
        temp = len(A)
        if len(A) % 2 != 0 or len(A[0]) % 2 != 0:
            A = self.pad(A)
            B = self.pad(B)
        
        # split the matrices into quarters
        A11, A12, A21, A22 = self.split(A)
        B11, B12, B21, B22 = self.split(B)

        # calculate 7 products
        M1 = self.strassen(self.add(A11, A22), self.add(B11, B22))
        M2 = self.strassen(self.add(A21, A22), B11)
        M3 = self.strassen(A11, self.sub(B12, B22))
        M4 = self.strassen(A22, self.sub(B21, B11))
        M5 = self.strassen(self.add(A11, A12), B22)
        M6 = self.strassen(self.sub(A21, A11), self.add(B11, B12))
        M7 = self.strassen(self.sub(A12, A22), self.add(B21, B22))

        # calculate 4 quadrants of the final matrix
        C11 = self.add(self.sub(self.add(M1, M4), M5), M7)
        C12 = self.add(M3, M5)
        C21 = self.add(M2, M4)
        C22 = self.add(self.sub(self.add(M1, M3), M2), M6)

        # join the 4 quadrants into a single matrix
        n = len(A)
        m = int(n/2)
        C = [[0 for i in range(n)] for j in range(n)]
        for i in range(m):
            for j in range(m):
                C[i][j] = C11[i][j]
                C[i][j+m] = C12[i][j]
                C[i+m][j] = C21[i][j]
                C[i+m][j+m] = C22[i][j]
        
        # if the matrix was padded, remove the extra rows and columns
        if temp != n:
            C = C[:temp]
            for i in range(temp):
                C[i] = C[i][:temp]

        return C

    # print the matrix
    def printMatrix(self, A):
        n = len(A)
        for i in range(n):
            print (A[i])
    
    # run the algorithm
    def run(self):
        print ("A = ")
        self.printMatrix(self.A)
        print ("B = ")
        self.printMatrix(self.B)
        print ("C = A * B = ")
        C = self.strassen(self.A, self.B)
        self.printMatrix(C)

    def testError(self, size, m):
        D = np.eye(size)
        # The first third of the matrix is multiplied by 1
        # The second third of the matrix is multiplied by m
        # The last third of the matrix is multiplied by m^2
        for i in range(size):
            if (i < size/3):
                D[i][i] = 1
            elif (i < size*2/3):
                D[i][i] = m
            else:
                D[i][i] = m**2
        
        A = D @ self.A
        B = self.B @ D
        C = A @ B
        C_strassen = self.strassen(A, B)
        error = abs((C[0][0]-C_strassen[0][0])/C[0][0])

        # import machine epsilon
        from numpy import finfo
        eps = finfo(float).eps
        threshold = eps * m**4
    
        D = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                error = abs((C[i][j] - C_strassen[i][j])/C[i][j])
                
                if (error > threshold):
                    D[i][j] = 20

                elif (error > threshold/(m**2)):
                    D[i][j] = 10

                elif (error > threshold/(m**3)):
                    D[i][j] = 5
            
            # visualize the matrix where x-axis is the row and y-axis is the column and z-axis is the value of the matrix
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x = np.arange(0, size, 1)
            y = np.arange(0, size, 1)
            X, Y = np.meshgrid(x, y)
            Z = D[X,Y]
            ax.plot_surface(X, Y, Z)
            ax.set_xlabel('Row')
            ax.set_ylabel('Column')
            ax.set_zlabel('Value')

            # save the figure in the folder
            fig.savefig("tests/strassen_" + str(size) + "_" + str(m) + ".png")

if __name__ == "__main__":
    # pipeline for testing
    sizes = [16, 32, 64, 128]
    values = [10, 100, 1000]
    
    for size in tqdm(sizes):
        for value in tqdm(values):
            # generate random matrices whose elements are between -1 and 1
            A = np.random.rand(size, size) * 2 - 1
            B = np.random.rand(size, size) * 2 - 1
            s = strassen(A, B)
            s.testError(size, value)
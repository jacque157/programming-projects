class MatrixError(Exception):
    pass

class Matrix:

    def __init__(self, two_dimensional_sequence):
        self.check_input(two_dimensional_sequence)
       
        self.matrix = [None] * len(two_dimensional_sequence)
        
        for i, row in enumerate(two_dimensional_sequence):
            self.matrix[i] = [0] * len(row)
            for j, value in enumerate(row):
                self.matrix[i][j] = value
                
    def __repr__(self):
        rep = "\n"
        for row in self.matrix:
            rep += "|"
            for value in row:
                rep += " " + str(value)
            rep += " |\n"

        return rep

    def __mul__(self, other_matrix):
        if isinstance(other_matrix, int) or isinstance(other_matrix, float):
            return self.scalar_mul(other_matrix)
        
        if not isinstance(other_matrix, Matrix):
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other_matrix).__name__}'")

        k, l = self.size()
        m, n = other_matrix.size()

        if l != m:
            raise MatrixError(f"Cannot multiply matrices of size {k}x{l} and {m}x{n}.")

        seq = [[0 for i in range(n)] for j in range(k)]

        for i in range(k):
            for j in range(n):
                for p in range(l):
                    seq[i][j] += self.matrix[i][p] * other_matrix.matrix[p][j]
        return Matrix(seq)

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self.scalar_mul(other)
        if isinstance(other, Matrix):
            return other * self
        raise TypeError(f"unsupported operand type(s) for *: '{type(other).__name__}' and '{type(self).__name__}'")
        

    def __getitem__(self, index):
        return [ value for value in self.matrix[index]]

    def __add__(self, other_matrix):
        if not isinstance(other_matrix, Matrix):
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other_matrix).__name__}'")

        k, l = self.size()
        m, n = other_matrix.size()

        if k != m or l != n:
            raise MatrixError(f"Cannot add matrices of size {k}x{l} and {m}x{n}.")

        seq = [[0 for i in range(n)] for j in range(k)]

        for i in range(k):
            for j in range(l):
                seq[i][j] += self.matrix[i][j] + other_matrix.matrix[i][j]
        return Matrix(seq)

    def __sub__(self, other_matrix):
        if not isinstance(other_matrix, Matrix):
            raise TypeError(f"unsupported operand type(s) for *: '{type(self).__name__}' and '{type(other_matrix).__name__}'")
        return -1 * other_matrix + self

    def scalar_mul(self, s):
        m, n = self.size()
        return Matrix([[ s * self[j][i] for i in range(n)] for j in range(m)])
        
    
    def check_input(self, input_sequence):
        if len(input_sequence) == 0:
            raise MatrixError("Sequence cannot be empty.")

        try:
            size = len(input_sequence[0])
            for row in input_sequence:
                if len(row) != size:
                    raise MatrixError("Rows cannot vary in size.")
        except TypeError:
            raise MatrixError("Sequence must have iterable rows.")

    def size(self):
        return (len(self.matrix), len(self.matrix[0]))


class Mat4(Matrix):
    
    def __init__(self, four_by_four_sequence):
        self.check_4x4_matrix(four_by_four_sequence)
        super().__init__(four_by_four_sequence)
        
    def check_4x4_matrix(self, sequence):
        if len(sequence) != 4:
            raise MatrixError(f"Sequence must have exactly 4 rows.")
        
        for row in sequence:
            try:
                if len(row) != 4:
                    raise MatrixError(f"Each row must have exactly 4 columns.")
            except TypeError:
                raise MatrixError("Sequence must have iterable rows.")

    def __mul__(self, other_matrix):
        if isinstance(other_matrix, Vec4):
            seq = [0 for j in range(4)]
            for i in range(4):
                for j in range(4):
                    seq[i] += self.matrix[i][j] * other_matrix.matrix[j][0]
            return Vec4(*seq)
        
        if isinstance(other_matrix, Mat4):
            seq = [[0 for i in range(4)] for j in range(4)]

            for i in range(4):
                for j in range(4):
                    for p in range(4):
                        seq[i][j] += self.matrix[i][p] * other_matrix.matrix[p][j]
            return Mat4(seq)

        if isinstance(other_matrix, int) or isinstance(other_matrix, float):
            s = other_matrix
            seq = [[0 for i in range(4)] for j in range(4)]
            
            for i in range(4):
                for j in range(4):
                    seq[i][j] += self.matrix[i][j] + s                 
            return Mat4(seq)

        return super().__mul__(other_matrix)
    
    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self * other
        raise TypeError(f"unsupported operand type(s) for *: '{type(other).__name__}' and '{type(self).__name__}'")

    def __add__(self, other_matrix):
        if isinstance(other_matrix, Mat4):
            seq = [[0 for i in range(4)] for j in range(4)]
            
            for i in range(4):
                for j in range(4):
                    seq[i][j] += self.matrix[i][j] + other_matrix.matrix[i][j]
            return Mat4(seq)
        
        return super().__add__(other_matrix)
    
        
class Vec4(Matrix):
    
    def __init__(self, x, y, z, w):
        super().__init__(((x,),
                        (y,),
                        (z,),
                        (w,)))

    def __mul__(self, other_vector):
        if isinstance(other_vector, Vec4):
            x1, y1, z1, w = self.components()
            x2, y2, z2, w = other_vector.components()
            return Vec4((y1 * z2) - (z1 * y2), (z1 * x2) - (x1 * z2), (x1 * y2) - (y1 * x2), w)
        
        if isinstance(other_vector, int) or isinstance(other_vector, float):
            s = other_vector
            x1, y1, z1, w = self.components()
            return Vec4(x1 * s, y1 * s, z1 * s, w * s)
        
        return super().__mul__(other_vector)

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self * other
        raise TypeError(f"unsupported operand type(s) for *: '{type(other).__name__}' and '{type(self).__name__}'")

    def __add__(self, other_vector):
        if isinstance(other_vector, Vec4):
            seq = [[0 for i in range(4)] for j in range(1)]
            x1, y1, z1, w1 = self.components()
            x2, y2, z2, w2 = other_vector.components()

            return Vec4(x1 + x2, y1 + y2, z1 + z2, w1 + w2)
        
        return super().__add__(other_matrix)

    def __repr__(self):
        return repr(self.components())

    def length(self):
        x, y, z, w = self.components()
        return ((x * x) + (y * y) + (z * z) + (w * w)) ** (1 / 2)

    def dot_product(self, other_vector):
        x1, y1, z1, w1 = self.components()
        x2, y2, z2, w2 = other_vector.components()
        return (x1 * x2) + (y1 * y2) + (z1 * z2) + (w1 * w2)

    def normalise(self):
        x, y, z, w = self.components()
        d = self.length()
        return Vec4(x / d, y / d, z / d, w / d)
        
    def components(self):
        return self.matrix[0][0], self.matrix[1][0], self.matrix[2][0], self.matrix[3][0]



class Mesh: 
    
    def __init__(self, file_name):
        self.vertices = [None]
        self.indices = []
        self.name = "No name"
        self.read_file(file_name)

    def read_file(self, name):
        with open(name, "r") as file:
            for line in file:
                elements = line.split()
                if elements[0] == "v":
                    x, y, z = map(float, elements[1::])
                    self.vertices.append(Vec4(x, y, z, 1))
                elif elements[0] == "f":
                    i, j, k = map(int, elements[1::])
                    self.indices.append((i, j, k))
                elif elements[0] == "o":
                    self.name = " ".join(elements[1::])
    
if __name__ == "__main__":
    A = Matrix([[3, 5],
                [1, 2]])
    
    B = Matrix([[4, 1],
                [2, -1]])
    
    C = Matrix([[0, 2],
                [2, 3]])
    print()
    print(A * B)
    print()
    print(B * A)
    print()
    print((A * B) * C)
    print()
    print(A * (B * C))
    
    A = Matrix([[3, 5, 1],
                [2, 1, 0],
                [1, 3, 1]])
    
    B = Matrix([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    
    C = Matrix([[1],
                [0],
                [0]])
    print()
    print(A * C)
    print()
    print(A * B)
    print()
    #print(C * A)
    #print()

    ### Addition Test
    # expected  4 5 1
    #           2 2 0
    #           1 3 2
    print()
    print(A + B)

    p = Vec4(1, 2, 3, 1)
    v = Vec4(1, 2, 3, 0)
    mt = Mat4([[1, 0, 0, 52],
               [0, 1, 0, 18],
               [0, 0, 1, 2],
               [0, 0, 0, 1]])

    ms = Mat4([[2, 0, 0, 0],
               [0, 2, 0, 0],
               [0, 0, 2, 0],
               [0, 0, 0, 1]])
    print(mt * mt * mt * p)
    print()
    print(mt * v)
    print(ms * v)
    print(ms * p)

    ### CrossProduct Test
    # expected (-14, -6, 10)
    a = Vec4(2, -3, 1, 0)
    b = Vec4(4, -1, 5, 0)
    print()
    print(a * b)


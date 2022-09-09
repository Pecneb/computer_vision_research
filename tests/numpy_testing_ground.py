import numpy

vec = numpy.array([1, 2, 3, 4])
vec_square = numpy.square(vec)
print(vec_square)

mat = numpy.mat([[1,2,3],
                 [4,5,6],
                 [7,8,9]])

print(numpy.diag(mat))
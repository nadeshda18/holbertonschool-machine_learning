# Welcome to Linear Algebra

## Things you should be able to answer without the help of Google

### What is a vector?
Vectors are a fundamental element of linear algebra but simply, vectors are ordinary numbers.

### What is a matrix?
A matrix is an array of numbers. Those numbers are contained within square brackets. In other words, a matrix is a 2D array, made up of rows and columns.

### What is a transpose?
Transposing a matrix is flipping the rows and columns of the matrix, effectively swapping elements across the diagonal.

### What is the shape of a matrix?
When referring to the shape of a matrix, we refer to the rows and columns in that order. A 2 x 3 matrix, is a rectangular arrangement of 2 rows with 3 colums.

### What is an axis?
We have axis 0 which represents the rows and axis 1 which represents the columns.

### What is a slice?
A slice is a subset of elements from a vector or matrix. For example, slicing a vector or matrix can involve selecting a range of elements along one or more axes.

### How do you slice a vector/matrix?
To slice a vector or matrix in Python using NumPy, you can use slicing notation. For instance, to select the first three elements of a vector v, you can use v[:3].

### What are the element-wise operations?
Element-wise operations are operations performed independently on each element of a vector or matrix. Examples include addition, subtraction, multiplication, and division.

### How do you concatenate vectors/matrices?
To concatenate vectors or matrices in NumPy, you can use functions like numpy.concatenate() or simply use the vertical or horizontal stack functions (numpy.vstack() or numpy.hstack()).

### What is the dot product?
The dot product of two vectors is the sum of the products of their corresponding elements.

### What is matrix multiplication?
Matrix multiplication involves multiplying each element of a row of the first matrix by each element of a column of the second matrix and summing up these products. The result is a new matrix. For example, if A and B are two matrices, their product C is calculated as C = A*B.

### What is NumPy?
NumPy is a powerful library for numerical computing in Python. It provides support for arrays, matrices, and a wide range of mathematical functions, making it essential for tasks involving numerical operations.

### What is parallelization and why is it important?
Parallelization is the process of splitting a task into smaller parts that can be executed simultaneously by multiple processing units, such as CPU cores or GPUs. It's important because it can significantly reduce computation time, especially for tasks that are computationally intensive.

### What is broadcasting?
Broadcasting is a mechanism in NumPy that allows arrays of different shapes to be combined together in arithmetic operations. NumPy automatically broadcasts the smaller array to match the shape of the larger array, making the operation possible even if the shapes are not identical.


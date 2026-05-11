def mandelbrot(c, max_iter):
    """ This function determines whether a complex number belongs
    to the mandelbrot set.
    parameters:
    c: complex, complex number being tested
    max_iter: int, max number of iteration

    return iteration number where divergence occurs
    return max_iter if bounded.
    """
    z = 0
    for i in range(max_iter):
        z=z**2 + c
        if abs(z) > 2:
            return i
    return max_iter

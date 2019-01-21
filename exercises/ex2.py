def adders(n):
    """Return a list of n functions, adding 1, 2, ..., n to their argument.

    Parameters
    ----------
    n : int
        The number of functions to return.

    Returns
    -------
    functions : list of (int -> int)
        List of functions.

    Examples
    --------
    >>> num = 9
    >>> [f(num) for f in adders(4)]
    [13, 14, 15, 16]
    """
    functions = []
    for i in range(n):
        function = lambda x: x + i
        functions.append(function)
    return functions

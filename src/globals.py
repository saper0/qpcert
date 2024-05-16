def init(other_params):
    """Define global variables for use. """
    global debug, zero_tol, grad_tol
    if "zero_tol" in other_params:
        zero_tol = other_params["zero_tol"]
    else:
        zero_tol = 1e-6
    if "debug" in other_params:
        debug = other_params["debug"]
    else:
        debug = False
    grad_tol = 1e-8
    
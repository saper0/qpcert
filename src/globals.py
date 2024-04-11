def init(other_params):
    """Define global variables for use. """
    global debug, zero_tol
    zero_tol = 1e-6
    if "debug" in other_params:
        debug = other_params["debug"]
    else:
        debug = False
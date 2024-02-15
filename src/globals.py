def init(other_params):
    """Define global variables for use. """
    global debug
    if "debug" in other_params:
        debug = other_params["debug"]
    else:
        debug = False
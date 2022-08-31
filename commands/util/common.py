import os
import numpy as np
from .cli_report import report_arguments


@report_arguments("Set seed")
def set_seed(seed) -> int:
    """"
        This function loads the seed or generate it, if required
        Return: None
        Post: Sets all the random seed for generators to the same value
    """
    to_seed = int(seed) if seed else int.from_bytes(os.urandom(4), byteorder='big')
    np.random.seed(to_seed)
    return to_seed



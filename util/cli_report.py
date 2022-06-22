"""
    This file contains some functions which can be used to
    create a cli report for example for some experemnts which
    can go through a long set of steps.

    Author: Enrique Fernandez-Blanco

    This library requires the rich library for its behaviout
"""
from rich import print
from functools import wraps

def screen_header(label, decoration='='):
    """
        This function creates a header to separate parts of
        the different steps.

        label: tect ot be on then header
        decoration: simbols used to complete the remaining space
                    in the header not filled with the label

        ouput: None

    """
    text = f"{label.upper()}".center(80, decoration)
    print(f"[bold white]{text}[/bold white]")


def report(header:str, tail:str) -> None:
    """
        This functions writes a line in the cli which is completed with
        dots the remaning space.

        header: Text to appear as label
        output: the other end of the line

        output: None
    """
    label = f"{header}".ljust(80-len(str(tail)), '.')
    print(f"[dim white]{label}[/dim white][green]{tail}[/green]")

def report_arguments(label=None):
    """
        Decorator which prints the NAMED arguments in lines to an easy debug
        if the name of the arguments wants to be the header of the line
        the description has to be set to None. The unnamed arguments are ignored

        description: Text for the head of the line, if None, the name of the argument is used
        Output: decorated funtion
    """
    def _function_decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            for key in kwargs:
                header = label if label else key
                report(header,kwargs[key])
            return function(*args, **kwargs)
        return wrapper
    return _function_decorator

def report_output(label):
    """"
        This decorator print in a certain fashion based on report the output of a
        function for an easier debug

        description: text to be used as label of the line
        Output: Decorated function
    """
    def _function_decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            output = function(*args, **kwargs)
            report(label, output)
            return output
        return wrapper
    return _function_decorator


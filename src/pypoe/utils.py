"""Kitchen sink of utility classes and functions.

Code first, sort later.
"""


# Python's `Count` doesn't give an alternative way of removing entries,
# and `del` is an operator so it's a PITA to work with it in a functional
# pipeline (like toolz's)
def delfn(x, key):
    del x[key]
    return x


class RGB:
    def __init__(self, kwargs):
        self._r = kwargs.get("r", 0)
        self._g = kwargs.get("g", 0)
        self._b = kwargs.get("b", 0)

    def __str__(self):
        # the `__str__` method will let us call `str(rgb_instance)` and get
        # a formatted representation of the crafting option.
        r = f"{self._r}R" if self._r > 0 else ""
        g = f"{self._g}G" if self._g > 0 else ""
        b = f"{self._b}B" if self._b > 0 else ""

        return "chromatic" if (r + g + b) == "" else "bench " + r + g + b

    def __iter__(self):
        # Python's `__iter__` lets us use algorithms on the objects of this class
        # like sorting, transforming to a list and getting min and max, but also
        # zipping and using in for loops.
        for e in (self._r, self._g, self._b):
            yield e

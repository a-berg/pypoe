"""
Socket currency related functions.

E.g. compute the chance to get off colors, etc.
"""


import math
from itertools import repeat, combinations_with_replacement
from collections import Counter
from dataclasses import dataclass
from toolz import pipe, curry
from toolz.curried import map, filter, get

import numpy as np
from scipy.stats import multinomial, geom

from .utils import delfn, RGB

rng = np.random.default_rng()  # no seed, we want "true" randomness

# Dictionary with chromatic costs for a given generic color combination
CHROMATIC_COSTS = {
    (0, 0, 0): 1,
    (0, 0, 1): 4,
    (0, 1, 1): 15,
    (0, 0, 2): 25,
    (0, 1, 2): 100,
    (0, 0, 3): 120,
}


@dataclass
class Item:
    n_sockets: int = 6
    max_sockets: int = 6
    str_req: int = 0
    dex_req: int = 0
    int_req: int = 0

    def __iter__(self):
        # Python's `__iter__` lets us use algorithms on the objects of this class
        # like sorting, transforming to a list and getting min and max, but also
        # zipping and using in for loops.
        for e in (self.str_req, self.dex_req, self.int_req):
            yield e

    def __post_init__(self):
        self.n_req = pipe(
            self,  # hopefully this triggers __iter__
            filter(lambda x: x > 0),
            list,
            len,
        )


class ColorChances:
    """This is the 'hardest' part because PoE has lots of attribute permutations and
    3 special cases for them.
    The class will compute the on-color and off-color chances for each attribute.
    """

    def __init__(self, item):
        _attr_names = ("str", "dex", "int")
        # this will be used to sort the chances in the correct attribute order
        # last element is max
        sorted_attrs = sorted(zip(_attr_names, item), key=get(1))
        # by default triple requirements or no requirements (e.g. Dialla's Malefaction)
        # all colors are equally probable
        self._chances = dict(zip(_attr_names, np.ones(3) / 3))
        # for convenience
        hi_attr = sorted_attrs[2][1]  # primary attribute
        lo_attr = sorted_attrs[1][1]  # the "secondary" attribute

        if item.n_req == 1:
            on_chance = self._on_color_chance_1req(hi_attr)
            off_chance = (1 - on_chance) / 2
            for (attr, _), p in zip(sorted_attrs, [off_chance, off_chance, on_chance]):
                self._chances[attr] = p
        elif item.n_req == 2:
            on_chance = self._on_color_chance_2req(hi_attr, lo_attr)
            for (attr, _), p in zip(sorted_attrs, [0.1, 0.9 - on_chance, on_chance]):
                self._chances[attr] = p

    def _on_color_chance_1req(self, R):
        return 0.9 * (R + 10) / (R + 20)

    def _on_color_chance_2req(self, R1, R2):
        return 0.9 * R1 / (R1 + R2)

    def __iter__(self):
        for e in self._chances.values():
            yield e

    def __getitem__(self, key):
        return self._chances[key]


class ChromaticCalculator:
    """This is the easiest part once you realize socketing follows a
    multinomial distribution and the resulting chances computed via
    the PMF of the multinomial, can be used as tje parameters of n
    geometric distributions in order to get mean attempts, standard
    deviation and percentiles of success.
    """

    def __init__(self, item, desired_socket_colors):
        self._base_chances = np.array(list(ColorChances(item)))
        self._available_sockets = item.n_sockets
        self._desired_colors = np.array(desired_socket_colors)

        # I will use this to tune the multinomial distribution
        self._chromatic_options_modifiers = pipe(
            # the 3 possible colors and the "null" color as 0, to get combinations like 1R
            "rgb0",
            # use combinatorics to get all possible combinations of 3 elements
            lambda x: combinations_with_replacement(x, r=3),
            # use counter to reduce combinations to counts, e.g. 2R1G
            map(Counter),
            # delete 0s from the counters entries (keys)
            map(curry(delfn, key="0")),
            # filter out the 1R1G1B combination, which doesn't exist for bench crafts (man, I wish)
            filter(lambda x: not all(x[i] == 1 for i in "rgb")),
            # transform to dictionaries
            map(dict),
            # custom class to ease formatting and getting the colors
            map(RGB),
            # materialize in memory as a list
            list,
        )

        # self._chromatic_options_modifiers = {
        #     "chromatic": [0, 0, 0],
        #     **dict(
        #         zip(
        #             map(lambda x: "bench " + str(x), all_combinations),
        #             map(list, all_combinations),
        #         )
        #     ),
        # }

    def get_base_chances_multinom(self):
        """The idea is to use a multidimensional Multinomial distribution, and use its probability mass function to
        get the exact chances of each result."""
        # so basically I need to iterate over all crafting options.
        # I will use a for-loop and try to be smart about it later.
        results = []
        for bench_opt in self._chromatic_options_modifiers:
            as_np = np.array(list(bench_opt))
            chance = multinomial.pmf(
                self._desired_colors - as_np,
                n=self._available_sockets - as_np.sum(),
                p=self._base_chances,
            )
            cost = CHROMATIC_COSTS[tuple(sorted(bench_opt))]
            percentiles = geom.ppf([0.66, 0.80, 0.9, 0.95, 0.99], chance)
            results.append([str(bench_opt), chance, cost, percentiles])

        # self.results = results
        return results

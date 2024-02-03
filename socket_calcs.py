"""
Socket currency related functions.

E.g. compute the chance to get off colors, etc.
"""


import math
from itertools import repeat
from dataclasses import dataclass

import numpy as np
from scipy.stats import multinomial, geom

rng = np.random.default_rng()  # no seed, we want "true" randomness


@dataclass
class Item:
    n_sockets: int = 6
    max_sockets: int = 6
    str_req: int = 0
    dex_req: int = 0
    int_req: int = 0


class ColorChances:
    """This is the 'hardest' part because PoE has lots of attribute permutations and
    3 special cases for them.
    The class will compute the on-color and off-color chances for each attribute.
    """

    def __init__(self, item):
        _str = item.str_req
        _dex = item.dex_req
        _int = item.int_req

        attrs = np.array([_str, _dex, _int])
        n_req = np.count_nonzero(attrs)
        # this will be used to sort the chances in the correct attribute order
        # note that it is reversed so it goes from greater (position 0) to lesser (position 2)
        ii = np.argsort(attrs)[::-1]
        self.hi_attr = attrs[ii[0]]  # the "dominant" attribute
        self.lo_attr = attrs[ii[1]]  # the "secondary" attribute
        # case 1: all are equal
        if (attrs == attrs[0]).all():
            self._chances = np.array([1 / 3, 1 / 3, 1 / 3])
        # case 2: there is 1 requirement
        elif n_req == 1:
            off_ = self._off_color_chance_1req(hi_attr)
            self._chances = np.array([self._on_color_chance_1req(hi_attr), off_, off_])[
                ii
            ]
        # case 3: there are 2 requirements
        elif n_req == 2:
            self._chances = np.array(
                [
                    self._on_color_chance_2req(hi_attr, lo_attr),
                    # could just do 0.9 - previous value
                    self._on_color_chance_2req(lo_attr, hi_attr),
                    # will just be 0.1
                    self._off_color_chance_2req(),
                ]
            )[ii]
        # triple requirements are not contemplated
        else:
            print("error")

    def _off_color_chance_1req(self, R):
        return 0.05 + 4.5 / (R + 20)

    def _off_color_chance_2req(self):
        return 0.1

    def _on_color_chance_1req(self, R):
        return 0.9 * (R + 10) / (R + 20)

    def _on_color_chance_2req(self, R1, R2):
        return 0.9 * R1 / (R1 + R2)

    def __call__(self):
        return self._chances


class ChromaticCalculator:
    """This is the easiest part once you realize socketing follows a
    multinomial distribution and the resulting chances computed via
    the PMF of the multinomial, can be used as the parameters of n
    geometric distributions in order to get mean attempts, standard
    deviation and percentiles of success.
    """

    def __init__(self, item, desired_sockets):
        self._base_chances = ColorChances(item)()
        self._available_sockets = item.n_sockets
        self._desired_colors = np.array(desired_sockets)

        self.bench_costs = dict(
            base_chromatic=1,
            single_color=4,
            two_color_mix=15,
            two_color_mono=25,
            three_color_mix=100,
            three_color_mono=120,
        )

        self._chromatic_options_modifiers = {
            "chromatic": [0, 0, 0],
            "bench 1R": [1, 0, 0],
            "bench 1G": [0, 1, 0],
            "bench 1B": [0, 0, 1],
            "bench 2R": [2, 0, 0],
            "bench 2G": [0, 2, 0],
            "bench 2B": [0, 0, 2],
            "bench 3R": [3, 0, 0],
            "bench 3G": [0, 3, 0],
            "bench 3B": [0, 0, 3],
            "bench 1R1G": [1, 1, 0],
            "bench 1R1B": [1, 0, 1],
            "bench 1B1G": [0, 1, 1],
        }

    def _get_base_chances_multinom(self):
        """The idea is to use a multidimensional Multinomial distribution, and use its probability mass function to
        get the exact chances of each result."""
        options_modifiers = np.array(
            [v for v in self._chromatic_options_modifiers.values()]
        )
        self.global_df = dict(
            zip(
                self._chromatic_options_modifiers.keys(),
                multinomial.pmf(
                    self._desired_colors - options_modifiers,
                    n=self._available_sockets,  # FIXME: this will vary with bench crafting options!
                    p=self._base_chances,
                ),
            )
        )

    def _get_percentiles(self, p):
        return geom.ppf([0.66, 0.80, 0.9, 0.95, 0.99], p)


def geometric_inverse_cdf(p: float, pin: float):
    log_1 = math.log1p(-p)
    log_2 = math.log1p(-pin)
    return math.ceil(log_1 / log_2)

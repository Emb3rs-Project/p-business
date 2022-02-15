from .tests.test_lib import defineArguments, processInput
from .tests.test_bm import test_bm
from .tests.test_heat_rec import test_int_heat


# Write Here all the available tests you want to run
availableTests = {
    "business:test": test_bm,
    "business:internal_heat_recovery:test": test_int_heat,
}


def init():
    # DO NOT CHANGE FROM THIS POINT BELOW
    # UNLESS YOU KNOW WHAT YOUR DOING
    args = defineArguments(availableTests)

    processInput(args, availableTests)

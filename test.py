from .tests.test_lib import defineArguments, processInput
from .tests.test_bm import test_bm
from .tests.test_new_test import new_test

# Write Here all the available tests you want to run
availableTests = {
    "business:test": test_bm,
    "business:new_test": new_test,
}


def init():
    # DO NOT CHANGE FROM THIS POINT BELOW
    # UNLESS YOU KNOW WHAT YOUR DOING
    args = defineArguments(availableTests)

    processInput(args, availableTests)

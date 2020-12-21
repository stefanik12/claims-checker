from pv211_utils.eval import mean_average_precision

from AWIRSystem import AWIRSystem
from SillyRandomIRSystem import SillyRandomIRSystem


if __name__ == "__main__":
    mean_average_precision(SillyRandomIRSystem(), submit_result=False)
    mean_average_precision(AWIRSystem(), submit_result=False)


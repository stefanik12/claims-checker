from pv211_utils.eval import mean_average_precision

from AWIRSystem import AWIRSystem
from baseline.SillyRandomIRSystem import SillyRandomIRSystem
from baseline.tfidf import CharNGramTfIdfIRSystem


if __name__ == "__main__":
    print("Evaluating RandomIRSystem")
    mean_average_precision(SillyRandomIRSystem(), submit_result=False)
    print("Evaluating Character N-gram TF-IDF IRSystem")
    mean_average_precision(CharNGramTfIdfIRSystem(), submit_result=False)
    print("Evaluating AWIRSystem")
    mean_average_precision(AWIRSystem(), submit_result=False)


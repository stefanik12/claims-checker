# mean_average_precision(AWIRSystem(), submit_result=False)

from numpy import mean
from pv211_utils import loader
from pv211_utils.eval import average_precision

import config
from AWIRSystem import AWIRSystem

queries = loader.load_queries()
documents = loader.load_documents()
relevant = loader.load_judgements(queries, documents)

num_relevant = {}
for query, _ in relevant:
    if query not in num_relevant:
        num_relevant[query] = 0
    num_relevant[query] += 1

average_precisions = {}
results = {}

for no_heads in range(2, 20):
    config.bestn_heads = no_heads
    ir_system_instance = AWIRSystem()
    average_precisions[no_heads] = []
    for query in queries.values():
        results_i = ir_system_instance.search(query)
        # print("results length: %s" % len(results))
        # print("Top 10 results: %s" % results[:10])
        # print("Top 10 results headers: %s" % [doc.title for doc in results[:10]])
        average_precisions[no_heads].append(average_precision(query, results_i, relevant, num_relevant))
        print("Query %s precision: %s" % (query.body, average_precisions[no_heads][-1]))

    results[no_heads] = float(mean(average_precisions[no_heads]))
    print(f'Mean average precision: {results[no_heads] * 100:.3f}% ')
    print("All precisions: %s" % results)

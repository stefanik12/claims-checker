# mean_average_precision(AWIRSystem(), submit_result=False)

from numpy import mean
from pv211_utils import loader
from pv211_utils.eval import average_precision

from AWIRSystem import AWIRSystem

ir_system_instance = AWIRSystem()

queries = loader.load_queries()
documents = loader.load_documents()
relevant = loader.load_judgements(queries, documents)

num_relevant = {}
for query, _ in relevant:
    if query not in num_relevant:
        num_relevant[query] = 0
    num_relevant[query] += 1

average_precisions = []
for query in queries.values():
    results = ir_system_instance.search(query)
    print("results length: %s" % len(results))
    print("Top 10 results: %s" % results[:10])
    print("Top 10 results headers: %s" % [doc.title for doc in results[:10]])
    average_precisions.append(average_precision(query, results, relevant, num_relevant))
    print("Query %s precision: %s" % (query.body, average_precisions[-1]))

result = float(mean(average_precisions))
print(f'Mean average precision: {result * 100:.3f}% ')

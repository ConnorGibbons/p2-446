import csv
import sys
import os
import math

# Gives the number of relevent documents in the qrels file, and the list of relevant documents
def calcNumRel(qid, qrels):
    numRel = 0
    relevant = []
    for doc in qrels[qid]:
        if qrels[qid][doc] > 0:
            relevant.append(doc)
            numRel += 1
    return (numRel, relevant)

# Computes number of relevant documents found, recall at 10, precision at 10, F1@10, AP, and reciprocal rank
def basicStats(query, relevant):
    numrelFound = 0
    i = 0
    foundBefore10 = 0
    firstRelevantRank = False
    precisionVals = []
    for doc in query:
        i = i+1
        if doc['docid'] in relevant:
            if not firstRelevantRank:
                firstRelevantRank = i
            numrelFound += 1
            if i < 11:
                foundBefore10 += 1
                precisionVals.append(foundBefore10/(i))
    recallAtTen = foundBefore10/len(relevant) if len(relevant) > 0 else 0
    precisionAtTen = foundBefore10/10.0
    F1AtTen = 2 * (precisionAtTen * recallAtTen) / (precisionAtTen + recallAtTen) if ((precisionAtTen > 0) and (recallAtTen > 0)) else 0
    recipRank = 1.0/firstRelevantRank if firstRelevantRank else 0
    avgPrecision = sum(precisionVals)/len(relevant) if len(relevant) > 0 else 0
    return (numrelFound, recipRank, precisionAtTen, recallAtTen, F1AtTen, avgPrecision)

def ngdc(query, relevant, qrels, queries):
    idealDCG = 0
    relDocs = []
    for doc in qrels[query]:
        if qrels[query][doc] > 0:
            relDocs.append((doc,(qrels[query][doc])))
    relDocs = sorted(relDocs, key=lambda x: x[1], reverse=True)

    i = 1 
    for doc in relDocs:
        if(i > 20):
            break
        if(i < 2):
            idealDCG += doc[1]
        else:
            idealDCG += doc[1]/math.log2(i)
        i += 1

    queryID = query # need this to index qrels 
    query = queries[query] # for iterating over query results
    dcg = 0
    i = 1
    for doc in query:
        if(i > 20):
            break
        elif doc['docid'] in relevant:
            if(i < 2):
                dcg += qrels[queryID][doc['docid']]
            else:
                dcg += qrels[queryID][doc['docid']]/math.log2(i)
        i += 1
    print(f"DCG: {dcg}, IdealDCG: {idealDCG}")
    
    return dcg/idealDCG


def eval(trecrunFile, qrelsFile, outputFile):
    # Load qrels
    qrels = {} # Format: {qid: {docid: rel}}
    with open(qrelsFile, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            qrels.setdefault(qid, {})
            qrels[qid][docid] = int(rel)
    print(f"Loaded {len(qrels)} queries from {qrelsFile}")

    # Load queries
    queries = {} # Format: {qid: [{rank, docid, score}]}
    with open(trecrunFile, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            queries.setdefault(qid, [])
            queries[qid].append({'rank': int(rank), 'docid': docid, 'score': float(score)})
    print(f"Loaded {len(queries)} queries from {trecrunFile}")

    
    # Evaluate
    stats = {} # Format: {qid: [NDGC@20, numRel, relFound, RR, P@10, R@10, F1@10, AP]}
    for query in queries:
        numRel, relevant = calcNumRel(query, qrels)
        queryStats = basicStats(queries[query], relevant)
        discountedCumulative = ngdc(query, relevant, qrels, queries)
        stats[query] = [discountedCumulative] + [numRel] + list(queryStats)
    print(f"Evaluated {len(stats)} queries")
    print(stats)

    # Writing to output
    with open(outputFile, 'w') as f:
        for query in stats:
            f.write(f"NDCG@20  {query:{8}} {format(stats[query][0],'.4f')}\n")
            f.write(f"numRel   {query:{8}} {format(stats[query][1],'d')}\n")
            f.write(f"relFound {query:{8}} {format(stats[query][2],'d')}\n")
            f.write(f"RR       {query:{8}} {format(stats[query][3],'.4f')}\n")
            f.write(f"P@10     {query:{8}} {format(stats[query][4],'.4f')}\n")
            f.write(f"R@10     {query:{8}} {format(stats[query][5],'.4f')}\n")
            f.write(f"F1@10    {query:{8}} {format(stats[query][6],'.4f')}\n")
            f.write(f"AP       {query:{8}} {format(stats[query][7],'.4f')}\n")
        f.write(f"NDCG@20  all      {format(sum([stats[query][0] for query in stats])/len(stats),'.4f')}\n")
        f.write(f"numRel   all      {format(sum([stats[query][1] for query in stats]),'d')}\n")
        f.write(f"relFound all      {format(sum([stats[query][2] for query in stats]),'d')}\n")
        f.write(f"MRR      all      {format(sum([stats[query][3] for query in stats])/len(stats),'.4f')}\n")
        f.write(f"P@10     all      {format(sum([stats[query][4] for query in stats])/len(stats),'.4f')}\n")
        f.write(f"R@10     all      {format(sum([stats[query][5] for query in stats])/len(stats),'.4f')}\n")
        f.write(f"F1@10    all      {format(sum([stats[query][6] for query in stats])/len(stats),'.4f')}\n")
        f.write(f"MAP      all      {format(sum([stats[query][7] for query in stats])/len(stats),'.4f')}\n")


    return







if __name__ == '__main__':
    argv_len = len(sys.argv)
    runFile = sys.argv[1] if argv_len >= 2 else "msmarcosmall-bm25.trecrun"
    qrelsFile = sys.argv[2] if argv_len >= 3 else "msmarco.qrels"
    outputFile = sys.argv[3] if argv_len >= 4 else "my-msmarcosmall-bm25.eval"

    eval(runFile, qrelsFile, outputFile)
    

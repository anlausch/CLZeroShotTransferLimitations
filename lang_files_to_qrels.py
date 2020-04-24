import json

qrel_dict = {}
for q in range(13049):
    qrel_dict["q" + str(q)] = {}
    for d in range(13049):
        if q==d:
            qrel_dict["q" + str(q)]["d" + str(d)] = 1
        #else:
        #    qrel_dict["q" + str(q)]["d" + str(d)] = 0

with open("/work/anlausch/DebunkMLBERT/data/qrel_small.json", 'w') as fp:
    json.dump(qrel_dict, fp)
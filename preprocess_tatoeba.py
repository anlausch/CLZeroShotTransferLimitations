import csv
import itertools
import networkx

def load_sents(path):
    with open(path, mode='r') as infile:
        reader = csv.DictReader(infile, delimiter="\t", fieldnames=["id", "lang", "text"])
        data = []
        for line in reader:
            data.append(line)
        return data


def filter_lang(data, lang):
    return [sent for sent in data if sent["lang"] == lang]



def filter_langs(data, langs=["ara", "eus", "cmn", "eng", "fin", "heb", "hin", "ita", "jpn", "kor", "rus", "swe", "tur"]):
    """
    Arabic (ara), Basque (eus), Chinese (cmn), English (eng), Finnish (fin), Hebrew (heb), Hindi (hin), Italian (ita),
    Japanese (jpn), Korean (kor), Russian (rus), Swedish (swe), Turkish (tur)
    """
    # get all langs and num sents per lang
    sent_dict = {}
    for lang in langs:
        sent_dict[lang] = filter_lang(data, lang)

    # empty data
    output_csv(sent_dict, "./data/sentences_filtered.csv")


def find_sent_by_lang_id(sents, lang, id):
    for sent in sents:
        if sent[lang + "_id"] == id:
            return sent
    return None


def match_single_sent(sent, final_sents, key_min, ids, link_dict_st, link_dict_ts, lang):
    if sent["id"] in link_dict_ts:
        for id in link_dict_ts[sent["id"]]:
            if id in ids:
                trans_sent = find_sent_by_lang_id(final_sents, key_min, id)
                trans_sent[lang] = sent["text"]
                trans_sent[lang + "_id"] = sent["id"]
                break
    if sent["id"] in link_dict_st:
        for id in link_dict_st[sent["id"]]:
            if id in ids:
                trans_sent = find_sent_by_lang_id(final_sents, key_min, id)
                trans_sent[lang] = sent["text"]
                trans_sent[lang + "_id"] = sent["id"]
                break


def match_single_sent_indirect(sent, final_sents, key_min, ids, all_translations, lang):
    for trans_set in all_translations:
        if sent["id"] in trans_set:
            for id in trans_set:
                if id in ids:
                    trans_sent = find_sent_by_lang_id(final_sents, key_min, id)
                    trans_sent[lang] = sent["text"]
                    trans_sent[lang + "_id"] = sent["id"]
                    break


def align_langs_direct(data, link_dict_st, link_dict_ts, langs=["ara", "eus", "cmn", "eng", "fin", "heb", "hin", "ita", "jpn", "kor", "rus", "swe", "tur"]):
    """
    the smallest lang is eus;

    :param sent_dict:
    :param langs:
    :return:
    """
    size_dict = {}
    # get all langs and num sents per lang
    sent_dict = {}
    for lang in langs:
        sent_dict[lang] = filter_lang(data, lang)
        if not len(sent_dict[lang]) == 0:
            size_dict[lang] = len(sent_dict[lang])

    # get min
    key_min = min(size_dict.keys(), key=(lambda k: size_dict[k]))
    print('Minimum sents %d for lang %s' % (size_dict[key_min], key_min))

    # save all ids we need in new dict
    ids = {}
    final_sents = []
    for sent in sent_dict[key_min]:
        ids[sent["id"]] = ""
        final_sents.append({key_min: sent["text"], key_min + "_id": sent["id"]})

    # now align
    for lang in langs:
        if lang == key_min:
            continue
        else:
            #final_sents[lang] = []
            for sent in sent_dict[lang]:
                match_single_sent(sent, final_sents, key_min, ids, link_dict_ts, link_dict_st, lang)

    # filter only for sents which have all langs inside
    max_num_keys = max([len(sent.keys()) for sent in final_sents])
    final_sents = [sent for sent in final_sents if len(sent.keys()) == max_num_keys]
    return final_sents


def align_langs_indirect(data, all_translations, langs=["ara", "eus", "cmn", "eng", "fin", "heb", "hin", "ita", "jpn", "kor", "rus", "swe", "tur"]):
    """
    the smallest lang is eus;

    :param sent_dict:
    :param langs:
    :return:
    """
    size_dict = {}
    # get all langs and num sents per lang
    sent_dict = {}
    for lang in langs:
        sent_dict[lang] = filter_lang(data, lang)
        if not len(sent_dict[lang]) == 0:
            size_dict[lang] = len(sent_dict[lang])

    # get min
    key_min = min(size_dict.keys(), key=(lambda k: size_dict[k]))
    print('Minimum sents %d for lang %s' % (size_dict[key_min], key_min))

    # save all ids we need in new dict
    ids = {}
    final_sents = []
    for sent in sent_dict[key_min]:
        ids[sent["id"]] = ""
        final_sents.append({key_min: sent["text"], key_min + "_id": sent["id"]})

    # now align
    for lang in langs:
        if lang == key_min:
            continue
        else:
            counter = 0
            #final_sents[lang] = []
            for sent in sent_dict[lang]:
                if counter < 10000:
                    match_single_sent_indirect(sent, final_sents, key_min, ids, all_translations, lang)
                    if lang in sent:
                        counter += 1
                else:
                    break

    # filter only for sents which have all langs inside
    max_num_keys = max([len(sent.keys()) for sent in final_sents])
    final_sents = [sent for sent in final_sents if len(sent.keys()) == max_num_keys]
    return final_sents



def output_csv(data, path):
    with open(path, mode='w') as outfile:
        writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=["id", "lang", "text"])
        for lang, sents in data.items():
            print("Writing %s" % lang)
            for sent in sents:
                writer.writerow(sent)


def output_csv_translated(data, path):
    with open(path, mode='w') as outfile:
        writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=data[0].keys())
        writer.writeheader()
        for i,sents in enumerate(data):
            if i % 1000 == 0:
                print("Writing %d" % i)
            writer.writerow(sents)


def load_links(path):
    with open(path, mode='r') as infile:
        reader = csv.DictReader(infile, delimiter="\t", fieldnames=["source", "target"])
        # these dicts should link from every sentence to all its translations
        # this creates redundancy but is more efficient for looking up the translations
        link_dict_st = {}
        link_dict_ts = {}
        for line in reader:
            if line["source"] in link_dict_st:
                link_dict_st[line["source"]].append(line["target"])
            else:
                link_dict_st[line["source"]] = [line["target"]]
            if line["target"] in link_dict_ts:
                link_dict_ts[line["target"]].append(line["source"])
            else:
                link_dict_ts[line["target"]] = [line["source"]]
        return link_dict_st, link_dict_ts


def load_links_as_graph(path):
    with open(path, mode='r') as infile:
        reader = csv.DictReader(infile, delimiter="\t", fieldnames=["source", "target"])
        # these dicts should link from every sentence to all its translations
        # this creates redundancy but is more efficient for looking up the translations
        direct_trans = []
        for line in reader:
            direct_trans.append((line["source"], line["target"]))
        g = networkx.Graph(direct_trans)
        all_translations = []
        for subgraph in connected_component_subgraphs(g):
            sub = {}
            for node in subgraph.nodes():
                sub[str(node)] = ""
            all_translations.append(sub)
        return all_translations


def compute_pairwise_direct_alignment():
    print("Process started")
    data = load_sents("./data/sentences_filtered.csv")
    link_dict_st, link_dict_ts = load_links("./data/links.csv")
    for comb in itertools.combinations(["ara", "eus", "cmn", "eng", "fin", "heb", "hin", "ita", "jpn", "kor", "rus", "swe", "tur"],2):
        suff = "_" + comb[0] + "_" + comb[1]
        print("Working on: %s" % suff)
        final_data = align_langs_direct(data, link_dict_st, link_dict_ts, list(comb))
        output_csv_translated(final_data, "./data/sentences_aligned" + suff + ".csv")
    print(len(final_data))


def connected_component_subgraphs(G):
    for c in networkx.connected_components(G):
        yield G.subgraph(c)

def compute_pairwise_indirect_alignment():
    print("Process started")
    data = load_sents("./data/sentences_filtered.csv")
    all_translations = load_links_as_graph("./data/links.csv")
    for comb in itertools.combinations(["ara", "eus", "cmn", "eng", "fin", "heb", "hin", "ita", "jpn", "kor", "rus", "swe", "tur"],2):
        suff = "_" + comb[0] + "_" + comb[1]
        print("Working on: %s" % suff)
        final_data = align_langs_indirect(data, all_translations, list(comb))
        output_csv_translated(final_data, "./data/sentences_aligned_indirect3" + suff + ".csv")
    print(len(final_data))


def main():
    compute_pairwise_indirect_alignment()



if __name__ == "__main__":
    main()

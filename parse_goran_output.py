from itertools import combinations



def output_stats_as_csv(stats):
    """
    :param stats:
    :return:
    >>> output_stats_as_csv({'ar_cmn-Hans': 0, 'ar_en': 547134, 'ar_fi': 541712, 'ar_he': 441649, 'ar_hi': 375047, 'ar_it': 543946, 'ar_ja': 577143, 'ar_ko': 557919, 'ar_ru': 537072, 'ar_sv': 544851, 'ar_tr': 474603, 'eu_cmn-Hans': 0, 'eu_en': 0, 'eu_fi': 0, 'eu_he': 0, 'eu_hi': 0, 'eu_it': 0, 'eu_ja': 0, 'eu_ko': 0, 'eu_ru': 0, 'eu_sv': 0, 'eu_tr': 0, 'cmn-Hans_en': 0, 'cmn-Hans_fi': 0, 'cmn-Hans_he': 0, 'cmn-Hans_hi': 0, 'cmn-Hans_it': 0, 'cmn-Hans_ja': 0, 'cmn-Hans_ko': 0, 'cmn-Hans_ru': 0, 'cmn-Hans_sv': 0, 'cmn-Hans_tr': 0, 'en_fi': 2124604, 'en_hi': 634477, 'en_it': 2312206, 'en_ja': 2072288, 'en_ko': 1871621, 'en_ru': 1028908, 'en_sv': 1764517, 'en_tr': 514084, 'fi_he': 621020, 'fi_hi': 627680, 'fi_it': 2070638, 'fi_ja': 1974219, 'fi_ko': 1777219, 'fi_ru': 1020378, 'fi_sv': 1700331, 'en_he': 628051, 'ar_eu': 0, 'fi_tr': 507159, 'he_hi': 512988, 'he_it': 620520, 'he_ja': 692567, 'he_ko': 651469, 'he_ru': 611869, 'he_sv': 624141, 'he_tr': 434601, 'hi_it': 633930, 'hi_ja': 626357, 'hi_ko': 626427, 'hi_ru': 595639, 'hi_sv': 633483, 'hi_tr': 388140, 'it_ja': 2040647, 'it_ko': 1835990, 'it_ru': 1024941, 'it_sv': 1731658, 'it_tr': 505047, 'ja_ko': 1912770, 'ja_ru': 1080421, 'ja_sv': 1966896, 'ja_tr': 581052, 'ko_ru': 1025142, 'ko_sv': 1765532, 'ko_tr': 547268, 'ru_sv': 1020580, 'ru_tr': 515635, 'sv_tr': 513528})
    """
    langs = set()
    for key in stats.keys():
        langa, langb = key.split("_")
        langs.add(langa)
        langs.add(langb)
    langs = sorted(list(langs))

    columns = "\t"
    for lang in langs:
        columns += lang + "\t"
    print(columns)
    line_str = ""
    current_row = ""
    counter = 1
    for comb in combinations(langs, 2):
        if current_row != comb[0]:
            current_row = comb[0]
            if line_str != "":
                print(line_str)
                line_str = comb[0]
            else:
                line_str = comb[0]
            line_str += "\t"
            for i in range(counter):
                line_str += "\t"
            counter += 1
        key = comb[0] + "_" + comb[1]
        #line_str += comb[0] + "\t"
        if key in stats:
            line_str += str(stats[key]) + "\t"
        else:
            try:
                line_str += str(stats[comb[1] + "_" + comb[0]]) + "\t"
            except Exception as e:
                line_str += "-\t"
                continue





with open("/work/anlausch/DebunkMLBERT/data/goran_3.txt", "r") as f:
    results = {}
    for line in f.readlines():
        if "/work/anlausch/DebunkMLBERT/data/" in line:
            output_stats_as_csv(results)
            results = {}
        if len(line.split(" ")) == 5:
            parts = line.split(" ")
            lang_a = parts[0]
            lang_b = parts[1]
            score = parts[3]
            results[str(lang_a) + "_" + str(lang_b)] = score
    output_stats_as_csv(results)

import os
import itertools

def read_lines_from_file(path, filename):
    with open(os.path.join(path, filename), "r") as f:
        return list(f.readlines())

def output_lines_to_file(path, filename, lines):
    with open(os.path.join(path, filename), "w") as f:
        f.writelines(lines)

def jw300_align(path="/work/anlausch/DebunkMLBERT/data/JW300"):
    print("Aligning JW 300 files")
    for comb in itertools.combinations(["ar", "eu", "cmn-Hans", "en", "fi", "he", "hi", "it", "ja", "ko", "ru", "sv", "tr"], 2):
        # for all combinations, build the file names of the two alignment files
        filename_a = "jw300_" + comb[0] + "_" + comb[1] + "." + comb[0]
        filename_b = "jw300_" + comb[0] + "_" + comb[1] + "." + comb[1]
        lines_a = read_lines_from_file(path, filename_a)
        lines_b = read_lines_from_file(path, filename_b)
        if len(lines_a) > 0 and len(lines_b) > 0:
            lines_filtered = []
            lines = list(zip(lines_a, lines_b))
            count = 0
            for l in lines:
                if l[0] != "" and l[1] != "" and l[0] != "\n" and l[1] != "\n" and l[0] != "\r\n" and l[1] != "\r\n" and count < 13049:
                    lines_filtered.append(l)
                    count += 1
                elif count >= 13049:
                    break
            lines_a_new, lines_b_new = list(zip(*lines_filtered))
            filename_a_new = "jw300_filtered_" + comb[0] + "_" + comb[1] + "." + comb[0]
            filename_b_new = "jw300_filtered_" + comb[0] + "_" + comb[1] + "." + comb[1]
            output_lines_to_file(path + "/filtered", filename_a_new, lines_a_new)
            output_lines_to_file(path + "/filtered", filename_b_new, lines_b_new)


def wikimatrix_filter(path="/work/anlausch/DebunkMLBERT/data/Wikimatrix"):
    print("Filtering Wikimatrix files")
    for comb in [("tr","zh"), ("sv","zh"), ("ru","zh"), ("ko","zh"), ("ja","zh"), ("it","zh"), ("hi","zh"), ("he","zh"),
                 ("fi", "zh"), ("en","zh"), ("ar","zh"), ("eu","tr"), ("eu","sv"), ("eu","ru"), ("eu","ko"), ("eu","ja"),
                 ("eu", "it"), ("eu","hi"), ("eu","he"), ("eu","fi"), ("en","eu"), ("eu","zh"), ("ar","eu"),]:
        # for all combinations, build the file names of the two alignment files
        filename_a = "wikimatrix_" + comb[0] + "_" + comb[1] + ".txt." + comb[0]
        filename_b = "wikimatrix_" + comb[0] + "_" + comb[1] + ".txt." + comb[1]
        lines_a = read_lines_from_file(path, filename_a)
        lines_b = read_lines_from_file(path, filename_b)
        if len(lines_a) > 0 and len(lines_b) > 0:
            lines_filtered = []
            lines = list(zip(lines_a, lines_b))
            count = 0
            for l in lines:
                if l[0] != "" and l[1] != "" and l[0] != "\n" and l[1] != "\n" and l[0] != "\r\n" and l[1] != "\r\n" and count < 13049:
                    lines_filtered.append(l)
                    count += 1
                elif count >= 13049:
                    break
            lines_a_new, lines_b_new = list(zip(*lines_filtered))
            filename_a_new = "wikimatrix_filtered_" + comb[0] + "_" + comb[1] + "." + comb[0]
            filename_b_new = "wikimatrix_filtered_" + comb[0] + "_" + comb[1] + "." + comb[1]
            output_lines_to_file(path + "/filtered", filename_a_new, lines_a_new)
            output_lines_to_file(path + "/filtered", filename_b_new, lines_b_new)


def main():
    #jw300_align()
    wikimatrix_filter()

if __name__ == "__main__":
    main()





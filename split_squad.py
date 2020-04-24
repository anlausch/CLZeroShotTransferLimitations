import json
import os


def split_data(lang):
    data_dir = "/work/anlausch/DebunkMLBERT/finetune_data/xquad"
    train_file = ("xquad.%s.json" % lang)
    with open(
        os.path.join(data_dir, train_file), "r", encoding="utf-8"
    ) as reader:
        input_data = json.load(reader)["data"]
        print(len(input_data))

        # we take 10 articles for few shot
        train_data = input_data[:10]
        test_data = input_data[10:]

        train_train_file = ("xquad-train.%s.json" % lang)
        with open(
            os.path.join(data_dir, train_train_file), "w", encoding="utf-8"
        ) as writer:
            writer.writelines(json.dumps(train_data))

        train_test_file = ("xquad-test.%s.json" % lang)
        with open(
            os.path.join(data_dir, train_test_file), "w", encoding="utf-8"
        ) as writer:
            writer.writelines(json.dumps(test_data))

def main():
    for lang in ["en", "zh", "vi", "tr", "th", "ru", "hi", "es", "el", "de", "ar"]:
        split_data(lang)

if __name__ == "__main__":
    main()




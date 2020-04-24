#!/usr/bin/env bash
# "ar", "eu", "cmn-Hans", "en", "fi", "he", "hi", "it", "ja", "ko", "ru", "sv", "tr"

for file in WikiMatrix.ar-eu.tsv.gz WikiMatrix.eu-zh.tsv.gz WikiMatrix.en-eu.tsv.gz WikiMatrix.eu-fi.tsv.gz WikiMatrix.eu-he.tsv.gz WikiMatrix.eu-hi.tsv.gz WikiMatrix.eu-it.tsv.gz WikiMatrix.eu-ja.tsv.gz WikiMatrix.eu-ko.tsv.gz WikiMatrix.eu-ru.tsv.gz WikiMatrix.eu-sv.tsv.gz WikiMatrix.eu-tr.tsv.gz WikiMatrix.ar-zh.tsv.gz WikiMatrix.en-zh.tsv.gz WikiMatrix.fi-zh.tsv.gz WikiMatrix.he-zh.tsv.gz WikiMatrix.hi-zh.tsv.gz WikiMatrix.it-zh.tsv.gz WikiMatrix.ja-zh.tsv.gz WikiMatrix.ko-zh.tsv.gz WikiMatrix.ru-zh.tsv.gz WikiMatrix.sv-zh.tsv.gz WikiMatrix.tr-zh.tsv.gz
do
    echo $file
    IFS='-' # hyphen (-) is set as delimiter
    read -ra ADDR <<< "${file}" #
    echo ${ADDR[0]}
    echo ${ADDR[1]}

    IFS='.' # hyphen (-) is set as delimiter

    read -ra ADDR1 <<< ${ADDR[0]}
    echo ${ADDR1[1]}

    read -ra ADDR2 <<< ${ADDR[1]}
    echo ${ADDR2[0]}

	langa=${ADDR1[1]}
	langb=${ADDR2[0]}

	echo $langa
	echo $langb
	echo $file
    unset IFS;
	python ./../../extract_wikimatrix.py \
      --tsv ${file} \
      --bitext "wikimatrix_${langa}_${langb}.txt" \
      --src-lang $langa --trg-lang $langb \
      --threshold 1.04
done

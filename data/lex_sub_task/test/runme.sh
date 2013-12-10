#! /bin/bash
# carefully check lexsub_test_with_mark.txt, since there could be some strange things occur (like "Decembe. 2003", sent. 641, etc.)

cat lexsub_test_with_mark.txt | sed 's/^[a-z]\+\.[a-z]//g;s/^\([0-9]\+\)$/_\1\_./g' > lexsub_test.txt

java -cp stanford-corenlp-3.3.0.jar:stanford-corenlp-3.3.0-models.jar:xom.jar:joda-time.jar:jollyday.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit -file lexsub_test.txt 

cat lexsub_test.txt.xml | sed 's/<\/sentence>/<word>xxxxx/g' | grep '<word>' | sed 's/<\/sentence>/\n/g;s/ \+<word>//g;s/<\/word>//g' | sed -e "s///g" | awk '{if ($0 == "xxxxx") {print "";} else {printf "%s ",$1} ;}' | grep -e "_ [0-9]\+ _\|xxx[^ ]\+xxx" > temp

cat temp | awk '{if (match($0, "_ [0-9]+ _")) {print $2;} else {for (i=1;i<=NF;i++) {if (match($i, "xxx[^ ]+xxx")) print i;}}}' > lexsub_word_pos.txt

cat temp | sed 's/^_ \([0-9]\+\) _ .\+/\1/g;s/xxx\([^ ]\+\)xxx/\1/g' > lexsub_sentence.txt

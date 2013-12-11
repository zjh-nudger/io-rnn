cat original/ratings.txt | sed 's/\t/ \.\n/g;s/<b> /xxx/g;s/ <\/b>/xxx/g;s/^\([0-9]\+\)/_\1_/g' | awk '{u = NR % 18; if (u == 1 || u == 6 || u == 7) {print $0};}' > context.txt

#java -cp stanford-corenlp-3.3.0.jar:stanford-corenlp-3.3.0-models.jar:xom.jar:joda-time.jar:jollyday.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit -file context.txt

#cat context.txt.xml | sed 's/<\/sentence>/<word>xxxxx/g' | grep '<word>' | sed 's/<\/sentence>/\n/g;s/ \+<word>//g;s/<\/word>//g' | sed -e "s///g" | awk '{if ($0 == "xxxxx") {print "";} else {printf "%s ",$1} ;}' | grep -e "_ [0-9]\+ _\|xxx[^ ]\+xxx" > temp

#cat temp | awk '{if (match($0, "_ [0-9]+ _")) {print $2;} else {for (i=1;i<=NF;i++) {if (match($i, "xxx[^ ]+xxx")) print i;}}}' > word_pos.txt

#cat temp | sed 's/^_ \([0-9]\+\) _ .\+/\1/g;s/xxx\([^ ]\+\)xxx/\1/g' > sentence.txt


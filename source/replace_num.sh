FILE=train25
cat /tmp/$FILE.mst.conll | awk '{if(NF > 2) {printf "%s\t%s\t%s\t%s\t",$4,$5,$6,$7; if ($7=="0") printf "ROOT"; else printf "NOLABEL"} printf "\n"}' > /tmp/$FILE.partial
cat ../data/wsj10/data/$FILE.gold.conll | awk '{if(NF > 2) printf "%s\t%s\t%s",$1,$2,$3; printf "\n"}' > /tmp/$FILE.word.conll
paste /tmp/$FILE.word.conll /tmp/$FILE.partial | sed 's/^\t//g' > ../data/wsj10/data/$FILE.mst110iter.conll

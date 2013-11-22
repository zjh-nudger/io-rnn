cat ../trees/dev.txt | sed 's/(\([^ )]\+\) \([^ )]\+\))/\2#\1/g' | sed 's/(\([0-9]\+\) /(X#\1 /g'  > dev.txt
cat ../trees/train.txt | sed 's/(\([^ )]\+\) \([^ )]\+\))/\2#\1/g' | sed 's/(\([0-9]\+\) /(X#\1 /g'  > train.txt
cat ../trees/test.txt | sed 's/(\([^ )]\+\) \([^ )]\+\))/\2#\1/g' | sed 's/(\([0-9]\+\) /(X#\1 /g'  > test.txt

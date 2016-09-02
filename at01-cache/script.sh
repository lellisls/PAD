for ex in ex01 ex02 ex03 ex04; do
    python calcstats.py ${ex}/results.csv > results/${ex}.csv;
done;
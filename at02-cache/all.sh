for ex in ex01 ex02 ex03 ex04 ex05 ex06; do
    cd $ex
    echo $ex
    sh script.sh > results.csv
    cat results.csv
    cd ..
    python ${ex}/calcstats.py ${ex}/results.csv latex/tables/${ex}.csv latex/plots/${ex}-time.txt;
done;

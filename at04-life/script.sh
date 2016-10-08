make life life2 quiet=1

rm -rf results/life.txt;
for k in 1 2 3 4 5; do
	bin/life 1000 >> results/life.txt;
done;

for k in 1 2 3 4 5; do
	bin/life2 1000 >> results/life.txt;
done;
cat results/life.txt;

python calcstats.py results/life.txt latex/tables/ex01.csv
cat latex/tables/ex01.csv

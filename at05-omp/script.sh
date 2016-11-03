
make ex01 quiet=1
rm -rf results/ex01.txt;
for i in 1024 4096; do
	for j in 1 2 4 8 12; do
		for k in 1 2 3 4 5; do
			bin/ex01 $i 128 $j >> results/ex01.txt;
		done;
	done;
done;
cat results/ex01.txt;

make ex02 quiet=1
rm -rf results/ex02.txt;
for i in 40000; do
	for j in 1 2 4 8 12; do
		for k in 1 2 3 4 5; do
			bin/ex02 $i $j >> results/ex02.txt;
		done;
	done;
done;
cat results/ex02.txt;

make ex03 quiet=1
rm -rf results/ex03.txt;
for i in 100000000; do
	for j in 1 2 4 8 12; do
		for k in 1 2 3 4 5; do
			bin/ex03 $i $j >> results/ex03.txt;
		done;
	done;
done;
cat results/ex03.txt;

rm -rf results/ex04.txt;

make ex04 schedule="static,10" quiet=1
for j in 1 2 4 8 12; do
	for k in 1 2 3 4 5; do
		bin/ex04 $j >> results/ex04.txt;
	done;
done;

make ex04 schedule="static,5" quiet=1
for j in 1 2 4 8 12; do
	for k in 1 2 3 4 5; do
		bin/ex04 $j >> results/ex04.txt;
	done;
done;

make ex04 schedule="static,2" quiet=1
for j in 1 2 4 8 12; do
	for k in 1 2 3 4 5; do
		bin/ex04 $j >> results/ex04.txt;
	done;
done;

make ex04 schedule="dynamic" quiet=1
for j in 1 2 4 8 12; do
	for k in 1 2 3 4 5; do
		bin/ex04 $j >> results/ex04.txt;
	done;
done;

make ex04 schedule="guided" quiet=1
for j in 1 2 4 8 12; do
	for k in 1 2 3 4 5; do
		bin/ex04 $j >> results/ex04.txt;
	done;
done;

cat results/ex04.txt;

make ex05 quiet=1
rm -rf results/ex05.txt;
for i in 500; do
	for j in 1 2 4 8 12; do
		for k in 1 2 3 4 5; do
			bin/ex05 $i $j >> results/ex05.txt;
		done;
	done;
done;
cat results/ex05.txt;

# python calcstats.py results/life.txt latex/tables/ex01.csv
# cat latex/tables/ex01.csv

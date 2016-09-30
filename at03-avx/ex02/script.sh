make quiet=1

rm -rf results/ex2.txt

for test in least-squares least-squares-avx; do
  for k in 1 2 3 4 5; do
  	bin/${test} >> results/ex2.txt;
  done;
done;
cat results/ex2.txt

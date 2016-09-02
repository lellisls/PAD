for i in 1000000; do
for j in 1 2; do
for k in 1 2 3 4 5; do
  	gcc ex05.c -lpapi -lm -o bin/ex05 -O3 -DLENGTH=$i -DMODE=$j -DQUIET;
	bin/ex05;
done;
done;
done;

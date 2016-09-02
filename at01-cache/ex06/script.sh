for i in 1000000; do
for j in 1 2 3; do
for k in 1 2 3 4 5; do
  	gcc ex06.c -lpapi -lm -o bin/ex06 -O3 -DLENGTH=$i -DMODE$j -DQUIET;
	bin/ex06;
done;
done;
done;

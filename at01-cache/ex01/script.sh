for i in 128 512 1024; do
for j in 0 1 2 3 4 5; do
for k in 1 2 3 4 5; do
  gcc ex01.c -lpapi -o bin/ex01 -O3 -DMATLEN=$i -DMODE=$j -DQUIET;
	bin/ex01;
done;
done;
done;

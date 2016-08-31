for i in 128 512 1024; do
for k in 1 2 3 4 5; do
  gcc ex04.c -lpapi -lblas -o bin/ex04 -O3 -DMATLEN=$i -DQUIET;
	bin/ex04;
done;
done;

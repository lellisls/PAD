for i in 128 512 1024; do
for k in 1 2 3 4 5; do
  gcc ex03.c -lpapi -o bin/ex03 -O3 -DMATLEN=$i -DBLOCK=64 -DQUIET;
	bin/ex03;
done;
done;

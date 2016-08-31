for i in 128 512 1024; do
for j in 2 4 16 64; do
for k in 1 2 3 4 5; do
  gcc ex03.c -lpapi -o bin/ex03 -O3 -DMATLEN=$i -DBLOCK=$j -DQUIET;
	bin/ex03;
done;
done;
done;

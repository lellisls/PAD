for i in 128 512 1024; do
for j in 2 4 16 64; do
for k in 1 2 3 4 5; do
  gcc ex02.c -lpapi -o bin/ex02 -O3 -DMATLEN=$i -DBLOCK=$j -DQUIET;
	bin/ex02;
done;
done;
done;

ex_a=(exercicio_a exercicio_a_avx)
ex_b=(exercicio_b exercicio_b_mod exercicio_b_avx)
ex_c=(exercicio_c exercicio_c_mod exercicio_c_avx)
ex_d=(exercicio_d exercicio_d_avx)
ex_e=(exercicio_e exercicio_e_avx)
ex_f=(exercicio_f)
make stats exercicios quiet=1

rm -rf results/ex_a.txt;
for test in ${ex_a[@]}; do
  for k in 1 2 3 4 5; do
  	bin/${test} >> results/ex_a.txt;
  done;
done;
cat results/ex_a.txt;

rm -rf results/ex_b.txt;
for test in ${ex_b[@]}; do
  for k in 1 2 3 4 5; do
  	bin/${test} >> results/ex_b.txt;
  done;
done;
cat results/ex_b.txt;

rm -rf results/ex_c.txt;
for test in ${ex_c[@]}; do
  for k in 1 2 3 4 5; do
  	bin/${test} >> results/ex_c.txt;
  done;
done;
cat results/ex_c.txt;

rm -rf results/ex_d.txt;
for test in ${ex_d[@]}; do
  for k in 1 2 3 4 5; do
  	bin/${test} >> results/ex_d.txt;
  done;
done;
cat results/ex_d.txt;

rm -rf results/ex_e.txt;
for test in ${ex_e[@]}; do
  for k in 1 2 3 4 5; do
  	bin/${test} >> results/ex_e.txt;
  done;
done;
cat results/ex_e.txt;

rm -rf results/ex_f.txt;
for test in ${ex_f[@]}; do
  for k in 1 2 3 4 5; do
  	bin/${test} >> results/ex_f.txt;
  done;
done;
cat results/ex_f.txt;

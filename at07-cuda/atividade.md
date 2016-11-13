Lista 7 - Programação CUDA
------------

#### Ex 01
O programa serial *"ftcs.c"* disponível para download no moodle é utilizado para cálculo da **difusão de calor em um domínio unidimensional** através de diferenças finitas.
Paralelize-o com **CUDA**. Verifiquem o tempo de execução do trecho que compreende:
- Transferência de memória do Host para Device para o array A;
- Lançamento do Kernel para cálculo no device;
- Transferência de memória do Device para Host para o array B;

#### Ex 02
Utilize os utilitários **"nvprof"** e **"nvvp"** para extrair informações sobre o desempenho do programa em GPU para o exercício 1. Procure descobrir os tempos utilizados para transferência de dados entre Host e Device e entre Device e Host, alem do tempo utilizado apenas para cálculo.
Obs: Para o nvprof existe um tutorial em: http://lacad.uefs.br/wp-content/uploads/2016/07/tutorial-nvprof.pdf

#### Ex 03
Crie um programa em CUDA para **preencher na GPU um vetor com números aleatórios** do tipo **float**. Verifique como implementar a geração de números aleatótios em GPU através da função **"cuRand"** (não utilize a mesma função usada em CPU). Meça tempos de execução e compare com o tempo para geração de números aleatórios em **CPU**.

#### Ex 04
Use o programa desenvolvido no exercício 3 e implemente **outro kernel em em CUDA** que **calcule a somatoria** de valores contidos no array de números aleatórios, fazendo um **processo de redução**.

#### Ex 05
Implemente uma versão do **Jogo da Vida**, conforme já descrita em exercícios anteriores em **CUDA** (com as **mesmas condições iniciais**). Compare o desempenho com a versão em **OpenMP** para arrays de mesmo tamanho.

--------------------------
Entregue o código-fonte e um relatório que contenha comentários e análises dos resultados obtidos. Caso sua solução faça uso de estruturas de dados com variáveis __shared__ explique a razão do fato.
Faça medições de tempos separadas para transferencia de dados e de computação na GPU.

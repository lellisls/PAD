# Atividade 2 - Otimizando o desempenho de códigos para afinidade de memória

## Ex01
Implemente, em linguagem C ou Fortran, todas as 6 diferentes formas de se calcular
a multiplicação de matriz tradicional sem blocagem (no-blocking), **mudando a hierarquia
dos laços**: ijk, ikj, jik, jki, kij, kji.

Faça medições de desempenho cada uma delas, **variando-se o tamanho N do vetor** em 128x128,
512x512, 1024x1024. Crie um ou mais gráficos com as medidas de desempenho constando:

- Tempo total de processamento
- Quantidade de Cache-Misses em memória Cache L2
- MFLOPS
- CPI (Cycles per instructions).

Destaque a forma de organização do laço mais eficiente.

PAPI PRESETS: https://icl.cs.utk.edu/projects/papi/wiki/PAPI3:PAPI_presets.3

## Ex02
Repita os testes para um **algoritmo com blocagem** (blocking) com a variação mais eficiente
definindo o tamanho do bloco para 2, 4, 16 e 64, variando-se o tamanho N do vetor da mesma
maneira que o exercício 1.
- Destaque o tamanho do bloco mais eficiente para cada tamanho total da matriz.
- _Importante:_ demonstre que o código está livre de erros no acesso aos dados pra qualquer tamanho do bloco.

## Ex03
Implemente o algoritmo de Strassen para multiplicação de matrizes e refaça os experimentos.
Houve melhora em relação ao desempenho obtido nos casos anteriores?

## Ex04
Utilize uma função para multiplicação de matrizes do **ATLAS** (http://math-atlas.sourceforge.net/)
com as otimizações possíveis disponíveis na biblioteca e refaça os experimentos.
No laboratório da universidade basta compilar com a opção **-lblas** para que se habilite seu uso.
- _Dica:_ no link: http://ead.unifesp.br/posgraduacao/mod/resource/view.php?id=13533
tem um exemplo de uso da função "sgemm" que faz a multiplicação de matrizes.

## Ex05
Fusão de laços: Teste o desempenho da técnica de fusão de laços:

```Fortran
// codigo original
for i = 1 to n do
b[i] = sin(a[i]*2.0) + 1:0;
end for
for i = 1 to n do
c[i] = cos(b[i]+a[i]) + 4:0;
end for
```

```Fortran
// Depois da fusão dos lacos
for i = 1 to n do
b[i] = sin(a[i]*2.0) + 1:0;
c[i] = cos(b[i]+a[i]) + 4:0;
end for
```

- Use valores de N de forma a gerar tempos de execução da ordem de poucos segundos para o código original.
- Os arrays devem ser do tipo double.
- Verifique se houve ganho no uso de memória cache e desempenho global com o uso de fusão de laços.

## Ex06

Modifique o código com fusão de laços do exercício 5 (segundo algoritmo)
trabalhando com estruturas de dados diferentes da seguinte forma:

```c
// Array merging 1 - Forma de acesso
double abc[ ? ][3] //alocacao deve ser dinamica para matrizes
```
```c
// Array merging 2 - Forma de acesso
double abc[3][ ? ] //alocacao deve ser dinamica para matrizes
```
```c
// Array de estruturas
struct {
double a, b, c; } est_abc;
struct *est_abc; ////alocacao deve ser dinamica
```
- Use o mesmo valor de N para o tamanho do vetor utilizado no exercício 5.
- Verifique se houve ganho no uso de memória cache e desempenho global para cada nova estrutura de dados.

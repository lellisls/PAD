# Atividade 5 - Programação paralela em Sistemas de Mem. Compartilhada

**Forma de entrega:** Arquivo contendo relatório em PDF e código-fonte.
*Obs:* Não utilizar caracteres acentuados em nomes de arquivos e pastas.

---

## Ex01
Construir um programa **com vetorização e blocagem** para utilização efetiva de cache (quando possível), utilizando **programação paralela** através de OpenMP para **multiplicar duas matrizes** (arrays 2D) quadradas de NxN elementos, onde **N=10³ e 10⁴ de números do tipo float**. Variar a quantidade de threads, no mínimo, da seguinte forma: **1, 2 e 4**.

Resultados requisitados no relatório:
- Mostre um pequeno teste com valores conhecidos dispostos em um vetor de poucos elementos provando que o algoritmo desenvolvido funciona para duas ou mais threads.
- Analise e coloque em um gráfico os tempos de processamento, Speedup, Eficiência e MFlops obtidos variando-se o número de threads conforme pedido, bem como a utilização da memória Cache através do PAPI, ou seja, informe a quantidade de Cache Misses para cada simulação.

---

## Ex02

Paralelize o algoritmo de ordenação Odd-Even Sort (baseado no bubble-sort) com OpenMP:
```C
void OddEven (int a[], int n) {
   int i, j, nHalf, lastEven, lastOdd;

   nHalf = n/2;
   if (n%2==0) {lastEven=nHalf-1; lastOdd=nHalf-2;}
   else        {lastEven=nHalf-1; lastOdd=nHalf-1;}

   for (i=0; i<n-1; i++) {
       for (j=0; j<=lastOdd; j++)
          ce(&a[2*j+1], &a[2*j+2]); // odd
       for (j=0; j<=lastEven; j++)
          ce(&a[2*j], &a[2*j+1]); // even
   }
}
```

Teste com diferentes números de threads (mínimo: 1, 2 e 4). Use tamanhos de conjuntos de dados onde o tempo de processamento serial demore, se possível, até um segundo.

---

## Ex03

Construa um algoritmo que, dado um vetor preenchido com números inteiros aleatórios cujos valores variam no intervalo de 0 até 999, construa um algoritmo paralelo em OpenMP que **conte quantas ocorrências de cada número foram encontrados**.

- Utilize vetores com tamanho **N=10^8**. Faça testes variando o número de threads em **1, 2 e 4** e mostre tempo de processamento, Speedup e Eficiência.
- Ao final do programa, deve-se somar a quantidade de ocorrências para cada número encontrado e este valor deverá ser igual ao tamanho do vetor, ou seja, 10^8.

_Dica:_ Ao paralelizar, tome cuidado para que duas ou mais threads ao achar a ocorrência de um dado número ao mesmo tempo não tenha problemas de inconsistência ao computar esta ocorrência.

---

## Ex04

**Christian Goldbach (1690-1764)** foi um matemático nascido na Prussia contemporânio de Euler. Um das suas mais famosas conjecturas ainda não devidamente provadas estabelece que todo número par maior que dois é a soma de dois números primos, por exemplo, 28 = 5 + 23.

O programa disponibilizado no moodle calcula a primeira ocorrência de cada soma de dois primos encontrados para números que variam de 2 até 32000. **Paralelize-o usando OpenMP** introduzindo a cláusula "for" no primeiro laço (linha 19), usando a **cláusula Schedule** para modificar as diferentes formas de balanceamento de carga entre os processadores:
- STATIC,10
- STATIC, 5
- STATIC, 2
- DYNAMIC
- GUIDED

Compare os tempos de processamento com 1, 2 e 4 threads. Compare ainda com o uso do laço paralelo sem nenhuma cláusula que mude o escalonamento.

---

## Ex05

Paralelize em OpenMP o programa desenvolvido para executar o Jogo da Vida (Atividade 4). Mostre o desempenho através de Speedup e Eficiência para 1, 2 e 4 threads.

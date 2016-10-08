# Atividade 4 - Otimizações independentes de compilador ou maquina

O **Jogo da Vida**, criado por John H. Conway, utiliza um **autômato celular** para simular gerações sucessivas de uma sociedade de organismos vivos. É composto por um **tabuleiro bi-dimensional**, infinito em qualquer direção, de células quadradas idênticas.
Cada célula tem exatamente oito células vizinhas _(todas as células que compartilham, com a célula original, uma aresta ou um vértice)_.
Cada célula está em um de dois estados: **viva ou morta** _(correspondentes aos valores 1 ou 0)_.

Uma geração da sociedade é representada pelo conjunto dos estados das células do tabuleiro. Sociedades evoluem de uma geração para a próxima aplicando simultaneamente, a todas as células do tabuleiro, regras que estabelecem o próximo estado de cada célula. As regras são:

- Células vivas com menos de 2 vizinhas vivas morrem por abandono;
- Células vivas com mais de 3 vizinhas vivas morrem de superpopulação;
- Células mortas com exatamente 3 vizinhas vivas tornam-se vivas;
- As demais células mantêm seu estado anterior.

#### Item 1

Programe, em linguagem C, um algoritmo que implemente o Jogo da Vida sobre um tabuleiro finito, NxN, orlado por células eternamente mortas (ou seja, tendo em suas bordas células permanetmente em estado de mortas).
Admita que (0,0) identifica a célula no canto superior esquerdo do tabuleiro e que (N-1,N-1) identifica a célula no canto inferior direito. Estruture seu programa da forma abaixo:

**i)** Aloque dinâmicamente a matriz necessária (de numeros inteiros) com tamanho N*N como um array unidimensional:

```c
(int *) malloc((n*n)*sizeof(int));
```

**ii)** Identifique cada posição (i,j) do array criado da seguinte maneira:
```
 Aij --> A[n*i + j]
```

**iii)** Crie um procedimento que retorne o próximo estado de uma célula dadas as coordenadas da célula e o estado atual de todas as células no tabuleiro que implemente o seguinte algoritmo:

```c
up = val[ ( i - 1 ) * n + j ];
upright = val[ ( i - 1 ) * n + j + 1 ];
right = val[ i * n + j + 1 ];
rightdown = val[ ( i + 1 ) * n + j + 1 ];
down = val[ ( i + 1 ) * n + j ];
downleft = val[ ( i + 1 ) * n + j - 1 ];
left = val[ i * n + j - 1 ];
leftup = val[ ( i - 1 ) * n + j - 1 ];
sum = up + upright + right + rightdown + down + downleft + left + leftup;
if( sum == 3 ) {
  proxestado = 1;
}
else if( ( ( val[ n * i + j ] ) == 1 ) && ( ( sum < 2 ) || ( sum > 3 ) ) ) {
  proxestado = 0;
}
```

**iv)** Crie um segundo procedimento que, dada a geração atual, calcule a próxima geração, invocando repetidamente o procedimento descrito no item anterior;

Considere a **configuração inicial** um tabuleiro com **células vivas nas posições (1,2), (2,3), (3,1), (3,2), (3,3) e as demais células mortas.**

Para demonstrar a correção dos procedimentos, crie um programa principal que imprime o **estado inicial** do tabuleiro **e das quatro gerações sucessivas para N=10**.

Em seguida, **modifique o programa para computar 4(N-3)** gerações, imprimindo as **10 primeiras linhas e colunas** da configuração inicial do tabuleiro e as **10 últimas linhas e colunas** da geração final.

Iniba a impressão e meça o tempo de execução do trecho do programa que execute as **4(N-3) iterações**. **Escolha um valor para N** tal que o tempo de execução seja maior que 5 segundos.

---
#### Item 2
Modifique o programa acima mudando o algoritmo descrito em (iii) para:
```
int inj = i*n + j; up = val[inj - n]; upright = val[inj - n + 1]; right = val[inj + 1]; rightdown = val[inj + n + 1]; down = val[inj + n]; downleft = val[inj + n - 1]; left = val[inj - 1]; leftup = val[inj - n - 1];
```

- Mantendo o restante do código sem modificações.

- Verifique o tempo de processamento para a mesma configuração do exercício 1 e compare os dois resultados.

- Utilize o PAPI e verifique o índice CPE (Cycles per Element) para os dois casos medidos. Utilize o evento: PAPI_TOT_CYC.

- Escreva sobre suas conclusões em relação as medidas efetuadas, comparando os programas desenvolvidos nos itens 1 e 2.

- Escolha um dos códigos desenvolvidos e utilize o GPROF para verificar os tempos de execução por rotina.

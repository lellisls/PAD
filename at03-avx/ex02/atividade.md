# Atividade 3 - Otimização por vetorização
## Ex02

O método dos **mínimos quadrados** é uma técnica padrão de otimização matemática para encontrar o **melhor ajuste para um conjunto de dados** tentando **minimizar a soma dos quadrados** das diferenças entre o valor estimado e os dados observados. O algoritmo consiste em encontrar uma equação do tipo: **y = mx + b**, onde, dado um conjunto de _n_ pontos _{(x1,y1), x2,y2),...,xn,yn)}_, deve-se calcular:

```c++
SUMx = x1 + x2 + ... + xn
SUMy = y1 + y2 + ... + yn
SUMxy = x1*y1 + x2*y2 + ... + xn*yn
SUMxx = x1*x1 + x2*x2 + ... + xn*xn
//Sendo os valores de "m" e "b" calculados da seguinte forma:
slope (m) = (SUMx*SUMy - n*SUMxy) / (SUMx*SUMx - n*SUMxx)
y-intercept (b) = (SUMy - m*SUMx) / n
```
Pode-se obter um programa anexado no moodle que é uma **versão serial** do código o qual lê um arquivo contendo:
- Na primeira linha a quantidade total de elementos.
- Nas linhas seguintes o conjunto de dados para os valores de x e y.

---

- Mostre o funcionamento para um **conjunto pequeno de valores conhecidos**.
- Construa uma versão **sem vetorização**;
- Outra versão **com vetorização automática** pelo compilador (em SSE2, AVX ou AVX2, através de opções de compilação que certifiquem que está ocorrendo);
- E outra com **funções intrinsecas** em SSE2, AVX ou AVX2 (usar a melhor funcionalidade).
- Verifique o **desempenho** de todas as versões **em tempo de execução, MFLOPS e CPE** (Ciclos por elementos do vetor).
- Mostre o resultado através de uma tabela ou gráfico.

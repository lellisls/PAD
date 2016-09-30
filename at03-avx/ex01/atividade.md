# Atividade 3 - Otimização por vetorização
## Ex01

Cheque no compilador GCC (usando as opções: **-O3 -msse -ftree-vectorize -fopt-info-vec-all**  ,  procure pela frase "LOOP VECTORIZED"):

  1. Quais dos laços abaixo são **automaticamente vetorizáveis**;
  2. Para os quais **não for possível a vetorização**, indique uma forma de permitir a vetorização automática pelo compilador através de **refatorações/modificações** no algoritmo que permitam ativar as operações vetoriais em SSE2, AVX ou AVX2 (verificar configuração da máquina que permita a maior funcionalidade);
  3. Indique uma forma de vetorização por chamada a **funções intrínsecas** do C/C++ em SSE2, AVX ou AVX2 (verificar configuração da máquina que permita a melhor funcionalidade);
  4. Faça **medições de desempenho** com todas as versões desenvolvidas dos códigos (determine **dados de entrada aleatórios** e especifique quantidades de elementos cujo tempo de processamento seja da ordem de **1 segundo pelo menos**, se possível) indicando **tempo de execução, MFLOPS e CPE** (Ciclos por elementos do vetor).



  _**Observações:**_
  - Deve-se medir o tempo **apenas do trecho responsável pelo cálculo**, desprezando-se os tempos relativos a alocação e inicialização dos dados.
  - Mostre o resultado através do relatório emitido pelo compilador.
  - Caso seja **impossível** vetorizar através de mudanças no código, informe o ocorrido.

---

```c++
for (i=1; i<N; i++) {
  x[i] = y[i] + z[i];
  a[i] = x[i-1] + 1.0;
}
```
```c++
for (i=0; i<N-1; i++) {
  x[i] = y[i] + z[i];
  a[i] = x[i+1] + 1.0;
}
```
```c++
for (i=0; i<N-1; i++) {
  x[i] = a[i] + z[i];
  a[i] = x[i+1] + 1.0;
}
```
```c++
for (i=0; i<N; i++) {
  t = y[i] + z[i];
  a[i] = t + 1.0/t;
}
```
```c++
int s=0.0;
for (i=0; i<N; i++) {
  s += z[i];
}
```
```c++
for (i=1; i<N; i++) {
  a[i] = a[i-1] + b[i];
}
```

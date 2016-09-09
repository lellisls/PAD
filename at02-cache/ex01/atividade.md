Implemente, em linguagem C ou Fortran, todas as 6 diferentes formas de se calcular
a multiplicação de matriz tradicional sem blocagem (no-blocking), mudando a hierarquia
dos laços: ijk, ikj, jik, jki, kij, kji.
Faça medições de desempenho cada uma delas, variando-se o tamanho N do vetor em 128x128,
 512x512, 1024x1024. Crie um ou mais gráficos com as medidas de desempenho constando:
  Tempo total de processamento, Quantidade de Cache-Misses em memória Cache L2, MFLOPS
  e CPI (Cycles per instructions). Destaque a forma de organização do laço mais eficiente.


PAPI PRESETS: https://icl.cs.utk.edu/projects/papi/wiki/PAPI3:PAPI_presets.3

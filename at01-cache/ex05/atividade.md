Fusão de laços: Teste o desempenho da técnica de fusão de laços:
// codigo original
for i = 1 to n do
b[i] = sin(a[i]*2.0) + 1:0;
end for
for i = 1 to n do
c[i] = cos(b[i]+a[i]) + 4:0;
end for

// Depois da fusão dos lacos
for i = 1 to n do
b[i] = sin(a[i]*2.0) + 1:0;
c[i] = cos(b[i]+a[i]) + 4:0;
end for

Use valores de N de forma a gerar tempos de execução da ordem 
de poucos segundos para o código original.
Os arrays devem ser do tipo double.
Verifique se houve ganho no uso de memória cache e desempenho 
global com o uso de fusão de laços.

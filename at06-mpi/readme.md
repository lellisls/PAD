## Ex01
Relate brevemente as novidades existentes no padrão **MPI-3** (ou 3.1).
Destaque as **principais novas funcionalidades**, destacando o uso com dispositivos de **memória compartilhada**.

## Ex02
Implemente **duas versões do Jogo da Vida** em paralelo, conforme especificado na lista de exercícios anterior, em MPI em linguagem C. Utilize as mesmas configurações e premissas já definidas anteriormente na lista anterior.

- Demonstre a correção do programa.
- Uma versão deverá ser implementada com **operações tradicionais de ponto-a-ponto** (send-receive).
- Outra versão deverá ser implementada utilizando-se **comunicação unidirecional** baseada no MPI-2.
- Verifique a **correção** dos dois códigos.
- **Teste o desempenho de ambos** para os mesmos valores de N e iterações já demonstrados na lista anterior.
- Entregue tabela ou gráfico com os **tempos de execução** e os **ganhos** (“speed-up” e eficiência) obtidos. Analise os resultados **comparando as duas versões**.
- Utilize **pelo menos dois computadores**.

## Ex03
Modifique o código acima para utilizar **OpenMP**, onde se tem OpenMP dentro de uma máquina e **MPI apenas entre diferentes máquinas**. Escolha a versão MPI de melhor desempenho. Compare o resultado com o código do exercício acima.

## Ex04
Escolha um (1) exercício da **Maratona de Programação Paralela do WSCAD 2016** (com exceção do próprio Jogo da Vida) e **implemente em MPI e OpenMP**.

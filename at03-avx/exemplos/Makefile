all: least-squares somavet

soma_vet_esc:
	gcc -c soma_vet_esc.c -o obj/soma_vet_esc.o

soma_vet_sse:
	gcc -c -mavx soma_vet_sse.c -o obj/soma_vet_sse.o

least-squares: soma_vet_esc soma_vet_sse
	gcc -o bin/least-squares.x -mavx somavetmain.c obj/*
	bin/least-squares.x

somavet:
	gcc -o bin/somavet.x -mavx somavetmain.c obj/*
	bin/somavet.x

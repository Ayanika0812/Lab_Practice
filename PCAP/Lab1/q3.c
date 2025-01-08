#include<stdio.h>
#include<mpi.h>
#include<stdlib.h>

int main(int argc, char *argv[]){
    int a = atoi(argv[1]), b = atoi(argv[2]), x, rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    switch(rank){
    case 0:
    	x = a + b;
    	printf("Rank %d gives a + b = %d\n", rank, x);
    	break;
    case 1:
    	x = a - b;
    	printf("Rank %d gives a - b = %d\n", rank, x);
    	break;
    case 2:
    	x = a * b;
    	printf("Rank %d gives a * b = %d\n", rank, x);
    	break;
    case 3:
    	x = a / b;
    	printf("Rank %d gives a / b = %d\n", rank, x);
    	break;
    case 4:
    	x = a % b;
    	printf("Rank %d gives a %% b = %d\n", rank, x);
    	break;
    default:
    	printf("Invalid rank number %d :(\n", rank);
    	break;
    }
    MPI_Finalize();
    return 0;
}

// Sample Input:
// mpirun -n 5 ./q3 10 5
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
// Q1. DAXPY Loop Implementation
void daxpy(double a, double *X, double *Y, int n, int rank, int size) {
    int chunk = n / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? n : start + chunk;
    for (int i = start; i < end; i++) {
        X[i] = a * X[i] + Y[i];
    }
}
// Q2. Calculation of Pi using MPI_Bcast and MPI_Reduce
double calculate_pi(int num_steps, int rank, int size) {
    double step = 1.0 / (double)num_steps;
    double sum = 0.0;
    for (int i = rank; i < num_steps; i += size) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    double global_sum;
    MPI_Reduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) return step * global_sum;
    return 0.0;
}
// Q3. Find Primes Using MPI_Recv and MPI_Send
int is_prime(int n) {
    if (n < 2) return 0;
    for (int i = 2; i <= sqrt(n); i++)
        if (n % i == 0) return 0;
    return 1;
}
void find_primes(int max_val, int rank, int size) {
    if (rank == 0) {
        for (int num = 2; num <= max_val; num++) {
            int source, prime_flag;
            MPI_Recv(&prime_flag, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (prime_flag >= 0 && is_prime(prime_flag))
                printf("%d is prime\n", prime_flag);

            MPI_Send(&num, 1, MPI_INT, source, 0, MPI_COMM_WORLD);
        }

        for (int i = 1; i < size; i++) {
            int stop_signal = -1;
            MPI_Send(&stop_signal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        while (1) {
            int number;
            MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (number < 0) break;
            int result = is_prime(number) ? number : -number;
            MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
}
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n = 1 << 16;
    double a = 2.5;
    double *X = (double *)malloc(n * sizeof(double));
    double *Y = (double *)malloc(n * sizeof(double));
    if (rank == 0) printf("Running DAXPY Loop...\n");
    daxpy(a, X, Y, n, rank, size);
    int num_steps = 100000;
    if (rank == 0) printf("Calculating Pi...\n");
    double pi = calculate_pi(num_steps, rank, size);
    if (rank == 0) printf("Approximate value of Pi: %f\n", pi);
    if (rank == 0) printf("Finding Primes up to 100...\n");
    find_primes(100, rank, size);
    free(X);
    free(Y);
    MPI_Finalize();
    return 0;
}

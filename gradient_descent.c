#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void get_permutation_count(int alphabet_length, int string_length, int number_of_permutations, int permutations[][alphabet_length], double permutation_count[], double variables[][alphabet_length]);
double get_objective_function(int number_of_permutations, double permutation_count[]);
double get_log_barrier(int alphabet_length, int string_length, double mu, double variables[][alphabet_length]);
double get_quadratic_penalty(int alphabet_length, int string_length, double mu, double variables[][alphabet_length]);
void get_objective_gradient(int alphabet_length, int string_length, int number_of_permutations, double gradient[][alphabet_length], double variables[][alphabet_length], int permutations[][alphabet_length], double permutation_count[]);
void get_log_barrier_gradient(int alphabet_length, int string_length, double mu, double gradient[][alphabet_length], double variables[][alphabet_length]);
void get_quadratic_penalty_gradient(int alphabet_length, int string_length, double mu, double gradient[][alphabet_length], double variables[][alphabet_length]);
double get_gradient_norm(int alphabet_length, int string_length, double gradient[][alphabet_length]);
void update_variables(int alphabet_length, int string_length, double step_length, double variables_trial[][alphabet_length], double variables[][alphabet_length], double direction[][alphabet_length]);
int is_feasible_barrier(int alphabet_length, int string_length, double variables[][alphabet_length]);
int is_feasible_no_barrier(int alphabet_length, int string_length, double variables[][alphabet_length]);
int permutation_array_to_index(int result[], int start_index, int permutation_length, int alphabet_length);
void permutation_index_to_array(int permutation[], int index, int permutation_length, int alphabet_length);
void fill_permutation_matrix(int number_of_permutations, int permutation_length, int permutations[][permutation_length], int alphabet_length);
int check_permutation_matrix(int number_of_permutations, int permutation_length, int permutations[][permutation_length], int alphabet_length);
int is_permutation(int result[], int start_index, int permutation_length, int alphabet_length);
int is_superpermutation(int result[], int alphabet_length, int string_length, long check_length);
long factorial(int n);
long n_permute_k(int n, int k);

int main()
{
	const int alphabet_length = 3;
	const int string_length = 9;
	const int number_of_permutations = factorial(alphabet_length);

	int permutations[number_of_permutations][alphabet_length]; 	// List of permutations.
	double variables[string_length][alphabet_length];
	double variables_trial[string_length][alphabet_length];
	double gradient[string_length][alphabet_length];
	double permutation_count[number_of_permutations];
	int result[string_length];                         			// The string that results from rounding the variables.

	fill_permutation_matrix(number_of_permutations, alphabet_length, permutations, alphabet_length);

	if (!check_permutation_matrix(number_of_permutations, alphabet_length, permutations, alphabet_length))
	{
		printf("Error with permutation matrix.\n");
		exit(1);
	}

	// Fill variable matrix with random initial values.
	srand(time(NULL));
	for (int i = 0; i < string_length; i++)
	{
		double S = 0;
		for (int j = 0; j < alphabet_length - 1; j++)
		{
			variables[i][j] = 1.0 / alphabet_length - 0.1 + 0.2 * (double) rand() / RAND_MAX;
			S += variables[i][j];
		}
		variables[i][alphabet_length - 1] = 1.0 - S;
	}

	// Print initial values.
	if (alphabet_length <= 3)
	{
		printf("\nInitial values:\n");
		for (int i = 0; i < string_length; i++)
		{
			for (int j = 0; j < alphabet_length; j++)
			{
				printf("%f ", variables[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}

	// Iterate 100 times or until a superpermutation is found.
	int superpermutation_found = 0;
	double mu = 0.001;
	int gradient_test = 0;
	long outer_iteration_count = 0;
	while (outer_iteration_count < 100 > 1e-9 && !superpermutation_found)
	{
		outer_iteration_count++;
		long inner_iteration_count = 0;
		double gradient_norm;

		do // Iterate solver.
		{
			get_permutation_count(alphabet_length, string_length, number_of_permutations, permutations, permutation_count, variables);
			get_objective_gradient(alphabet_length, string_length, number_of_permutations, gradient, variables, permutations, permutation_count);
			get_log_barrier_gradient(alphabet_length, string_length, mu, gradient, variables);
			get_quadratic_penalty_gradient(alphabet_length, string_length, mu, gradient, variables);

			// Step.
			gradient_norm = get_gradient_norm(alphabet_length, string_length, gradient);
			double step_length = 0.01 / gradient_norm;
			double c1 = 0.001;

			double f_old = get_objective_function(number_of_permutations, permutation_count);
			f_old += get_log_barrier(alphabet_length, string_length, mu, variables);
			f_old += get_quadratic_penalty(alphabet_length, string_length, mu, variables);
			double f_new = f_old;

			// Armijo condition.
			long armijo_count = 0;
			while (f_new - f_old <= c1 * 1.0 * step_length)
			{
				armijo_count++;
				step_length *= 0.5;
				update_variables(alphabet_length, string_length, step_length, variables_trial, variables, gradient);

				if (is_feasible_barrier(alphabet_length, string_length, variables_trial))
				{
					get_permutation_count(alphabet_length, string_length, number_of_permutations, permutations, permutation_count, variables_trial);
					f_new = get_objective_function(number_of_permutations, permutation_count);
					f_new += get_log_barrier(alphabet_length, string_length, mu, variables_trial);
					f_new += get_quadratic_penalty(alphabet_length, string_length, mu, variables_trial);
				}
			}

			// Update variables.
			for (int i = 0; i < string_length; i++)
			{
				for (int j = 0; j < alphabet_length; j++)
				{
					variables[i][j] = variables_trial[i][j];
				}
			}

			// Update string.
			for (int i = 0; i < string_length; i++)
			{
				int j_max = 0;
				for (int j = 1; j < alphabet_length; j++)
				{
					if (variables[i][j] > variables[i][j_max])
					{
						j_max = j;
					}
				}

				result[i] = j_max;
			}

			// Check if superpermutation.
			if (is_superpermutation(result, alphabet_length, string_length, number_of_permutations))
			{
				superpermutation_found = 1;
			}

			inner_iteration_count++;

		} while (gradient_norm > 1.0 && inner_iteration_count < 1e6 && !superpermutation_found);

		mu *= 0.9; // Reduce barrier.
	}

	// Print results.
	if (alphabet_length <= 3)
	{
		printf("Final values:\n");
		for (int i = 0; i < string_length; i++)
		{
			for (int j = 0; j < alphabet_length; j++)
			{
				printf("%f ", variables[i][j]);
			}
			printf("\n");
		}
		printf("\nFinal permutation counts:\n");
		for (int i = 0; i < number_of_permutations; i++)
		{
			printf("%f ", permutation_count[i]);
		}
		printf("\n\n");
	}

	if (is_superpermutation(result, alphabet_length, string_length, number_of_permutations))
	{
		printf("Found a superpermutation.\n");
	}
	else
	{
		printf("Did not find a superpermutation.\n");
	}
	printf("\nResult: ");
	for (int i = 0; i < string_length; i++)
	{
		printf("%d", result[i]);
	}
	printf("\n\n");

	return 0;
}

// Count permutations as products of variables.
void get_permutation_count(int alphabet_length, int string_length, int number_of_permutations, int permutations[][alphabet_length], double permutation_count[], double variables[][alphabet_length])
{
	for (int index = 0; index < number_of_permutations; index++)
	{
		permutation_count[index] = 0;
		for (int i = 0; i < string_length - (alphabet_length - 1); i++)	// Count for every substring of length alphabet_length.
		{
			double count = 1.0; // Empty product.
			for (int j = 0; j < alphabet_length; j++)
			{
				count *= variables[i + j][permutations[index][j]];
			}
			permutation_count[index] += count;
		}
	}
}

// Objective function as sum of logarithms.
double get_objective_function(int number_of_permutations, double permutation_count[])
{
	double f = 0;
	for (int i = 0; i < number_of_permutations; i++)
	{
		f += log(permutation_count[i]);
	}

	return f;
}

double get_log_barrier(int alphabet_length, int string_length, double mu, double variables[][alphabet_length])
{
	double f = 0;
	for (int i = 0; i < string_length; i++)
	{
		for (int j = 0; j < alphabet_length; j++)
		{
			f += log(variables[i][j]);
		}
	}
	f *= mu;

	return f;
}

double get_quadratic_penalty(int alphabet_length, int string_length, double mu, double variables[][alphabet_length])
{
	double f = 0;
	for (int i = 0; i < string_length; i++)
	{
		double S = 0;
		for (int j = 0; j < alphabet_length; j++)
		{
			S += variables[i][j];
		}
		f -= (S - 1.0) * (S - 1.0);
	}
	f *= 0.5 / mu;

	return f;
}

void get_objective_gradient(int alphabet_length, int string_length, int number_of_permutations, double gradient[][alphabet_length], double variables[][alphabet_length], int permutations[][alphabet_length], double permutation_count[])
{
	for (int i = 0; i < string_length; i++)
	{
		for (int j = 0; j < alphabet_length; j++)
		{
			double dfdx = 0;

			// Compute how much the permutation counts change with variables[i][j].
			for (int index = 0; index < number_of_permutations; index++)
			{
				double dfdp = 1.0 / permutation_count[index]; // Objective function is a sum of logarithms.
				double dpdx = 0;

				// First and last index that a substring of length alphabet_length containing index i can start with.
				int first_substring_start = (i - alphabet_length + 1 > 0) ? i - alphabet_length + 1 : 0;
				int last_substring_start  = (i < string_length - alphabet_length) ? i : string_length - alphabet_length;

				for (int substring_start = first_substring_start; substring_start <= last_substring_start; substring_start++)
				{
					// Compute inner derivative.
					double derivative = 0;
					if (permutations[index][i - substring_start] == j) // The derivative wrt variables[i][j] is zero unless the (i - substring_start):th character of the substring matches the j:th character of the permutation.
					{
						derivative = 1.0;
						for (int k = 0; k < alphabet_length; k++)
						{
							if (substring_start + k != i)
							{
								derivative *= variables[substring_start + k][permutations[index][k]];
							}
						}
					}

					dpdx += derivative;
				}

				dfdx += dfdp * dpdx;
			}

			gradient[i][j] = dfdx;
		}
	}
}

void get_log_barrier_gradient(int alphabet_length, int string_length, double mu, double gradient[][alphabet_length], double variables[][alphabet_length])
{
	for (int i = 0; i < string_length; i++)
	{
		for (int j = 0; j < alphabet_length; j++)
		{
			double dfdx = mu / variables[i][j];

			gradient[i][j] += dfdx;
		}
	}
}

void get_quadratic_penalty_gradient(int alphabet_length, int string_length, double mu, double gradient[][alphabet_length], double variables[][alphabet_length])
{
	for (int i = 0; i < string_length; i++)
	{
		double S = 0;
		for (int j = 0; j < alphabet_length; j++)
		{
			S += variables[i][j];
		}
		
		double dfdx = -(S - 1.0) / mu;
		for (int j = 0; j < alphabet_length; j++)
		{
			gradient[i][j] += dfdx;
		}
	}
}

double get_gradient_norm(int alphabet_length, int string_length, double gradient[][alphabet_length])
{
	double S = 0;
	for (int i = 0; i < string_length; i++)
	{
		for (int j = 0; j < alphabet_length; j++)
		{
			S += gradient[i][j] * gradient[i][j];
		}
	}

	return sqrt(S);
}

// Update variable matrix in the given direction.
void update_variables(int alphabet_length, int string_length, double step_length, double variables_trial[][alphabet_length], double variables[][alphabet_length], double direction[][alphabet_length])
{
	for (int i = 0; i < string_length; i++)
	{
		for (int j = 0; j < alphabet_length; j++)
		{
			variables_trial[i][j] = variables[i][j] + step_length * direction[i][j];
		}
	}
}

// Return 1 if all variables are positive and 0 otherwise.
int is_feasible_barrier(int alphabet_length, int string_length, double variables[][alphabet_length])
{
	for (int i = 0; i < string_length; i++)
	{
		for (int j = 0; j < alphabet_length; j++)
		{
			if (variables[i][j] <= 0.0)
			{
				return 0;
			}
		}
	}

	return 1;
}

// Return 1 if all variables are non-negative and zero otherwise.
int is_feasible_no_barrier(int alphabet_length, int string_length, double variables[][alphabet_length])
{
	for (int i = 0; i < string_length; i++)
	{
		for (int j = 0; j < alphabet_length; j++)
		{
			if (variables[i][j] < 0.0)
			{
				return 0;
			}
		}
	}

	return 1;
}

// Return the index of the permutation result[start : start + permutation_length - 1], or -1 if it isn't a permutation.
// Not tested.
int permutation_array_to_index(int result[], int start_index, int permutation_length, int alphabet_length)
{
	if (!is_permutation(result, start_index, permutation_length, alphabet_length))
	{
		return -1;
	}

	int rang[alphabet_length];
	for (int i = 0; i < alphabet_length; i++)
	{
		rang[i] = i;
	}

	int index = 0;
	for (int i = 0; i < permutation_length; i++)
	{
		index += rang[result[start_index + i]] * n_permute_k(alphabet_length - 1 - i, permutation_length - 1 - i);
		for (int j = result[start_index + i] + 1; j < alphabet_length; j++)
		{
			rang[j]--;
		}
	}

	return index;
}

// Write the permutation of the given length and index to the array result.
void permutation_index_to_array(int result[], int index, int permutation_length, int alphabet_length)
{
	int rang[alphabet_length];
	for (int i = 0; i < alphabet_length; i++)
	{
		rang[i] = i;
	}

	for (int i = 0; i < permutation_length; i++)
	{
		int quotient = index / n_permute_k(alphabet_length - 1 - i, permutation_length - 1 - i);
		index = index % n_permute_k(alphabet_length - 1 - i, permutation_length - 1 - i);

		for (int j = 0; j < alphabet_length; j++)
		{
			if (rang[j] == quotient)
			{
				rang[j] = -1;
				for (int k = j + 1; k < alphabet_length; k++)
				{
					rang[k]--;
				}

				result[i] = j;
				break;
			}
		}
	}
}

// Fill the permutation matrix with all permutations of length permutation_length from an alphabet of alphabet_length characters.
// Not tested.
void fill_permutation_matrix(int number_of_permutations, int permutation_length, int permutations[][permutation_length], int alphabet_length)
{
	for (int index = 0; index < number_of_permutations; index++)
	{
		permutation_index_to_array(permutations[index], index, permutation_length, alphabet_length);
	}
}

// Return 1 if the permutation matrix contains all permutations of length permutation_length from an alphabet of alphabet_length characters.
// Not tested.
int check_permutation_matrix(int number_of_permutations, int permutation_length, int permutations[][permutation_length], int alphabet_length)
{
	for (int index = 0; index < number_of_permutations; index++)
	{
		if (index != permutation_array_to_index(permutations[index], 0, permutation_length, alphabet_length))
		{
			return 0;
		}
	}

	return 1;
}

// Return 1 if result[start : start + permutation_length - 1] is a permutation of permutation_length characters from an alphabet of alphabet_length characters.
// Return 0 otherwise.
int is_permutation(int result[], int start_index, int permutation_length, int alphabet_length)
{
	int check[alphabet_length];
	for (int i = 0; i < alphabet_length; i++)
	{
		check[i] = 0;
	}

	for (int i = 0; i < permutation_length; i++)
	{
		if (check[result[start_index + i]] == 1)
		{
			return 0;
		}
		check[result[start_index + i]]++;
	}

	return 1;
}

// Return 1 if result is a superpermutation of length string_length and alphabet_length symbols. Return 0 otherwise.
int is_superpermutation(int result[], int alphabet_length, int string_length, long check_length)
{
	int check[check_length];
	for (long i = 0; i < check_length; i++)
	{
		check[i] = 0;
	}
	for (long start_index = 0; start_index < string_length - (alphabet_length - 1); start_index++)
	{
		int index = permutation_array_to_index(result, start_index, alphabet_length, alphabet_length);
		if (index >= 0)
		{
			check[index] = 1;
		}
	}
	for (long i = 0; i < check_length; i++)
	{
		if (check[i] == 0)
		{
			return 0;
		}
	}

	return 1;
}

long factorial(int n)
{
	long f = 1;
	for (int i = 2; i <= n; i++)
	{
		f *= i;
	}
	return f;
}

// n! / (n - k)!
long n_permute_k(int n, int k)
{
	long p = 1;
	for (int i = n - k + 1; i <= n; i++)
	{
		p *= i;
	}
	return p;
}

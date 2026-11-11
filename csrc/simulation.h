#ifndef SIMULATION_H
#define SIMULATION_H

#include <stddef.h>

#define INITIAL_CAPACITY 16
typedef struct {
    double x;
    double y;
    double z;
} Vector3;

typedef struct {
    Vector3 *data;
    size_t count;
    size_t capacity;
} ElectronStack;

typedef struct {
    Vector3 *data;
    size_t count;
    size_t capacity;
} PointArray;

typedef struct {
    PointArray points;
    size_t electrons_processed;
    size_t secondary_electrons_created;
    int truncated;
} SimulationResult;

/* Returns 0 on success, -1 on allocation failure, -2 for invalid inputs.
   Pass a fresh result, or free its previous contents before reuse.
   Uses global srand/rand state: native callers must serialize simulations. */
int simulate_avalanche(
    double voltage,
    double start_radius,
    unsigned int seed,
    size_t max_electrons,
    SimulationResult *result
);

void simulation_result_free(SimulationResult *result);

int stack_init(ElectronStack *stack);
int stack_push(ElectronStack *stack, Vector3 item);
int stack_pop(ElectronStack *stack, Vector3 *item);
void stack_free(ElectronStack *stack);
int point_array_init(PointArray *points);
int point_array_append(PointArray *points, Vector3 position);
void point_array_free(PointArray *points);

#endif
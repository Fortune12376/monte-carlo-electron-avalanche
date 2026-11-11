#include <stdlib.h>
#include<math.h>
#include<stdint.h>

#include "physics.h"
#include "simulation.h"

#define PI 3.14159265358979323846

#define ANODE_RADIUS 8.0e-5
#define CATHODE_RADIUS 1.0e-2
#define SUPPRESSION_RADIUS 5.0e-4

#define ELECTRON_MASS 9.1093837015e-31
#define ELECTRON_CHARGE 1.602176634e-19
#define THERMAL_ENERGY 0.025

#define HELIUM_DENSITY 0.1664
#define HELIUM_MOLAR_MASS 0.0040026
#define AVOGADRO 6.022e23
#define K_MAX 2.0e-12

int stack_init(ElectronStack *stack)
{
    stack->count = 0;
    stack->capacity = INITIAL_CAPACITY;

    stack->data = malloc(stack->capacity * sizeof(Vector3));

    if (stack->data == NULL) {
        stack->capacity = 0;
        return -1;
    }

    return 0;
}

int stack_push(ElectronStack *stack, Vector3 item)
{
    if (stack->count == stack->capacity) {
        if(stack->capacity > SIZE_MAX / sizeof(Vector3) / 2) return -1;
        size_t new_capacity = stack->capacity == 0
                            ? INITIAL_CAPACITY
                            : stack->capacity * 2;

        Vector3 *temp = realloc(
            stack->data,
            new_capacity * sizeof(Vector3)
        );

        if (temp == NULL) {
            return -1;
        }

        stack->data = temp;
        stack->capacity = new_capacity;
    }

    stack->data[stack->count] = item;
    stack->count++;

    return 0;
}

int stack_pop(ElectronStack *stack, Vector3 *item)
{
    if (stack->count == 0) {
        return -1;
    }

    stack->count--;
    *item = stack->data[stack->count];

    return 0;
}

void stack_free(ElectronStack *stack)
{
    free(stack->data);

    stack->data = NULL;
    stack->count = 0;
    stack->capacity = 0;
}



int point_array_init(PointArray *points)
{
    points->count = 0;
    points->capacity = INITIAL_CAPACITY;

    points->data = malloc(points->capacity * sizeof(Vector3));

    if (points->data == NULL) {
        points->capacity = 0;
        return -1;
    }

    return 0;
}

int point_array_append(PointArray *points, Vector3 position)
{
    if (points->count == points->capacity) {
        if(points->capacity > SIZE_MAX / sizeof(Vector3) / 2) return -1;
        size_t new_capacity = points->capacity == 0
                            ? INITIAL_CAPACITY
                            : points->capacity * 2;

        Vector3 *temp = realloc(
            points->data,
            new_capacity * sizeof(Vector3)
        );

        if (temp == NULL) {
            return -1;
        }

        points->data = temp;
        points->capacity = new_capacity;
    }

    points->data[points->count] = position;
    points->count++;

    return 0;
}

void point_array_free(PointArray *points)
{
    free(points->data);

    points->data = NULL;
    points->count = 0;
    points->capacity = 0;
}

static Vector3 electron_acceleration(Vector3 position, double voltage){
    double r = sqrt(position.x * position.x
              + position.y * position.y);
    double field = electric_field(r,voltage);
    Vector3 acceleration = {(position.x / r) *(-ELECTRON_CHARGE*field/ELECTRON_MASS),(position.y / r) *(-ELECTRON_CHARGE*field/ELECTRON_MASS),0.0};
    return acceleration;
}

static double random_uniform(void)
{
    return ((double)rand() + 0.5) / ((double)RAND_MAX + 1.0);
}

static Vector3 random_velocity(double speed){
    double u1 = random_uniform();
    double u2 = random_uniform();
    double phi = 2* PI *u1;
    double costheta = 2*u2 - 1;
    Vector3 random_vel = {speed*sqrt(1-costheta*costheta)*cos(phi),speed*sqrt(1-costheta*costheta)*sin(phi),speed*costheta};
    return random_vel;
}

static int transport_electron(
    Vector3 pos,
    double voltage,
    double tau,
    double thermal_speed,
    size_t max_electrons,
    ElectronStack *electrons,
    SimulationResult *result
){
    double r_init = hypot(pos.x,pos.y);
    if(r_init < SUPPRESSION_RADIUS){
        return point_array_append(&result->points,pos);
    }
    Vector3 vel = random_velocity(thermal_speed);
    Vector3 acc = electron_acceleration(pos,voltage);
    Vector3 displacement = {0,0,0};

    while(1){
        double dt = -tau*log(random_uniform());
        displacement.x += vel.x*dt + 0.5*acc.x*dt*dt;
        displacement.y += vel.y*dt + 0.5*acc.y*dt*dt;
        displacement.z += vel.z*dt;
        vel.x += acc.x*dt;
        vel.y += acc.y*dt;
        Vector3 candidate = {pos.x+displacement.x,pos.y+displacement.y,pos.z+displacement.z};
        double r = hypot(candidate.x,candidate.y);
        if(r <= ANODE_RADIUS){
            if(r > 0.0){
                double scale = ANODE_RADIUS/r;
                candidate.x *= scale;
                candidate.y *= scale;
            }
            else{
                candidate.x = ANODE_RADIUS;
                candidate.y = 0.0;
            }
            return point_array_append(&result->points,candidate);
        }
        if(r >= CATHODE_RADIUS) return 0;

        double speed_sq = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
        double E_k = ELECTRON_MASS*speed_sq/(2*ELECTRON_CHARGE);
        double el_cross = elastic_cross_section(E_k)*1e-4;
        double in_cross = inelastic_cross_section(E_k)*1e-4;
        double P_in = in_cross/(el_cross+in_cross);
        int is_inelastic = in_cross > 0.0 && random_uniform() < P_in;
        if(!is_inelastic){
            double e_speed = sqrt(speed_sq);
            double P_acc = e_speed*el_cross/K_MAX;
            /* Null proposals keep accumulating the same free flight. */
            if(random_uniform() > P_acc) continue;
            pos = candidate;
            displacement = (Vector3){0,0,0};
            acc = electron_acceleration(pos,voltage);
            if(random_uniform() < 0.001){
                if(point_array_append(&result->points,pos) != 0) return -1;
            }
            vel = random_velocity(e_speed);
            /* Flip the entire direction into the inward hemisphere. */
            if(vel.x*pos.x + vel.y*pos.y > 0.0){
                vel.x = -vel.x;
                vel.y = -vel.y;
                vel.z = -vel.z;
            }
            continue;
        }

        /* Inelastic proposals are accepted directly, as in simulation.py. */
        pos = candidate;
        displacement = (Vector3){0,0,0};
        vel = random_velocity(thermal_speed);
        acc = electron_acceleration(pos,voltage);
        if(electrons->count < max_electrons-result->electrons_processed){
            if(stack_push(electrons,pos) != 0) return -1;
            result->secondary_electrons_created++;
        }
        else result->truncated = 1;
        if(point_array_append(&result->points,pos) != 0) return -1;
    }
}

int simulate_avalanche(double voltage, double start_radius, unsigned int seed,
                       size_t max_electrons, SimulationResult *result)
{
    if(result == NULL) return -2;
    *result = (SimulationResult){0};
    if(!isfinite(voltage) || voltage <= 0.0 || !isfinite(start_radius) ||
       start_radius <= SUPPRESSION_RADIUS || start_radius >= CATHODE_RADIUS ||
       max_electrons < 1) return -2;

    ElectronStack electrons = {0};
    if(stack_init(&electrons) != 0) return -1;
    if(point_array_init(&result->points) != 0){
        stack_free(&electrons);
        return -1;
    }
    Vector3 pos = {start_radius,0,0};
    if(stack_push(&electrons,pos) != 0) goto allocation_error;
    srand(seed);
    double number_density = HELIUM_DENSITY*AVOGADRO/HELIUM_MOLAR_MASS;
    double tau = 1.0/(number_density*K_MAX);
    double thermal_speed = sqrt(2*THERMAL_ENERGY*ELECTRON_CHARGE/ELECTRON_MASS);
    while(electrons.count > 0){
        if(result->electrons_processed >= max_electrons){
            result->truncated = 1;
            break;
        }
        stack_pop(&electrons,&pos);
        result->electrons_processed++;
        if(transport_electron(pos,voltage,tau,thermal_speed,max_electrons,
                              &electrons,result) != 0) goto allocation_error;
    }
    stack_free(&electrons);
    return 0;

allocation_error:
    stack_free(&electrons);
    simulation_result_free(result);
    return -1;
}

void simulation_result_free(SimulationResult *result)
{
    if(result == NULL) return;
    point_array_free(&result->points);
    result->electrons_processed = 0;
    result->secondary_electrons_created = 0;
    result->truncated = 0;
}

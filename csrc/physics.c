#include<math.h>
#include "physics.h"

double elastic_cross_section(double energy){
    /*Energy is all in electron volts*/
    if(energy <= 2.0){
        return 7.0e-16;
    }
    if(energy <= 10.0){
        return energy * (-0.27231e-16) + 6.455375e-16;
    }
    if(energy <= 20.0){
        return energy * (-0.20977e-16) + 6.8801e-16;
    }
    if(energy <= 100.0){
        return energy * (-0.02981e-16) + 3.2809e-16;
    }
    return 3.0e-17;
}

double inelastic_cross_section(double energy){
    /*Energy all in eV*/
    double slope = 0.047745e-17;
    double intercept = -1.174527e-17;
    double threshold_ev = 24.6;
    if(energy < threshold_ev){
        return 0.0;
    }
    if(energy > 100.0){
        return 3.6e-17;
    }
    return energy*slope + intercept;
}

double electric_field(double r, double voltage){
    double ra = 0.00008;
    double rc = 0.01;

    return voltage / (r * log(rc / ra));
}
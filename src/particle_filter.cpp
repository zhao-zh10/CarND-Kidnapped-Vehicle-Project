/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * Completed on: June 29, 2019
 * Completed by: zhao-zh10
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  this->num_particles = 1000;  // TODO: Set the number of particles

  this->particles.clear();
  this->weights.clear();
  this->particles.resize(this->num_particles);
  this->weights.resize(this->num_particles);

  std::default_random_engine gen;
  //Create normal(gaussian) distributions for x, y, theta
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for(int i = 0; i < this->num_particles; ++ i){
    double sample_x, sample_y, sample_theta, sample_weight;
    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);
    sample_weight = 1.0;

    Particle part;
    part.id = i;
    part.x = sample_x;
    part.y = sample_y;
    part.theta = sample_theta;
    part.weight = sample_weight;

    this->particles[i] = part;
    this->weights[i] = sample_weight;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  const double eps = 1.0e-8;
  // Generate random Gaussian noise to each particle
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0.0, std_pos[0]);
  std::normal_distribution<double> dist_y(0.0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0.0, std_pos[2]);
  // Run over each particle 
  for(std::vector<Particle>::iterator iter = particles.begin(); iter != particles.end(); iter ++){
    double x_final, x_previous, y_final, y_previous, theta_final, theta_previous;
    x_previous = (*iter).x;
    y_previous = (*iter).y;
    theta_previous = (*iter).theta;
    // Decide which motion model to use(CV model or CTRV model?)
    if(fabs(yaw_rate) < eps){
      x_final = x_previous + velocity * delta_t * cos(theta_previous);
      y_final = y_previous + velocity * delta_t * sin(theta_previous);
      theta_final = theta_previous;
    }
    else{
      x_final = x_previous + velocity / yaw_rate * (sin(theta_previous + yaw_rate * delta_t)-sin(theta_previous));
      y_final = y_previous + velocity / yaw_rate * (cos(theta_previous)-cos(theta_previous + yaw_rate * delta_t));
      theta_final = theta_previous + yaw_rate * delta_t;
    }
    // Add random Gaussian noise to each particle
    x_final += dist_x(gen);
    y_final += dist_y(gen);
    theta_final += dist_theta(gen);
    // Update particle position state after prediction step
    (*iter).x = x_final;
    (*iter).y = y_final;
    (*iter).theta = theta_final;
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  // Run over each observed measurement
  for(std::vector<LandmarkObs>::iterator obs = observations.begin(); obs != observations.end(); obs ++){
    double closest_distance = std::numeric_limits<double>::max();
    int closest_id = -1;
    // Run over each predicted measurement for the chosen observed measurement and find the closest landmark
    for(std::vector<LandmarkObs>::iterator pred = predicted.begin(); pred != predicted.end(); pred ++){
      double distance = dist((*obs).x, (*obs).y, (*pred).x, (*pred).y);
      if(distance < closest_distance){
        closest_distance = distance;
        closest_id = (*pred).id;
      }
    }
    // assign the observed measurement to this particular landmark
    (*obs).id = closest_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  // Run over each particle
  int particle_Index = 0;
  for(std::vector<Particle>::iterator part = particles.begin(); part != particles.end(); part ++){
    double xp = (*part).x;
    double yp = (*part).y;
    double theta = (*part).theta;
    double xc = 0.0, yc = 0.0, xm = 0.0, ym = 0.0;
    std::vector<LandmarkObs> observations_transformed;
    // Perform Homogenous Transformation
    // Run over each landmark observation measurements
    for(auto obs = observations.begin(); obs != observations.end(); obs ++){
      xc = (*obs).x;
      yc = (*obs).y;
      xm = xp + cos(theta) * xc - sin(theta) * yc;
      ym = yp + sin(theta) * xc + cos(theta) * yc;
      LandmarkObs trans;
      trans.x = xm;
      trans.y = ym;
      observations_transformed.push_back(trans);
    }
    // Run over each map landmark position to find those in the range of sensor for this chosen particle
    std::vector<LandmarkObs> predicted;
    for(auto single_lm = map_landmarks.landmark_list.begin(); single_lm != map_landmarks.landmark_list.end(); single_lm ++){
      double x_landmark = (*single_lm).x_f;
      double y_landmark = (*single_lm).y_f;
      double id_landmark = (*single_lm).id_i;
      double distance = dist(x_landmark, y_landmark, xp, yp);
      if(distance <= sensor_range){
        LandmarkObs pred;
        pred.id = id_landmark;
        pred.x = x_landmark;
        pred.y = y_landmark;
        predicted.push_back(pred);
      }
    }
    // Perform data association step
    dataAssociation(predicted, observations_transformed);
    // Calculate particle weights using multivariate gaussian probability
    // Run over each landmark associated observation measurements
    double final_weight = 1.0;
    for(std::vector<LandmarkObs>::iterator obs = observations_transformed.begin(); obs != observations_transformed.end(); obs ++){
      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];
      double x_obs = (*obs).x;
      double y_obs = (*obs).y;
      LandmarkObs pos_associated;
      for(std::vector<LandmarkObs>::iterator pred = predicted.begin(); pred != predicted.end(); pred ++){
        if((*pred).id == (*obs).id){
          pos_associated = *pred;
        }
      }
      double mu_x = pos_associated.x;
      double mu_y = pos_associated.y;
      final_weight *= multiv_prob(sig_x, sig_y, x_obs, y_obs, mu_x, mu_y);
    }
    (*part).weight = final_weight;
    this->weights[particle_Index ++] = final_weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // Run over the weight vector to find the largest weight.
  double largest_weight = 0.0;
  for(std::vector<double>::iterator iter = weights.begin(); iter != weights.end(); iter ++){
    if(*iter > largest_weight){
      largest_weight = *iter;
    }    
  }
  std::default_random_engine random(time(NULL));
  std::uniform_int_distribution<int> dist_initial_index(0, num_particles);
  std::uniform_real_distribution<double> dist_weight(0.0, 2*largest_weight);
  double beta = 0.0;
  int index = dist_initial_index(random);
  std::vector<Particle> particles_resample;
  std::vector<double> weights_resample;
  // Run over each particle
  for(int i = 0; i < num_particles; ++ i){
    beta += dist_weight(random);
    while(weights[index] < beta){
      beta -= weights[index];
      index = (index + 1)% num_particles;
    }
    Particle part_resample;
    part_resample.id = i;
    part_resample.x = particles[index].x;
    part_resample.y = particles[index].y;
    part_resample.theta = particles[index].theta;
    part_resample.weight = particles[index].weight;
    particles_resample.push_back(part_resample);
    weights_resample.push_back(part_resample.weight);
  }
  particles = particles_resample;
  weights = weights_resample;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}


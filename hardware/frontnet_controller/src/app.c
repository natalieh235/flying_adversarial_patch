#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"

#include "commander.h"
#include "peer_localization.h"

#include "FreeRTOS.h"
#include "task.h"

#include "debug.h"

#include "log.h"
#include "param.h"
#include "math3d.h"

#define DEBUG_MODULE "APP"


const float safety_distance = 0.5;  // min distance between UAV and target in m
const float angle = 0.0;   // angle between UAV and target
const float tau = 1/10.0; // update rate

bool start_main = false;
uint8_t peer_id = 5;
float max_velo_v = 0.5f;

// Normalize radians to be in range [-pi,pi]
// See https://stackoverflow.com/questions/4633177/c-how-to-wrap-a-float-to-the-interval-pi-pi
static inline float normalize_radians(float radians)
{
	// Copy the sign of the value in radians to the value of pi.
	float signed_pi = copysignf(M_PI_F, radians);
	// Set the value of difference to the appropriate signed value between pi and -pi.
	radians = fmodf(radians + signed_pi, 2 * M_PI_F) - signed_pi;
	return radians;
}

// modulo operation that uses the floored definition (as in Python), rather than
// the truncated definition used for the % operator in C
// See https://en.wikipedia.org/wiki/Modulo_operation
static inline float fmodf_floored(float x, float n)
{
	return x - floorf(x / n) * n;
}

// compute shortest signed angle between two given angles (in range [-pi, pi])
// See https://stackoverflow.com/questions/1878907/how-can-i-find-the-difference-between-two-angles
static inline float shortest_signed_angle_radians(float start, float goal)
{
	float diff = goal - start;
	float signed_diff = fmodf_floored(diff + M_PI_F, 2 * M_PI_F) - M_PI_F;
	return signed_diff;
}

// helper for calculating the angle between two vectors
float calcAngleBetweenVectors(struct vec vec1, struct vec vec2)
{
  struct vec vec1_norm = vnormalize(vec1);
  struct vec vec2_norm = vnormalize(vec2);
  float dot = vdot(vec1_norm, vec2_norm);
  return acosf(dot);
}

struct vec calcHeadingVec(float radius, float angle)
{
  float x = (float) 0.0f + radius * cosf(angle);
  float y = (float) 0.0f + radius * sinf(angle);
  return mkvec(x, y, 0.0f);
}

void appMain()
{

  //TODO: add function to lift off and set start_main to true

  static setpoint_t setpoint;

  // struct vec max_velocity = mkvec(max_velo_v, max_velo_v, max_velo_v);
  
  struct vec p_D = vzero();  // current position
  struct vec p_D_prime = vzero(); // next position

  struct vec e_D = vzero(); // unit vector of UAV
  struct vec e_H_delta = vzero(); // unit vector of target multiplied by safety distance
  
  struct vec v_D = vzero(); // velocity of UAV
  
  float estYawRad;  // estimate of current yaw angle
  float theta_prime_D; // angle between current UAV heading and desired heading
  float theta_D; // angle between current UAV heading and last heading ???
  float omega_prime_D; // attitude rate



  //vTaskDelay(M2T(3000));

  logVarId_t idStabilizerYaw = logGetVarId("stabilizer", "yaw");

  logVarId_t idXEstimate = logGetVarId("stateEstimate", "x");
  logVarId_t idYEstimate = logGetVarId("stateEstimate", "y");
  logVarId_t idZEstimate = logGetVarId("stateEstimate", "z");



  while(1) {
    vTaskDelay(M2T(100));

    if (peerLocalizationIsIDActive(peer_id) && start_main) {

      p_D.x = logGetFloat(idXEstimate);
      p_D.y = logGetFloat(idYEstimate);
      p_D.z = logGetFloat(idZEstimate);

      struct vec max_velocity = mkvec(max_velo_v, max_velo_v, max_velo_v);

      // Query position of our target
      peerLocalizationOtherPosition_t* target = peerLocalizationGetPositionByID(peer_id);
      struct vec target_pos = mkvec(target->pos.x, target->pos.y, target->pos.z);

      // z is kept at same height as target
      setpoint.mode.z = modeAbs;
      setpoint.position.z = target_pos.z;

      // position is handled given velocity and attitude rate
      setpoint.mode.x = modeVelocity;
      setpoint.mode.y = modeVelocity;
      // setpoint.position.x = p_D_prime.x;
      // setpoint.position.y = p_D_prime.y;
      // setpoint.position.z = p_D_prime.z;


      // eq 6
      e_H_delta = calcHeadingVec(1.0f*safety_distance, angle); 
      // radius is 1 * safety_distance
      // angle is set to 0°, since the target should always be in the center of the camera image

      p_D_prime = vadd(target_pos, e_H_delta);

      // eq 7
      struct vec v_H = vzero(); // target velocity, set to 0 since we don't have this information yet
      v_D = vdiv(vsub(p_D_prime, p_D), tau);
      v_D = vadd(v_D, v_H);
      v_D = vclamp(v_D, vneg(max_velocity), max_velocity);

      setpoint.mode.x = modeAbs;
      setpoint.mode.y = modeAbs;
      setpoint.position.x = p_D_prime.x;
      setpoint.position.y = p_D_prime.y;

      // eq 8
      estYawRad = radians(logGetFloat(idStabilizerYaw));    // get the current yaw in degrees
      e_D = calcHeadingVec(1.0f, estYawRad);           // radius is 1, angle is current yaw 
      
      theta_prime_D = calcAngleBetweenVectors(e_D, vsub(target_pos, p_D));
      
      theta_D = estYawRad;//calcAngleBetweenVectors(e_D, vsub(target_pos, p_D_prime));
      
      // eq 9
      //omega_prime_D = (theta_prime_D - theta_D) / tau;     // <-- incorrect, the difference does not always provide the shortest angle between the two thetas
      omega_prime_D = shortest_signed_angle_radians(theta_prime_D, theta_D) / tau;
      omega_prime_D = clamp(omega_prime_D, -0.8f, 0.8f);

      setpoint.mode.yaw = modeVelocity;
      setpoint.attitudeRate.yaw = degrees(omega_prime_D);

      setpoint.velocity_body = false;  // world frame

      // send a new setpoint to the UAV
      commanderSetSetpoint(&setpoint, 3);

      // update current position
      p_D = p_D_prime;


      }//end if
    else{

      p_D.x = logGetFloat(idXEstimate);
      p_D.y = logGetFloat(idYEstimate);
      p_D.z = logGetFloat(idZEstimate);

    }//end else
  }//end while
}//end main


/**
 * Parameters to set the estimator and controller type
 * for the stabilizer module, or to do an emergency stop
 */
PARAM_GROUP_START(frontnet)
/**
 * @brief Estimator type Any(0), complementary(1), kalman(2) (Default: 0)
 */
PARAM_ADD_CORE(PARAM_UINT8, start, &start_main)
PARAM_ADD_CORE(PARAM_UINT8, cfid, &peer_id)
PARAM_ADD_CORE(PARAM_FLOAT, maxvelo, &max_velo_v)


PARAM_GROUP_STOP(frontnet)

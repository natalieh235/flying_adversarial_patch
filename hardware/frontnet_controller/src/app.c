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

#include "uart1.h"
#include "debug.h"

#define DEBUG_MODULE "APP"


const float distance = 0.5;  // distance between UAV and target in m
const float angle = 0.0;   // angle between UAV and target
const float tau = 1/100.0; // update rate

bool start_main = false;
uint8_t peer_id = 5;
float max_velo_v = 1.0f;

struct vec target_pos;
float target_yaw;

const uint32_t baudrate_esp32 = 115200;

// helper for calculating the angle between two vectors
float calcAngleBetweenVectors(struct vec vec1, struct vec vec2)
{
  struct vec vec1_norm = vnormalize(vec1);
  struct vec vec2_norm = vnormalize(vec2);
  float dot = vdot(vec1_norm, vec2_norm);
  return acosf(dot);
}

// calculate the vector of the heading, angle in radians
struct vec calcHeadingVec(float radius, float angle)
{
  float x = (float) 0.0f + radius * cosf(angle);
  float y = (float) 0.0f + radius * sinf(angle);
  return mkvec(x, y, 0.0f);
}

void appMain()
{

  static setpoint_t setpoint;
  
  struct vec p_D = vzero();  // current position
  struct vec p_D_prime = vzero(); // next position

  struct vec e_D = vzero(); // heading vector of UAV
  struct vec e_H_delta = vzero(); // heading vector of target multiplied by safety distance
  
  // struct vec v_D = vzero(); // velocity of UAV
  
  float estYawRad;  // estimate of current yaw angle
  //float theta_prime_D; // angle between current UAV heading and desired heading
  //float theta_D; // angle between current UAV heading and last heading
  float omega_prime_D; // attitude rate


  logVarId_t idStabilizerYaw = logGetVarId("stabilizer", "yaw");

  logVarId_t idXEstimate = logGetVarId("stateEstimate", "x");
  logVarId_t idYEstimate = logGetVarId("stateEstimate", "y");
  logVarId_t idZEstimate = logGetVarId("stateEstimate", "z");

  // init UART
  DEBUG_PRINT("[DEBUG] Init UART...\n");
  uart1Init(baudrate_esp32);
  DEBUG_PRINT("[DEBUG] done!\n");

  while(1) {
    vTaskDelay(M2T(10));

    uint8_t dummy = 0x00;
    uint8_t uart_buffer[16];

    // if (start_main) {
    
    // DEBUG_PRINT("[DEBUG] Waiting for UART message...\n");
    while (dummy != 0xBC)
    {
      uart1GetDataWithDefaultTimeout((uint8_t*)&dummy);
    }
    // DEBUG_PRINT("[DEBUG] Got package from !\n");
    
    uart1GetDataWithDefaultTimeout((uint8_t*)&dummy);
    uint8_t length = dummy;
    if (length == sizeof(uart_buffer))
    {
      for (uint8_t i = 0; i < length+1; i++)
      {
        uart1GetDataWithDefaultTimeout((uint8_t*)&uart_buffer[i]);
      }

      // DEBUG_PRINT("[DEBUG] Read package from UART!:\n");
      for (uint8_t i = 0; i < length; i++)
      {
        // DEBUG_PRINT("%02X", uart_buffer[i]);
      }
      // DEBUG_PRINT("\n");
    }    

      p_D.x = logGetFloat(idXEstimate);
      p_D.y = logGetFloat(idYEstimate);
      p_D.z = logGetFloat(idZEstimate);

      double x_d = 0;
      double y_d = 0;
      double z_d = 0;
      double phi_d = 0;

      uint32_t x = (uart_buffer[3] << 24) | (uart_buffer[2] << 16) | (uart_buffer[1] << 8) | uart_buffer[0];

      if (uart_buffer[3]>=0xF0)
      {
         x_d = (-(double)~x) * 2.46902e-05 + 1.02329e+00;
      }
      else
      {
         x_d = (double)x * 2.46902e-05 + 1.02329e+00;
      }
      


      uint32_t y = (uart_buffer[7] << 24) | (uart_buffer[6] << 16) | (uart_buffer[5] << 8) | uart_buffer[4];
      if (uart_buffer[7]>=0xF0)
      {
         y_d = (-(double)~y) * 2.46902e-05 + 7.05523e-04;
      }
      else
      {
         y_d = (double)y * 2.46902e-05 + 7.05523e-04;
      }


      uint32_t z = (uart_buffer[11] << 24) | (uart_buffer[10] << 16) | (uart_buffer[9] << 8) | uart_buffer[8];
      if (uart_buffer[11]>=0xF0)
      {
         z_d = (-(double)~z) * 2.46902e-05 + 2.68245e-01;
      }
      else
      {
         z_d = (double)z * 2.46902e-05 + 2.68245e-01;
      }


      uint32_t phi = (uart_buffer[15] << 24) | (uart_buffer[14] << 16) | (uart_buffer[13] << 8) | uart_buffer[12];
      if (uart_buffer[15]>=0xF0)
      {
         phi_d = (-(double)~phi) * 2.46902e-05 + 5.60173e-04;
      }
      else
      {
         phi_d = (double)phi * 2.46902e-05 + 5.60173e-04;
      }

      // DEBUG_PRINT("[DEBUG] Conversion worked?: %ld, %ld, %ld, %ld\n", x, y, z, phi);
      DEBUG_PRINT("[DEBUG] Received coordinates: %f, %f, %f, %f\n", x_d, y_d, z_d, phi_d);

      // DEBUG_PRINT("[DEBUG] Conversion to uint32 worked? %lu\n", x);  
      // velocity control
      // Query position of our target
      if (peerLocalizationIsIDActive(peer_id))
      {
      // peerLocalizationOtherPosition_t* target = peerLocalizationGetPositionByID(peer_id);

      // target_pos = mkvec(target->pos.x, target->pos.y, target->pos.z);
      // target_yaw = target->yaw;
      target_pos = mkvec((float)x_d, (float)y_d, (float)z_d);
      target_yaw = (float)phi_d;

      // z is kept at same height as target
      setpoint.mode.z = modeAbs;
      setpoint.position.z = target_pos.z;

      // position is handled given position and attitude rate

      setpoint.mode.x = modeAbs;
      setpoint.mode.y = modeAbs;


      // eq 6
      e_H_delta = calcHeadingVec(1.0f*distance, target_yaw);//angle); 
      // radius is 1 * distance
      // angle is set to the current yaw angle (rad) of the target

      p_D_prime = vadd(target_pos, e_H_delta);
      setpoint.position.x = p_D_prime.x;
      setpoint.position.y = p_D_prime.y;
      setpoint.position.z = p_D_prime.z;

      // velocity control -> produces oscillation of the CF in x and y direction
      // // eq 7
      // struct vec v_H = vzero(); // target velocity, set to 0 since we don't have this information yet
      // v_D = vdiv(vsub(p_D_prime, p_D), tau);
      // v_D = vadd(v_D, v_H);
      // //v_D = vclamp(v_D, vneg(max_velocity), max_velocity); // -> debugging, doesn't preserve the direction of the vector
      // v_D = vclampnorm(v_D, max_velo_v);

      // setpoint.velocity.x = v_D.x;
      // setpoint.velocity.y = v_D.y;


      // heading control
      // eq 8
      estYawRad = radians(logGetFloat(idStabilizerYaw));    // get the current yaw in degrees
      e_D = calcHeadingVec(1.0f, estYawRad);           // radius is 1, angle is current yaw 

      struct vec target_vector = vsub(target_pos, p_D);
      float angle_target = atan2f(target_vector.y, target_vector.x);

      //theta_prime_D = calcAngleBetweenVectors(e_D, vsub(target_pos, p_D));
      
      //theta_D = calcAngleBetweenVectors(e_D, vsub(target_pos, p_D_prime));  //estYawRad;
      
      // eq 9
      // omega_prime_D = (theta_prime_D - theta_D) / tau;     // <-- incorrect, the difference does not always provide the shortest angle between the two thetas
      omega_prime_D = shortest_signed_angle_radians(estYawRad, angle_target) / tau;
      omega_prime_D = clamp(omega_prime_D, -0.8f, 0.8f);

      setpoint.mode.yaw = modeVelocity;
      setpoint.attitudeRate.yaw = degrees(omega_prime_D);

      setpoint.velocity_body = false;  // world frame

      // send a new setpoint to the UAV
      commanderSetSetpoint(&setpoint, 3);

      // update current position
      //p_D = p_D_prime;
      //}//end debug if

      }//end if
  }//end while
}//end main


/**
 * Parameters to set the start flag, peer id and max velocity
 * for the frontnet-like controller.
 */
PARAM_GROUP_START(frontnet)
/**
 * @brief Estimator type Any(0), complementary(1), kalman(2) (Default: 0)
 */
PARAM_ADD_CORE(PARAM_UINT8, start, &start_main)
PARAM_ADD_CORE(PARAM_UINT8, cfid, &peer_id)
PARAM_ADD_CORE(PARAM_FLOAT, maxvelo, &max_velo_v)


PARAM_GROUP_STOP(frontnet)

// add new log group for local variables
LOG_GROUP_START(frontnet)
LOG_ADD(LOG_FLOAT, targetx, &target_pos.x)
LOG_ADD(LOG_FLOAT, targety, &target_pos.y)
LOG_ADD(LOG_FLOAT, targetz, &target_pos.z)
LOG_ADD(LOG_FLOAT, targetyaw, &target_yaw)
LOG_GROUP_STOP(frontnet)

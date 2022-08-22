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

const uint8_t peer_id = 255;
const float safety_distance = 0.5;
const float rad_to_deg = 57.295779578552;

// helper for calculating the angle between two vectors
float calcAngleBetweenVectors(struct vec vec1, struct vec vec2)
{
  float dot = vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
  float det = vec1.x * vec2.x - vec1.y * vec2.y - vec1.z * vec2.z;
  float angle = atan2(det, dot);
  return angle;
}

void appMain()
{
  static setpoint_t setpoint;

  struct vec unit_vec = mkvec(1, 1, 1);

  vTaskDelay(M2T(3000));

  while(1) {
    vTaskDelay(M2T(10));

    if (peerLocalizationIsIDActive(peer_id)) {

      // Query position of our target
      peerLocalizationOtherPosition_t* target = peerLocalizationGetPositionByID(peer_id);
      struct vec target_pos = mkvec(target->pos.x, target->pos.y, target->pos.z);
      struct vec current_pos = mkvec(setpoint.position.x, setpoint.position.y, setpoint.position.z);
      
      
      // TODO: add actual setpoint logic here

      // z is the same as the target position, since we want to maintain the same hight
      setpoint.mode.z = modeAbs;
      setpoint.position.z = target_pos.z;

      // x is the same as the target position, since we want stay right in front of the target
      setpoint.mode.x = modeAbs;
      setpoint.position.x = target_pos.x;

      // y is set to target position + safety distance
      setpoint.mode.y = modeAbs;
      setpoint.position.y = target_pos.y + safety_distance;




      // compute target orientation theta_D
      struct vec diff = mkvec(target_pos.x - current_pos.x, target_pos.y - current_pos.y, target_pos.z - current_pos.z);
      float theta_D = calcAngleBetweenVectors(unit_vec, diff);
  
      struct vec diff_curr = mkvec(target_pos.x - setpoint.position.x, target_pos.y - setpoint.position.y, target_pos.z - setpoint.position.z);
      float theta_D_curr = calcAngleBetweenVectors(unit_vec, diff_curr);

      // setpoint.attitude = theta_D_curr * rad_to_deg;
      // setpoint.attitudeRate = 





      // setpoint.mode.yaw = modeVelocity;
      // setpoint->attitudeRate.yaw = yawrate;


      // setpoint->mode.x = modeVelocity;
      // setpoint->mode.y = modeVelocity;
      // setpoint->velocity.x = vx;
      // setpoint->velocity.y = vy;

      setpoint.velocity_body = true;

      // send a new setpoint to the UAV
      commanderSetSetpoint(&setpoint, 3);


    }
  }
}

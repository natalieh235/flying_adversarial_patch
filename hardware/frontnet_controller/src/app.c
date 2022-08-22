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

void appMain()
{
  static setpoint_t setpoint;

  vTaskDelay(M2T(3000));

  while(1) {
    vTaskDelay(M2T(10));

    if (peerLocalizationIsIDActive(peer_id)) {

      // Query position of our target
      peerLocalizationOtherPosition_t* target = peerLocalizationGetPositionByID(peer_id);
      struct vec target_pos = mkvec(target->pos.x, target->pos.y, target->pos.z);

      // TODO: add actual setpoint logic here

      setpoint.mode.z = modeAbs;
      setpoint.position.z = target_pos.z;
      // setpoint->mode.yaw = modeVelocity;
      // setpoint->attitudeRate.yaw = yawrate;


      // setpoint->mode.x = modeVelocity;
      // setpoint->mode.y = modeVelocity;
      // setpoint->velocity.x = vx;
      // setpoint->velocity.y = vy;

      // setpoint->velocity_body = true;

      // send a new setpoint to the UAV
      commanderSetSetpoint(&setpoint, 3);


    }
  }
}

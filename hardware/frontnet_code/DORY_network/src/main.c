/*
 * test_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 */



#include "pmsis.h"
#include "bsp/bsp.h"
#include "bsp/camera.h"
#include "bsp/camera/himax.h"


// #include "gaplib/ImgIO.h"

// #include "img_proc.h"

#include "network.h"

PI_L2 unsigned char *buff;

// PI_L2 unsigned char *croppedImg;
PI_L2 unsigned char *outputNN;
char* L2_memory_buffer;

static struct pi_device camera;
static volatile int done;

#define WIDTH 162
#define HEIGHT 162

#define BUFF_SIZE (WIDTH*HEIGHT)

static struct pi_hyperflash_conf flash_conf;
static struct pi_hyper_conf ram_conf;
static struct pi_device ram;
static int activations_input;


// filesystem management functions
void open_filesystem_and_ram(struct pi_device *flash, struct pi_device *fs)
{
  struct pi_readfs_conf conf;
  struct pi_hyperflash_conf flash_conf;

  /* Init & open flash. */
  pi_hyperflash_conf_init(&flash_conf);
  pi_open_from_conf(flash, &flash_conf);
  if (pi_flash_open(flash))
  {
      // cpxPrintToConsole(LOG_TO_CRTP, "Error flash open !\n");
      printf("Error flash open !\n");
      pmsis_exit(-1);
  }

  /* Open filesystem on flash. */
  pi_readfs_conf_init(&conf);
  conf.fs.flash = flash;
  pi_open_from_conf(fs, &conf);
  if (pi_fs_mount(fs))
  {
      // cpxPrintToConsole(LOG_TO_CRTP, "Error FS mounting !\n");
      printf("Error FS mounting !\n");
      pmsis_exit(-2);
  }
  pi_task_t task = {0};
  pi_task_block(&task);
  pi_hyperram_conf_init(&ram_conf);
  pi_open_from_conf(&ram, &ram_conf);
  pi_ram_open(&ram);
}

void cropImage(unsigned char *imgBuff, unsigned char *L2_input_buffer)
{
  uint32_t offset = 35;
  unsigned char *curr_adr = L2_input_buffer;

  for (uint8_t idx_h = 0; idx_h <97; idx_h++)
  {
    memcpy(curr_adr, imgBuff + (offset * sizeof(char)), 161*sizeof(char));
    offset += 163;
    curr_adr += 161 * sizeof(char);
  }
}

static void handle_transfer_end(void *arg)
{
    done = 1;
}

static int open_camera(struct pi_device *device)
{
    printf("Opening Himax camera\n");
    struct pi_himax_conf cam_conf;
    pi_himax_conf_init(&cam_conf);

    cam_conf.format = PI_CAMERA_QQVGA;

    pi_open_from_conf(device, &cam_conf);
    if (pi_camera_open(device))
        return -1;

    // Rotate camera orientation
    pi_camera_control(&camera, PI_CAMERA_CMD_START, 0);
    uint8_t set_value = 3;
    uint8_t reg_value;

    pi_camera_reg_set(&camera, IMG_ORIENTATION, &set_value);
    pi_time_wait_us(1000000);
    pi_camera_reg_get(&camera, IMG_ORIENTATION, &reg_value);
    if (set_value!=reg_value)
    {
        printf("Failed to rotate camera image\n");
        return -1;
    }
    pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);

    pi_camera_control(device, PI_CAMERA_CMD_AEG_INIT, 0);

    return 0;
}


int prediction_task(void)
{
    printf("Entering main controller\n");

    printf("Testing normal camera capture\n");

    // Open the Himax camera
    if (open_camera(&camera))
    {
        printf("Failed to open camera\n");
        pmsis_exit(-1);
    }

    /*
    Opening of Filesystem and Ram
    */
    struct pi_device fs;
    struct pi_device flash;
    open_filesystem_and_ram(&flash, &fs);
    pi_ram_alloc(&ram, &activations_input, (uint32_t) 500000);

    //allocate network
    network_alloc(fs, ram);


    // Reserve buffer space for image
    buff = pmsis_l2_malloc(WIDTH*HEIGHT*sizeof(unsigned char));
    if (buff == NULL){ return -1;}

    L2_memory_buffer = pi_l2_malloc((uint32_t) 380000);


    // croppedImg = (unsigned char *)pmsis_l2_malloc(96*160*sizeof(unsigned char));
    // if (croppedImg == NULL){ return -1;}

    outputNN = (unsigned char *)pmsis_l2_malloc(4*sizeof(unsigned int));
    if (outputNN == NULL){ return -1;}

    // buff_demosaick = pmsis_l2_malloc(BUFF_SIZE);
    // if (buff_demosaick == NULL){ return -1;}
    printf("Initialized buffers\n");

    while(1)
    {
        // Start the camera
        pi_camera_control(&camera, PI_CAMERA_CMD_START, 0);
        pi_camera_capture(&camera, buff, BUFF_SIZE);

        // Stop the camera and immediately close it
        pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);
        pi_camera_close(&camera);


        cropImage(buff, L2_memory_buffer);
        // memcpy(L2_memory_buffer, croppedImg, sizeof(croppedImg));
        printf("\nL2 Buffer alloc initial\t@ 0x%08x:\t%s\n", (unsigned int)L2_memory_buffer, L2_memory_buffer?"Ok":"Failed");

        int begin_end = 1;
        network_run(L2_memory_buffer, 380000, outputNN, begin_end, ram);
        printf("Network Output: ");
        // cpxPrintToConsole(LOG_TO_CRTP, "Network Output: ");
        for(int i = 0; i < 16; i+=4)
        {
        printf("%d ", *(int32_t *)(outputNN + i));
        // cpxPrintToConsole(LOG_TO_CRTP, "%d", *(int32_t *)(L2_output + i));
        }
        printf("\n");
    }

    pi_ram_free(&ram, activations_input, 500000);
    network_free(ram);  
    // pi_l2_free(, (uint32_t) 380000);

    pmsis_exit(0);

}

int main(void)
{
    printf("\n\t*** PMSIS Camera with LCD Example ***\n\n");
    return pmsis_kickoff((void *) prediction_task);
}

// #include "network.h"

// #define FLASH_BUFF_SIZE 128
// #define VERBOSE 1

// #define IMG_ORIENTATION 0x0101
// #define CAM_WIDTH 162
// #define CAM_HEIGHT 162

// static struct pi_hyperflash_conf flash_conf;
// static struct pi_hyper_conf ram_conf;
// static struct pi_device ram;
// static int activations_input;
// static uint8_t flashBuffer[FLASH_BUFF_SIZE];

// char* L2_output;

// static pi_task_t led_task;
// static int led_val = 0;
// static struct pi_device gpio_device;


// static int open_pi_camera_himax(struct pi_device *device)
// {
//   struct pi_himax_conf cam_conf;

//   pi_himax_conf_init(&cam_conf);

//   cam_conf.format = PI_CAMERA_QQVGA;

//   pi_open_from_conf(device, &cam_conf);
//   if (pi_camera_open(device))
//     return -1;

//   // rotate image
//   pi_camera_control(&camera, PI_CAMERA_CMD_START, 0);
//   uint8_t set_value = 3;
//   uint8_t reg_value;
//   pi_camera_reg_set(&camera, IMG_ORIENTATION, &set_value);
//   pi_time_wait_us(1000000);
//   pi_camera_reg_get(&camera, IMG_ORIENTATION, &reg_value);
//   if (set_value != reg_value)
//   {
//     cpxPrintToConsole(LOG_TO_CRTP, "Failed to rotate camera image\n");
//     return -1;
//   }
//   pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);

//   pi_camera_control(device, PI_CAMERA_CMD_AEG_INIT, 0);

//   return 0;
// }

// void cropImage(unsigned char *imgBuff, unsigned char *croppedImg)
// {
//   uint32_t offset = 35;
//   unsigned char *curr_adr = croppedImg;

//   for (uint8_t idx_h = 0; idx_h <97; idx_h++)
//   {
//     memcpy(curr_adr, imgBuff + (offset * sizeof(char)), 161*sizeof(char));
//     offset += 163;
//     curr_adr += 161 * sizeof(char);
//   }
// }

// // filesystem management functions
// // void open_filesystem_and_ram(struct pi_device *flash, struct pi_device *fs)
// // {
// //   struct pi_readfs_conf conf;
// //   struct pi_hyperflash_conf flash_conf;

// //   /* Init & open flash. */
// //   pi_hyperflash_conf_init(&flash_conf);
// //   pi_open_from_conf(flash, &flash_conf);
// //   if (pi_flash_open(flash))
// //   {
// //       // cpxPrintToConsole(LOG_TO_CRTP, "Error flash open !\n");
// //       printf("Error flash open !\n");
// //       pmsis_exit(-1);
// //   }

// //   /* Open filesystem on flash. */
// //   pi_readfs_conf_init(&conf);
// //   conf.fs.flash = flash;
// //   pi_open_from_conf(fs, &conf);
// //   if (pi_fs_mount(fs))
// //   {
// //       // cpxPrintToConsole(LOG_TO_CRTP, "Error FS mounting !\n");
// //       printf("Error FS mounting !\n");
// //       pmsis_exit(-2);
// //   }
// //   pi_task_t task = {0};
// //   pi_task_block(&task);
// //   pi_hyperram_conf_init(&ram_conf);
// //   pi_open_from_conf(&ram, &ram_conf);
// //   pi_ram_open(&ram);
// // }

// int main () {
//   char* L2_memory_buffer;
//   char* L2_input;


//   uint32_t resolution = CAM_WIDTH * CAM_HEIGHT;
//   uint32_t captureSize = resolution * sizeof(unsigned char);
//   // imgBuff = (unsigned char *)pmsis_l2_malloc(captureSize);
//   // croppedImg = (unsigned char *)pmsis_l2_malloc(96*160*sizeof(unsigned char));
//   imgBuff = (unsigned char *)pi_l2_malloc(captureSize);
//   croppedImg = (unsigned char *)pi_l2_malloc(96*160*sizeof(unsigned char));

//   if (imgBuff == NULL)
//   {
//     printf("[RAM] failed to allocate memory for camera image buffer\n");
//     return;
//   }

//   if (croppedImg == NULL)
//   {
//     printf("[RAM] failed to allocate memory for input to network\n");
//     return;
//   }

//   if (open_pi_camera_himax(&camera))
//   {
//     printf("[CAMERA] Failed to open camera\n");
//     return;
//   }

//   pi_camera_control(&camera, PI_CAMERA_CMD_START, 0);
//   pi_camera_capture_async(&camera, imgBuff, resolution, pi_task_callback(&task1, capture_done_cb, NULL));
  
//   pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);

//   cropImage(imgBuff, croppedImg);
//   imgSize = 96 * 160;

//   network_run(croppedImg, 380000, L2_output, begin_end, ram);


//   // PMU_set_voltage(1000, 0);
//   // pi_time_wait_us(10000);
//   // pi_freq_set(PI_FREQ_DOMAIN_FC, 100000000);
//   // pi_time_wait_us(10000);
//   // pi_freq_set(PI_FREQ_DOMAIN_CL, 100000000);
//   // pi_time_wait_us(10000);


//   // send_gap8("debug: start main, own print");
// /*
//     Opening of Filesystem and Ram
// */
//   // struct pi_device fs;
//   // struct pi_device flash;
//   // open_filesystem_and_ram(&flash, &fs);
//   // pi_ram_alloc(&ram, &activations_input, (uint32_t) 500000);
//   // pi_fs_file_t *file;
//   // file = pi_fs_open(&fs, "inputs.hex", 0);
//   // if (file == NULL)
//   // {
//   //   // // cpxPrintToConsole(LOG_TO_CRTP, "file open failed\n");
//   //   printf("file open failed\n");
//   //   return -1;
//   // }
// /*
//     Copying the input file from flash to ram
// */
//   // int flashBuffSize = FLASH_BUFF_SIZE * sizeof(char);
//   // int rdDone = 0;
//   // // loop on chunk in file
//   // while(rdDone < (15360 / sizeof(char)))
//   // {
//   //   // read from HyperFlash
//   //   int size = pi_fs_read(file, flashBuffer, flashBuffSize);
//   //   // write to HyperRam
//   //   pi_ram_write(&ram, activations_input+rdDone, flashBuffer, (uint32_t) size);
//   //   rdDone += size / sizeof(char);
//   // }
// /*
//     Allocating space for input and copying it
// */
// //   L2_memory_buffer = pi_l2_malloc((uint32_t) 380000);
// //   int begin_end = 1;
// //   L2_input = L2_memory_buffer + (1 - begin_end) * (380000 - rdDone);
// //   L2_output = L2_memory_buffer;
// //   // cpxPrintToConsole(LOG_TO_CRTP, "\nL2 Buffer alloc initial\t@ 0x%08x:\t%s\n", (unsigned int)L2_memory_buffer, L2_memory_buffer?"Ok":"Failed");
// // #ifdef VERBOSE
// //   printf("\nL2 Buffer alloc initial\t@ 0x%08x:\t%s\n", (unsigned int)L2_memory_buffer, L2_memory_buffer?"Ok":"Failed");
// // #endif
// /*
//     Allocation
// */
//     // pi_ram_read(&ram, activations_input, L2_input, 15360);
//     // network_alloc(fs, ram);  
// /*
//     Running of the network
// */
//   	// network_run(L2_memory_buffer, 380000, L2_output, begin_end, ram);
//     // cpxPrintToConsole(LOG_TO_CRTP, "Hello world, frontnet is working!");
// // #ifdef VERBOSE
//     printf("Network Output: ");
//     // cpxPrintToConsole(LOG_TO_CRTP, "Network Output: ");
//     for(int i = 0; i < 16; i+=4)
//     {
//       printf("%d ", *(int32_t *)(L2_output + i));
//       // cpxPrintToConsole(LOG_TO_CRTP, "%d", *(int32_t *)(L2_output + i));
//     }
//     printf("\n");
//     // cpxPrintToConsole(LOG_TO_CRTP, "Network Output: ");
// // #endif

// // /*
// //     Send network output via UART to GAP8
// // */
// //     send_gap8(L2_output);
// /*
//     Deallocation
// */
//     pi_ram_free(&ram, activations_input, 500000);
//     network_free(ram);  
//     pi_l2_free(L2_memory_buffer, (uint32_t) 380000);
// }

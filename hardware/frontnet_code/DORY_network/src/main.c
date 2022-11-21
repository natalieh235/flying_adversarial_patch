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
#include "network.h"

#define FLASH_BUFF_SIZE 128
#define VERBOSE 1

static struct pi_hyperflash_conf flash_conf;
static struct pi_hyper_conf ram_conf;
static struct pi_device ram;
static int activations_input;
static uint8_t flashBuffer[FLASH_BUFF_SIZE];

char* L2_output;

static pi_task_t led_task;
static int led_val = 0;
static struct pi_device gpio_device;

static void led_handle(void *arg)
{
  pi_gpio_pin_write(&gpio_device, 2, led_val);
  led_val ^= 1;
  pi_task_push_delayed_us(pi_task_callback(&led_task, led_handle, NULL), 500000);
}


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

static void send_gap8(char* L2_output)
{
  printf("Entering main controller...\n");

  // set configurations in uart
  struct pi_uart_conf conf;
  struct pi_device device;
  pi_uart_conf_init(&conf);
  conf.baudrate_bps =115200;

  // // configure LED
  // pi_gpio_pin_configure(&gpio_device, 2, PI_GPIO_OUTPUT);

  // Open uart
  pi_open_from_conf(&device, &conf);
  printf("[UART] Open\n");
  if (pi_uart_open(&device))
  {
    printf("[UART] open failed !\n");
    pmsis_exit(-1);
  }

  pi_uart_open(&device);

  // toggle LED when sending information
  pi_gpio_pin_write(&gpio_device, 2, led_val);
  led_val ^= 1;
  pi_task_push_delayed_us(pi_task_callback(&led_task, led_handle, NULL), 500000);  
  // Write the value to uart
  pi_uart_write(&device, &L2_output, 1);

  pi_gpio_pin_write(&gpio_device, 2, led_val);
  led_val ^= 1;
  pi_task_push_delayed_us(pi_task_callback(&led_task, led_handle, NULL), 500000);

}

int main () {
  char* L2_memory_buffer;
  char* L2_input;
  // PMU_set_voltage(1000, 0);
  // pi_time_wait_us(10000);
  // pi_freq_set(PI_FREQ_DOMAIN_FC, 100000000);
  // pi_time_wait_us(10000);
  // pi_freq_set(PI_FREQ_DOMAIN_CL, 100000000);
  // pi_time_wait_us(10000);


  printf("debug: start main");
  // send_gap8("debug: start main, own print");
/*
    Opening of Filesystem and Ram
*/
  struct pi_device fs;
  struct pi_device flash;
  open_filesystem_and_ram(&flash, &fs);
  pi_ram_alloc(&ram, &activations_input, (uint32_t) 500000);
  pi_fs_file_t *file;
  file = pi_fs_open(&fs, "inputs.hex", 0);
  if (file == NULL)
  {
    // // cpxPrintToConsole(LOG_TO_CRTP, "file open failed\n");
    printf("file open failed\n");
    return -1;
  }
/*
    Copying the input file from flash to ram
*/
  int flashBuffSize = FLASH_BUFF_SIZE * sizeof(char);
  int rdDone = 0;
  // loop on chunk in file
  while(rdDone < (15360 / sizeof(char)))
  {
    // read from HyperFlash
    int size = pi_fs_read(file, flashBuffer, flashBuffSize);
    // write to HyperRam
    pi_ram_write(&ram, activations_input+rdDone, flashBuffer, (uint32_t) size);
    rdDone += size / sizeof(char);
  }
/*
    Allocating space for input and copying it
*/
  L2_memory_buffer = pi_l2_malloc((uint32_t) 380000);
  int begin_end = 1;
  L2_input = L2_memory_buffer + (1 - begin_end) * (380000 - rdDone);
  L2_output = L2_memory_buffer;
  // cpxPrintToConsole(LOG_TO_CRTP, "\nL2 Buffer alloc initial\t@ 0x%08x:\t%s\n", (unsigned int)L2_memory_buffer, L2_memory_buffer?"Ok":"Failed");
#ifdef VERBOSE
  printf("\nL2 Buffer alloc initial\t@ 0x%08x:\t%s\n", (unsigned int)L2_memory_buffer, L2_memory_buffer?"Ok":"Failed");
#endif
/*
    Allocation
*/
    pi_ram_read(&ram, activations_input, L2_input, 15360);
    network_alloc(fs, ram);  
/*
    Running of the network
*/
  	network_run(L2_memory_buffer, 380000, L2_output, begin_end, ram);
    // cpxPrintToConsole(LOG_TO_CRTP, "Hello world, frontnet is working!");
// #ifdef VERBOSE
    printf("Network Output: ");
    // cpxPrintToConsole(LOG_TO_CRTP, "Network Output: ");
    for(int i = 0; i < 16; i+=4)
    {
      printf("%d ", *(int32_t *)(L2_output + i));
      // cpxPrintToConsole(LOG_TO_CRTP, "%d", *(int32_t *)(L2_output + i));
    }
    printf("\n");
    // cpxPrintToConsole(LOG_TO_CRTP, "Network Output: ");
// #endif

// /*
//     Send network output via UART to GAP8
// */
//     send_gap8(L2_output);
/*
    Deallocation
*/
    pi_ram_free(&ram, activations_input, 500000);
    network_free(ram);  
    pi_l2_free(L2_memory_buffer, (uint32_t) 380000);
}

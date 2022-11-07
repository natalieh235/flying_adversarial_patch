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
      printf("Error flash open !\n");
      pmsis_exit(-1);
  }

  /* Open filesystem on flash. */
  pi_readfs_conf_init(&conf);
  conf.fs.flash = flash;
  pi_open_from_conf(fs, &conf);
  if (pi_fs_mount(fs))
  {
      printf("Error FS mounting !\n");
      pmsis_exit(-2);
  }
  pi_task_t task = {0};
  pi_task_block(&task);
  pi_hyperram_conf_init(&ram_conf);
  pi_open_from_conf(&ram, &ram_conf);
  pi_ram_open(&ram);
}

int main () {
  char* L2_memory_buffer;
  char* L2_input;
  PMU_set_voltage(1000, 0);
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_FC, 100000000);
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_CL, 100000000);
  pi_time_wait_us(10000);

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
#ifdef VERBOSE
    printf("Network Output: ");
    for(int i = 0; i < 16; i+=4)
    {
      printf("%d ", *(int32_t *)(L2_output + i));
    }
    printf("\n");
#endif
/*
    Deallocation
*/
    pi_ram_free(&ram, activations_input, 500000);
    network_free(ram);  
    pi_l2_free(L2_memory_buffer, (uint32_t) 380000);
}

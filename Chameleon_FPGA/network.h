/*
header for conv module
Conv 2 layer
Separable conv layer 
*/

// #ifndef CONV_H
// #define CONV_H

#define paratype float 
#define IN 
#define OUT 
#define graphinW 128
#define graphinH 128
#define graphinC 3
#define Conv2Kernel 3
#define Conv2Channel 32
#define Conv2outH 64
#define Conv2Stride 2

#define Conv1_depKernel 3
#define Conv1_depChannel 32 
#define Conv1_depoutH 64
#define Conv1_depStride 1
#define Conv1_poiKernel 1
#define Conv1_poiChannel 64
#define Conv1_poioutH 64
#define Conv1_poiStride 1

#define Conv2_depKernel 3
#define Conv2_depChannel 64
#define Conv2_depoutH 64
#define Conv2_depStride 2
#define Conv2_poiKernel 1
#define Conv2_poiChannel 128
#define Conv2_poioutH 32
#define Conv2_poiStride 1

#define Conv3_depKernel 3
#define Conv3_depChannel 128
#define Conv3_depoutH 32
#define Conv3_depStride 1
#define Conv3_poiKernel 1
#define Conv3_poiChannel 128
#define Conv3_poioutH 32
#define Conv3_poiStride 1

#define Conv4_depKernel 3
#define Conv4_depChannel 128
#define Conv4_depoutH 16
#define Conv4_depStride 2
#define Conv4_poiKernel 1
#define Conv4_poiChannel 256
#define Conv4_poioutH 16
#define Conv4_poiStride 1

#define Conv5_depKernel 3
#define Conv5_depChannel 256
#define Conv5_depoutH 16
#define Conv5_depStride 1
#define Conv5_poiKernel 1
#define Conv5_poiChannel 256
#define Conv5_poioutH 16
#define Conv5_poiStride 1

#define Conv6_depKernel 3
#define Conv6_depChannel 256
#define Conv6_depoutH 16
#define Conv6_depStride 2
#define Conv6_poiKernel 1
#define Conv6_poiChannel 512
#define Conv6_poioutH 8
#define Conv6_poiStride 1

#define Conv7_depKernel 3
#define Conv7_depChannel 512
#define Conv7_depoutH 8
#define Conv7_depStride 1
#define Conv7_poiKernel 1
#define Conv7_poiChannel 512
#define Conv7_poioutH 8
#define Conv7_poiStride 1

#define Conv8_depKernel 3
#define Conv8_depChannel 512
#define Conv8_depoutH 8
#define Conv8_depStride 1
#define Conv8_poiKernel 1
#define Conv8_poiChannel 512
#define Conv8_poioutH 8
#define Conv8_poiStride 1

#define Conv9_depKernel 3
#define Conv9_depChannel 512
#define Conv9_depoutH 8
#define Conv9_depStride 1
#define Conv9_poiKernel 1
#define Conv9_poiChannel 512
#define Conv9_poioutH 8
#define Conv9_poiStride 1

#define Conv10_depKernel 3
#define Conv10_depChannel 512
#define Conv10_depoutH 8
#define Conv10_depStride 1
#define Conv10_poiKernel 1
#define Conv10_poiChannel 512
#define Conv10_poioutH 8
#define Conv10_poiStride 1

#define Conv11_depKernel 3
#define Conv11_depChannel 512
#define Conv11_depoutH 8
#define Conv11_depStride 1
#define Conv11_poiKernel 1
#define Conv11_poiChannel 512
#define Conv11_poioutH 8
#define Conv11_poiStride 1

#define Conv12_depKernel 3
#define Conv12_depChannel 512
#define Conv12_depoutH 8
#define Conv12_depStride 2
#define Conv12_poiKernel 1
#define Conv12_poiChannel 1024
#define Conv12_poioutH 4
#define Conv12_poiStride 1

#define Conv13_depKernel 3
#define Conv13_depChannel 1024
#define Conv13_depoutH 4
#define Conv13_depStride 1
#define Conv13_poiKernel 1
#define Conv13_poiChannel 1024
#define Conv13_poioutH 4
#define Conv13_poiStride 1

void read_module(paratype graphin[128][128][3], paratype grahout[50]);
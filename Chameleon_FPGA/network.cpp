#include "network.h"
#include <cstddef>  // this is to fix the bug for <'::max_align_t’ has not been declared>
#include <iostream>
#include <fstream>
#include <math.h>
// Includes
#include <stdint.h>
#include <hls_stream.h>

//implement the convolution
const char* Conv2weightdir = "./wbin/Conv2d_0_weights.bin";

const char* dep1weightdir = "./wbin/Conv2d_1_depthwise_depthwise_weights.bin";
const char* poi1weightdir = "./wbin/Conv2d_1_pointwise_weights.bin";

const char* dep2weightdir = "./wbin/Conv2d_2_depthwise_depthwise_weights.bin";
const char* poi2weightdir = "./wbin/Conv2d_2_pointwise_weights.bin";

const char* dep3weightdir = "./wbin/Conv2d_3_depthwise_depthwise_weights.bin";
const char* poi3weightdir = "./wbin/Conv2d_3_pointwise_weights.bin";

const char* dep4weightdir = "./wbin/Conv2d_4_depthwise_depthwise_weights.bin";
const char* poi4weightdir = "./wbin/Conv2d_4_pointwise_weights.bin";

const char* dep5weightdir = "./wbin/Conv2d_5_depthwise_depthwise_weights.bin";
const char* poi5weightdir = "./wbin/Conv2d_5_pointwise_weights.bin";

const char* dep6weightdir = "./wbin/Conv2d_6_depthwise_depthwise_weights.bin";
const char* poi6weightdir = "./wbin/Conv2d_6_pointwise_weights.bin";

const char* dep7weightdir = "./wbin/Conv2d_7_depthwise_depthwise_weights.bin";
const char* poi7weightdir = "./wbin/Conv2d_7_pointwise_weights.bin";

const char* dep8weightdir = "./wbin/Conv2d_8_depthwise_depthwise_weights.bin";
const char* poi8weightdir = "./wbin/Conv2d_8_pointwise_weights.bin";

const char* dep9weightdir = "./wbin/Conv2d_9_depthwise_depthwise_weights.bin";
const char* poi9weightdir = "./wbin/Conv2d_9_pointwise_weights.bin";

const char* dep10weightdir = "./wbin/Conv2d_10_depthwise_depthwise_weights.bin";
const char* poi10weightdir = "./wbin/Conv2d_10_pointwise_weights.bin";

const char* dep11weightdir = "./wbin/Conv2d_11_depthwise_depthwise_weights.bin";
const char* poi11weightdir = "./wbin/Conv2d_11_pointwise_weights.bin";

const char* dep12weightdir = "./wbin/Conv2d_12_depthwise_depthwise_weights.bin";
const char* poi12weightdir = "./wbin/Conv2d_12_pointwise_weights.bin";

const char* dep13weightdir = "./wbin/Conv2d_13_depthwise_depthwise_weights.bin";
const char* poi13weightdir = "./wbin/Conv2d_13_pointwise_weights.bin";

const char* fcweightdir = "./wbin/Final_Layer.bin";

void conv1(float* input,
	float* weight,
	float* output)
{
   float local_input_buffer[3][128][128];
   float local_weight_buffer[32][3][64][64];
   float local_output_buffer[32][64][64];

//#pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
//#pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
//#pragma HLS ARRAY_PARTITION variable=local_output_buffer complete


    for (int i=0; i<3; i++) {
       	for (int j=0; j<128; j++) {
		    for (int k=0; k<128; k++) {
//#pragma HLS pipeline
	   			local_input_buffer[i][j][k] = input[i*128*128 + j*128 + k];
		    }
	    }
    }

	for (int i=0; i<32; i++) {
		for (int j=0; j<3; j++) {
			for (int k=0; k<64; k++) {
				for (int l=0; l<64; l++) {
//#pragma HLS pipeline
					local_weight_buffer[i][j][k][l] = weight[i*64*64*3 + j*64*64 + k*64+ l];
				}
			}
		}
	}

    for (int i=0; i<32; i++) {
	   for (int j=0; j<64; j++) {
		   for (int k=0; k<64; k++) {
//#pragma HLS pipeline
	   			local_output_buffer[i][j][k] = 0;
		   }
	   }
   	}

	for(int co = 0;co<32;co++){
		for(int h = 0;h<64;h++){
//#pragma HLS pipeline
			for(int w = 0;w<64;w++){

				float sum = 0;
				for(int ci = 0;ci<3;ci++){

					for(int m = 0;m<3;m++){
						for(int n = 0;n<3;n++){
//#pragma HLS unroll

							sum += local_weight_buffer[co][ci][m][n] * ((2*h+m < 128 && 2*w+n < 128) ? local_input_buffer[ci][2*h+m][2*w+n]:0);
						    // printf("%f", sum);
						}
					}
				}
				local_output_buffer[co][h][w] = (sum > 0)? sum : 0.0f;
				// printf("%f", output[co*64*64 + h*64 + w]);
			}
		}
	}

	for (int i=0; i<32; i++) {
		for (int j=0; j<64; j++) {
			for (int k=0; k<64; k++) {
//#pragma HLS pipeline
				output[i*64*64 + j*64 + k] = local_output_buffer[i][j][k];
			}
		}

	}
}

void dwconv_layer2(float* input,
	float* weight,
	float* output, 
	int depChannel, int depoutH){

   int Tr = 8;
   int Tc = 8;
   int Tm = 8;

   float local_input_buffer[8][8][8];
   float local_weight_buffer[8][3][3];
   float local_output_buffer[8][8][8];


#pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

		dep_H: for(int h = 0;h<64;h+=8){
#pragma HLS_TRIPCOUNT max=8
			dep_W: for(int w = 0;w<64;w+=8){
#pragma HLS_TRIPCOUNT max=8
				dep_C:for(int co = 0;co<32;co+=8){
#pragma HLS_TRIPCOUNT max=4
//#pragma HLS pipeline
					//								int cii_k = 0;
					weight_load:for (int cii=0; cii<Tm; cii++) {
//#pragma HLS pipeline
									for (int ck=0; ck<3; ck++) {
										for (int cl=0; cl<3; cl++) {
//#pragma HLS pipelin
										local_weight_buffer[cii][ck][cl] = weight[(cii + co)*9 + ck*3 + cl];
										}
									}
								}

					input_load:for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
						for (int j=0; j<Tr; j++) {
										for (int k=0; k<Tc; k++) {
//#pragma HLS pipeline
											local_input_buffer[i][j][k] = input[(i + co)*4096 + (j + h)*64 + k + w];
											//local_output_buffer[i][j][k] = 0;
										}
									}
								}
				compute_conv: for(int m = 0;m<3;m++){
#pragma HLS TRIPCOUNT
						for(int n = 0;n<3;n++){
#pragma HLS TRIPCOUNT
					tile_compute:for (int trr = 0; trr<Tr; trr++) {
//#pragma HLS pipeline
							for (int tcc = 0; tcc <Tc; tcc++) {
#pragma HLS pipeline
								for (int too=0; too<Tm; too++) {
#pragma HLS UNROLL
	//								float sum = 0;
						//				local_input_buffer[co][h][w] = input[co*64*64 + h*64 + w];
									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][m][n] * (( trr+m-1 >= 0 && tcc+n-1 >= 0 && trr+m-1 < depoutH && tcc+n-1 < depoutH) ?local_input_buffer[too][trr+m-1][tcc+n-1]:0);
												}
											}
//local_output_buffer[too][trr][tcc] = sum;
								}
							}
						}
				output_store: for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
					for (int j=0; j<Tr; j++) {
									for (int k=0; k<Tc; k++) {
						//#pragma HLS pipeline
										output[(i + co)*4096 + (j + h)*64 + (k + w)] = local_output_buffer[i][j][k];
									}
								}
							}
			}
		}
	}

}


void dwconv_layer6(float* input,
	float* weight,
	float* output,
	int depChannel, int depoutH){

   int Tr = 8;
   int Tc = 8;
   int Tm = 8;

   float local_input_buffer[8][8][8];
   float local_weight_buffer[8][3][3];
   float local_output_buffer[8][8][8];


#pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

		dep_H: for(int h = 0;h<32;h+=8){
#pragma HLS_TRIPCOUNT max=4
			dep_W: for(int w = 0;w<32;w+=8){
#pragma HLS_TRIPCOUNT max=4
				dep_C:for(int co = 0;co<128;co+=8){
#pragma HLS_TRIPCOUNT max=4
//#pragma HLS pipeline
					//								int cii_k = 0;
					weight_load:for (int cii=0; cii<Tm; cii++) {
//#pragma HLS pipeline
									for (int ck=0; ck<3; ck++) {
										for (int cl=0; cl<3; cl++) {
//#pragma HLS pipelin
										local_weight_buffer[cii][ck][cl] = weight[(cii + co)*9 + ck*3 + cl];
										}
									}
								}

					input_load:for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
						for (int j=0; j<Tr; j++) {
										for (int k=0; k<Tc; k++) {
//#pragma HLS pipeline
											local_input_buffer[i][j][k] = input[(i + co)*1024 + (j + h)*32 + k + w];
											//local_output_buffer[i][j][k] = 0;
										}
									}
								}
				compute_conv: for(int m = 0;m<3;m++){
#pragma HLS TRIPCOUNT
						for(int n = 0;n<3;n++){
#pragma HLS TRIPCOUNT
					tile_compute:for (int trr = 0; trr<Tr; trr++) {
//#pragma HLS pipeline
							for (int tcc = 0; tcc <Tc; tcc++) {
#pragma HLS pipeline
								for (int too=0; too<Tm; too++) {
#pragma HLS UNROLL
	//								float sum = 0;
						//				local_input_buffer[co][h][w] = input[co*64*64 + h*64 + w];
									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][m][n] * (( trr+m-1 >= 0 && tcc+n-1 >= 0 && trr+m-1 < depoutH && tcc+n-1 < depoutH) ?local_input_buffer[too][trr+m-1][tcc+n-1]:0);
												}
											}
//local_output_buffer[too][trr][tcc] = sum;
								}
							}
						}
				output_store: for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
					for (int j=0; j<Tr; j++) {
									for (int k=0; k<Tc; k++) {
						//#pragma HLS pipeline
										output[(i + co)*1024 + (j + h)*32 + (k + w)] = local_output_buffer[i][j][k];
									}
								}
							}
			}
		}
	}

}


void dwconv_layer10(float* input,
	float* weight,
	float* output,
	int depChannel, int depoutH){

   int Tr = 8;
   int Tc = 8;
   int Tm = 8;

   float local_input_buffer[8][8][8];
   float local_weight_buffer[8][3][3];
   float local_output_buffer[8][8][8];


#pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

		dep_H: for(int h = 0;h<16;h+=8){
#pragma HLS_TRIPCOUNT max=4
			dep_W: for(int w = 0;w<16;w+=8){
#pragma HLS_TRIPCOUNT max=4
				dep_C:for(int co = 0;co<256;co+=8){
#pragma HLS_TRIPCOUNT max=32
//#pragma HLS pipeline
					//								int cii_k = 0;
					weight_load:for (int cii=0; cii<Tm; cii++) {
//#pragma HLS pipeline
									for (int ck=0; ck<3; ck++) {
										for (int cl=0; cl<3; cl++) {
//#pragma HLS pipelin
										local_weight_buffer[cii][ck][cl] = weight[(cii + co)*9 + ck*3 + cl];
										}
									}
								}

					input_load:for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
						for (int j=0; j<Tr; j++) {
										for (int k=0; k<Tc; k++) {
//#pragma HLS pipeline
											local_input_buffer[i][j][k] = input[(i + co)*256 + (j + h)*16 + k + w];
											//local_output_buffer[i][j][k] = 0;
										}
									}
								}
				compute_conv: for(int m = 0;m<3;m++){
#pragma HLS TRIPCOUNT
						for(int n = 0;n<3;n++){
#pragma HLS TRIPCOUNT
					tile_compute:for (int trr = 0; trr<Tr; trr++) {
//#pragma HLS pipeline
							for (int tcc = 0; tcc <Tc; tcc++) {
#pragma HLS pipeline
								for (int too=0; too<Tm; too++) {
#pragma HLS UNROLL
	//								float sum = 0;
						//				local_input_buffer[co][h][w] = input[co*64*64 + h*64 + w];
									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][m][n] * (( trr+m-1 >= 0 && tcc+n-1 >= 0 && trr+m-1 < depoutH && tcc+n-1 < depoutH) ?local_input_buffer[too][trr+m-1][tcc+n-1]:0);
												}
											}
//local_output_buffer[too][trr][tcc] = sum;
								}
							}
						}
				output_store: for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
					for (int j=0; j<Tr; j++) {
									for (int k=0; k<Tc; k++) {
						//#pragma HLS pipeline
										output[(i + co)*256 + (j + h)*16 + (k + w)] = local_output_buffer[i][j][k];
									}
								}
							}
			}
		}
	}

}

void dwconv_layer_common(float* input,
	float* weight,
	float* output,
	int depChannel, int depoutH){

   int Tr = 4;
   int Tc = 4;
   int Tm = 8;

   float local_input_buffer[8][4][4];
   float local_weight_buffer[8][3][3];
   float local_output_buffer[8][4][4];


#pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

		dep_H: for(int h = 0;h<8;h+=4){
#pragma HLS_TRIPCOUNT max=2
			dep_W: for(int w = 0;w<8;w+=4){
#pragma HLS_TRIPCOUNT max=2
				dep_C:for(int co = 0;co<512;co+=8){
#pragma HLS_TRIPCOUNT max=64
//#pragma HLS pipeline
					//								int cii_k = 0;
					weight_load:for (int cii=0; cii<Tm; cii++) {
//#pragma HLS pipeline
									for (int ck=0; ck<3; ck++) {
										for (int cl=0; cl<3; cl++) {
//#pragma HLS pipelin
										local_weight_buffer[cii][ck][cl] = weight[(cii + co)*9 + ck*3 + cl];
										}
									}
								}

					input_load:for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
						for (int j=0; j<Tr; j++) {
										for (int k=0; k<Tc; k++) {
//#pragma HLS pipeline
											local_input_buffer[i][j][k] = input[(i + co)*64 + (j + h)*8 + k + w];
											//local_output_buffer[i][j][k] = 0;
										}
									}
								}
				compute_conv: for(int m = 0;m<3;m++){
#pragma HLS TRIPCOUNT
						for(int n = 0;n<3;n++){
#pragma HLS TRIPCOUNT
					tile_compute:for (int trr = 0; trr<Tr; trr++) {
//#pragma HLS pipeline
							for (int tcc = 0; tcc <Tc; tcc++) {
#pragma HLS pipeline
								for (int too=0; too<Tm; too++) {
#pragma HLS UNROLL
	//								float sum = 0;
						//				local_input_buffer[co][h][w] = input[co*64*64 + h*64 + w];
									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][m][n] * (( trr+m-1 >= 0 && tcc+n-1 >= 0 && trr+m-1 < depoutH && tcc+n-1 < depoutH) ?local_input_buffer[too][trr+m-1][tcc+n-1]:0);
												}
											}
//local_output_buffer[too][trr][tcc] = sum;
								}
							}
						}
				output_store: for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
					for (int j=0; j<Tr; j++) {
									for (int k=0; k<Tc; k++) {
						//#pragma HLS pipeline
										output[(i + co)*64 + (j + h)*8 + (k + w)] = local_output_buffer[i][j][k];
									}
								}
							}
			}
		}
	}

}


void dwconv_layer26(float* input,
	float* weight,
	float* output,
	int depChannel, int depoutH){

   int Tr = 4;
   int Tc = 4;
   int Tm = 8;

   float local_input_buffer[8][4][4];
   float local_weight_buffer[8][3][3];
   float local_output_buffer[8][4][4];


#pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

//		dep_H: for(int h = 0;h<4;h+=2){
//#pragma HLS_TRIPCOUNT max=2
//			dep_W: for(int w = 0;w<4;w+=2){
//#pragma HLS_TRIPCOUNT max=2
				dep_C:for(int co = 0;co<1024;co+=8){
#pragma HLS_TRIPCOUNT max=128
//#pragma HLS pipeline
					//								int cii_k = 0;
					weight_load:for (int cii=0; cii<Tm; cii++) {
//#pragma HLS pipeline
									for (int ck=0; ck<3; ck++) {
										for (int cl=0; cl<3; cl++) {
//#pragma HLS pipelin
										local_weight_buffer[cii][ck][cl] = weight[(cii + co)*9 + ck*3 + cl];
										}
									}
								}

					input_load:for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
						for (int j=0; j<Tr; j++) {
										for (int k=0; k<Tc; k++) {
//#pragma HLS pipeline
											local_input_buffer[i][j][k] = input[(i + co)*4 + (j)*4 + k];
											//local_output_buffer[i][j][k] = 0;
										}
									}
								}
				compute_conv: for(int m = 0;m<3;m++){
#pragma HLS TRIPCOUNT
						for(int n = 0;n<3;n++){
#pragma HLS TRIPCOUNT
					tile_compute:for (int trr = 0; trr<Tr; trr++) {
//#pragma HLS pipeline
							for (int tcc = 0; tcc <Tc; tcc++) {
#pragma HLS pipeline
								for (int too=0; too<Tm; too++) {
#pragma HLS UNROLL
	//								float sum = 0;
						//				local_input_buffer[co][h][w] = input[co*64*64 + h*64 + w];
									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][m][n] * (( trr+m-1 >= 0 && tcc+n-1 >= 0 && trr+m-1 < depoutH && tcc+n-1 < depoutH) ?local_input_buffer[too][trr+m-1][tcc+n-1]:0);
												}
											}
//local_output_buffer[too][trr][tcc] = sum;
								}
							}
						}
				output_store: for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
					for (int j=0; j<Tr; j++) {
									for (int k=0; k<Tc; k++) {
						//#pragma HLS pipeline
										output[(i + co)*4 + (j )*4 + (k)] = local_output_buffer[i][j][k];
									}
								}
							}
			}
		}
//	}
//
//}




void dwconv_3x3_pooling(float* input,
	float* weight,
	float* output,
	int depChannel, int depoutH){


//   float local_input_buffer[depChannel][depoutH][depoutH];
//   float local_weight_buffer[depChannel][3][3];
//   float local_output_buffer[depChannel][depoutH][depoutH];


//#pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
//#pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
//#pragma HLS ARRAY_PARTITION variable=local_output_buffer complete


//    for (int i=0; i<depChannel; i++) {
//       	for (int j=0; j<depoutH; j++) {
//		    for (int k=0; k<depoutH; k++) {
////#pragma HLS pipeline
//	   			local_input_buffer[i][j][k] = input[i*depoutH*depoutH + j*depoutH + k];
//		    }
//	    }
//    }
//
//	for (int i=0; i<depChannel; i++) {
////		for (int j=0; j<depChannel; j++) {
//			for (int k=0; k<depoutH; k++) {
//				for (int l=0; l<depoutH; l++) {
////#pragma HLS pipeline
//					local_weight_buffer[i][k][l] = weight[i*depChannel*depoutH + k*depoutH+ l];
//				}
//			}
//		//}
//	}
//
//    for (int i=0; i<depChannel; i++) {
//	   for (int j=0; j<depoutH; j++) {
//		   for (int k=0; k<depoutH; k++) {
////#pragma HLS pipeline
//	   			local_output_buffer[i][j][k] = 0;
//		   }
//	   }
//   	}


	for(int co = 0;co<depChannel;co++){
		for(int h = 0;h<depoutH;h++){
			for(int w = 0;w<depoutH;w++){
				float sum = 0;
				for(int m = 0;m<3;m++){
						for(int n = 0;n<3;n++){
							sum += weight[co*9 + m*3 + n] * (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < depoutH && w+n-1 < depoutH) ?input[co*depoutH*depoutH + (h*depoutH+m-1) + (w+n-1)]:0);
						}
					}
				output[co*depoutH*depoutH + h*depoutH + w] = sum;
			}
		}
	}

//	for(int co = 0;co<depChannel;co++){
//		for(int h = 0;h<depoutH;h++){
//			for(int w = 0;w<depoutH;w++){
//				float sum = 0;
//				for(int m = 0;m<3;m++){
//						for(int n = 0;n<3;n++){
//							sum += local_weight_buffer[co][m][n] * (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < depoutH && w+n-1 < depoutH) ?local_input_buffer[co][h+m-1][w+n-1]:0);
//						}
//					}
//				local_output_buffer[co][h][w] = sum;
//			}
//		}
//	}
//
//    for (int i=0; i<depChannel; i++) {
//		for (int j=0; j<depoutH; j++) {
//			for (int k=0; k<depoutH; k++) {
////#pragma HLS pipeline
//				output[i*depoutH*depoutH + j*depoutH + k] = local_output_buffer[i][j][k];
//			}
//		}
//
//	}
//


}

void dwconv_layer4_stride(float* input,
	float* weight,
	float* output,
	int depChannel, int depoutH){

   int Tr = 8;
   int Tc = 8;
   int Tm = 8;

   float local_input_buffer[8][8][8];
   float local_weight_buffer[8][3][3];
   float local_output_buffer[8][8][8];


#pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

		dep_H: for(int h = 0;h<32;h+=8){
#pragma HLS_TRIPCOUNT max=8
			dep_W: for(int w = 0;w<32;w+=8){
#pragma HLS_TRIPCOUNT max=8
				dep_C:for(int co = 0;co<64;co+=8){
#pragma HLS_TRIPCOUNT max=4
//#pragma HLS pipeline
					//								int cii_k = 0;
					weight_load:for (int cii=0; cii<Tm; cii++) {
//#pragma HLS pipeline
									for (int ck=0; ck<3; ck++) {
										for (int cl=0; cl<3; cl++) {
//#pragma HLS pipelin
										local_weight_buffer[cii][ck][cl] = weight[(cii + co)*9 + ck*3 + cl];
										}
									}
//									cii_k++;
								}

					input_load:for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
						for (int j=0; j<Tr; j++) {
										for (int k=0; k<Tc; k++) {
//#pragma HLS pipeline
											local_input_buffer[i][j][k] = input[(i + co)*4096 + (j + h)*64 + k + w];
											//local_output_buffer[i][j][k] = 0;
										}
									}
								}
				compute_conv: for(int m = 0;m<3;m++){
#pragma HLS TRIPCOUNT
						for(int n = 0;n<3;n++){
#pragma HLS TRIPCOUNT
					tile_compute:for (int trr = 0; trr<Tr; trr++) {
//#pragma HLS pipeline
							for (int tcc = 0; tcc <Tc; tcc++) {
#pragma HLS pipeline
								for (int too=0; too<Tm; too++) {
#pragma HLS UNROLL
						 			local_output_buffer[too][trr][tcc] += local_weight_buffer[too][m][n] * (( trr*2+m >= 0 && tcc*2+n >= 0 && trr*2+m < depoutH && tcc*2+n < depoutH) ?local_input_buffer[too][trr*2+m][tcc*2+n]:0);
												}
											}
								}
							}
						}
				output_store: for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
					for (int j=0; j<Tr; j++) {
									for (int k=0; k<Tc; k++) {
						//#pragma HLS pipeline
										output[(i + co)*1024 + (j + h)*32 + (k + w)] = local_output_buffer[i][j][k];
									}
								}
							}
			}
		}
	}

}

void dwconv_layer8_stride(float* input,
	float* weight,
	float* output,
	int depChannel, int depoutH){

   int Tr = 4;
   int Tc = 4;
   int Tm = 8;

   float local_input_buffer[8][4][4];
   float local_weight_buffer[8][3][3];
   float local_output_buffer[8][4][4];


#pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

		dep_H: for(int h = 0;h<16;h+=4){
#pragma HLS_TRIPCOUNT max=4
			dep_W: for(int w = 0;w<16;w+=4){
#pragma HLS_TRIPCOUNT max=4
				dep_C:for(int co = 0;co<128;co+=8){
#pragma HLS_TRIPCOUNT max=16
//#pragma HLS pipeline
					//								int cii_k = 0;
					weight_load:for (int cii=0; cii<Tm; cii++) {
//#pragma HLS pipeline
									for (int ck=0; ck<3; ck++) {
										for (int cl=0; cl<3; cl++) {
//#pragma HLS pipelin
										local_weight_buffer[cii][ck][cl] = weight[(cii + co)*9 + ck*3 + cl];
										}
									}
//									cii_k++;
								}

					input_load:for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
						for (int j=0; j<Tr; j++) {
										for (int k=0; k<Tc; k++) {
//#pragma HLS pipeline
											local_input_buffer[i][j][k] = input[(i + co)*1024 + (j + h)*32 + k + w];
											//local_output_buffer[i][j][k] = 0;
										}
									}
								}
				compute_conv: for(int m = 0;m<3;m++){
#pragma HLS TRIPCOUNT
						for(int n = 0;n<3;n++){
#pragma HLS TRIPCOUNT
					tile_compute:for (int trr = 0; trr<Tr; trr++) {
//#pragma HLS pipeline
							for (int tcc = 0; tcc <Tc; tcc++) {
#pragma HLS pipeline
								for (int too=0; too<Tm; too++) {
#pragma HLS UNROLL
									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][m][n] * (( trr*2+m >= 0 && tcc*2+n >= 0 && trr*2+m < depoutH && tcc*2+n < depoutH) ?local_input_buffer[too][trr*2+m][tcc*2+n]:0);
												}
											}
								}
							}
						}
				output_store: for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
					for (int j=0; j<Tr; j++) {
									for (int k=0; k<Tc; k++) {
						//#pragma HLS pipeline
										output[(i + co)*512 + (j + h)*16 + (k + w)] = local_output_buffer[i][j][k];
									}
								}
							}
			}
		}
	}

}

void dwconv_layer12_stride(float* input,
	float* weight,
	float* output,
	int depChannel, int depoutH){

   int Tr = 4;
   int Tc = 4;
   int Tm = 8;

   float local_input_buffer[8][4][4];
   float local_weight_buffer[8][3][3];
   float local_output_buffer[8][4][4];


#pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

		dep_H: for(int h = 0;h<8;h+=4){
#pragma HLS_TRIPCOUNT max=2
			dep_W: for(int w = 0;w<8;w+=4){
#pragma HLS_TRIPCOUNT max=2
				dep_C:for(int co = 0;co<256;co+=8){
#pragma HLS_TRIPCOUNT max=32
//#pragma HLS pipeline
					//								int cii_k = 0;
					weight_load:for (int cii=0; cii<Tm; cii++) {
//#pragma HLS pipeline
									for (int ck=0; ck<3; ck++) {
										for (int cl=0; cl<3; cl++) {
//#pragma HLS pipelin
										local_weight_buffer[cii][ck][cl] = weight[(cii + co)*9 + ck*3 + cl];
										}
									}
//									cii_k++;
								}

					input_load:for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
						for (int j=0; j<Tr; j++) {
										for (int k=0; k<Tc; k++) {
//#pragma HLS pipeline
											local_input_buffer[i][j][k] = input[(i + co)*1024 + (j + h)*32 + k + w];
											//local_output_buffer[i][j][k] = 0;
										}
									}
								}
				compute_conv: for(int m = 0;m<3;m++){
#pragma HLS TRIPCOUNT
						for(int n = 0;n<3;n++){
#pragma HLS TRIPCOUNT
					tile_compute:for (int trr = 0; trr<Tr; trr++) {
//#pragma HLS pipeline
							for (int tcc = 0; tcc <Tc; tcc++) {
#pragma HLS pipeline
								for (int too=0; too<Tm; too++) {
#pragma HLS UNROLL
									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][m][n] * (( trr*2+m >= 0 && tcc*2+n >= 0 && trr*2+m < depoutH && tcc*2+n < depoutH) ?local_input_buffer[too][trr*2+m][tcc*2+n]:0);
												}
											}
								}
							}
						}
				output_store: for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
					for (int j=0; j<Tr; j++) {
									for (int k=0; k<Tc; k++) {
						//#pragma HLS pipeline
										output[(i + co)*512 + (j + h)*16 + (k + w)] = local_output_buffer[i][j][k];
									}
								}
							}
			}
		}
	}

}




void dwconv_3x3_pooling_stride(float* input,
	float* weight,
	float* output,
	int depChannel, 
	int depoutH){
	int outputH = depoutH / 2;	
	for(int co = 0;co<depChannel;co++){
		for(int h = 0;h<outputH;h++){
			for(int w = 0;w<outputH;w++){
				float sum = 0;
				for(int m = 0;m<3;m++){
						for(int n = 0;n<3;n++){
							sum += weight[co*9 + m*3 + n] * ((h*2+m < depoutH && w*2+n < depoutH) ?input[co*depoutH*depoutH + (2*h*depoutH+m) + (w*2+n)]:0);
						}
					}
				output[co*outputH*outputH + h*outputH + w] = sum;
			}
		}
	}
}

//void pwconv_layer3(float* input,
//	float* weight,
//	float* output,
//	int Cin,
//	int depChannel,
//	int depoutH){
//
//   float local_input_buffer[4][8][8];
//   float local_weight_buffer[8][4];
//   float local_output_buffer[8][8][8];
//
//   int Tr = 8;
//   int Tc = 8;
//   int Tm = 8;
//   int Tn = 4;
//
//   #pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
//   #pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
//   #pragma HLS ARRAY_PARTITION variable=local_output_buffer complete
//
//   		dep_H: for(int h = 0;h<64;h+=8){
//   #pragma HLS_TRIPCOUNT max=8
//   			dep_W: for(int w = 0;w<64;w+=8){
//   #pragma HLS_TRIPCOUNT max=8
//   				dep_Cout:for(int co = 0;co<64;co+=8){
//   #pragma HLS_TRIPCOUNT max=8
//   	   				dep_Cin:for(int ci = 0;ci<32;ci+=4){
//   #pragma HLS_TRIPCOUNT max=8
//   					weight_load:for (int coo=0; coo<Tm; coo++) {
//									for (int cii=0; cii<Tn; cii++) {
//									local_weight_buffer[coo][cii] = weight[(coo + co)*32 + (cii + ci)];
//									}
//   								}
//
//   					input_load:for (int i=0; i<Tn; i++) {
//   						for (int j=0; j<Tr; j++) {
//   										for (int k=0; k<Tc; k++) {
//   											local_input_buffer[i][j][k] = input[(i + ci)*4096 + (j + h)*64 + k + w];
//   										}
//   									}
//   								}
//   					tile_compute:for (int trr = 0; trr<Tr; trr++) {
//   //#pragma HLS pipeline
//   							for (int tcc = 0; tcc <Tc; tcc++) {
//  #pragma HLS pipeline
//   								for (int too=0; too<Tm; too++) {
//   #pragma HLS UNROLL
//   									for (int tii=0; tii<Tn; tii++) {
//#pragma HLS UNROLL
//   									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][tii] * local_input_buffer[tii][trr][tcc];
//   									}
//   								}
//   							}
//   						}
//   				output_store: for (int i=0; i<Tm; i++) {
//   //#pragma HLS pipeline
//   					for (int j=0; j<Tr; j++) {
//   									for (int k=0; k<Tc; k++) {
//   						//#pragma HLS pipeline
//   										output[(i + co)*4096 + (j + h)*64 + (k + w)] = local_output_buffer[i][j][k];
//																												 //> 0)? local_output_buffer[i][j][k]:0.0f;
//   									}
//   								}
//   							}
//   	   				}
//   			}
//   		}
//   	}
//}

void pwconv_layer3(float* input,
	float* weight,
	float* output,
	int Cin,
	int depChannel,
	int depoutH){

   float local_input_buffer[4][8][8];
   float local_weight_buffer[4][4];
   float local_output_buffer[4][8][8];

   int Tr = 8;
   int Tc = 8;
   int Tm = 4;
   int Tn = 4;

   #pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

   		dep_H: for(int h = 0;h<64;h+=8){
   #pragma HLS_TRIPCOUNT max=8
   			dep_W: for(int w = 0;w<64;w+=8){
   #pragma HLS_TRIPCOUNT max=8
   				dep_Cout:for(int co = 0;co<64;co+=4){
   #pragma HLS_TRIPCOUNT max=16
   	   				dep_Cin:for(int ci = 0;ci<32;ci+=4){
   #pragma HLS_TRIPCOUNT max=8
   					weight_load:for (int coo=0; coo<Tm; coo++) {
									for (int cii=0; cii<Tn; cii++) {
									local_weight_buffer[coo][cii] = weight[(coo + co)*32 + (cii + ci)];
									}
   								}

   					input_load:for (int i=0; i<Tn; i++) {
   						for (int j=0; j<Tr; j++) {
   										for (int k=0; k<Tc; k++) {
   											local_input_buffer[i][j][k] = input[(i + ci)*4096 + (j + h)*64 + k + w];
   										}
   									}
   								}
   					tile_compute:for (int trr = 0; trr<Tr; trr++) {
   //#pragma HLS pipeline
   							for (int tcc = 0; tcc <Tc; tcc++) {
  #pragma HLS pipeline
   								for (int too=0; too<Tm; too++) {
   #pragma HLS UNROLL
   									for (int tii=0; tii<Tn; tii++) {
#pragma HLS UNROLL
   									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][tii] * local_input_buffer[tii][trr][tcc];
   									}
   								}
   							}
   						}
   				output_store: for (int i=0; i<Tm; i++) {
   //#pragma HLS pipeline
   					for (int j=0; j<Tr; j++) {
   									for (int k=0; k<Tc; k++) {
   						//#pragma HLS pipeline
   										output[(i + co)*4096 + (j + h)*64 + (k + w)] = local_output_buffer[i][j][k];
																												 //> 0)? local_output_buffer[i][j][k]:0.0f;
   									}
   								}
   							}
   	   				}
   			}
   		}
   	}
}


void pwconv_layer5(float* input,
	float* weight,
	float* output,
	int Cin,
	int depChannel,
	int depoutH){

   float local_input_buffer[4][8][8];
   float local_weight_buffer[4][4];
   float local_output_buffer[4][8][8];

   int Tr = 8;
   int Tc = 8;
   int Tm = 4;
   int Tn = 4;

   #pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

   		dep_H: for(int h = 0;h<32;h+=8){
   #pragma HLS_TRIPCOUNT max=4
   			dep_W: for(int w = 0;w<32;w+=8){
   #pragma HLS_TRIPCOUNT max=4
   				dep_Cout:for(int co = 0;co<128;co+=4){
   #pragma HLS_TRIPCOUNT max=32
   	   				dep_Cin:for(int ci = 0;ci<64;ci+=4){
   #pragma HLS_TRIPCOUNT max=16
   					weight_load:for (int coo=0; coo<Tm; coo++) {
									for (int cii=0; cii<Tn; cii++) {
									local_weight_buffer[coo][cii] = weight[(coo + co)*64 + (cii + ci)];
									}
   								}

   					input_load:for (int i=0; i<Tn; i++) {
   						for (int j=0; j<Tr; j++) {
   										for (int k=0; k<Tc; k++) {
   											local_input_buffer[i][j][k] = input[(i + ci)*4096+ (j + h)*64 + k + w];
   										}
   									}
   								}
   					tile_compute:for (int trr = 0; trr<Tr; trr++) {
   //#pragma HLS pipeline
   							for (int tcc = 0; tcc <Tc; tcc++) {
  #pragma HLS pipeline
   								for (int too=0; too<Tm; too++) {
   #pragma HLS UNROLL
   									for (int tii=0; tii<Tn; tii++) {
#pragma HLS UNROLL
   									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][tii] * local_input_buffer[tii][trr][tcc];
   									}
   								}
   							}
   						}
   				output_store: for (int i=0; i<Tm; i++) {
   //#pragma HLS pipeline
   					for (int j=0; j<Tr; j++) {
   									for (int k=0; k<Tc; k++) {
   						//#pragma HLS pipeline
   										output[(i + co)*1024 + (j + h)*32 + (k + w)] = local_output_buffer[i][j][k];
																												 //> 0)? local_output_buffer[i][j][k]:0.0f;
   									}
   								}
   							}
   	   				}
   			}
   		}
   	}
}

void pwconv_layer7(float* input,
	float* weight,
	float* output,
	int Cin,
	int depChannel,
	int depoutH){

   float local_input_buffer[4][8][8];
   float local_weight_buffer[4][4];
   float local_output_buffer[4][8][8];

   int Tr = 8;
   int Tc = 8;
   int Tm = 4;
   int Tn = 4;

   #pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

   		dep_H: for(int h = 0;h<32;h+=8){
   #pragma HLS_TRIPCOUNT max=4
   			dep_W: for(int w = 0;w<32;w+=8){
   #pragma HLS_TRIPCOUNT max=4
   				dep_Cout:for(int co = 0;co<128;co+=4){
   #pragma HLS_TRIPCOUNT max=32
   	   				dep_Cin:for(int ci = 0;ci<128;ci+=4){
   #pragma HLS_TRIPCOUNT max=32
   					weight_load:for (int coo=0; coo<Tm; coo++) {
									for (int cii=0; cii<Tn; cii++) {
									local_weight_buffer[coo][cii] = weight[(coo + co)*128 + (cii + ci)];
									}
   								}

   					input_load:for (int i=0; i<Tn; i++) {
   						for (int j=0; j<Tr; j++) {
   										for (int k=0; k<Tc; k++) {
   											local_input_buffer[i][j][k] = input[(i + ci)*1024 + (j + h)*32 + k + w];
   										}
   									}
   								}
   					tile_compute:for (int trr = 0; trr<Tr; trr++) {
   //#pragma HLS pipeline
   							for (int tcc = 0; tcc <Tc; tcc++) {
  #pragma HLS pipeline
   								for (int too=0; too<Tm; too++) {
   #pragma HLS UNROLL
   									for (int tii=0; tii<Tn; tii++) {
#pragma HLS UNROLL
   									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][tii] * local_input_buffer[tii][trr][tcc];
   									}
   								}
   							}
   						}
   				output_store: for (int i=0; i<Tm; i++) {
   //#pragma HLS pipeline
   					for (int j=0; j<Tr; j++) {
   									for (int k=0; k<Tc; k++) {
   						//#pragma HLS pipeline
   										output[(i + co)*1024 + (j + h)*32 + (k + w)] = local_output_buffer[i][j][k];
																												 //> 0)? local_output_buffer[i][j][k]:0.0f;
   									}
   								}
   							}
   	   				}
   			}
   		}
   	}
}

void pwconv_layer9(float* input,
	float* weight,
	float* output,
	int Cin,
	int depChannel,
	int depoutH){

   float local_input_buffer[4][8][8];
   float local_weight_buffer[8][4];
   float local_output_buffer[8][8][8];

   int Tr = 8;
   int Tc = 8;
   int Tm = 8;
   int Tn = 4;

   #pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

   		dep_H: for(int h = 0;h<16;h+=8){
   #pragma HLS_TRIPCOUNT max=2
   			dep_W: for(int w = 0;w<16;w+=8){
   #pragma HLS_TRIPCOUNT max=2
   				dep_Cout:for(int co = 0;co<256;co+=8){
   #pragma HLS_TRIPCOUNT max=32
   	   				dep_Cin:for(int ci = 0;ci<128;ci+=4){
   #pragma HLS_TRIPCOUNT max=32
   					weight_load:for (int coo=0; coo<Tm; coo++) {
									for (int cii=0; cii<Tn; cii++) {
									local_weight_buffer[coo][cii] = weight[(coo + co)*128 + (cii + ci)];
									}
   								}

   					input_load:for (int i=0; i<Tn; i++) {
   						for (int j=0; j<Tr; j++) {
   										for (int k=0; k<Tc; k++) {
   											local_input_buffer[i][j][k] = input[(i + ci)*256 + (j + h)*16 + k + w];
   										}
   									}
   								}
   					tile_compute:for (int trr = 0; trr<Tr; trr++) {
   //#pragma HLS pipeline
   							for (int tcc = 0; tcc <Tc; tcc++) {
   #pragma HLS pipeline
   								for (int too=0; too<Tm; too++) {
   #pragma HLS UNROLL
   									for (int tii=0; tii<Tn; tii++) {
#pragma HLS UNROLL
   									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][tii] * local_input_buffer[tii][trr][tcc];
   									}
   								}
   							}
   						}
   				output_store: for (int i=0; i<Tm; i++) {
   //#pragma HLS pipeline
   					for (int j=0; j<Tr; j++) {
   									for (int k=0; k<Tc; k++) {
   						//#pragma HLS pipeline
   										output[(i + co)*256 + (j + h)*16 + (k + w)] = local_output_buffer[i][j][k];
																												 //> 0)? local_output_buffer[i][j][k]:0.0f;
   									}
   								}
   							}
   	   				}
   			}
   		}
   	}
}

void pwconv_layer11(float* input,
	float* weight,
	float* output,
	int Cin,
	int depChannel,
	int depoutH){

   float local_input_buffer[4][8][8];
   float local_weight_buffer[8][4];
   float local_output_buffer[8][8][8];

   int Tr = 8;
   int Tc = 8;
   int Tm = 4;
   int Tn = 4;

   #pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

   		dep_H: for(int h = 0;h<16;h+=8){
   #pragma HLS_TRIPCOUNT max=2
   			dep_W: for(int w = 0;w<16;w+=8){
   #pragma HLS_TRIPCOUNT max=2
   				dep_Cout:for(int co = 0;co<256;co+=4){
   #pragma HLS_TRIPCOUNT max=64
   	   				dep_Cin:for(int ci = 0;ci<256;ci+=4){
   #pragma HLS_TRIPCOUNT max=64
   					weight_load:for (int coo=0; coo<Tm; coo++) {
									for (int cii=0; cii<Tn; cii++) {
									local_weight_buffer[coo][cii] = weight[(coo + co)*256 + (cii + ci)];
									}
   								}

   					input_load:for (int i=0; i<Tn; i++) {
   						for (int j=0; j<Tr; j++) {
   										for (int k=0; k<Tc; k++) {
   											local_input_buffer[i][j][k] = input[(i + ci)*256 + (j + h)*16 + k + w];
   										}
   									}
   								}
   					tile_compute:for (int trr = 0; trr<Tr; trr++) {
   //#pragma HLS pipeline
   							for (int tcc = 0; tcc <Tc; tcc++) {
  #pragma HLS pipeline
   								for (int too=0; too<Tm; too++) {
   #pragma HLS UNROLL
   									for (int tii=0; tii<Tn; tii++) {
#pragma HLS UNROLL
   									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][tii] * local_input_buffer[tii][trr][tcc];
   									}
   								}
   							}
   						}
   				output_store: for (int i=0; i<Tm; i++) {
   //#pragma HLS pipeline
   					for (int j=0; j<Tr; j++) {
   									for (int k=0; k<Tc; k++) {
   						//#pragma HLS pipeline
   										output[(i + co)*256 + (j + h)*16 + (k + w)] = local_output_buffer[i][j][k];
																												 //> 0)? local_output_buffer[i][j][k]:0.0f;
   									}
   								}
   							}
   	   				}
   			}
   		}
   	}
}

void pwconv_layer13(float* input,
	float* weight,
	float* output,
	int Cin,
	int depChannel,
	int depoutH){

   float local_input_buffer[4][4][4];
   float local_weight_buffer[8][4];
   float local_output_buffer[8][4][4];

   int Tr = 4;
   int Tc = 4;
   int Tm = 8;
   int Tn = 4;

   #pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

   		dep_H: for(int h = 0;h<8;h+=4){
   #pragma HLS_TRIPCOUNT max=2
   			dep_W: for(int w = 0;w<8;w+=4){
   #pragma HLS_TRIPCOUNT max=2
   				dep_Cout:for(int co = 0;co<512;co+=8){
   #pragma HLS_TRIPCOUNT max=64
   	   				dep_Cin:for(int ci = 0;ci<256;ci+=4){
   #pragma HLS_TRIPCOUNT max=64
   					weight_load:for (int coo=0; coo<Tm; coo++) {
									for (int cii=0; cii<Tn; cii++) {
									local_weight_buffer[coo][cii] = weight[(coo + co)*256 + (cii + ci)];
									}
   								}

   					input_load:for (int i=0; i<Tn; i++) {
   						for (int j=0; j<Tr; j++) {
   										for (int k=0; k<Tc; k++) {
   											local_input_buffer[i][j][k] = input[(i + ci)*64 + (j + h)*8 + k + w];
   										}
   									}
   								}
   					tile_compute:for (int trr = 0; trr<Tr; trr++) {
   //#pragma HLS pipeline
   							for (int tcc = 0; tcc <Tc; tcc++) {
  #pragma HLS pipeline
   								for (int too=0; too<Tm; too++) {
   #pragma HLS UNROLL
   									for (int tii=0; tii<Tn; tii++) {
#pragma HLS UNROLL
   									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][tii] * local_input_buffer[tii][trr][tcc];
   									}
   								}
   							}
   						}
   				output_store: for (int i=0; i<Tm; i++) {
   //#pragma HLS pipeline
   					for (int j=0; j<Tr; j++) {
   									for (int k=0; k<Tc; k++) {
   						//#pragma HLS pipeline
   										output[(i + co)*64 + (j + h)*8 + (k + w)] = local_output_buffer[i][j][k];
																												 //> 0)? local_output_buffer[i][j][k]:0.0f;
   									}
   								}
   							}
   	   				}
   			}
   		}
   	}
}

void pwconv_layer_common(float* input,
	float* weight,
	float* output,
	int Cin,
	int depChannel,
	int depoutH){

   float local_input_buffer[16][4][4];
   float local_weight_buffer[16][16];
   float local_output_buffer[16][4][4];

   int Tr = 4;
   int Tc = 4;
   int Tm = 16;
   int Tn = 16;

   #pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

   		dep_H: for(int h = 0;h<8;h+=4){
   #pragma HLS_TRIPCOUNT max=2
   			dep_W: for(int w = 0;w<8;w+=4){
   #pragma HLS_TRIPCOUNT max=2
   				dep_Cout:for(int co = 0;co<512;co+=16){
   #pragma HLS_TRIPCOUNT max=32
   	   				dep_Cin:for(int ci = 0;ci<512;ci+=16){
   #pragma HLS_TRIPCOUNT max=32
   					weight_load:for (int coo=0; coo<Tm; coo++) {
									for (int cii=0; cii<Tn; cii++) {
									local_weight_buffer[coo][cii] = weight[(coo + co)*512 + (cii + ci)];
									}
   								}

   					input_load:for (int i=0; i<Tn; i++) {
   						for (int j=0; j<Tr; j++) {
   										for (int k=0; k<Tc; k++) {
   											local_input_buffer[i][j][k] = input[(i + ci)*64 + (j + h)*8 + k + w];
   										}
   									}
   								}
   					tile_compute:for (int trr = 0; trr<Tr; trr++) {
   //#pragma HLS pipeline
   							for (int tcc = 0; tcc <Tc; tcc++) {
  #pragma HLS pipeline
   								for (int too=0; too<Tm; too++) {
   #pragma HLS UNROLL
   									for (int tii=0; tii<Tn; tii++) {
#pragma HLS UNROLL
   									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][tii] * local_input_buffer[tii][trr][tcc];
   								 	}
   								}
   							}
   						}


   					output_store: for (int i=0; i<Tm; i++) {
   //#pragma HLS pipeline
   					for (int j=0; j<Tr; j++) {
   									for (int k=0; k<Tc; k++) {
   						//#pragma HLS pipeline
   										output[(i + co)*64 + (j + h)*8 + (k + w)] = local_output_buffer[i][j][k];
																												 //> 0)? local_output_buffer[i][j][k]:0.0f;
   									}
   								}
   							}
   	   				}
   			}
   		}
   	}
}

void pwconv_layer_optimized(float* input,
	float* weight,
	float* output,
	int Cin,
	int depChannel,
	int depoutH){

   float local_input_buffer[4][8][8];
   float local_weight_buffer[8][4];
   float local_output_buffer[8][8][8];

   int Tr = 8;
   int Tc = 8;
   int Tm = 8;
   int Tn = 4;

   #pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

//	   		dep_H: for(int h = 0;h<8;h+=4){
//	   #pragma HLS_TRIPCOUNT max=2
//	   			dep_W: for(int w = 0;w<8;w+=4){
//	   #pragma HLS_TRIPCOUNT max=2
   				dep_Cout:for(int co = 0;co<512;co+=8){
   #pragma HLS_TRIPCOUNT max=64
   	   				dep_Cin:for(int ci = 0;ci<512;ci+=4){
   #pragma HLS_TRIPCOUNT max=128

   	   					weight_load:for (int coo=0; coo<Tm; coo++) {
									for (int cii=0; cii<Tn; cii++) {
									local_weight_buffer[coo][cii] = weight[(coo + co)*512 + (cii + ci)];
									}
   								}

   					input_load:for (int i=0; i<Tn; i++) {
   						for (int j=0; j<Tr; j++) {
   										for (int k=0; k<Tc; k++) {
   											local_input_buffer[i][j][k] = input[(i + ci)*64 + (j)*8 + k];
   										}
   									}
   								}
   					tile_compute:for (int trr = 0; trr<Tr; trr++) {

   							for (int tcc = 0; tcc <Tc; tcc++) {
  #pragma HLS pipeline
   								for (int too=0; too<Tm; too++) {
   #pragma HLS UNROLL
   									for (int tii=0; tii<Tn; tii++) {
#pragma HLS UNROLL
   									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][tii] * local_input_buffer[tii][trr][tcc];
   								 	}
   								}
   							}
   						}
  					tile_compute1:for (int trr = 0; trr<Tr; trr++) {

    				pixel_comp:	for (int tcc = 0; tcc <Tc; tcc++) {
  #pragma HLS pipeline
   							output_comp:	for (int too=0; too<Tm; too++) {
   #pragma HLS UNROLL
   							input_comp:		for (int tii=0; tii<Tn; tii++) {
#pragma HLS UNROLL
   									local_output_buffer[too][trr][tcc] = local_input_buffer[tii][trr][tcc];
   								 	}
   								}
   							}
   						}


   					output_store: for (int i=0; i<Tm; i++) {
   					for (int j=0; j<Tr; j++) {
   									for (int k=0; k<Tc; k++) {
   						//#pragma HLS pipeline
   										output[(i + co)*64 + (j)*8 + (k)] = local_output_buffer[i][j][k];
																												 //> 0)? local_output_buffer[i][j][k]:0.0f;
   									}
   								}
   							}
   	   				}
   				}


//   		}
//   	}
}

void pwconv_layer25(float* input,
	float* weight,
	float* output,
	int Cin,
	int depChannel,
	int depoutH){

   float local_input_buffer[8][4][4];
   float local_weight_buffer[8][4];
   float local_output_buffer[8][4][4];

   int Tr = 4;
   int Tc = 4;
   int Tm = 8;
   int Tn = 8;

   #pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

//   		dep_H: for(int h = 0;h<4;h+=2){
//   #pragma HLS_TRIPCOUNT max=2
//   			dep_W: for(int w = 0;w<4;w+=2){
//   #pragma HLS_TRIPCOUNT max=2
   				dep_Cout:for(int co = 0;co<1024;co+=8){
   #pragma HLS_TRIPCOUNT max=128
   	   				dep_Cin:for(int ci = 0;ci<512;ci+=8){
   #pragma HLS_TRIPCOUNT max=64
   					weight_load:for (int coo=0; coo<Tm; coo++) {
									for (int cii=0; cii<Tn; cii++) {
									local_weight_buffer[coo][cii] = weight[(coo + co)*512 + (cii + ci)];
									}
   								}

   					input_load:for (int i=0; i<Tn; i++) {
   						for (int j=0; j<Tr; j++) {
   										for (int k=0; k<Tc; k++) {
   											local_input_buffer[i][j][k] = input[(i + ci)*16 + (j)*4 + k];
   										}
   									}
   								}
   					tile_compute:for (int trr = 0; trr<Tr; trr++) {
 #pragma HLS pipeline
   							for (int tcc = 0; tcc <Tc; tcc++) {
  #pragma HLS pipeline
   								for (int too=0; too<Tm; too++) {
   #pragma HLS UNROLL
   									for (int tii=0; tii<Tn; tii++) {
#pragma HLS UNROLL
   									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][tii] * local_input_buffer[tii][trr][tcc];
   								 	}
   								}
   							}
   						}


   					output_store: for (int i=0; i<Tm; i++) {
   					for (int j=0; j<Tr; j++) {
   									for (int k=0; k<Tc; k++) {
   						//#pragma HLS pipeline
   										output[(i + co)*16 + (j)*4 + (k)] = local_output_buffer[i][j][k];
																												 //> 0)? local_output_buffer[i][j][k]:0.0f;
   									}
   								}
   							}
   	   				}
   			}
//   		}
//   	}
}

//void dwconv_layer24_stride(float* input,
//	float* weight,
//	float* output,
//	int depChannel, int depoutH){
//
//   int Tr = 4;
//   int Tc = 4;
//   int Tm = 8;
//
//   float local_input_buffer[8][4][4];
//   float local_weight_buffer[8][3][3];
//   float local_output_buffer[8][4][4];
//
//
//#pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
//#pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
//#pragma HLS ARRAY_PARTITION variable=local_output_buffer complete
//
//		dep_H: for(int h = 0;h<8;h+=4){
//#pragma HLS_TRIPCOUNT max=2
//			dep_W: for(int w = 0;w<8;w+=4){
//#pragma HLS_TRIPCOUNT max=2
//				dep_C:for(int co = 0;co<256;co+=8){
//#pragma HLS_TRIPCOUNT max=32
////#pragma HLS pipeline
//					//								int cii_k = 0;
//					weight_load:for (int cii=0; cii<Tm; cii++) {
////#pragma HLS pipeline
//									for (int ck=0; ck<3; ck++) {
//										for (int cl=0; cl<3; cl++) {
////#pragma HLS pipelin
//										local_weight_buffer[cii][ck][cl] = weight[(cii + co)*9 + ck*3 + cl];
//										}
//									}
////									cii_k++;
//								}
//
//					input_load:for (int i=0; i<Tm; i++) {
////#pragma HLS pipeline
//						for (int j=0; j<Tr; j++) {
//										for (int k=0; k<Tc; k++) {
////#pragma HLS pipeline
//											local_input_buffer[i][j][k] = input[(i + co)*1024 + (j + h)*32 + k + w];
//											//local_output_buffer[i][j][k] = 0;
//										}
//									}
//								}
//				compute_conv: for(int m = 0;m<3;m++){
//#pragma HLS TRIPCOUNT
//						for(int n = 0;n<3;n++){
//#pragma HLS TRIPCOUNT
//					tile_compute:for (int trr = 0; trr<Tr; trr++) {
////#pragma HLS pipeline
//							for (int tcc = 0; tcc <Tc; tcc++) {
//#pragma HLS pipeline
//								for (int too=0; too<Tm; too++) {
//#pragma HLS UNROLL
//									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][m][n] * (( trr*2+m >= 0 && tcc*2+n >= 0 && trr*2+m < depoutH && tcc*2+n < depoutH) ?local_input_buffer[too][trr*2+m][tcc*2+n]:0);
//												}
//											}
//								}
//							}
//						}
//				output_store: for (int i=0; i<Tm; i++) {
////#pragma HLS pipeline
//					for (int j=0; j<Tr; j++) {
//									for (int k=0; k<Tc; k++) {
//						//#pragma HLS pipeline
//										output[(i + co)*512 + (j + h)*16 + (k + w)] = local_output_buffer[i][j][k];
//									}
//								}
//							}
//			}
//		}
//	}
//
//}
void dwconv_layer24_stride(float* input,
	float* weight,
	float* output,
	int depChannel, int depoutH){

   int Tr = 4;
   int Tc = 4;
   int Tm = 8;

   float local_input_buffer[8][8][8];
   float local_weight_buffer[8][3][3];
   float local_output_buffer[8][4][4];


#pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
#pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

//		dep_H: for(int h = 0;h<4;h+=2){
//#pragma HLS_TRIPCOUNT max=2
//			dep_W: for(int w = 0;w<4;w+=2){
//#pragma HLS_TRIPCOUNT max=2
				dep_C:for(int co = 0;co<512;co+=8){
#pragma HLS_TRIPCOUNT max=32
//#pragma HLS pipeline
					//								int cii_k = 0;
					weight_load:for (int cii=0; cii<Tm; cii++) {
//#pragma HLS pipeline
									for (int ck=0; ck<3; ck++) {
										for (int cl=0; cl<3; cl++) {
//#pragma HLS pipelin
										local_weight_buffer[cii][ck][cl] = weight[(cii + co)*9 + ck*3 + cl];
										}
									}
//									cii_k++;
								}

					input_load:for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
						for (int j=0; j<Tr*2; j++) {
										for (int k=0; k<Tc*2; k++) {
//#pragma HLS pipeline
											local_input_buffer[i][j][k] = input[(i + co)*64 + (j)*8 + k];
											//local_output_buffer[i][j][k] = 0;
										}
									}
								}
				compute_conv: for(int m = 0;m<3;m++){
#pragma HLS TRIPCOUNT
						for(int n = 0;n<3;n++){
#pragma HLS TRIPCOUNT
					tile_compute:for (int trr = 0; trr<Tr; trr++) {
//#pragma HLS pipeline
							for (int tcc = 0; tcc <Tc; tcc++) {
#pragma HLS pipeline
								for (int too=0; too<Tm; too++) {
#pragma HLS UNROLL
									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][m][n] * (( trr*2+m >= 0 && tcc*2+n >= 0 && trr*2+m < depoutH && tcc*2+n < depoutH) ?local_input_buffer[too][trr*2+m][tcc*2+n]:0);
												}
											}
								}
							}
						}
				output_store: for (int i=0; i<Tm; i++) {
//#pragma HLS pipeline
					for (int j=0; j<Tr; j++) {
									for (int k=0; k<Tc; k++) {
						//#pragma HLS pipeline
										output[(i + co)*16 + (j)*4 + (k)] = local_output_buffer[i][j][k];
									}
								}
							}
			}
//		}
//	}

}



void pwconv_layer27(float* input,
	float* weight,
	float* output,
	int Cin,
	int depChannel,
	int depoutH){

   float local_input_buffer[8][4][4];
   float local_weight_buffer[8][8];
   float local_output_buffer[8][4][4];

   int Tr = 4;
   int Tc = 4;
   int Tm = 16;
   int Tn = 8;

   #pragma HLS ARRAY_PARTITION variable=local_input_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_weight_buffer complete
   #pragma HLS ARRAY_PARTITION variable=local_output_buffer complete

//   		dep_H: for(int h = 0;h<4;h+=2){
//   #pragma HLS_TRIPCOUNT max=2
//   			dep_W: for(int w = 0;w<4;w+=2){
//   #pragma HLS_TRIPCOUNT max=2
   				dep_Cout:for(int co = 0;co<1024;co+=16){
   #pragma HLS_TRIPCOUNT max=128
   	   				dep_Cin:for(int ci = 0;ci<1024;ci+=8){
   #pragma HLS_TRIPCOUNT max=128
   					weight_load:for (int coo=0; coo<Tm; coo++) {
									for (int cii=0; cii<Tn; cii++) {
									local_weight_buffer[coo][cii] = weight[(coo + co)*1024 + (cii + ci)];
									}
   								}

   					input_load:for (int i=0; i<Tn; i++) {
   						for (int j=0; j<Tr; j++) {
   										for (int k=0; k<Tc; k++) {
   											local_input_buffer[i][j][k] = input[(i + ci)*16 + (j)*4  + k];
   										}
   									}
   								}
   					tile_compute:for (int trr = 0; trr<Tr; trr++) {
 #pragma HLS pipeline
   							for (int tcc = 0; tcc <Tc; tcc++) {
  #pragma HLS pipeline
   								for (int too=0; too<Tm; too++) {
   #pragma HLS UNROLL
   									for (int tii=0; tii<Tn; tii++) {
#pragma HLS UNROLL
   									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][tii] * local_input_buffer[tii][trr][tcc];
   								 	}
   								}
   							}
   						}

//   					tile_compute:for (int too = 0; too<Tm; too++) {
// #pragma HLS pipeline
//   							for (int tii = 0; tii <Tn; tii++) {
//  #pragma HLS pipeline
//   								for (int trr=0; trr<Tr; trr++) {
//   #pragma HLS UNROLL
//   									for (int tcc=0; tcc<Tc; tcc++) {
//#pragma HLS UNROLL
//   									local_output_buffer[too][trr][tcc] += local_weight_buffer[too][tii] * local_input_buffer[tii][trr][tcc];
//   								 	}
//   								}
//   							}
//   						}



   					output_store: for (int i=0; i<Tm; i++) {
   					for (int j=0; j<Tr; j++) {
   									for (int k=0; k<Tc; k++) {
   						//#pragma HLS pipeline
   										output[(i + co)*16 + (j)*4 + (k)] = local_output_buffer[i][j][k];
																												 //> 0)? local_output_buffer[i][j][k]:0.0f;
   									}
   								}
   							}
   	   				}
   			}
//   		}
//   	}
}


void pwconv_1x1(float* input,
	float* weight,
	float* output, 
	int Cin, 
	int depChannel, 
	int depoutH){
	for(int co = 0;co<depChannel;co++){
		for(int h = 0;h<depoutH;h++){
			for(int w = 0;w<depoutH;w++){
				float sum = 0;
				for(int ci = 0;ci<Cin;ci++){
					sum += weight[co*Cin + ci]*input[ci*depoutH*depoutH + h*depoutH + w];
				}
                output[co*depoutH*depoutH + h*depoutH + w] = (sum > 0)? sum : 0.0f;
				// printf("%f", sum);
			}
		}
	}
}


void average_pooling(float* input, 
	float* output){
	for(int co=0; co<1024; co++){
		float sum = 0;
		for (int h=0; h<4; h++) {
			for (int w=0; w<4; w++) {
				sum += input[co*16 + h*4 + w];
			}
		}
		output[co] = sum / 16;
		// printf("%f", output[co]);
	}
}

 void fc_layer(float* input,
 	float* weight,
 	float* output){
 	for (int  i=0; i<50; i++) {
 		float temp = 0;
 		for (int j=0; j<1024; j++) {
 			temp += input[j]*weight[i*1024 + j];
 			// printf("%f", weight[i*1024 + j]);
 		}
 		output[i] = temp;
		
 		// printf("%f", output[i]);
 	}
 	for (int i=0; i<50; i++) {
 		printf("%f ", output[i]);
 	}
 }

static int firstread = 0;
void read_module(paratype* graphin, OUT paratype* graphout)
{
	/*read parameters*/
	static paratype Conv2W[Conv2Channel*graphinC*Conv2Kernel*Conv2Kernel];

	static paratype Conv1_depW[Conv1_depKernel*Conv1_depKernel*Conv1_depChannel];
	static paratype Conv1_poiW[Conv1_poiKernel*Conv1_poiKernel*Conv1_depChannel*Conv1_poiChannel];

	static paratype Conv2_depW[Conv2_depKernel*Conv2_depKernel*Conv2_depChannel];
	static paratype Conv2_poiW[Conv2_poiKernel*Conv2_poiKernel*Conv2_depChannel*Conv2_poiChannel];

	static paratype Conv3_depW[Conv3_depKernel*Conv3_depKernel*Conv3_depChannel];
	static paratype Conv3_poiW[Conv3_poiKernel*Conv3_poiKernel*Conv3_depChannel*Conv3_poiChannel];

	static paratype Conv4_depW[Conv4_depKernel*Conv4_depKernel*Conv4_depChannel];
	static paratype Conv4_poiW[Conv4_poiKernel*Conv4_poiKernel*Conv4_depChannel*Conv4_poiChannel];

	static paratype Conv5_depW[Conv5_depKernel*Conv5_depKernel*Conv5_depChannel];
	static paratype Conv5_poiW[Conv5_poiKernel*Conv5_poiKernel*Conv5_depChannel*Conv5_poiChannel];

	static paratype Conv6_depW[Conv6_depKernel*Conv6_depKernel*Conv6_depChannel];
	static paratype Conv6_poiW[Conv6_poiKernel*Conv6_poiKernel*Conv6_depChannel*Conv6_poiChannel];

	static paratype Conv7_depW[Conv7_depKernel*Conv7_depKernel*Conv7_depChannel];
	static paratype Conv7_poiW[Conv7_poiKernel*Conv7_poiKernel*Conv7_depChannel*Conv7_poiChannel];

	static paratype Conv8_depW[Conv8_depKernel*Conv8_depKernel*Conv8_depChannel];
	static paratype Conv8_poiW[Conv8_poiKernel*Conv8_poiKernel*Conv8_depChannel*Conv8_poiChannel];

	static paratype Conv9_depW[Conv9_depKernel*Conv9_depKernel*Conv9_depChannel];
	static paratype Conv9_poiW[Conv9_poiKernel*Conv9_poiKernel*Conv9_depChannel*Conv9_poiChannel];

	static paratype Conv10_depW[Conv10_depKernel*Conv10_depKernel*Conv10_depChannel];
	static paratype Conv10_poiW[Conv10_poiKernel*Conv10_poiKernel*Conv10_depChannel*Conv10_poiChannel];

	static paratype Conv11_depW[Conv11_depKernel*Conv11_depKernel*Conv11_depChannel];
	static paratype Conv11_poiW[Conv11_poiKernel*Conv11_poiKernel*Conv11_depChannel*Conv11_poiChannel];

	static paratype Conv12_depW[Conv12_depKernel*Conv12_depKernel*Conv12_depChannel];
	static paratype Conv12_poiW[Conv12_poiKernel*Conv12_poiKernel*Conv12_depChannel*Conv12_poiChannel];

	static paratype Conv13_depW[Conv13_depKernel*Conv13_depKernel*Conv13_depChannel];
	static paratype Conv13_poiW[Conv13_poiKernel*Conv13_poiKernel*Conv13_depChannel*Conv13_poiChannel];

	static paratype FClayer_W[1024*50];
	static paratype FClayer_B[50];

	paratype* conv1_result = new paratype[32*64*64];
	paratype* dep1_result = new paratype[64 * 64 * 32];
	paratype* poi1_result = new paratype[64 * 64 * 64];

	paratype* dep2_result = new paratype[Conv2_poioutH * Conv2_poioutH * Conv2_depChannel];
	paratype* poi2_result = new paratype[Conv2_poioutH * Conv2_poioutH * Conv2_poiChannel];

	paratype* dep3_result = new paratype[Conv3_depoutH * Conv3_depoutH * Conv3_depChannel];
	paratype* poi3_result = new paratype[Conv3_poioutH * Conv3_poioutH * Conv3_poiChannel];

	paratype* dep4_result = new paratype[Conv4_poioutH * Conv4_poioutH * Conv4_depChannel];
	paratype* poi4_result = new paratype[Conv4_poioutH * Conv4_poioutH * Conv4_poiChannel];

	paratype* dep5_result = new paratype[Conv5_depoutH * Conv5_depoutH * Conv5_depChannel];
	paratype* poi5_result = new paratype[Conv5_poioutH * Conv5_poioutH * Conv5_poiChannel];

	paratype* dep6_result = new paratype[Conv6_poioutH * Conv6_poioutH * Conv6_depChannel];
	
	paratype* dep7_result = new paratype[Conv7_depoutH * Conv7_depoutH * Conv7_depChannel];
	paratype* poi7_result = new paratype[Conv7_poioutH * Conv7_poioutH * Conv7_poiChannel];

	paratype* poi8_result = new paratype[Conv11_poioutH * Conv11_poioutH * Conv12_depChannel];
	paratype* dep12_result = new paratype[Conv13_depoutH * Conv13_depoutH * Conv12_depChannel];

	paratype* poi13_result = new paratype[Conv13_poioutH * Conv13_poioutH * Conv13_depChannel];
	paratype* dep13_result = new paratype[Conv13_depoutH * Conv13_depoutH * Conv13_depChannel];

	paratype* inter_output = new paratype[1024];

	//read in the params read once
	if(!firstread)
	{
		firstread = 1;
	FILE *fr = fopen(Conv2weightdir, "rb");
	fread(Conv2W, sizeof(paratype), Conv2Kernel*Conv2Kernel*graphinC*Conv2Channel, fr);
	fclose(fr);

	// for (int i=0; i<Conv2Kernel*Conv2Kernel*graphinC*Conv2Channel; i++) {
	// 	printf("%f ", Conv2W[i]);
	// }

	fr = fopen(dep1weightdir, "rb");
	fread(Conv1_depW, sizeof(paratype), Conv1_depKernel*Conv1_depKernel*Conv1_depChannel, fr);
	fclose(fr);

	fr = fopen(poi1weightdir, "rb");
	fread(Conv1_poiW, sizeof(paratype), Conv1_poiKernel*Conv1_poiKernel*Conv1_depChannel*Conv1_poiChannel, fr);
	fclose(fr);

	fr = fopen(dep2weightdir, "rb");
	fread(Conv2_depW, sizeof(paratype), Conv2_depKernel*Conv2_depKernel*Conv2_depChannel, fr);
	fclose(fr);

	fr = fopen(poi2weightdir, "rb");
	fread(Conv2_poiW, sizeof(paratype), Conv2_poiKernel*Conv2_poiKernel*Conv2_depChannel*Conv2_poiChannel, fr);
	fclose(fr);

	fr = fopen(dep3weightdir, "rb");
	fread(Conv3_depW, sizeof(paratype), Conv3_depKernel*Conv3_depKernel*Conv3_depChannel, fr);
	fclose(fr);

	fr = fopen(poi3weightdir, "rb");
	fread(Conv3_poiW, sizeof(paratype), Conv3_poiKernel*Conv3_poiKernel*Conv3_depChannel*Conv3_poiChannel, fr);
	fclose(fr);

	fr = fopen(dep4weightdir, "rb");
	fread(Conv4_depW, sizeof(paratype), Conv4_depKernel*Conv4_depKernel*Conv4_depChannel, fr);
	fclose(fr);

	fr = fopen(poi4weightdir, "rb");
	fread(Conv4_poiW, sizeof(paratype), Conv4_poiKernel*Conv4_poiKernel*Conv4_depChannel*Conv4_poiChannel, fr);
	fclose(fr);

	fr = fopen(dep5weightdir, "rb");
	fread(Conv5_depW, sizeof(paratype), Conv5_depKernel*Conv5_depKernel*Conv5_depChannel, fr);
	fclose(fr);

	fr = fopen(poi5weightdir, "rb");
	fread(Conv5_poiW, sizeof(paratype), Conv5_poiKernel*Conv5_poiKernel*Conv5_depChannel*Conv5_poiChannel, fr);
	fclose(fr);

	fr = fopen(dep6weightdir, "rb");
	fread(Conv6_depW, sizeof(paratype), Conv6_depKernel*Conv6_depKernel*Conv6_depChannel, fr);
	fclose(fr);

	fr = fopen(poi6weightdir, "rb");
	fread(Conv6_poiW, sizeof(paratype), Conv6_poiKernel*Conv6_poiKernel*Conv6_depChannel*Conv6_poiChannel, fr);
	fclose(fr);

	fr = fopen(dep7weightdir, "rb");
	fread(Conv7_depW, sizeof(paratype), Conv7_depKernel*Conv7_depKernel*Conv7_depChannel, fr);
	fclose(fr);

	fr = fopen(poi7weightdir, "rb");
	fread(Conv7_poiW, sizeof(paratype), Conv7_poiKernel*Conv7_poiKernel*Conv7_depChannel*Conv7_poiChannel, fr);
	fclose(fr);

	fr = fopen(dep8weightdir, "rb");
	fread(Conv8_depW, sizeof(paratype), Conv8_depKernel*Conv8_depKernel*Conv8_depChannel, fr);
	fclose(fr);

	fr = fopen(poi8weightdir, "rb");
	fread(Conv8_poiW, sizeof(paratype), Conv8_poiKernel*Conv8_poiKernel*Conv8_depChannel*Conv8_poiChannel, fr);
	fclose(fr);

	fr = fopen(dep9weightdir, "rb");
	fread(Conv9_depW, sizeof(paratype), Conv9_depKernel*Conv9_depKernel*Conv9_depChannel, fr);
	fclose(fr);

	fr = fopen(poi9weightdir, "rb");
	fread(Conv9_poiW, sizeof(paratype), Conv9_poiKernel*Conv9_poiKernel*Conv9_depChannel*Conv9_poiChannel, fr);
	fclose(fr);

	fr = fopen(dep10weightdir, "rb");
	fread(Conv10_depW, sizeof(paratype), Conv10_depKernel*Conv10_depKernel*Conv10_depChannel, fr);
	fclose(fr);

	fr = fopen(poi10weightdir, "rb");
	fread(Conv10_poiW, sizeof(paratype), Conv10_poiKernel*Conv10_poiKernel*Conv10_depChannel*Conv10_poiChannel, fr);
	fclose(fr);

	fr = fopen(dep11weightdir, "rb");
	fread(Conv11_depW, sizeof(paratype), Conv11_depKernel*Conv11_depKernel*Conv11_depChannel, fr);
	fclose(fr);

	fr = fopen(poi11weightdir, "rb");
	fread(Conv11_poiW, sizeof(paratype), Conv11_poiKernel*Conv11_poiKernel*Conv11_depChannel*Conv11_poiChannel, fr);
	fclose(fr);

	fr = fopen(dep12weightdir, "rb");
	fread(Conv12_depW, sizeof(paratype), Conv12_depKernel*Conv12_depKernel*Conv12_depChannel, fr);
	fclose(fr);

	fr = fopen(poi12weightdir, "rb");
	fread(Conv12_poiW, sizeof(paratype), Conv12_poiKernel*Conv12_poiKernel*Conv12_depChannel*Conv12_poiChannel, fr);
	fclose(fr);

	fr = fopen(dep13weightdir, "rb");
	fread(Conv13_depW, sizeof(paratype), Conv13_depKernel*Conv13_depKernel*Conv13_depChannel, fr);
	fclose(fr);

	fr = fopen(poi13weightdir, "rb");
	fread(Conv13_poiW, sizeof(paratype), Conv13_poiKernel*Conv13_poiKernel*Conv13_depChannel*Conv13_poiChannel, fr);
	fclose(fr);

	fr = fopen(fcweightdir, "rb");
	fread(FClayer_W, sizeof(paratype), 1024*50, fr);
	fclose(fr);

	// fr = fopen(fcbiasdir, "rb");
	// fread(FClayer_B, sizeof(paratype), 50, fr);
	// fclose(fr);
    }
	
    //prelayers
	conv1(graphin, Conv2W, conv1_result);

	// /**stage1**/
	 dwconv_layer2(conv1_result, Conv1_depW, dep1_result, Conv1_depChannel, Conv1_depoutH);
	 pwconv_layer3(dep1_result, Conv1_poiW, poi1_result, Conv1_depChannel, Conv1_poiChannel, Conv1_depoutH);
	 delete[] conv1_result;
	 delete[] dep1_result;

	 // for (int i=0; i<Conv1_depoutH*Conv1_depoutH*Conv1_poiChannel; i++) {
	 // 	printf("%f ", poi1_result[i]);
	 // }

	 // /**stage2**/
	 dwconv_layer4_stride(poi1_result, Conv2_depW, dep2_result, Conv2_depChannel, Conv2_depoutH);
	 delete[] poi1_result;
	 pwconv_layer5(dep2_result, Conv2_poiW, poi2_result, Conv2_depChannel, Conv2_poiChannel, Conv2_poioutH);
	 delete[] dep2_result;

	 dwconv_layer6(poi2_result, Conv3_depW, dep3_result, Conv3_depChannel, Conv3_depoutH);
	 pwconv_layer7(dep3_result, Conv3_poiW, poi3_result, Conv3_depChannel, Conv3_poiChannel, Conv3_poioutH);
	 delete[] poi2_result;
	 delete[] dep3_result;

	 // // /**stage3**/
	 dwconv_layer8_stride(poi3_result, Conv4_depW, dep4_result, Conv4_depChannel, Conv4_depoutH);
	 pwconv_layer9(dep4_result, Conv4_poiW, poi4_result, Conv4_depChannel, Conv4_poiChannel, Conv4_poioutH);
	 delete[] poi3_result;
	 delete[] dep4_result;

	 dwconv_layer10(poi4_result, Conv5_depW, dep5_result, Conv5_depChannel, Conv5_depoutH);
	 pwconv_layer11(dep5_result, Conv5_poiW, poi5_result, Conv5_depChannel, Conv5_poiChannel, Conv5_poioutH);
	 delete[] poi4_result;
	 delete[] dep5_result;

	 // // /**stage4**/
	 // //unit1
	 dwconv_layer12_stride(poi5_result, Conv6_depW, dep6_result, Conv6_depChannel, Conv6_depoutH);
	 pwconv_layer13(dep6_result, Conv6_poiW, poi7_result, Conv6_depChannel, Conv6_poiChannel, Conv6_poioutH);
	 delete[] poi5_result;
	 delete[] dep6_result;

     // //unit2
	 dwconv_layer_common(poi7_result, Conv7_depW, dep7_result, Conv7_depChannel, Conv7_depoutH);
	 pwconv_layer_optimized(dep7_result, Conv7_poiW, poi7_result, Conv7_depChannel, Conv7_poiChannel, Conv7_poioutH);

     // //unit3
	 dwconv_layer_common(poi7_result, Conv8_depW, dep7_result, Conv8_depChannel, Conv8_depoutH);
	 pwconv_layer_optimized(dep7_result, Conv8_poiW, poi7_result, Conv8_depChannel, Conv8_poiChannel, Conv8_poioutH);

     // //unit4
	 dwconv_layer_common(poi7_result, Conv9_depW, dep7_result, Conv9_depChannel, Conv9_depoutH);
	 pwconv_layer_optimized(dep7_result, Conv9_poiW, poi7_result, Conv9_depChannel, Conv9_poiChannel, Conv9_poioutH);

     // //unit5
	 dwconv_layer_common(poi7_result, Conv10_depW, dep7_result, Conv10_depChannel, Conv10_depoutH);
	 pwconv_layer_optimized(dep7_result, Conv10_poiW, poi7_result, Conv10_depChannel, Conv10_poiChannel, Conv10_poioutH);

     // // //unit6
	 dwconv_layer_common(poi7_result, Conv11_depW, dep7_result, Conv11_depChannel, Conv11_depoutH);
	 pwconv_layer_optimized(dep7_result, Conv11_poiW, poi8_result, Conv11_depChannel, Conv11_poiChannel, Conv11_poioutH);
	 delete[] poi7_result;
	 delete[] dep7_result;

	 // // /**stage5**/
	 // //unit1
	 dwconv_layer24_stride(poi8_result, Conv12_depW, dep12_result, Conv12_depChannel, Conv12_depoutH);
	 pwconv_layer25(dep12_result, Conv12_poiW, poi13_result, Conv12_depChannel, Conv12_poiChannel, Conv12_poioutH);
	 delete[] poi8_result;
	 delete[] dep12_result;

     // //unit2
	 dwconv_layer26(poi13_result, Conv13_depW, dep13_result, Conv13_depChannel, Conv13_depoutH);
	 pwconv_layer27(dep13_result, Conv13_poiW, poi13_result, Conv13_depChannel, Conv13_poiChannel, Conv13_poioutH);
	 delete[] dep13_result;

     // // for (int i = 0; i<Conv13_depoutH*Conv13_depoutH*Conv13_depChannel; i++) {
	 // // 	printf("%f", poi13_result[i]);
	 // // }


	 average_pooling(poi13_result, inter_output);
	 delete[] poi13_result;

	 // for (int i=0; i<1000; i++) {
	 // 	printf("%f", FClayer_W[i]);
	 // }
	 fc_layer(inter_output, FClayer_W,  graphout);


}


int main() 
{   
	paratype* graphin;
	paratype * graphout,*graphtest,
		*heatmap,*offset2, *displacement_fwd, *displacement_bwd,
		*heatmap_2_weight, *heatmap_2_bias,
		*offset_2_weight, *offset_2_bias, *displacement_fwd_weight, *displacement_fwd_bias, *displacement_bwd_weight, *displacement_bwd_bias;

    graphin = new paratype[3*128*128];
	for(int i=0; i<3; i++) {
		for (int j=0; j<128; j++) {
			for (int k=0; k<128; k++) {
				graphin[i*128*128 + j*128 + k] = k * 1000;
			}
		}
	}

	graphout = new paratype[50];
	read_module(graphin, graphout);
}

// using namespace std;
// /**prelayers**/
// void conv1(float input[1][3][128][128],
// 	float weight[32][3][3][3],
// 	float output[1][32][64][64])
// {
// 	for(int co = 0;co<32;co++){
// 		for(int h = 0;h<64;h++){
// 			for(int w = 0;w<64;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<3;ci++){
// 					for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][ci][m][n] * ((2*h+m < 128 && 2*w+n < 128) ? input[0][ci][2*h+m][2*w+n]:0);
// 						}
// 					}
// 				}
// 				output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**stage1**/
// void layer2_dwconv_3x3_pooling(float input[1][32][64][64],
// 	float weight[32][1][3][3],
// 	float output[1][32][64][64]){
// 	for(int co = 0;co<32;co++){
// 		for(int h = 0;h<64;h++){
// 			for(int w = 0;w<64;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * (( h*2+m-1 >= 0 && w*2+n-1 >= 0 && h*2+m-1 < 64 && w*2+n-1 < 64) ?input[0][co][h*2+m-1][w*2+n-1]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }


// void layer3_pwconv_1x1(float input[1][32][64][64],
// 	float weight[64][32][1][1],
// 	float output[1][64][64][64]){
// 	for(int co = 0;co<64;co++){
// 		for(int h = 0;h<64;h++){
// 			for(int w = 0;w<64;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<32;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}

// }

// /**stage2**/
// void layer4_dwconv_3x3_pooling_stride(float input[1][64][64][64],
// 	float weight[64][1][3][3],
// 	float output[1][64][32][32]){
// 	for(int co = 0;co<64;co++){
// 		for(int h = 0;h<32;h++){
// 			for(int w = 0;w<32;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * ((h*2+m < 32 && w*2+n < 32) ?input[0][co][h*2+m][w*2+n]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer5_pwconv_1x1(float input[1][64][32][32],
// 	float weight[128][64][1][1],
// 	float output[1][128][32][32]){
// 	for(int co = 0;co<128;co++){
// 		for(int h = 0;h<32;h++){
// 			for(int w = 0;w<32;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<64;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// void layer6_dwconv_3x3_pooling(float input[1][128][64][64],
// 	float weight[128][1][3][3],
// 	float output[1][128][64][64]){
// 	for(int co = 0;co<128;co++){
// 		for(int h = 0;h<64;h++){
// 			for(int w = 0;w<64;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * (( h*2+m-1 >= 0 && w*2+n-1 >= 0 && h*2+m-1 < 64 && w*2+n-1 < 64) ?input[0][co][h*2+m-1][w*2+n-1]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer7_pwconv_1x1(float input[1][128][32][32],
// 	float weight[128][128][1][1],
// 	float output[1][128][32][32]){
// 	for(int co = 0;co<128;co++){
// 		for(int h = 0;h<32;h++){
// 			for(int w = 0;w<32;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<128;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**stage3**/
// void layer8_dwconv_3x3_pooling_stride(float input[1][128][32][32],
// 	float weight[128][1][3][3],
// 	float output[1][128][16][16]){
// 	for(int co = 0;co<128;co++){
// 		for(int h = 0;h<16;h++){
// 			for(int w = 0;w<16;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * ((h*2+m < 16 && w*2+n < 16) ?input[0][co][h*2+m][w*2+n]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer9_pwconv_1x1(float input[1][128][16][16],
// 	float weight[256][128][1][1],
// 	float output[1][256][16][16]){
// 	for(int co = 0;co<256;co++){
// 		for(int h = 0;h<16;h++){
// 			for(int w = 0;w<16;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<128;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// void layer10_dwconv_3x3_pooling(float input[1][256][16][16],
// 	float weight[256][1][3][3],
// 	float output[1][256][16][16]){
// 	for(int co = 0;co<256;co++){
// 		for(int h = 0;h<16;h++){
// 			for(int w = 0;w<16;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * (( h*2+m-1 >= 0 && w*2+n-1 >= 0 && h*2+m-1 <16 && w*2+n-1 < 16) ?input[0][co][h*2+m-1][w*2+n-1]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer11_pwconv_1x1(float input[1][256][16][16],
// 	float weight[256][256][1][1],
// 	float output[1][256][16][16]){
// 	for(int co = 0;co<256;co++){
// 		for(int h = 0;h<16;h++){
// 			for(int w = 0;w<16;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<256;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**stage4**/
// /**unit1**/
// void layer12_dwconv_3x3_pooling_stride(float input[1][256][16][16],
// 	float weight[256][1][3][3],
// 	float output[1][256][8][8]){
// 	for(int co = 0;co<256;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * ((h*2+m < 8 && w*2+n < 8) ?input[0][co][h*2+m][w*2+n]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer13_pwconv_1x1(float input[1][256][8][8],
// 	float weight[512][256][1][1],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<256;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**unit2**/
// void layer14_dwconv_3x3_pooling(float input[1][512][8][8],
// 	float weight[512][1][3][3],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * (( h*2+m-1 >= 0 && w*2+n-1 >= 0 && h*2+m-1 <8 && w*2+n-1 < 8) ?input[0][co][h*2+m-1][w*2+n-1]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer15_pwconv_1x1(float input[1][512][8][8],
// 	float weight[512][512][1][1],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<512;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**unit3**/
// void layer16_dwconv_3x3_pooling_stride(float input[1][512][8][8],
// 	float weight[512][1][3][3],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * ((h*2+m < 8 && w*2+n < 8) ?input[0][co][h*2+m][w*2+n]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer17_pwconv_1x1(float input[1][512][8][8],
// 	float weight[512][512][1][1],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<512;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**unit4**/
// void layer18_dwconv_3x3_pooling(float input[1][512][8][8],
// 	float weight[512][1][3][3],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * (( h*2+m-1 >= 0 && w*2+n-1 >= 0 && h*2+m-1 <8 && w*2+n-1 < 8) ?input[0][co][h*2+m-1][w*2+n-1]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer19_pwconv_1x1(float input[1][512][8][8],
// 	float weight[512][512][1][1],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<512;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**unit5**/
// void layer20_dwconv_3x3_pooling(float input[1][512][8][8],
// 	float weight[512][1][3][3],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * (( h*2+m-1 >= 0 && w*2+n-1 >= 0 && h*2+m-1 <8 && w*2+n-1 < 8) ?input[0][co][h*2+m-1][w*2+n-1]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer21_pwconv_1x1(float input[1][512][8][8],
// 	float weight[512][512][1][1],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<512;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**unit6**/
// void layer22_dwconv_3x3_pooling(float input[1][512][8][8],
// 	float weight[512][1][3][3],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * (( h*2+m-1 >= 0 && w*2+n-1 >= 0 && h*2+m-1 <8 && w*2+n-1 < 8) ?input[0][co][h*2+m-1][w*2+n-1]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer23_pwconv_1x1(float input[1][512][8][8],
// 	float weight[512][512][1][1],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<512;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**stage5**/
// /**unit1**/
// void layer24_dwconv_3x3_pooling_stride(float input[1][512][8][8],
// 	float weight[512][1][3][3],
// 	float output[1][512][4][4]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * ((h*2+m < 4 && w*2+n < 4) ?input[0][co][h*2+m][w*2+n]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer25_pwconv_1x1(float input[1][512][4][4],
// 	float weight[1024][512][1][1],
// 	float output[1][1024][4][4]){
// 	for(int co = 0;co<1024;co++){
// 		for(int h = 0;h<4;h++){
// 			for(int w = 0;w<4;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<512;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**unit2**/
// void layer26_dwconv_3x3_pooling(float input[1][1024][4][4],
// 	float weight[1024][1][3][3],
// 	float output[1][1024][4][4]){
// 	for(int co = 0;co<1024;co++){
// 		for(int h = 0;h<4;h++){
// 			for(int w = 0;w<4;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * (( h*2+m-1 >= 0 && w*2+n-1 >= 0 && h*2+m-1 < 4 && w*2+n-1 < 4) ?input[0][co][h*2+m-1][w*2+n-1]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer27_pwconv_1x1(float input[1][1024][4][4],
// 	float weight[1024][1024][1][1],
// 	float output[1][1024][4][4]){
// 	for(int co = 0;co<1024;co++){
// 		for(int h = 0;h<4;h++){
// 			for(int w = 0;w<4;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<1024;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// void layer28_average_pooling(float input[1][1024][4][4], 
// 	float output[1024]){
// 	for(int co=0; co<1024; co++){
// 		float sum = 0;
// 		for (int h=0; h<4; h++) {
// 			for (int w=0; w<4; w++) {
// 				sum += input[1][co][h][w];
// 			}
// 		}
// 		output[co] = sum / 16;
// 	}
// }

// void layer29_fclayer(float input[1024], 
// 	float weight[50][1024], 
// 	float bias[50],
// 	float output[50]){
// 	for (int  i=0; i<50; i++) {
// 		float temp = 0;
// 		for (int j=0; j<1024; j++) {
// 			temp += input[j]*weight[i][j];
// 		}
// 		output[i] = temp + bias[i];
// 	}
// }

////implement the convolution
//const char* Conv2weightdir = "./wbin/Conv2d_0_weights.bin";
//
//const char* dep1weightdir = "./wbin/Conv2d_1_depthwise_depthwise_weights.bin";
//const char* poi1weightdir = "./wbin/Conv2d_1_pointwise_weights.bin";
//
//const char* dep2weightdir = "./wbin/Conv2d_2_depthwise_depthwise_weights.bin";
//const char* poi2weightdir = "./wbin/Conv2d_2_pointwise_weights.bin";
//
//const char* dep3weightdir = "./wbin/Conv2d_3_depthwise_depthwise_weights.bin";
//const char* poi3weightdir = "./wbin/Conv2d_3_pointwise_weights.bin";
//
//const char* dep4weightdir = "./wbin/Conv2d_4_depthwise_depthwise_weights.bin";
//const char* poi4weightdir = "./wbin/Conv2d_4_pointwise_weights.bin";
//
//const char* dep5weightdir = "./wbin/Conv2d_5_depthwise_depthwise_weights.bin";
//const char* poi5weightdir = "./wbin/Conv2d_5_pointwise_weights.bin";
//
//const char* dep6weightdir = "./wbin/Conv2d_6_depthwise_depthwise_weights.bin";
//const char* poi6weightdir = "./wbin/Conv2d_6_pointwise_weights.bin";
//
//const char* dep7weightdir = "./wbin/Conv2d_7_depthwise_depthwise_weights.bin";
//const char* poi7weightdir = "./wbin/Conv2d_7_pointwise_weights.bin";
//
//const char* dep8weightdir = "./wbin/Conv2d_8_depthwise_depthwise_weights.bin";
//const char* poi8weightdir = "./wbin/Conv2d_8_pointwise_weights.bin";
//
//const char* dep9weightdir = "./wbin/Conv2d_9_depthwise_depthwise_weights.bin";
//const char* poi9weightdir = "./wbin/Conv2d_9_pointwise_weights.bin";
//
//const char* dep10weightdir = "./wbin/Conv2d_10_depthwise_depthwise_weights.bin";
//const char* poi10weightdir = "./wbin/Conv2d_10_pointwise_weights.bin";
//
//const char* dep11weightdir = "./wbin/Conv2d_11_depthwise_depthwise_weights.bin";
//const char* poi11weightdir = "./wbin/Conv2d_11_pointwise_weights.bin";
//
//const char* dep12weightdir = "./wbin/Conv2d_12_depthwise_depthwise_weights.bin";
//const char* poi12weightdir = "./wbin/Conv2d_12_pointwise_weights.bin";
//
//const char* dep13weightdir = "./wbin/Conv2d_13_depthwise_depthwise_weights.bin";
//const char* poi13weightdir = "./wbin/Conv2d_13_pointwise_weights.bin";
//
//const char* fcweightdir = "./wbin/Final_Layer.bin";
//
//void conv1(float input[1*3*128*128],
//	float weight[32*3*3*3],
//	float output[32*64*64])
//{
//#pragma HLS ARRAY_PARTITION variable=input complete
//#pragma HLS ARRAY_PARTITION variable=weight complete
//#pragma HLS ARRAY_PARTITION variable=output complete
//
//	for(int co = 0;co<32;co++){
//		for(int h = 0;h<64;h++){
//			for(int w = 0;w<64;w++){
//#pragma HLS pipeline
//				float sum = 0;
//				for(int ci = 0;ci<3;ci++){
//					for(int m = 0;m<3;m++){
//						for(int n = 0;n<3;n++){
//
//							sum += weight[co*(9*3) + ci*9 + 3*m + n] * ((2*h+m < 128 && 2*w+n < 128) ? input[ci*128*128 + (2*h)*128 + m + 2*w+n]:0);
//						    // printf("%f", sum);
//						}
//					}
//				}
//				output[co*64*64 + h*64 + w] = (sum > 0)? sum : 0.0f;
//				// printf("%f", output[co*64*64 + h*64 + w]);
//			}
//		}
//	}
//}
//
//void dwconv_3x3_pooling(float* input,
//	float* weight,
//	float* output,
//	int depChannel, int depoutH){
//	for(int co = 0;co<depChannel;co++){
//		for(int h = 0;h<depoutH;h++){
//			for(int w = 0;w<depoutH;w++){
//				float sum = 0;
//				for(int m = 0;m<3;m++){
//						for(int n = 0;n<3;n++){
//							sum += weight[co*9 + m*3 + n] * (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < depoutH && w+n-1 < depoutH) ?input[co*depoutH*depoutH + (h*depoutH+m-1) + (w+n-1)]:0);
//						}
//					}
//				output[co*depoutH*depoutH + h*depoutH + w] = sum;
//			}
//		}
//	}
//}
//
//void dwconv_3x3_pooling_stride(float* input,
//	float* weight,
//	float* output,
//	int depChannel,
//	int depoutH){
//	int outputH = depoutH / 2;
//	for(int co = 0;co<depChannel;co++){
//		for(int h = 0;h<outputH;h++){
//			for(int w = 0;w<outputH;w++){
//				float sum = 0;
//				for(int m = 0;m<3;m++){
//						for(int n = 0;n<3;n++){
//							sum += weight[co*9 + m*3 + n] * ((h*2+m < depoutH && w*2+n < depoutH) ?input[co*depoutH*depoutH + (2*h*depoutH+m) + (w*2+n)]:0);
//						}
//					}
//				output[co*outputH*outputH + h*outputH + w] = sum;
//			}
//		}
//	}
//}
//
//void pwconv_1x1(float* input,
//	float* weight,
//	float* output,
//	int Cin,
//	int depChannel,
//	int depoutH){
//	for(int co = 0;co<depChannel;co++){
//		for(int h = 0;h<depoutH;h++){
//			for(int w = 0;w<depoutH;w++){
//				float sum = 0;
//				for(int ci = 0;ci<Cin;ci++){
//					sum += weight[co*Cin + ci]*input[ci*depoutH*depoutH + h*depoutH + w];
//				}
//                output[co*depoutH*depoutH + h*depoutH + w] = (sum > 0)? sum : 0.0f;
//				// printf("%f", sum);
//			}
//		}
//	}
//}
//
//void average_pooling(float* input,
//	float* output){
//	for(int co=0; co<1024; co++){
//		float sum = 0;
//		for (int h=0; h<4; h++) {
//			for (int w=0; w<4; w++) {
//				sum += input[co*16 + h*4 + w];
//			}
//		}
//		output[co] = sum / 16;
//		// printf("%f", output[co]);
//	}
//}
//
//void fc_layer(float* input,
//	float* weight,
//	float* output){
//	for (int  i=0; i<50; i++) {
//		float temp = 0;
//		for (int j=0; j<1024; j++) {
//			temp += input[j]*weight[i*1024 + j];
//			// printf("%f", weight[i*1024 + j]);
//		}
//		output[i] = temp;
//
//		// printf("%f", output[i]);
//	}
//	for (int i=0; i<50; i++) {
//		printf("%f ", output[i]);
//	}
//}
//
//static int firstread = 0;
//void read_module(IN paratype* graphin, OUT paratype* graphout)
//{
//	/*read parameters*/
//	static paratype Conv2W[Conv2Kernel*Conv2Kernel*graphinC*Conv2Channel];
//
//	static paratype Conv1_depW[Conv1_depKernel*Conv1_depKernel*Conv1_depChannel];
//	static paratype Conv1_poiW[Conv1_poiKernel*Conv1_poiKernel*Conv1_depChannel*Conv1_poiChannel];
//
//	static paratype Conv2_depW[Conv2_depKernel*Conv2_depKernel*Conv2_depChannel];
//	static paratype Conv2_poiW[Conv2_poiKernel*Conv2_poiKernel*Conv2_depChannel*Conv2_poiChannel];
//
//	static paratype Conv3_depW[Conv3_depKernel*Conv3_depKernel*Conv3_depChannel];
//	static paratype Conv3_poiW[Conv3_poiKernel*Conv3_poiKernel*Conv3_depChannel*Conv3_poiChannel];
//
//	static paratype Conv4_depW[Conv4_depKernel*Conv4_depKernel*Conv4_depChannel];
//	static paratype Conv4_poiW[Conv4_poiKernel*Conv4_poiKernel*Conv4_depChannel*Conv4_poiChannel];
//
//	static paratype Conv5_depW[Conv5_depKernel*Conv5_depKernel*Conv5_depChannel];
//	static paratype Conv5_poiW[Conv5_poiKernel*Conv5_poiKernel*Conv5_depChannel*Conv5_poiChannel];
//
//	static paratype Conv6_depW[Conv6_depKernel*Conv6_depKernel*Conv6_depChannel];
//	static paratype Conv6_poiW[Conv6_poiKernel*Conv6_poiKernel*Conv6_depChannel*Conv6_poiChannel];
//
//	static paratype Conv7_depW[Conv7_depKernel*Conv7_depKernel*Conv7_depChannel];
//	static paratype Conv7_poiW[Conv7_poiKernel*Conv7_poiKernel*Conv7_depChannel*Conv7_poiChannel];
//
//	static paratype Conv8_depW[Conv8_depKernel*Conv8_depKernel*Conv8_depChannel];
//	static paratype Conv8_poiW[Conv8_poiKernel*Conv8_poiKernel*Conv8_depChannel*Conv8_poiChannel];
//
//	static paratype Conv9_depW[Conv9_depKernel*Conv9_depKernel*Conv9_depChannel];
//	static paratype Conv9_poiW[Conv9_poiKernel*Conv9_poiKernel*Conv9_depChannel*Conv9_poiChannel];
//
//	static paratype Conv10_depW[Conv10_depKernel*Conv10_depKernel*Conv10_depChannel];
//	static paratype Conv10_poiW[Conv10_poiKernel*Conv10_poiKernel*Conv10_depChannel*Conv10_poiChannel];
//
//	static paratype Conv11_depW[Conv11_depKernel*Conv11_depKernel*Conv11_depChannel];
//	static paratype Conv11_poiW[Conv11_poiKernel*Conv11_poiKernel*Conv11_depChannel*Conv11_poiChannel];
//
//	static paratype Conv12_depW[Conv12_depKernel*Conv12_depKernel*Conv12_depChannel];
//	static paratype Conv12_poiW[Conv12_poiKernel*Conv12_poiKernel*Conv12_depChannel*Conv12_poiChannel];
//
//	static paratype Conv13_depW[Conv13_depKernel*Conv13_depKernel*Conv13_depChannel];
//	static paratype Conv13_poiW[Conv13_poiKernel*Conv13_poiKernel*Conv13_depChannel*Conv13_poiChannel];
//
//	static paratype FClayer_W[1024*50];
//	static paratype FClayer_B[50];
//
//	paratype* conv1_result = new paratype[64 * 64 * 32];
//	paratype* dep1_result = new paratype[64 * 64 * 32];
//	paratype* poi1_result = new paratype[64 * 64 * 64];
//
//	paratype* dep2_result = new paratype[Conv2_poioutH * Conv2_poioutH * Conv2_depChannel];
//	paratype* poi2_result = new paratype[Conv2_poioutH * Conv2_poioutH * Conv2_poiChannel];
//
//	paratype* dep3_result = new paratype[Conv3_depoutH * Conv3_depoutH * Conv3_depChannel];
//	paratype* poi3_result = new paratype[Conv3_poioutH * Conv3_poioutH * Conv3_poiChannel];
//
//	paratype* dep4_result = new paratype[Conv4_poioutH * Conv4_poioutH * Conv4_depChannel];
//	paratype* poi4_result = new paratype[Conv4_poioutH * Conv4_poioutH * Conv4_poiChannel];
//
//	paratype* dep5_result = new paratype[Conv5_depoutH * Conv5_depoutH * Conv5_depChannel];
//	paratype* poi5_result = new paratype[Conv5_poioutH * Conv5_poioutH * Conv5_poiChannel];
//
//	paratype* dep6_result = new paratype[Conv6_poioutH * Conv6_poioutH * Conv6_depChannel];
//
//	paratype* dep7_result = new paratype[Conv7_depoutH * Conv7_depoutH * Conv7_depChannel];
//	paratype* poi7_result = new paratype[Conv7_poioutH * Conv7_poioutH * Conv7_poiChannel];
//
//	paratype* poi8_result = new paratype[Conv11_poioutH * Conv11_poioutH * Conv12_depChannel];
//	paratype* dep12_result = new paratype[Conv13_depoutH * Conv13_depoutH * Conv12_depChannel];
//
//	paratype* poi13_result = new paratype[Conv13_poioutH * Conv13_poioutH * Conv13_depChannel];
//	paratype* dep13_result = new paratype[Conv13_depoutH * Conv13_depoutH * Conv13_depChannel];
//
//	paratype* inter_output = new paratype[1024];
//
//	//read in the params read once
//	if(!firstread)
//	{
//		firstread = 1;
//	FILE *fr = fopen(Conv2weightdir, "rb");
//	fread(Conv2W, sizeof(paratype), Conv2Kernel*Conv2Kernel*graphinC*Conv2Channel, fr);
//	fclose(fr);
//
//	// for (int i=0; i<Conv2Kernel*Conv2Kernel*graphinC*Conv2Channel; i++) {
//	// 	printf("%f ", Conv2W[i]);
//	// }
//
//	fr = fopen(dep1weightdir, "rb");
//	fread(Conv1_depW, sizeof(paratype), Conv1_depKernel*Conv1_depKernel*Conv1_depChannel, fr);
//	fclose(fr);
//
//	fr = fopen(poi1weightdir, "rb");
//	fread(Conv1_poiW, sizeof(paratype), Conv1_poiKernel*Conv1_poiKernel*Conv1_depChannel*Conv1_poiChannel, fr);
//	fclose(fr);
//
//	fr = fopen(dep2weightdir, "rb");
//	fread(Conv2_depW, sizeof(paratype), Conv2_depKernel*Conv2_depKernel*Conv2_depChannel, fr);
//	fclose(fr);
//
//	fr = fopen(poi2weightdir, "rb");
//	fread(Conv2_poiW, sizeof(paratype), Conv2_poiKernel*Conv2_poiKernel*Conv2_depChannel*Conv2_poiChannel, fr);
//	fclose(fr);
//
//	fr = fopen(dep3weightdir, "rb");
//	fread(Conv3_depW, sizeof(paratype), Conv3_depKernel*Conv3_depKernel*Conv3_depChannel, fr);
//	fclose(fr);
//
//	fr = fopen(poi3weightdir, "rb");
//	fread(Conv3_poiW, sizeof(paratype), Conv3_poiKernel*Conv3_poiKernel*Conv3_depChannel*Conv3_poiChannel, fr);
//	fclose(fr);
//
//	fr = fopen(dep4weightdir, "rb");
//	fread(Conv4_depW, sizeof(paratype), Conv4_depKernel*Conv4_depKernel*Conv4_depChannel, fr);
//	fclose(fr);
//
//	fr = fopen(poi4weightdir, "rb");
//	fread(Conv4_poiW, sizeof(paratype), Conv4_poiKernel*Conv4_poiKernel*Conv4_depChannel*Conv4_poiChannel, fr);
//	fclose(fr);
//
//	fr = fopen(dep5weightdir, "rb");
//	fread(Conv5_depW, sizeof(paratype), Conv5_depKernel*Conv5_depKernel*Conv5_depChannel, fr);
//	fclose(fr);
//
//	fr = fopen(poi5weightdir, "rb");
//	fread(Conv5_poiW, sizeof(paratype), Conv5_poiKernel*Conv5_poiKernel*Conv5_depChannel*Conv5_poiChannel, fr);
//	fclose(fr);
//
//	fr = fopen(dep6weightdir, "rb");
//	fread(Conv6_depW, sizeof(paratype), Conv6_depKernel*Conv6_depKernel*Conv6_depChannel, fr);
//	fclose(fr);
//
//	fr = fopen(poi6weightdir, "rb");
//	fread(Conv6_poiW, sizeof(paratype), Conv6_poiKernel*Conv6_poiKernel*Conv6_depChannel*Conv6_poiChannel, fr);
//	fclose(fr);
//
//	fr = fopen(dep7weightdir, "rb");
//	fread(Conv7_depW, sizeof(paratype), Conv7_depKernel*Conv7_depKernel*Conv7_depChannel, fr);
//	fclose(fr);
//
//	fr = fopen(poi7weightdir, "rb");
//	fread(Conv7_poiW, sizeof(paratype), Conv7_poiKernel*Conv7_poiKernel*Conv7_depChannel*Conv7_poiChannel, fr);
//	fclose(fr);
//
//	fr = fopen(dep8weightdir, "rb");
//	fread(Conv8_depW, sizeof(paratype), Conv8_depKernel*Conv8_depKernel*Conv8_depChannel, fr);
//	fclose(fr);
//
//	fr = fopen(poi8weightdir, "rb");
//	fread(Conv8_poiW, sizeof(paratype), Conv8_poiKernel*Conv8_poiKernel*Conv8_depChannel*Conv8_poiChannel, fr);
//	fclose(fr);
//
//	fr = fopen(dep9weightdir, "rb");
//	fread(Conv9_depW, sizeof(paratype), Conv9_depKernel*Conv9_depKernel*Conv9_depChannel, fr);
//	fclose(fr);
//
//	fr = fopen(poi9weightdir, "rb");
//	fread(Conv9_poiW, sizeof(paratype), Conv9_poiKernel*Conv9_poiKernel*Conv9_depChannel*Conv9_poiChannel, fr);
//	fclose(fr);
//
//	fr = fopen(dep10weightdir, "rb");
//	fread(Conv10_depW, sizeof(paratype), Conv10_depKernel*Conv10_depKernel*Conv10_depChannel, fr);
//	fclose(fr);
//
//	fr = fopen(poi10weightdir, "rb");
//	fread(Conv10_poiW, sizeof(paratype), Conv10_poiKernel*Conv10_poiKernel*Conv10_depChannel*Conv10_poiChannel, fr);
//	fclose(fr);
//
//	fr = fopen(dep11weightdir, "rb");
//	fread(Conv11_depW, sizeof(paratype), Conv11_depKernel*Conv11_depKernel*Conv11_depChannel, fr);
//	fclose(fr);
//
//	fr = fopen(poi11weightdir, "rb");
//	fread(Conv11_poiW, sizeof(paratype), Conv11_poiKernel*Conv11_poiKernel*Conv11_depChannel*Conv11_poiChannel, fr);
//	fclose(fr);
//
//	fr = fopen(dep12weightdir, "rb");
//	fread(Conv12_depW, sizeof(paratype), Conv12_depKernel*Conv12_depKernel*Conv12_depChannel, fr);
//	fclose(fr);
//
//	fr = fopen(poi12weightdir, "rb");
//	fread(Conv12_poiW, sizeof(paratype), Conv12_poiKernel*Conv12_poiKernel*Conv12_depChannel*Conv12_poiChannel, fr);
//	fclose(fr);
//
//	fr = fopen(dep13weightdir, "rb");
//	fread(Conv13_depW, sizeof(paratype), Conv13_depKernel*Conv13_depKernel*Conv13_depChannel, fr);
//	fclose(fr);
//
//	fr = fopen(poi13weightdir, "rb");
//	fread(Conv13_poiW, sizeof(paratype), Conv13_poiKernel*Conv13_poiKernel*Conv13_depChannel*Conv13_poiChannel, fr);
//	fclose(fr);
//
//	fr = fopen(fcweightdir, "rb");
//	fread(FClayer_W, sizeof(paratype), 1024*50, fr);
//	fclose(fr);
//
//	// fr = fopen(fcbiasdir, "rb");
//	// fread(FClayer_B, sizeof(paratype), 50, fr);
//	// fclose(fr);
//    }
//
//    //prelayers
//	conv1(graphin, Conv2W, conv1_result);
//
//	// /**stage1**/
//	dwconv_3x3_pooling(conv1_result, Conv1_depW, dep1_result, Conv1_depChannel, Conv1_depoutH);
//	pwconv_1x1(dep1_result, Conv1_poiW, poi1_result, Conv1_depChannel, Conv1_poiChannel, Conv1_depoutH);
//	delete[] conv1_result;
//	delete[] dep1_result;
//
//	// for (int i=0; i<Conv1_depoutH*Conv1_depoutH*Conv1_poiChannel; i++) {
//	// 	printf("%f ", poi1_result[i]);
//	// }
//
//	// /**stage2**/
//	dwconv_3x3_pooling_stride(poi1_result, Conv2_depW, dep2_result, Conv2_depChannel, Conv2_depoutH);
//	delete[] poi1_result;
//	pwconv_1x1(dep2_result, Conv2_poiW, poi2_result, Conv2_depChannel, Conv2_poiChannel, Conv2_poioutH);
//	delete[] dep2_result;
//
//	dwconv_3x3_pooling(poi2_result, Conv3_depW, dep3_result, Conv3_depChannel, Conv3_depoutH);
//	pwconv_1x1(dep3_result, Conv3_poiW, poi3_result, Conv3_depChannel, Conv3_poiChannel, Conv3_poioutH);
//	delete[] poi2_result;
//	delete[] dep3_result;
//
//	// // /**stage3**/
//	dwconv_3x3_pooling_stride(poi3_result, Conv4_depW, dep4_result, Conv4_depChannel, Conv4_depoutH);
//	pwconv_1x1(dep4_result, Conv4_poiW, poi4_result, Conv4_depChannel, Conv4_poiChannel, Conv4_poioutH);
//	delete[] poi3_result;
//	delete[] dep4_result;
//
//	dwconv_3x3_pooling(poi4_result, Conv5_depW, dep5_result, Conv5_depChannel, Conv5_depoutH);
//	pwconv_1x1(dep5_result, Conv5_poiW, poi5_result, Conv5_depChannel, Conv5_poiChannel, Conv5_poioutH);
//	delete[] poi4_result;
//	delete[] dep5_result;
//
//	// // /**stage4**/
//	// //unit1
//	dwconv_3x3_pooling_stride(poi5_result, Conv6_depW, dep6_result, Conv6_depChannel, Conv6_depoutH);
//	pwconv_1x1(dep6_result, Conv6_poiW, poi7_result, Conv6_depChannel, Conv6_poiChannel, Conv6_poioutH);
//	delete[] poi5_result;
//	delete[] dep6_result;
//
//    // //unit2
//	dwconv_3x3_pooling(poi7_result, Conv7_depW, dep7_result, Conv7_depChannel, Conv7_depoutH);
//	pwconv_1x1(dep7_result, Conv7_poiW, poi7_result, Conv7_depChannel, Conv7_poiChannel, Conv7_poioutH);
//
//    // //unit3
//	dwconv_3x3_pooling(poi7_result, Conv8_depW, dep7_result, Conv8_depChannel, Conv8_depoutH);
//	pwconv_1x1(dep7_result, Conv8_poiW, poi7_result, Conv8_depChannel, Conv8_poiChannel, Conv8_poioutH);
//
//    // //unit4
//	dwconv_3x3_pooling(poi7_result, Conv9_depW, dep7_result, Conv9_depChannel, Conv9_depoutH);
//	pwconv_1x1(dep7_result, Conv9_poiW, poi7_result, Conv9_depChannel, Conv9_poiChannel, Conv9_poioutH);
//
//    // //unit5
//	dwconv_3x3_pooling(poi7_result, Conv10_depW, dep7_result, Conv10_depChannel, Conv10_depoutH);
//	pwconv_1x1(dep7_result, Conv10_poiW, poi7_result, Conv10_depChannel, Conv10_poiChannel, Conv10_poioutH);
//
//    // // //unit6
//	dwconv_3x3_pooling(poi7_result, Conv11_depW, dep7_result, Conv11_depChannel, Conv11_depoutH);
//	pwconv_1x1(dep7_result, Conv11_poiW, poi8_result, Conv11_depChannel, Conv11_poiChannel, Conv11_poioutH);
//	delete[] poi7_result;
//	delete[] dep7_result;
//
//	// // /**stage5**/
//	// //unit1
//	dwconv_3x3_pooling_stride(poi8_result, Conv12_depW, dep12_result, Conv12_depChannel, Conv12_depoutH);
//	pwconv_1x1(dep12_result, Conv12_poiW, poi13_result, Conv12_depChannel, Conv12_poiChannel, Conv12_poioutH);
//	delete[] poi8_result;
//	delete[] dep12_result;
//
//    // //unit2
//	dwconv_3x3_pooling(poi13_result, Conv13_depW, dep13_result, Conv13_depChannel, Conv13_depoutH);
//	pwconv_1x1(dep13_result, Conv13_poiW, poi13_result, Conv13_depChannel, Conv13_poiChannel, Conv13_poioutH);
//	delete[] dep13_result;
//
//    // // for (int i = 0; i<Conv13_depoutH*Conv13_depoutH*Conv13_depChannel; i++) {
//	// // 	printf("%f", poi13_result[i]);
//	// // }
//
//
//	average_pooling(poi13_result, inter_output);
//	delete[] poi13_result;
//
//	// for (int i=0; i<1000; i++) {
//	// 	printf("%f", FClayer_W[i]);
//	// }
//	fc_layer(inter_output, FClayer_W,  graphout);
//
//
//}
//
//
//int main()
//{
//	paratype* graphin, * graphout,*graphtest,
//		*heatmap,*offset2, *displacement_fwd, *displacement_bwd,
//		*heatmap_2_weight, *heatmap_2_bias,
//		*offset_2_weight, *offset_2_bias, *displacement_fwd_weight, *displacement_fwd_bias, *displacement_bwd_weight, *displacement_bwd_bias;
//
//        graphin = new paratype[128 * 128 * 3];
//		for(int i=0; i<128*128*3; i++) {
//			graphin[i] = i * 1000;
//		}
//
//        graphout = new paratype[50];
//        read_module(graphin, graphout);
//}

// using namespace std;
// /**prelayers**/
// void conv1(float input[1][3][128][128],
// 	float weight[32][3][3][3],
// 	float output[1][32][64][64])
// {
// 	for(int co = 0;co<32;co++){
// 		for(int h = 0;h<64;h++){
// 			for(int w = 0;w<64;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<3;ci++){
// 					for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][ci][m][n] * ((2*h+m < 128 && 2*w+n < 128) ? input[0][ci][2*h+m][2*w+n]:0);
// 						}
// 					}
// 				}
// 				output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**stage1**/
// void layer2_dwconv_3x3_pooling(float input[1][32][64][64],
// 	float weight[32][1][3][3],
// 	float output[1][32][64][64]){
// 	for(int co = 0;co<32;co++){
// 		for(int h = 0;h<64;h++){
// 			for(int w = 0;w<64;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * (( h*2+m-1 >= 0 && w*2+n-1 >= 0 && h*2+m-1 < 64 && w*2+n-1 < 64) ?input[0][co][h*2+m-1][w*2+n-1]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }


// void layer3_pwconv_1x1(float input[1][32][64][64],
// 	float weight[64][32][1][1],
// 	float output[1][64][64][64]){
// 	for(int co = 0;co<64;co++){
// 		for(int h = 0;h<64;h++){
// 			for(int w = 0;w<64;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<32;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}

// }

// /**stage2**/
// void layer4_dwconv_3x3_pooling_stride(float input[1][64][64][64],
// 	float weight[64][1][3][3],
// 	float output[1][64][32][32]){
// 	for(int co = 0;co<64;co++){
// 		for(int h = 0;h<32;h++){
// 			for(int w = 0;w<32;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * ((h*2+m < 32 && w*2+n < 32) ?input[0][co][h*2+m][w*2+n]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer5_pwconv_1x1(float input[1][64][32][32],
// 	float weight[128][64][1][1],
// 	float output[1][128][32][32]){
// 	for(int co = 0;co<128;co++){
// 		for(int h = 0;h<32;h++){
// 			for(int w = 0;w<32;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<64;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// void layer6_dwconv_3x3_pooling(float input[1][128][64][64],
// 	float weight[128][1][3][3],
// 	float output[1][128][64][64]){
// 	for(int co = 0;co<128;co++){
// 		for(int h = 0;h<64;h++){
// 			for(int w = 0;w<64;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * (( h*2+m-1 >= 0 && w*2+n-1 >= 0 && h*2+m-1 < 64 && w*2+n-1 < 64) ?input[0][co][h*2+m-1][w*2+n-1]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer7_pwconv_1x1(float input[1][128][32][32],
// 	float weight[128][128][1][1],
// 	float output[1][128][32][32]){
// 	for(int co = 0;co<128;co++){
// 		for(int h = 0;h<32;h++){
// 			for(int w = 0;w<32;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<128;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**stage3**/
// void layer8_dwconv_3x3_pooling_stride(float input[1][128][32][32],
// 	float weight[128][1][3][3],
// 	float output[1][128][16][16]){
// 	for(int co = 0;co<128;co++){
// 		for(int h = 0;h<16;h++){
// 			for(int w = 0;w<16;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * ((h*2+m < 16 && w*2+n < 16) ?input[0][co][h*2+m][w*2+n]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer9_pwconv_1x1(float input[1][128][16][16],
// 	float weight[256][128][1][1],
// 	float output[1][256][16][16]){
// 	for(int co = 0;co<256;co++){
// 		for(int h = 0;h<16;h++){
// 			for(int w = 0;w<16;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<128;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// void layer10_dwconv_3x3_pooling(float input[1][256][16][16],
// 	float weight[256][1][3][3],
// 	float output[1][256][16][16]){
// 	for(int co = 0;co<256;co++){
// 		for(int h = 0;h<16;h++){
// 			for(int w = 0;w<16;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * (( h*2+m-1 >= 0 && w*2+n-1 >= 0 && h*2+m-1 <16 && w*2+n-1 < 16) ?input[0][co][h*2+m-1][w*2+n-1]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer11_pwconv_1x1(float input[1][256][16][16],
// 	float weight[256][256][1][1],
// 	float output[1][256][16][16]){
// 	for(int co = 0;co<256;co++){
// 		for(int h = 0;h<16;h++){
// 			for(int w = 0;w<16;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<256;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**stage4**/
// /**unit1**/
// void layer12_dwconv_3x3_pooling_stride(float input[1][256][16][16],
// 	float weight[256][1][3][3],
// 	float output[1][256][8][8]){
// 	for(int co = 0;co<256;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * ((h*2+m < 8 && w*2+n < 8) ?input[0][co][h*2+m][w*2+n]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer13_pwconv_1x1(float input[1][256][8][8],
// 	float weight[512][256][1][1],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<256;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**unit2**/
// void layer14_dwconv_3x3_pooling(float input[1][512][8][8],
// 	float weight[512][1][3][3],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * (( h*2+m-1 >= 0 && w*2+n-1 >= 0 && h*2+m-1 <8 && w*2+n-1 < 8) ?input[0][co][h*2+m-1][w*2+n-1]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer15_pwconv_1x1(float input[1][512][8][8],
// 	float weight[512][512][1][1],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<512;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**unit3**/
// void layer16_dwconv_3x3_pooling_stride(float input[1][512][8][8],
// 	float weight[512][1][3][3],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * ((h*2+m < 8 && w*2+n < 8) ?input[0][co][h*2+m][w*2+n]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer17_pwconv_1x1(float input[1][512][8][8],
// 	float weight[512][512][1][1],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<512;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**unit4**/
// void layer18_dwconv_3x3_pooling(float input[1][512][8][8],
// 	float weight[512][1][3][3],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * (( h*2+m-1 >= 0 && w*2+n-1 >= 0 && h*2+m-1 <8 && w*2+n-1 < 8) ?input[0][co][h*2+m-1][w*2+n-1]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer19_pwconv_1x1(float input[1][512][8][8],
// 	float weight[512][512][1][1],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<512;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**unit5**/
// void layer20_dwconv_3x3_pooling(float input[1][512][8][8],
// 	float weight[512][1][3][3],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * (( h*2+m-1 >= 0 && w*2+n-1 >= 0 && h*2+m-1 <8 && w*2+n-1 < 8) ?input[0][co][h*2+m-1][w*2+n-1]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer21_pwconv_1x1(float input[1][512][8][8],
// 	float weight[512][512][1][1],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<512;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**unit6**/
// void layer22_dwconv_3x3_pooling(float input[1][512][8][8],
// 	float weight[512][1][3][3],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * (( h*2+m-1 >= 0 && w*2+n-1 >= 0 && h*2+m-1 <8 && w*2+n-1 < 8) ?input[0][co][h*2+m-1][w*2+n-1]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer23_pwconv_1x1(float input[1][512][8][8],
// 	float weight[512][512][1][1],
// 	float output[1][512][8][8]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<512;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**stage5**/
// /**unit1**/
// void layer24_dwconv_3x3_pooling_stride(float input[1][512][8][8],
// 	float weight[512][1][3][3],
// 	float output[1][512][4][4]){
// 	for(int co = 0;co<512;co++){
// 		for(int h = 0;h<8;h++){
// 			for(int w = 0;w<8;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * ((h*2+m < 4 && w*2+n < 4) ?input[0][co][h*2+m][w*2+n]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer25_pwconv_1x1(float input[1][512][4][4],
// 	float weight[1024][512][1][1],
// 	float output[1][1024][4][4]){
// 	for(int co = 0;co<1024;co++){
// 		for(int h = 0;h<4;h++){
// 			for(int w = 0;w<4;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<512;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// /**unit2**/
// void layer26_dwconv_3x3_pooling(float input[1][1024][4][4],
// 	float weight[1024][1][3][3],
// 	float output[1][1024][4][4]){
// 	for(int co = 0;co<1024;co++){
// 		for(int h = 0;h<4;h++){
// 			for(int w = 0;w<4;w++){
// 				float sum = 0;
// 				for(int m = 0;m<3;m++){
// 						for(int n = 0;n<3;n++){
// 							sum += weight[co][0][m][n] * (( h*2+m-1 >= 0 && w*2+n-1 >= 0 && h*2+m-1 < 4 && w*2+n-1 < 4) ?input[0][co][h*2+m-1][w*2+n-1]:0);
// 						}
// 					}
// 				output[0][co][h][w] = sum;
// 			}
// 		}
// 	}
// }

// void layer27_pwconv_1x1(float input[1][1024][4][4],
// 	float weight[1024][1024][1][1],
// 	float output[1][1024][4][4]){
// 	for(int co = 0;co<1024;co++){
// 		for(int h = 0;h<4;h++){
// 			for(int w = 0;w<4;w++){
// 				float sum = 0;
// 				for(int ci = 0;ci<1024;ci++){
// 					sum += weight[co][ci][0][0]*input[0][ci][h][w];
// 				}
//                 output[0][co][h][w] = (sum > 0)? sum : 0.0f;
// 			}
// 		}
// 	}
// }

// void layer28_average_pooling(float input[1][1024][4][4],
// 	float output[1024]){
// 	for(int co=0; co<1024; co++){
// 		float sum = 0;
// 		for (int h=0; h<4; h++) {
// 			for (int w=0; w<4; w++) {
// 				sum += input[1][co][h][w];
// 			}
// 		}
// 		output[co] = sum / 16;
// 	}
// }

// void layer29_fclayer(float input[1024],
// 	float weight[50][1024],
// 	float bias[50],
// 	float output[50]){
// 	for (int  i=0; i<50; i++) {
// 		float temp = 0;
// 		for (int j=0; j<1024; j++) {
// 			temp += input[j]*weight[i][j];
// 		}
// 		output[i] = temp + bias[i];
// 	}
// }




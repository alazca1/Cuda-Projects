#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 640
#define HEIGHT 480

int write_bmp(const char *filename, int width, int height, char *rgb);

__device__ void matrix_vector_mult(float *matrix, float *vector, float *vector_out) {

	vector_out[0] = matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2];
	vector_out[1] = matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2];
	vector_out[2] = matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2];
	
}

__global__ void traslacion(unsigned char *image, int h, int w) {

	int x_min =  450; 
	int x_max =  550; 
	int y_min =  200; 
	int y_max =  300; 
	int t_x =   -200; 
	int t_y =   100; 

	int posx = threadIdx.x;
	int posy = blockIdx.x;

	float matrix_tx[9] = {1, 0, t_x, 0, 1, t_y, 0, 0, 1};
	float vector[3] = {posx, posy, 1};

	if (posx <= x_max && posx >= x_min && posy <= y_max && posy >= y_min) {

		float vec_out[3];
		matrix_vector_mult(matrix_tx, vector, vec_out);
		
		int posx_out = vec_out[0];
		int posy_out = vec_out[1];

		image[(posx_out + posy_out * w) * 3 + 0] = image[(posx + posy * w) * 3 + 0];      //R
		image[(posx_out + posy_out * w) * 3 + 1] = image[(posx + posy * w) * 3 + 1];	  //G
		image[(posx_out + posy_out * w) * 3 + 2] = image[(posx + posy * w) * 3 + 2];	  //B
	}
}

__global__ void traslacion_rotacion(unsigned char* image, int h, int w) {

	int x_min = 450;
	int x_max = 550;
	int y_min = 200;
	int y_max = 300;
	//int t_x_1 = -posx - ((x_max - x_min) / 2);
	//int t_y_1 = -posy - ((y_max - y_min) / 2);
	//int t_x_2 = posx + ((x_max - x_min) / 2);
	//int t_y_2 = posy + ((y_max - y_min) / 2);
	//int angulo_giro = 45;

	int posx = threadIdx.x;
	int posy = blockIdx.x;

	float matrix_tx[9] = {0.5f, 0.f, 0.f, 0.5f, 1.f, 0.f, 0.f, 0.f, 1.f};
	float vector[3] = { posx, posy, 1 };

	if (posx <= x_max && posx >= x_min && posy <= y_max && posy >= y_min) {

		float vec_out[3];
		matrix_vector_mult(matrix_tx, vector, vec_out);

		int posx_out = vec_out[0];
		int posy_out = vec_out[1];

		image[(posx_out + posy_out * w) * 3 + 0] = image[(posx + posy * w) * 3 + 0];      //R
		image[(posx_out + posy_out * w) * 3 + 1] = image[(posx + posy * w) * 3 + 1];	  //G
		image[(posx_out + posy_out * w) * 3 + 2] = image[(posx + posy * w) * 3 + 2];	  //B
	}
}

__global__ void generaRGB(unsigned char *image, int h, int w) {
	int posx = threadIdx.x;
	int posy = blockIdx.x;

	image[(posx + posy * w) * 3 + 0] = posy;  //R
	image[(posx + posy * w) * 3 + 1] = posx;  //G
	image[(posx + posy * w) * 3 + 2] = 100;   //B
}

int main()
{
	const char *filename = "imagen_001.bmp";
	const int width = WIDTH;
	const int height = HEIGHT;
	unsigned char rgb[height*width * 3];
		

	/// Alojamiento GPU
	unsigned char *dev_rgb = 0;
	cudaMalloc((void**)&dev_rgb, width * height * 3 * sizeof(unsigned char));

	/// Copia datos CPU a GPU

	/// Kernel
	generaRGB << <height, width >> > (dev_rgb, height, width);

	//traslacion << <height, width >> > (dev_rgb, height, width);

	traslacion_rotacion << <height, width >> > (dev_rgb, height, width);

	/// Copia resultado GPU a CPU
	cudaMemcpy(rgb, dev_rgb, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	/// liberación memoria GPU 
	cudaFree(dev_rgb);
	
	/// escribe imagen
	write_bmp(filename, width, height, (char*)rgb);

	return 0;
}

struct BMPHeader
{
	char bfType[2];       /* "BM" */
	int bfSize;           /* Size of file in bytes */
	int bfReserved;       /* set to 0 */
	int bfOffBits;        /* Byte offset to actual bitmap data (= 54) */
	int biSize;           /* Size of BITMAPINFOHEADER, in bytes (= 40) */
	int biWidth;          /* Width of image, in pixels */
	int biHeight;         /* Height of images, in pixels */
	short biPlanes;       /* Number of planes in target device (set to 1) */
	short biBitCount;     /* Bits per pixel (24 in this case) */
	int biCompression;    /* Type of compression (0 if no compression) */
	int biSizeImage;      /* Image size, in bytes (0 if no compression) */
	int biXPelsPerMeter;  /* Resolution in pixels/meter of display device */
	int biYPelsPerMeter;  /* Resolution in pixels/meter of display device */
	int biClrUsed;        /* Number of colors in the color table (if 0, use
						  maximum allowed by biBitCount) */
	int biClrImportant;   /* Number of important colors.  If 0, all colors
						  are important */
};


int write_bmp(const char *filename, int width, int height, char *rgb)
{
	int i, j, ipos;
	int bytesPerLine;
	char *line;

	FILE *file;
	struct BMPHeader bmph;

	/* The length of each line must be a multiple of 4 bytes */

	bytesPerLine = (3 * (width + 1) / 4) * 4;

	strcpy(bmph.bfType, "BM");
	bmph.bfOffBits = 54;
	bmph.bfSize = bmph.bfOffBits + bytesPerLine * height;
	bmph.bfReserved = 0;
	bmph.biSize = 40;
	bmph.biWidth = width;
	bmph.biHeight = height;
	bmph.biPlanes = 1;
	bmph.biBitCount = 24;
	bmph.biCompression = 0;
	bmph.biSizeImage = bytesPerLine * height;
	bmph.biXPelsPerMeter = 0;
	bmph.biYPelsPerMeter = 0;
	bmph.biClrUsed = 0;
	bmph.biClrImportant = 0;

	file = fopen(filename, "wb");
	if (file == NULL) return(0);

	fwrite(&bmph.bfType, 2, 1, file);
	fwrite(&bmph.bfSize, 4, 1, file);
	fwrite(&bmph.bfReserved, 4, 1, file);
	fwrite(&bmph.bfOffBits, 4, 1, file);
	fwrite(&bmph.biSize, 4, 1, file);
	fwrite(&bmph.biWidth, 4, 1, file);
	fwrite(&bmph.biHeight, 4, 1, file);
	fwrite(&bmph.biPlanes, 2, 1, file);
	fwrite(&bmph.biBitCount, 2, 1, file);
	fwrite(&bmph.biCompression, 4, 1, file);
	fwrite(&bmph.biSizeImage, 4, 1, file);
	fwrite(&bmph.biXPelsPerMeter, 4, 1, file);
	fwrite(&bmph.biYPelsPerMeter, 4, 1, file);
	fwrite(&bmph.biClrUsed, 4, 1, file);
	fwrite(&bmph.biClrImportant, 4, 1, file);

	line = (char*)malloc(bytesPerLine);

	for (i = height - 1; i >= 0; i--)
	{
		for (j = 0; j < width; j++)
		{
			ipos = 3 * (width * i + j);
			line[3 * j] = rgb[ipos + 2];
			line[3 * j + 1] = rgb[ipos + 1];
			line[3 * j + 2] = rgb[ipos];
		}
		fwrite(line, bytesPerLine, 1, file);
	}

	free(line);
	fclose(file);

	return(1);
}

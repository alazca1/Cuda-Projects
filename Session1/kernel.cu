/**
 * EEAV 2022
 *
 * Template for Sesion 01
 *
 * @author Miguel Rodrigo-Bort
 * @author Antonio Martínez
 * @author Santiago Jiménez-Serrano
 */


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <string.h>

 // Image size
#define HEIGHT 480
#define WIDTH  640

// Types definitions
typedef unsigned char byte; // Number in range [0-255]


///////////////////////////////////////////////////////////////////////////////////////////////////
// Utilities
///////////////////////////////////////////////////////////////////////////////////////////////////

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
	int biClrUsed;        /* Number of colors in the color table (if 0, use maximum allowed by biBitCount) */
	int biClrImportant;   /* Number of important colors.  If 0, all colors are important */
};

int write_bmp(const char* filename, int width, int height, char* rgb)
{
	int i, j, ipos;
	int bytesPerLine;
	char* line;

	FILE* file;
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
	if (file == NULL)
		return 0;

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

	return 1;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Gray-Scale image (CPU)
///////////////////////////////////////////////////////////////////////////////////////////////////

void sample_01_CPU()
{
	const char* filename = "imagen_001_CPU.bmp";

	const int w = WIDTH;
	const int h = HEIGHT;
	byte* rgb = new byte[h * w * 3];

	int ipos;
	for (int j = 0; j < h; j++)
	{
		for (int i = 0; i < w; i++)
		{
			int Gray = (i + j) * 255 / (WIDTH + HEIGHT);
			ipos = 3 * (w * j + i);

			rgb[ipos] = Gray;   //R
			rgb[ipos + 1] = Gray;	//G		
			rgb[ipos + 2] = Gray;   //B	
		}
	}

	write_bmp(filename, w, h, (char*)rgb);
	delete[] rgb;
	printf("sample_01_CPU done... --> %s \n", filename);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Student code
///////////////////////////////////////////////////////////////////////////////////////////////////

///// Put your code in this section ... //////

__global__ void gray_scale_kernel(byte* rgb)
{
	int i = (threadIdx.x + blockIdx.x * blockDim.x) * 3;
	int gray;
	gray = (threadIdx.x + blockIdx.x) * 255 / (WIDTH + HEIGHT);
	rgb[i] = gray;
	rgb[i + 1] = gray;
	rgb[i + 2] = gray;

}

__global__ void rgb_scale_kernel(byte* rgb, int size_width, int size_height)
{
	int i = (threadIdx.x + blockIdx.x * blockDim.x) * 3;
	int r, g, b;
	r = threadIdx.x * 255 / size_width;
	g = blockIdx.x * 255 / size_height;
	b = 100;
	rgb[i] = r;
	rgb[i + 1] = g;
	rgb[i + 2] = b;
}

void sample_gpu()
{
	const char* filename = "imagen_001_GPU.bmp";

	const int w = WIDTH;
	const int h = HEIGHT;
	const int total_rgb_pixel = h * w * 3;
	const int size_width = 300;
	const int size_height = 100;
	int N_bloques = h;
	int N_threads = w;
	byte* rgb = new byte[h * w * 3];

	byte* dev_rgb = 0;
	cudaMalloc((void**)&dev_rgb, total_rgb_pixel * sizeof(byte));
	rgb_scale_kernel << <N_bloques, N_threads >> > (dev_rgb, size_width, size_height);
	cudaMemcpy(rgb, dev_rgb, total_rgb_pixel * sizeof(byte), cudaMemcpyDeviceToHost);

	write_bmp(filename, w, h, (char*)rgb);
	delete[] rgb;
	printf("sample_01_GPU done... --> %s \n", filename);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Main entry point
///////////////////////////////////////////////////////////////////////////////////////////////////

int main()
{
	// CPU sample function calling
	sample_01_CPU();

	// Call your functions here...
	// -->
	sample_gpu();
	///////////////////////////////

	return EXIT_SUCCESS;
}

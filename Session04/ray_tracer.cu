
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include <math.h> 
#include <windows.h>
#include <stdlib.h>
#include <sstream>
#include <string>

#define N_MAX 1078
#define BMP_HEADER_SIZE 1078

#define IM_WIDTH 640
#define IM_HEIGHT 480

void read_obj(const char *filename, float **vertex, float **normals, float **color, int *n_vertex, int **faces, int *n_faces);

int write_bmp(const char *filename, int width, int height, char *rgb);

/////////// utilities CPU

void unitary_vector(float *vec) {

	float norm = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
	vec[0] = vec[0] / norm;
	vec[1] = vec[1] / norm;
	vec[2] = vec[2] / norm;

}

void cross_product(float *res, float *a, float *b) {

	res[0] = a[1] * b[2] - a[2] * b[1];
	res[1] = -(a[0] * b[2] - a[2] * b[0]);
	res[2] = a[0] * b[1] - a[1] * b[0];

}

float norm_vector(float *vec) {

	float sum = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
	return sqrt(sum);

}

//////// utilities GPU

__device__ void unitary_vector_d(float *vec) {

	float norm = sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
	vec[0] = vec[0] / norm;
	vec[1] = vec[1] / norm;
	vec[2] = vec[2] / norm;

}

__device__ void cross_product_d(float *res, float *a, float *b) {

	res[0] = a[1] * b[2] - a[2] * b[1];
	res[1] = a[2] * b[0] - a[0] * b[2];
	res[2] = a[0] * b[1] - a[1] * b[0];

}

__device__ float dot_product_d(float *a, float *b) {

	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ float norm_vector_d(float* vec) {

	float sum = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
	return sqrt(sum);

}

__device__ void substract_vectors_d(float* res, float* a, float* b)
{
	res[0] = b[0] - a[0];
	res[1] = b[1] - a[1];
	res[2] = b[2] - a[2];
}
/////// ray_tracer GPU

__device__ void pixel_vertex_colors(float *out, float *view_point, float *dir_vector, int n_vertex, int n_faces, float *vertex, float *normals, float *color, int *faces) {

	// Completar (Ejercicio 3)
	int n_colisions = 0;

	float R, G, B;

	float distancia[3];
	float dist_min = 1000000;

	for (int n_f = 0; n_f < n_faces; n_f++) {

		float vert1[3];
		float vert2[3];
		float vert3[3];

		/*vert1 = Mod.vertices(Mod.faces(i, 1), :);
		vert2 = Mod.vertices(Mod.faces(i, 2), :);
		vert3 = Mod.vertices(Mod.faces(i, 3), :);*/
		for (int axis = 0; axis < 3; axis++)
		{
			vert1[axis] = vertex[(faces[(n_f * 3) + 0] * 3) + axis];
			vert2[axis] = vertex[(faces[(n_f * 3) + 1] * 3) + axis];
			vert3[axis] = vertex[(faces[(n_f * 3) + 2] * 3) + axis];
		}

		//normal_plane = cross(vert2 - vert1, vert3 - vert1);  (producto vectorial)
		float normal_plane[3];
		float substract_vert2_vert1[3];
		float substract_vert3_vert1[3];

		substract_vectors_d(substract_vert2_vert1, vert1, vert2);
		substract_vectors_d(substract_vert3_vert1, vert1, vert3);
		cross_product_d(normal_plane, substract_vert2_vert1, substract_vert3_vert1);

		//normal_plane = normal_plane. / norm(normal_plane);  (unitary vector)
		unitary_vector_d(normal_plane);

		//distance_plane= -sum( normal_plane .* vert1 ); (producto escalar)
		float distance_plane = -dot_product_d(normal_plane, vert1);

		//t = -(sum(normal_plane.*view_point) + distance_plane) / sum(normal_plane.*dir_vector); (productos escalares)
		float t = -(dot_product_d(normal_plane, view_point) + distance_plane) / dot_product_d(normal_plane, dir_vector);

		//coll_point=view_point + t * dir_vector;
		float coll_point[3];
		coll_point[0] = view_point[0] + (t * dir_vector[0]);
		coll_point[1] = view_point[1] + (t * dir_vector[1]);
		coll_point[2] = view_point[2] + (t * dir_vector[2]);

		//dir_1 = sum(cross(coll_point - vert1, vert2 - vert1).*normal_plane);  (producto escalar y vectorial)
		float dir_1, dir_2, dir_3;
		float alpha, beta, theta, total;
		float cross_product_coll_vert[3];
		float substract_coll_vert1[3];
		float substract_coll_vert2[3];
		float substract_coll_vert3[3];
		float substract_vert1_vert3[3];
		float substract_vert3_vert2[3];
		float cross_product_coll_vert_alpha[3];
		float cross_product_coll_vert_beta[3];
		float cross_product_coll_vert_theta[3];

		substract_vectors_d(substract_coll_vert1, vert1, coll_point);
		substract_vectors_d(substract_vert2_vert1, vert1, vert2);
		cross_product_d(cross_product_coll_vert, substract_coll_vert1, substract_vert2_vert1);
		dir_1 = dot_product_d(cross_product_coll_vert, normal_plane);

		//dir_2 = sum(cross(coll_point - vert2, vert3 - vert2).*normal_plane); (producto escalar y vectorial)
		substract_vectors_d(substract_coll_vert2, vert2, coll_point);
		substract_vectors_d(substract_vert3_vert2, vert2, vert3);
		cross_product_d(cross_product_coll_vert, substract_coll_vert2, substract_vert3_vert2);
		dir_2 = dot_product_d(cross_product_coll_vert, normal_plane);

		//dir_3 = sum(cross(coll_point - vert3, vert1 - vert3).*normal_plane); (producto escalar y vectorial)
		substract_vectors_d(substract_coll_vert3, vert3, coll_point);
		substract_vectors_d(substract_vert1_vert3, vert3, vert1);
		cross_product_d(cross_product_coll_vert, substract_coll_vert3, substract_vert1_vert3);
		dir_3 = dot_product_d(cross_product_coll_vert, normal_plane);

		//alpha = norm(cross(coll_point - vert2, coll_point - vert3) / 2;  (producto escalar y vectorial)
		substract_vectors_d(substract_coll_vert2, coll_point, vert2);
		substract_vectors_d(substract_coll_vert3, coll_point, vert3);
		cross_product_d(cross_product_coll_vert_alpha, substract_coll_vert2, substract_coll_vert3);

		//beta = norm(cross(coll_point - vert1, coll_point - vert3) / 2; (producto escalar y vectorial)
		substract_vectors_d(substract_coll_vert1, coll_point, vert1);
		substract_vectors_d(substract_coll_vert3, coll_point, vert3);
		cross_product_d(cross_product_coll_vert_beta, substract_coll_vert1, substract_coll_vert3);

		//theta = norm(cross(coll_point - vert1, coll_point - vert2) / 2; (producto escalar y vectorial)
		substract_vectors_d(substract_coll_vert1, coll_point, vert1);
		substract_vectors_d(substract_coll_vert2, coll_point, vert2);
		cross_product_d(cross_product_coll_vert_theta, substract_coll_vert1, substract_coll_vert2);

		// if (dir_1>0 & dir_2>0 & dir_3>0) ||  (dir_1<0 & dir_2<0 & dir_3<0), col++
		if ((dir_1 > 0 && dir_2 > 0 && dir_3 > 0) || (dir_1 < 0 && dir_2 < 0 && dir_3 < 0))
		{
			//++n_colisions;
			substract_vectors_d(distancia, view_point, coll_point);
			float dist = norm_vector_d(distancia);

			if (dist < dist_min) {
				dist_min = dist;

				alpha = norm_vector_d(cross_product_coll_vert_alpha) / 2;
				beta = norm_vector_d(cross_product_coll_vert_beta) / 2;
				theta = norm_vector_d(cross_product_coll_vert_theta) / 2;

				total = alpha + beta + theta;

				alpha = alpha / total;
				beta = beta / total;
				theta = theta / total;

				R = ((alpha * color[(faces[(n_f * 3) + 0] * 3) + 0]) + (beta * color[(faces[(n_f * 3) + 1] * 3) + 0]) + (theta * color[(faces[(n_f * 3) + 2] * 3) + 0])) * 255;
				G = ((alpha * color[(faces[(n_f * 3) + 0] * 3) + 1]) + (beta * color[(faces[(n_f * 3) + 1] * 3) + 1]) + (theta * color[(faces[(n_f * 3) + 2] * 3) + 1])) * 255;
				B = ((alpha * color[(faces[(n_f * 3) + 0] * 3) + 2]) + (beta * color[(faces[(n_f * 3) + 1] * 3) + 2]) + (theta * color[(faces[(n_f * 3) + 2] * 3) + 2])) * 255;

				out[0] = R;
				out[1] = G;
				out[2] = B;
			}
		}
	}

	if (dist_min == 1e6) {
		out[0] = 0;
		out[1] = 0;
		out[2] = 0;
	}
}

__device__ void pixel_coords(float *out, float *view_point, float *dir_vector, int n_vertex, int n_faces, float *vertex, float *normals, float *color, int *faces) {
	
	// Completar (Ejercicio 2)
	int n_colisions = 0;

	float R, G, B;

	float distancia[3];
	float dist_min = 1000000;

	for (int n_f = 0; n_f < n_faces; n_f++) {

		float vert1[3];
		float vert2[3];
		float vert3[3];

		/*vert1 = Mod.vertices(Mod.faces(i, 1), :);
		vert2 = Mod.vertices(Mod.faces(i, 2), :);
		vert3 = Mod.vertices(Mod.faces(i, 3), :);*/
		for (int axis = 0; axis < 3; axis++)
		{
			vert1[axis] = vertex[(faces[(n_f * 3) + 0] * 3) + axis];
			vert2[axis] = vertex[(faces[(n_f * 3) + 1] * 3) + axis];
			vert3[axis] = vertex[(faces[(n_f * 3) + 2] * 3) + axis];
		}

		//normal_plane = cross(vert2 - vert1, vert3 - vert1);  (producto vectorial)
		float normal_plane[3];
		float substract_vert2_vert1[3];
		float substract_vert3_vert1[3];

		substract_vectors_d(substract_vert2_vert1, vert1, vert2);
		substract_vectors_d(substract_vert3_vert1, vert1, vert3);
		cross_product_d(normal_plane, substract_vert2_vert1, substract_vert3_vert1);

		//normal_plane = normal_plane. / norm(normal_plane);  (unitary vector)
		unitary_vector_d(normal_plane);

		//distance_plane= -sum( normal_plane .* vert1 ); (producto escalar)
		float distance_plane = -dot_product_d(normal_plane, vert1);

		//t = -(sum(normal_plane.*view_point) + distance_plane) / sum(normal_plane.*dir_vector); (productos escalares)
		float t = -(dot_product_d(normal_plane, view_point) + distance_plane) / dot_product_d(normal_plane, dir_vector);

		//coll_point=view_point + t * dir_vector;
		float coll_point[3];
		coll_point[0] = view_point[0] + (t * dir_vector[0]);
		coll_point[1] = view_point[1] + (t * dir_vector[1]);
		coll_point[2] = view_point[2] + (t * dir_vector[2]);

		//dir_1 = sum(cross(coll_point - vert1, vert2 - vert1).*normal_plane);  (producto escalar y vectorial)
		float dir_1, dir_2, dir_3;
		float alpha, beta, theta, total;
		float cross_product_coll_vert[3];
		float substract_coll_vert1[3];
		float substract_coll_vert2[3];
		float substract_coll_vert3[3];
		float substract_vert1_vert3[3];
		float substract_vert3_vert2[3];
		float cross_product_coll_vert_alpha[3];
		float cross_product_coll_vert_beta[3];
		float cross_product_coll_vert_theta[3];

		substract_vectors_d(substract_coll_vert1, vert1, coll_point);
		substract_vectors_d(substract_vert2_vert1, vert1, vert2);
		cross_product_d(cross_product_coll_vert, substract_coll_vert1, substract_vert2_vert1);
		dir_1 = dot_product_d(cross_product_coll_vert, normal_plane);

		//dir_2 = sum(cross(coll_point - vert2, vert3 - vert2).*normal_plane); (producto escalar y vectorial)
		substract_vectors_d(substract_coll_vert2, vert2, coll_point);
		substract_vectors_d(substract_vert3_vert2, vert2, vert3);
		cross_product_d(cross_product_coll_vert, substract_coll_vert2, substract_vert3_vert2);
		dir_2 = dot_product_d(cross_product_coll_vert, normal_plane);

		//dir_3 = sum(cross(coll_point - vert3, vert1 - vert3).*normal_plane); (producto escalar y vectorial)
		substract_vectors_d(substract_coll_vert3, vert3, coll_point);
		substract_vectors_d(substract_vert1_vert3, vert3, vert1);
		cross_product_d(cross_product_coll_vert, substract_coll_vert3, substract_vert1_vert3);
		dir_3 = dot_product_d(cross_product_coll_vert, normal_plane);

		//alpha = norm(cross(coll_point - vert2, coll_point - vert3) / 2;  (producto escalar y vectorial)
		substract_vectors_d(substract_coll_vert2, coll_point, vert2);
		substract_vectors_d(substract_coll_vert3, coll_point, vert3);
		cross_product_d(cross_product_coll_vert_alpha, substract_coll_vert2, substract_coll_vert3);

		//beta = norm(cross(coll_point - vert1, coll_point - vert3) / 2; (producto escalar y vectorial)
		substract_vectors_d(substract_coll_vert1, coll_point, vert1);
		substract_vectors_d(substract_coll_vert3, coll_point, vert3);
		cross_product_d(cross_product_coll_vert_beta, substract_coll_vert1, substract_coll_vert3);

		//theta = norm(cross(coll_point - vert1, coll_point - vert2) / 2; (producto escalar y vectorial)
		substract_vectors_d(substract_coll_vert1, coll_point, vert1);
		substract_vectors_d(substract_coll_vert2, coll_point, vert2);
		cross_product_d(cross_product_coll_vert_theta, substract_coll_vert1, substract_coll_vert2);

		// if (dir_1>0 & dir_2>0 & dir_3>0) ||  (dir_1<0 & dir_2<0 & dir_3<0), col++
		if ((dir_1 > 0 && dir_2 > 0 && dir_3 > 0) || (dir_1 < 0 && dir_2 < 0 && dir_3 < 0))
		{
			//++n_colisions;
			substract_vectors_d(distancia, view_point, coll_point);
			float dist = norm_vector_d(distancia);

			if (dist < dist_min) {
				dist_min = dist;

				alpha = norm_vector_d(cross_product_coll_vert_alpha) / 2;
				beta = norm_vector_d(cross_product_coll_vert_beta) / 2;
				theta = norm_vector_d(cross_product_coll_vert_theta) / 2;

				total = alpha + beta + theta;

				R = alpha / total * 255;
				G = beta  / total * 255;
				B = theta / total * 255;

				out[0] = R;
				out[1] = G;
				out[2] = B;
			}
		}
	}
}

__device__ void pixel_collision(float *out, float *view_point, float *dir_vector, int n_vertex, int n_faces, float *vertex, float *normals, float *color, int *faces) {
	
	// Completar (Ejercicio 1)
	
	int n_colisions = 0;

	for (int n_f = 0; n_f < n_faces; n_f++) {

		float vert1[3];
		float vert2[3];
		float vert3[3];

		/*vert1 = Mod.vertices(Mod.faces(i, 1), :);
		vert2 = Mod.vertices(Mod.faces(i, 2), :);
		vert3 = Mod.vertices(Mod.faces(i, 3), :);*/
		for (int axis = 0; axis < 3; axis++)
		{
			vert1[axis] = vertex[(faces[(n_f * 3) + 0] * 3) + axis];
			vert2[axis] = vertex[(faces[(n_f * 3) + 1] * 3) + axis];
			vert3[axis] = vertex[(faces[(n_f * 3) + 2] * 3) + axis];
		}

		//normal_plane = cross(vert2 - vert1, vert3 - vert1);  (producto vectorial)
		float normal_plane[3];
		float substract_vert2_vert1[3];
		float substract_vert3_vert1[3];
		
		substract_vectors_d(substract_vert2_vert1, vert1, vert2);
		substract_vectors_d(substract_vert3_vert1, vert1, vert3);
		cross_product_d(normal_plane, substract_vert2_vert1, substract_vert3_vert1);
		
		//normal_plane = normal_plane. / norm(normal_plane);  (unitary vector)
		unitary_vector_d(normal_plane);

		//distance_plane= -sum( normal_plane .* vert1 ); (producto escalar)
		float distance_plane = -dot_product_d(normal_plane, vert1);

		//t = -(sum(normal_plane.*view_point) + distance_plane) / sum(normal_plane.*dir_vector); (productos escalares)
		float t = -(dot_product_d(normal_plane, view_point) + distance_plane) / dot_product_d(normal_plane, dir_vector);
		
		//coll_point=view_point + t * dir_vector;
		float coll_point[3];
		coll_point[0] = view_point[0] + (t * dir_vector[0]);
		coll_point[1] = view_point[1] + (t * dir_vector[1]);
		coll_point[2] = view_point[2] + (t * dir_vector[2]);
		
		//dir_1 = sum(cross(coll_point - vert1, vert2 - vert1).*normal_plane);  (producto escalar y vectorial)
		float dir_1;
		float cross_product_coll_vert[3];
		float substract_coll_vert1[3];

		substract_vectors_d(substract_coll_vert1, vert1, coll_point);
		substract_vectors_d(substract_vert2_vert1, vert1, vert2);
		cross_product_d(cross_product_coll_vert, substract_coll_vert1, substract_vert2_vert1);
		dir_1 = dot_product_d(cross_product_coll_vert, normal_plane);

		//dir_2 = sum(cross(coll_point - vert2, vert3 - vert2).*normal_plane); (producto escalar y vectorial)
		float dir_2;
		float substract_coll_vert2[3];
		float substract_vert3_vert2[3];

		substract_vectors_d(substract_coll_vert2, vert2, coll_point);
		substract_vectors_d(substract_vert3_vert2, vert2, vert3);
		cross_product_d(cross_product_coll_vert, substract_coll_vert2, substract_vert3_vert2);
		dir_2 = dot_product_d(cross_product_coll_vert, normal_plane);

		//dir_3 = sum(cross(coll_point - vert3, vert1 - vert3).*normal_plane); (producto escalar y vectorial)
		float dir_3;
		float substract_coll_vert3[3];
		float substract_vert1_vert3[3];

		substract_vectors_d(substract_coll_vert3, vert3, coll_point);
		substract_vectors_d(substract_vert1_vert3, vert3, vert1);
		cross_product_d(cross_product_coll_vert, substract_coll_vert3, substract_vert1_vert3);
		dir_3 = dot_product_d(cross_product_coll_vert, normal_plane);

		// if (dir_1>0 & dir_2>0 & dir_3>0) ||  (dir_1<0 & dir_2<0 & dir_3<0), col++
		if ((dir_1 > 0 && dir_2 > 0 && dir_3 > 0) || (dir_1 < 0 && dir_2 < 0 && dir_3 < 0))
			++n_colisions;
	}
		   
	// Completar: Pasar de # of colisions [0-2] a grayscale [0-255]
	out[0] = (n_colisions > 0) ? 255 : 0;
	out[1] = (n_colisions > 0) ? 255 : 0;
	out[2] = (n_colisions > 0) ? 255 : 0;
}

__global__ void kernel_raytracer(unsigned char *dev_rgb, int size, float *fov_00_d, float *fov_w0_d, float *fov_0h_d, float *view_point, int n_vertex, int n_faces, float *vertex, float *normals, float *color, int *faces) {
	   	 
	int pos_x = threadIdx.x + (blockDim.x * blockIdx.x);
	int pos_y = threadIdx.y + (blockDim.y * blockIdx.y);

	if (pos_x < IM_WIDTH && pos_y < IM_HEIGHT) {
		
		float pos_screen[3];
		pos_screen[0] = fov_00_d[0] + (fov_w0_d[0] - fov_00_d[0])*pos_x / (IM_WIDTH - 1) + (fov_0h_d[0] - fov_00_d[0])*pos_y / (IM_HEIGHT - 1);
		pos_screen[1] = fov_00_d[1] + (fov_w0_d[1] - fov_00_d[1])*pos_x / (IM_WIDTH - 1) + (fov_0h_d[1] - fov_00_d[1])*pos_y / (IM_HEIGHT - 1);
		pos_screen[2] = fov_00_d[2] + (fov_w0_d[2] - fov_00_d[2])*pos_x / (IM_WIDTH - 1) + (fov_0h_d[2] - fov_00_d[2])*pos_y / (IM_HEIGHT - 1);

		float dir_vector[3];
		dir_vector[0] = pos_screen[0] - view_point[0];
		dir_vector[1] = pos_screen[1] - view_point[1];
		dir_vector[2] = pos_screen[2] - view_point[2];

		float out[3];
		//pixel_collision(out, view_point, dir_vector, n_vertex, n_faces, vertex, normals, color, faces);  //Ejercicio 1
		pixel_coords(out, view_point, dir_vector, n_vertex, n_faces, vertex, normals, color, faces);  //Ejercicio 2
		//pixel_vertex_colors(out, view_point, dir_vector, n_vertex, n_faces, vertex, normals, color, faces);  //Ejercicio 3

		/////////

		dev_rgb[(pos_x + IM_WIDTH * pos_y) * 3] = out[0];
		dev_rgb[(pos_x + IM_WIDTH * pos_y) * 3 + 1] = out[1];
		dev_rgb[(pos_x + IM_WIDTH * pos_y) * 3 + 2] = out[2];
	}
}

void ray_tracer(int w, int h, char *rgb, int n_vertex, int n_faces, float *vertex, float *normals, float *color, int *faces) {
	
	float fov_00[3] = { -80, 20, 15 };	
	float fov_w0[3] = { -80, -15, 15 };
	float fov_0h[3] = { -80, 20, -15 };

	float center_screen[3];
	center_screen[0] = fov_00[0] + 0.5 * (fov_w0[0] - fov_00[0]) + 0.5 * (fov_0h[0] - fov_00[0]);
	center_screen[1] = fov_00[1] + 0.5 * (fov_w0[1] - fov_00[1]) + 0.5 * (fov_0h[1] - fov_00[1]);
	center_screen[2] = fov_00[2] + 0.5 * (fov_w0[2] - fov_00[2]) + 0.5 * (fov_0h[2] - fov_00[2]);

	float vec_h[3];
	vec_h[0] = fov_0h[0] - fov_00[0];
	vec_h[1] = fov_0h[1] - fov_00[1];
	vec_h[2] = fov_0h[2] - fov_00[2];

	float vec_w[3];
	vec_w[0] = fov_w0[0] - fov_00[0]; 
	vec_w[1] = fov_w0[1] - fov_00[1]; 
	vec_w[2] = fov_w0[2] - fov_00[2];
	
	float normal_screen[3];
	cross_product(normal_screen, vec_h, vec_w);
	unitary_vector(normal_screen);

	float view_point[3];
	view_point[0] = center_screen[0] + normal_screen[0] * 60;
	view_point[1] = center_screen[1] + normal_screen[1] * 60;
	view_point[2] = center_screen[2] + normal_screen[2] * 60;

	//////////// copio variables de vista

	unsigned char *dev_rgb;
	int size = w * h;

	float *fov_00_d;
	float *fov_w0_d;
	float *fov_0h_d;
	float *view_point_d;

	cudaMalloc((void**)&dev_rgb, size * 3 * sizeof(char));
	cudaMalloc((void**)&fov_00_d, 3 * sizeof(float));
	cudaMalloc((void**)&fov_w0_d, 3 * sizeof(float));
	cudaMalloc((void**)&fov_0h_d, 3 * sizeof(float));
	cudaMalloc((void**)&view_point_d, 3 * sizeof(float));

	cudaMemcpy(fov_00_d, fov_00, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fov_w0_d, fov_w0, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fov_0h_d, fov_0h, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(view_point_d, view_point, 3 * sizeof(float), cudaMemcpyHostToDevice);

	/////// copio variables de modelo

	float *vertex_d;
	float *normals_d;
	float *color_d;
	int *faces_d;
	
	cudaMalloc((void**)&vertex_d, 3 * n_vertex * sizeof(float));
	cudaMalloc((void**)&normals_d, 3 * n_vertex * sizeof(float));
	cudaMalloc((void**)&color_d, 3 * n_vertex * sizeof(float));
	cudaMalloc((void**)&faces_d, 3 * n_faces * sizeof(int));
	
	cudaMemcpy(vertex_d, vertex, 3 * n_vertex * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(normals_d, normals, 3 * n_vertex * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(color_d, color, 3 * n_vertex * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(faces_d, faces, 3 * n_faces * sizeof(int), cudaMemcpyHostToDevice);
	
	////// lanzo kernel

	dim3 grid_size(IM_WIDTH/32 + 1, IM_HEIGHT / 32 + 1);
	dim3 block_size(32,32);
		
	kernel_raytracer <<< grid_size, block_size >> >
		(dev_rgb, size, fov_00_d, fov_w0_d, fov_0h_d, view_point_d, n_vertex, n_faces, vertex_d, normals_d, color_d, faces_d);

	printf("Kernel done\n");

	cudaMemcpy(rgb, dev_rgb, size * 3 * sizeof(char), cudaMemcpyDeviceToHost);
	cudaFree(dev_rgb);
	   	  
}

/////// Main

int main()
{
			
	////// read obj
	
	const char *filename_in = "./colors.obj";
	
	int n_vertex, n_faces;
	float *vertex;
	float *normals;
	float *color;
	int *faces;
	   	
	read_obj(filename_in, &vertex, &normals, &color, &n_vertex, &faces, &n_faces);
	
	printf("\n %d Vertex: \n", n_vertex);
	for (int i_v= 0; i_v < n_vertex; i_v++) {
		printf("(#%d) x:%f , y:%f , z:%f ; r:%f , g:%f , b:%f \n", i_v, vertex[i_v*3], vertex[i_v * 3+1], vertex[i_v * 3+2], color[i_v*3], color[i_v * 3 + 1], color[i_v * 3 + 2]);
		printf("nx: %f , ny: %f , nz: %f \n", normals[i_v * 3], normals[i_v * 3 + 1], normals[i_v * 3 + 2]);
	}

	printf("\n %d Faces: \n", n_faces);
	for (int i_f = 0; i_f < n_faces; i_f++) {
		printf("(#%d) i:%d , j:%d , k:%d \n", i_f, faces[i_f * 3], faces[i_f * 3 + 1], faces[i_f * 3 + 2]);
	}

	printf("Read done \n");

	///// GPU 

	const char *filename = "image_out.bmp";
	int width = IM_WIDTH;
	int height = IM_HEIGHT;
	char *rgb = new char[height*width * 3];
	   	
	ray_tracer(width, height, rgb, n_vertex, n_faces, vertex, normals, color, faces);
	
	////// write bmp

	write_bmp(filename, width, height, rgb);
		
    return 0;
}

//////// read/write files

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

void read_obj(const char *filename, float **vertex, float **normals, float **color, int *n_vertex, int **faces, int *n_faces)
{
	FILE * file = fopen(filename, "r");
	if (file == NULL) {
		printf("Impossible to open the file !\n");
		return;
	}

	int n_faces_l = 0;
	int faces_l[N_MAX * 3];
	int n_vertex_l = 0;
	float vertex_l[N_MAX * 3];
	float colors_l[N_MAX * 3];
	int n_normals_l = 0;
	float normals_l[N_MAX * 3];

	while (1) {

		char lineHeader[128];
		// Lee la primera palabra de la línea
		int res = fscanf(file, "%s", lineHeader);

		if (res == EOF)
			break; // EOF = End Of File, es decir, el final del archivo. Se finaliza el ciclo.

		if (strcmp(lineHeader, "v") == 0) {
			float x, y, z, r, g, b;
			fscanf(file, "%f %f %f %f %f %f \n", &x, &y, &z, &r, &g, &b);

			// guardo vertices 
			vertex_l[n_vertex_l * 3] = x;
			vertex_l[n_vertex_l * 3 + 1] = y;
			vertex_l[n_vertex_l * 3 + 2] = z;

			// guardo colores
			colors_l[n_vertex_l * 3] = r;
			colors_l[n_vertex_l * 3 + 1] = g;
			colors_l[n_vertex_l * 3 + 2] = b;

			n_vertex_l++;
		}

		if (strcmp(lineHeader, "vn") == 0) {
			float x, y, z;
			fscanf(file, "%f %f %f\n", &x, &y, &z);

			// guardo normales 
			normals_l[n_normals_l * 3] = x;
			normals_l[n_normals_l * 3 + 1] = y;
			normals_l[n_normals_l * 3 + 2] = z;

			n_normals_l++;
		}

		if (strcmp(lineHeader, "f") == 0) {
			int i, j, k;
			fscanf(file, "%d//%d %d//%d %d//%d\n", &i, &i, &j, &j, &k, &k);

			// guardo caras (comenzando por 0)
			faces_l[n_faces_l * 3] = i - 1;
			faces_l[n_faces_l * 3 + 1] = j - 1;
			faces_l[n_faces_l * 3 + 2] = k - 1;

			n_faces_l++;
		}
	}

	if (n_normals_l != n_vertex_l) {
		printf("Different number of vertex and normals!!!\n");
		return;
	}

	/// copio datos

	n_vertex[0] = n_vertex_l;
	*vertex = new float[n_vertex_l * 3];
	*normals = new float[n_vertex_l * 3];
	*color = new float[n_vertex_l * 3];

	for (int i_v = 0; i_v < n_vertex_l; i_v++) {
		(*vertex)[i_v * 3] = vertex_l[i_v * 3];
		(*vertex)[i_v * 3 + 1] = vertex_l[i_v * 3 + 1];
		(*vertex)[i_v * 3 + 2] = vertex_l[i_v * 3 + 2];

		(*normals)[i_v * 3] = normals_l[i_v * 3];
		(*normals)[i_v * 3 + 1] = normals_l[i_v * 3 + 1];
		(*normals)[i_v * 3 + 2] = normals_l[i_v * 3 + 2];

		(*color)[i_v * 3] = colors_l[i_v * 3];
		(*color)[i_v * 3 + 1] = colors_l[i_v * 3 + 1];
		(*color)[i_v * 3 + 2] = colors_l[i_v * 3 + 2];
	}

	n_faces[0] = n_faces_l;
	*faces = new int[n_faces_l * 3];

	for (int i_f = 0; i_f < n_faces_l; i_f++) {
		(*faces)[i_f * 3] = faces_l[i_f * 3];
		(*faces)[i_f * 3 + 1] = faces_l[i_f * 3 + 1];
		(*faces)[i_f * 3 + 2] = faces_l[i_f * 3 + 2];
	}
}

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






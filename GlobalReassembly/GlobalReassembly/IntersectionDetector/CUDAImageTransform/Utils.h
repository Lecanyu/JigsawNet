#pragma once
#include "cuda_runtime.h"
class CUDAImage;
typedef unsigned char uint8;

struct UV_i {
	int row_i;
	int col_i;
	uint8 r;
	uint8 g;
	uint8 b;
};

struct IdAndPose
{
	int fragmentId;
	double* pose;		// R|t, length=6, 

	IdAndPose() :fragmentId(-1), pose(NULL) {}
	IdAndPose(int id, double po[6]) :fragmentId(id)
	{
		pose = new double[6];
		for (int i = 0; i < 6; ++i)
			pose[i] = po[i];
	}
	IdAndPose(const IdAndPose& item)
	{
		fragmentId = item.fragmentId;
		pose = new double[6];
		for (int i = 0; i < 6; ++i)
			pose[i] = item.pose[i];
	}
	IdAndPose& operator=(const IdAndPose& item)
	{
		if (this == &item)
			return *this;
		if (pose != NULL)
			delete[] pose;
		fragmentId = item.fragmentId;
		pose = new double[6];
		for (int i = 0; i < 6; ++i)
			pose[i] = item.pose[i];
		return *this;
	}
	~IdAndPose()
	{
		delete[] pose;
	}

};


/***
 * inv(R|t) = R^T|-R^T*t 
 */
double* InverseMat(double* affine_mat);
double* MatMul(double* mat1, double* mat2);
/*
 *affine_mat = R|t
 *	affine_mat[0], affine_mat[1], affine_mat[2],
 *	affine_mat[3], affine_mat[4], affine_mat[5],
 */
//UV_i transformUVi(UV_i src_uv, double* affine_mat);



/**
 * show uint8 image array
 */
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
void Showuint8Array(uint8* image_array, int rows, int cols);
void ShowUViArray(UV_i* image_array, int rows, int cols, int offset_row, int offset_col);
cv::Mat CUDAImage2CV(const CUDAImage& cuda_image);
CUDAImage CV2CUDAImage(const cv::Mat& cv_image, uchar bg_red = 0, uchar bg_green = 0, uchar bg_blue = 0);
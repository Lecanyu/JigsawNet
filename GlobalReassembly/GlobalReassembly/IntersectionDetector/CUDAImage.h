#pragma once
#include <Eigen/Core>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

#include "CUDAImageTransform/TransformImage.h"

typedef unsigned char uint8;
class CUDAImage
{
public:
	uint8* image_;
	int rows_;
	int cols_;
	std::vector<int> fragmentIds_;		// This image consists of fragments' Id.
	std::vector<int> countPixels_;
	std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> fragmentPoses_;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	CUDAImage() {image_ = NULL; rows_ = 0; cols_ = 0; }
	CUDAImage(const cv::Mat& img, uchar red=0, uchar green=0, uchar blue=0)
	{
		int color_pixels = 0;
		rows_ = img.rows;
		cols_ = img.cols;
		image_ = new uint8[img.rows*img.cols * 3];

#pragma omp parallel for reduction(+:color_pixels)
		for (int i = 0; i<img.rows; ++i)
		{
			for (int j = 0; j<img.cols; ++j)
			{
				cv::Vec3b intensity = img.at<cv::Vec3b>(i, j);
				uchar b = intensity.val[0];
				uchar g = intensity.val[1];
				uchar r = intensity.val[2];
				if(r==red && g == green && b == blue)
				{
					image_[i*cols_ * 3 + j * 3] = 0;
					image_[i*cols_ * 3 + j * 3 + 1] = 0;
					image_[i*cols_ * 3 + j * 3 + 2] = 0;
				}
				else
				{
					image_[i*cols_ * 3 + j * 3] = b;
					image_[i*cols_ * 3 + j * 3 + 1] = g;
					image_[i*cols_ * 3 + j * 3 + 2] = r;
					color_pixels++;
				}
			}
		}
		countPixels_.push_back(color_pixels);
		fragmentPoses_.push_back(Eigen::Matrix3d::Identity());
	}
	CUDAImage(std::string image_filename, uchar red = 0, uchar green = 0, uchar blue = 0)
	{
		cv::Mat img = cv::imread(image_filename);
		rows_ = img.rows;
		cols_ = img.cols;
		image_ = new uint8[img.rows*img.cols * 3];
		int color_pixels = 0;

#pragma omp parallel for reduction(+:color_pixels)
		for(int i=0;i<img.rows;++i)
		{
			for(int j=0;j<img.cols;++j)
			{
				cv::Vec3b intensity = img.at<cv::Vec3b>(i, j);
				uchar b = intensity.val[0];
				uchar g = intensity.val[1];
				uchar r = intensity.val[2];
				if (r == red && g == green && b == blue)
				{
					image_[i*cols_ * 3 + j * 3] = 0;
					image_[i*cols_ * 3 + j * 3 + 1] = 0;
					image_[i*cols_ * 3 + j * 3 + 2] = 0;
				}
				else
				{
					image_[i*cols_ * 3 + j * 3] = b;
					image_[i*cols_ * 3 + j * 3 + 1] = g;
					image_[i*cols_ * 3 + j * 3 + 2] = r;
					color_pixels++;
				}
			}
		}
		countPixels_.push_back(color_pixels);
		fragmentPoses_.push_back(Eigen::Matrix3d::Identity());
	}
	CUDAImage(const CUDAImage& cuda_image)
	{
		rows_ = cuda_image.rows_;
		cols_ = cuda_image.cols_;
		fragmentIds_ = cuda_image.fragmentIds_;
		countPixels_ = cuda_image.countPixels_;
		fragmentPoses_ = cuda_image.fragmentPoses_;
		image_ = new uint8[rows_*cols_ * 3];
		memcpy(image_, cuda_image.image_, rows_*cols_ * 3 * sizeof(uint8));
	}

	CUDAImage& operator=(const CUDAImage& cuda_image)
	{
		if (this == &cuda_image)
			return *this;
		if (image_ != NULL)
			delete[] image_;
		rows_ = cuda_image.rows_;
		cols_ = cuda_image.cols_;
		fragmentIds_ = cuda_image.fragmentIds_;
		countPixels_ = cuda_image.countPixels_;
		fragmentPoses_ = cuda_image.fragmentPoses_;
		image_ = new uint8[rows_*cols_ * 3];
		memcpy(image_, cuda_image.image_, rows_*cols_ * 3 * sizeof(uint8));
		return *this;
	}
	~CUDAImage()
	{
		delete[] image_;
	}


};
//////////////////////////////////////////////////////////////////////////
// Visualize assembled images according to pose
#pragma once

#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Utils.h"

namespace JigsawVisualNS
{
	class JigsawVisual
	{
	public:
		Alignments2d pose_;
		Alignments2d cropTranslation_;
		std::vector<cv::Mat> originalImgArr_;
		std::vector<cv::Mat> croppedImgArr_;
		cv::Mat assembledImg_;

		JigsawVisual(std::string root_dir, std::string pose_file, bool isLine);
		JigsawVisual(const std::unordered_map<int, Eigen::Matrix3d>& pose_map, const std::vector<cv::Mat>& all_fragment_images);

		// Crop out bounding box in original images 
		void CropImages(uchar r, uchar g, uchar b);
		// Assemble all image fragments and ignore background color 
		void AssembleAllImages(uchar r, uchar g, uchar b, Eigen::Matrix3d& out_offset);

	private:
		// according to background color to find bounding box
		void FindBoundingBox(const cv::Mat& img, uchar r, uchar g, uchar b, JigsawVisualNS::UV& out_min_uv, JigsawVisualNS::UV& out_max_uv);
	};
}

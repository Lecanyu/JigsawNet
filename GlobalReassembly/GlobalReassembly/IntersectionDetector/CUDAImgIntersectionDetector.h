#pragma once

#include "CUDAImage.h"
#include "../MultiGraph.h"
#include "../Parameters.h"


class CUDAImgIntersectionDetector
{
public:
	// given loop, detect intersection.
	CUDAImage DetectIntersection(const std::vector<CUDAImage>& images, const std::vector<int>& abstract_loop, const std::vector<int>& concrete_loop, MultiGraph& multi_graph);

	// Post-process to gurantee all of vertices are connected without intersection conflict. 
	std::unordered_map<int, cv::Mat> UnionFindSetSelectNoIntersection(MultiGraph& multiGraph, const LoopClosure& loop_closure, const std::vector<cv::Mat>& all_fragment_images, std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& all_fragment_poses, std::vector<int>& out_vertex2set);

	// only detect intersection
	void CheckIntersection(const CUDAImage& src, const CUDAImage& dst, const Eigen::Matrix3d& transform, double& out_overlap_ratio, int& out_overlap_pixels)
	{
		double* affine = new double[6];
		affine[0] = transform(0, 0);	affine[1] = transform(0, 1);	affine[2] = transform(0, 2);
		affine[3] = transform(1, 0);	affine[4] = transform(1, 1);	affine[5] = transform(1, 2);

		int fusion_rows = 0;
		int fusion_cols = 0;
		OnlyCalculateIntersection(src.image_, src.rows_, src.cols_, dst.image_, dst.rows_, dst.cols_, affine, out_overlap_pixels, out_overlap_ratio);
		delete[] affine;
	}

	// fusion two image
	cv::Mat Fusion(const cv::Mat& cv_src, const cv::Mat& cv_dst, const Eigen::Matrix3d& transform, double& out_overlap_ratio, int& out_offset_row, int& out_offset_col, int& out_overlap_pixels)
	{
		CUDAImage src(cv_src);
		CUDAImage dst(cv_dst);

		double* affine = new double[6];
		affine[0] = transform(0, 0);	affine[1] = transform(0, 1);	affine[2] = transform(0, 2);
		affine[3] = transform(1, 0);	affine[4] = transform(1, 1);	affine[5] = transform(1, 2);

		int fusion_rows = 0;
		int fusion_cols = 0;
		uint8* fusion_img_ptr = TransformImage(src.image_, src.rows_, src.cols_, dst.image_, dst.rows_, dst.cols_, affine, fusion_rows, fusion_cols, out_overlap_ratio, out_offset_row, out_offset_col, out_overlap_pixels);
		CUDAImage fusion_img;
		fusion_img.image_ = fusion_img_ptr;
		fusion_img.rows_ = fusion_rows;
		fusion_img.cols_ = fusion_cols;

		cv::Mat fusion_cv_img = CUDAImage2CV(fusion_img);
		return fusion_cv_img;
	}

	CUDAImage FusionLite(const CUDAImage& src, const CUDAImage& dst, const Eigen::Matrix3d& transform, double& out_overlap_ratio, int& out_offset_row, int& out_offset_col, int& out_overlap_pixels)
	{
		double* affine = new double[6];
		affine[0] = transform(0, 0);	affine[1] = transform(0, 1);	affine[2] = transform(0, 2);
		affine[3] = transform(1, 0);	affine[4] = transform(1, 1);	affine[5] = transform(1, 2);

		int fusion_rows = 0;
		int fusion_cols = 0;
		uint8* fusion_img_ptr = TransformImage(src.image_, src.rows_, src.cols_, dst.image_, dst.rows_, dst.cols_, affine, fusion_rows, fusion_cols, out_overlap_ratio, out_offset_row, out_offset_col, out_overlap_pixels);
		CUDAImage fusion_img;
		fusion_img.image_ = fusion_img_ptr;
		fusion_img.rows_ = fusion_rows;
		fusion_img.cols_ = fusion_cols;

		delete[] affine;
		return fusion_img;
	}
	
	CUDAImage Fusion(const CUDAImage& src, const CUDAImage& dst, const Eigen::Matrix3d& transform, double& out_overlap_ratio, int& out_offset_row, int& out_offset_col, int& out_overlap_pixels)
	{
		double* affine = new double[6];
		affine[0] = transform(0, 0);	affine[1] = transform(0, 1);	affine[2] = transform(0, 2);
		affine[3] = transform(1, 0);	affine[4] = transform(1, 1);	affine[5] = transform(1, 2);

		int fusion_rows = 0;
		int fusion_cols = 0;
		uint8* fusion_img_ptr = TransformImage(src.image_, src.rows_, src.cols_, dst.image_, dst.rows_, dst.cols_, affine, fusion_rows, fusion_cols, out_overlap_ratio, out_offset_row, out_offset_col, out_overlap_pixels);
		CUDAImage fusion_img;
		fusion_img.image_ = fusion_img_ptr;
		fusion_img.rows_ = fusion_rows;
		fusion_img.cols_ = fusion_cols;

		std::vector<int> fragments_id = src.fragmentIds_;
		fragments_id.insert(fragments_id.end(), dst.fragmentIds_.begin(), dst.fragmentIds_.end());
		fusion_img.fragmentIds_ = fragments_id;

		std::vector<int> pixel_count = src.countPixels_;
		pixel_count.insert(pixel_count.end(), dst.countPixels_.begin(), dst.countPixels_.end());
		fusion_img.countPixels_ = pixel_count;

		auto poses1 = src.fragmentPoses_;
		auto poses2 = dst.fragmentPoses_;
		Eigen::Matrix3d offset_trans = Eigen::Matrix3d::Identity();
		offset_trans(0, 2) = out_offset_row;
		offset_trans(1, 2) = out_offset_col;
		for (int i = 0; i<src.fragmentIds_.size(); ++i)
			poses1[i] = offset_trans*poses1[i];
		for (int i = 0; i < dst.fragmentIds_.size(); ++i)
			poses2[i] = offset_trans*transform*poses2[i];
		poses1.insert(poses1.end(), poses2.begin(), poses2.end());
		fusion_img.fragmentPoses_ = poses1;

		delete[] affine;
		return fusion_img;
	}
	

protected:
	CUDAImage DetectFromLoop(const std::vector<CUDAImage>& images, const std::vector<int>& vertexId_in_loop, const std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& transforms);

};

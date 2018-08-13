#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <unordered_map>

namespace JigsawVisualNS
{

	//////////////////////////////  File IO ////////////////////////////////////////////
	struct FramedTransformation2d {
		int frame1_;
		int frame2_;
		double score_;
		Eigen::Matrix3d transformation_;

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		FramedTransformation2d(int frame1, int frame2, double score, const Eigen::Matrix3d& t)
			: frame1_(frame1), frame2_(frame2), score_(score), transformation_(t)
		{}
		FramedTransformation2d() :frame1_(-1), frame2_(-1), score_(0.0), transformation_(Eigen::Matrix3d::Identity()) {}
	};

	struct SortFramedTransformation2d
	{
		bool operator () (const FramedTransformation2d &k1, const FramedTransformation2d &k2)
		{
			// check frame1
			if (k1.frame1_ < k2.frame1_)
				return true;
			else if (k1.frame1_ == k2.frame1_) {
				// check frame2
				if (k1.frame2_ < k2.frame2_)
					return true;
				else if (k1.frame2_ == k2.frame2_) {
					// check score
					if (k1.score_ > k2.score_)
						return true;
					else
						return false;
				}
				else
					return false;
			}
			else
				return false;

		}
	};

	struct SortScore
	{
		bool operator () (const FramedTransformation2d &k1, const FramedTransformation2d &k2)
		{
			return k1.score_ > k2.score_;
		}
	};

	struct Alignments2d {
		std::vector< FramedTransformation2d, Eigen::aligned_allocator<FramedTransformation2d>> data_;
		// total frame numbers
		int frame_num_;

		void LoadFromFileLine(std::string filename) {
			data_.clear();
			int frame1, frame2;
			double score;
			Eigen::Matrix3d trans;
			char temp[128];
			int n;

			FILE * f = fopen(filename.c_str(), "r");
			if (f != NULL) {
				char buffer[1024];
				while (fgets(buffer, 1024, f) != NULL) {
					if (strlen(buffer) > 0 && buffer[0] != '#') {
						sscanf(buffer, "%s %d", temp, &n);
						std::string head = temp;
						if (head == "Node")
						{
							frame_num_ = n + 1;
							continue;
						}
						else
						{
							sscanf(buffer, "%d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", &frame1, &frame2, &score,
								&trans(0, 0), &trans(0, 1), &trans(0, 2),
								&trans(1, 0), &trans(1, 1), &trans(1, 2),
								&trans(2, 0), &trans(2, 1), &trans(2, 2));
							data_.push_back(FramedTransformation2d(frame1, frame2, score, trans));
						}
					}
				}
				fclose(f);
			}
		}
		void LoadFromFile(std::string filename)
		{
			data_.clear();
			int frame1, frame2;
			double score;
			Eigen::Matrix3d trans;
			FILE * f = fopen(filename.c_str(), "r");
			if (f != NULL) {
				char buffer[1024];
				while (fgets(buffer, 1024, f) != NULL) {
					if (strlen(buffer) > 0 && buffer[0] != '#' && buffer[0] != '\n') {
						sscanf(buffer, "%d %d %lf", &frame1, &frame2, &score);
						fgets(buffer, 1024, f);
						sscanf(buffer, "%lf %lf %lf", &trans(0, 0), &trans(0, 1), &trans(0, 2));
						fgets(buffer, 1024, f);
						sscanf(buffer, "%lf %lf %lf", &trans(1, 0), &trans(1, 1), &trans(1, 2));
						fgets(buffer, 1024, f);
						sscanf(buffer, "%lf %lf %lf", &trans(2, 0), &trans(2, 1), &trans(2, 2));
						data_.push_back(FramedTransformation2d(frame1, frame2, score, trans));
					}
				}
				fclose(f);
			}
		}


		void LoadFromPose(std::string filename)
		{
			data_.clear();
			int frame1, frame2;
			Eigen::Matrix3d trans;
			FILE * f = fopen(filename.c_str(), "r");
			if (f != NULL) {
				char buffer[1024];
				while (fgets(buffer, 1024, f) != NULL) {
					if (strlen(buffer) > 0 && buffer[0] != '#') {
						sscanf(buffer, "%d %d", &frame1, &frame2);
						fgets(buffer, 1024, f);
						sscanf(buffer, "%lf %lf %lf", &trans(0, 0), &trans(0, 1), &trans(0, 2));
						fgets(buffer, 1024, f);
						sscanf(buffer, "%lf %lf %lf", &trans(1, 0), &trans(1, 1), &trans(1, 2));
						fgets(buffer, 1024, f);
						sscanf(buffer, "%lf %lf %lf", &trans(2, 0), &trans(2, 1), &trans(2, 2));

						data_.push_back(FramedTransformation2d(frame1, frame2, 0.0, trans));
					}
				}
				fclose(f);
			}
			frame_num_ = data_.size();
		}

		void LoadFromPoseLine(std::string filename)
		{
			data_.clear();
			int frameId;
			Eigen::Matrix3d trans;

			FILE * f = fopen(filename.c_str(), "r");
			if (f != NULL) {
				char buffer[1024];
				while (fgets(buffer, 1024, f) != NULL) {
					if (strlen(buffer) > 0 && buffer[0] != '#') {
						sscanf(buffer, "%d", &frameId);
						fgets(buffer, 1024, f);
						sscanf(buffer, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
							&trans(0, 0), &trans(0, 1), &trans(0, 2),
							&trans(1, 0), &trans(1, 1), &trans(1, 2),
							&trans(2, 0), &trans(2, 1), &trans(2, 2));
						data_.push_back(FramedTransformation2d(frameId - 1, frameId - 1, 0.0, trans));
					}
				}
				fclose(f);
			}
			frame_num_ = data_.size();
		}

		// Only save transformation for pose.
		void SaveToFile(std::string filename) {
			FILE * f = fopen(filename.c_str(), "w");
			for (int i = 0; i < (int)data_.size(); i++) {
				Eigen::Matrix3d & trans = data_[i].transformation_;
				fprintf(f, "%d\t%d\n", data_[i].frame1_, data_[i].frame2_);
				fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(0, 0), trans(0, 1), trans(0, 2));
				fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(1, 0), trans(1, 1), trans(1, 2));
				fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(2, 0), trans(2, 1), trans(2, 2));
			}
			fclose(f);
		}
	};


	//////////////////////////////// DummyImage structure //////////////////////////////////////////
	struct UV
	{
		int u, v;
		UV() :u(-1), v(-1) {}
		UV(int uu, int vv) :u(uu), v(vv) {}
		UV(float uu, float vv) { u = round(uu);	v = round(vv); }
		UV(double uu, double vv) { u = round(uu);	v = round(vv); }
		int round(double x) {
			return static_cast<int>(floor(x + 0.5));
		}

		void Transform(const Eigen::Matrix3d& mat)
		{
			Eigen::Vector3d uvi;
			uvi << u, v, 1;
			Eigen::Vector3d new_uvi = mat*uvi;
			u = round(new_uvi(0));
			v = round(new_uvi(1));
		}
	};

	struct FloatUV
	{
		float u, v;
		FloatUV() :u(-1), v(-1) {}
		FloatUV(int uu, int vv) { u = static_cast<float>(uu); v = static_cast<float>(vv); }
		FloatUV(float uu, float vv) :u(uu), v(vv) {}
		FloatUV(double uu, double vv) { u = static_cast<float>(uu);	v = static_cast<float>(vv); }
	};

	struct Pixel
	{
		UV uv;
		FloatUV f_uv;
		uchar r, g, b;
		Pixel() :uv(0, 0), f_uv(0.0, 0.0), r(0), g(0), b(0) {}
		Pixel(UV uv_pos, int rr, int gg, int bb)
			:r(rr), g(gg), b(bb)
		{
			uv.u = uv_pos.u;
			uv.v = uv_pos.v;
		}
		Pixel(UV uv_pos, FloatUV f_uv_pose, int rr, int gg, int bb) :r(rr), g(gg), b(bb)
		{
			uv.u = uv_pos.u;
			uv.v = uv_pos.v;
			f_uv.u = f_uv_pose.u;
			f_uv.v = f_uv_pose.v;
		}
	};
	struct PixelComp
	{
		bool operator()(const Pixel& p1, const Pixel& p2)
		{
			if (p1.uv.u < p2.uv.u)
				return true;
			else if (p1.uv.u == p2.uv.u)
			{
				if (p1.uv.v < p2.uv.v)
					return true;
				else
					return false;
			}
			else
				return false;
		}
	};

	//////////////////////////////// map data structure //////////////////////////////////////////
	struct hash_func {
		int operator()(const UV& k) const
		{
			return (std::hash<int>()(k.u) ^ (std::hash<int>()(k.v) << 1) >> 1);
		}
	};

	struct hash_big
	{
		int operator()(const UV& k) const
		{
			return k.u * 10000 + k.v;
		}
	};

	struct key_equal {
		bool operator () (const UV &k1, const UV &k2) const
		{
			return k1.u == k2.u && k1.v == k2.v;
		}
	};

	struct key_comp {
		bool operator () (const UV &k1, const UV &k2)
		{
			return k1.u < k2.u || (k1.u == k2.u && k1.v < k2.v);
		}
	};



	struct DummyImage
	{
		std::vector<Pixel> pixels_;
		DummyImage() {}
		DummyImage(std::vector<Pixel> pixels) :pixels_(pixels) {}

		// Initialize from opencv image and remove background
		DummyImage(const cv::Mat& img, uchar r = 0, uchar g = 0, uchar b = 0)
		{
			for (int u = 0; u < img.rows; ++u)
			{
				for (int v = 0; v < img.cols; ++v)
				{
					cv::Vec3b intensity = img.at<cv::Vec3b>(u, v);
					uchar blue = intensity.val[0];
					uchar green = intensity.val[1];
					uchar red = intensity.val[2];
					if (blue == b && green == g && red == r)
						continue;
					UV uv(u, v);
					FloatUV f_uv(u, v);
					Pixel pixel(uv, f_uv, red, green, blue);
					pixels_.push_back(pixel);
				}
			}
		}

		// tranform image
		void Transform_i(const Eigen::Matrix3d& mat)
		{
			for (int i = 0; i < pixels_.size(); ++i)
			{
				Eigen::Vector3d uvi;
				uvi << pixels_[i].uv.u, pixels_[i].uv.v, 1;
				Eigen::Vector3d new_uvi = mat*uvi;
				pixels_[i].uv.u = round(new_uvi(0));
				pixels_[i].uv.v = round(new_uvi(1));
			}
		}
		void Transform_f(const Eigen::Matrix3d& mat)
		{
			for (int i = 0; i < pixels_.size(); ++i)
			{
				Eigen::Vector3d uvi;
				uvi << pixels_[i].f_uv.u, pixels_[i].f_uv.v, 1.0;
				Eigen::Vector3d new_uvi = mat*uvi;
				pixels_[i].f_uv.u = new_uvi(0);
				pixels_[i].f_uv.v = new_uvi(1);
			}
		}


		// bounding box
		void BoundingBox_i(UV& out_min_uv, UV& out_max_uv)
		{
			int min_u = 99999, min_v = 99999;
			int max_u = -99999, max_v = -99999;
			for (int i = 0; i < pixels_.size(); ++i)
			{
				if (min_u > pixels_[i].uv.u) min_u = pixels_[i].uv.u;
				if (min_v > pixels_[i].uv.v) min_v = pixels_[i].uv.v;
				if (max_u < pixels_[i].uv.u) max_u = pixels_[i].uv.u;
				if (max_v < pixels_[i].uv.v) max_v = pixels_[i].uv.v;
			}
			out_min_uv.u = min_u;	out_min_uv.v = min_v;
			out_max_uv.u = max_u;	out_max_uv.v = max_v;
		}

		void BoundingBox_f(UV& out_min_uv, UV& out_max_uv)
		{
			float min_u = 99999.0, min_v = 99999.0;
			float max_u = -99999.0, max_v = -99999.0;
			for (int i = 0; i < pixels_.size(); ++i)
			{
				if (min_u > pixels_[i].f_uv.u) min_u = pixels_[i].f_uv.u;
				if (min_v > pixels_[i].f_uv.v) min_v = pixels_[i].f_uv.v;
				if (max_u < pixels_[i].f_uv.u) max_u = pixels_[i].f_uv.u;
				if (max_v < pixels_[i].f_uv.v) max_v = pixels_[i].f_uv.v;
			}
			out_min_uv.u = round(min_u);	out_min_uv.v = round(min_v);
			out_max_uv.u = round(max_u);	out_max_uv.v = round(max_v);
		}

		int round(double x) {
			return static_cast<int>(floor(x + 0.5));
		}
	};


	//////////////////////////////// IdPair map data structure //////////////////////////////////////////
	struct IdPair {
		int frame1, frame2;
		IdPair() :frame1(0), frame2(0) {}
		IdPair(int id1, int id2) :frame1(id1), frame2(id2) {}
	};

	struct hash_func_IdPair {
		int operator()(const IdPair& k) const
		{
			return (std::hash<int>()(k.frame1) ^ (std::hash<int>()(k.frame2) << 1) >> 1);
		}
	};

	struct key_equal_IdPair {
		bool operator () (const IdPair &k1, const IdPair &k2) const
		{
			return k1.frame1 == k2.frame1 && k1.frame2 == k2.frame2;
		}
	};

	struct key_comp_IdPair {
		bool operator () (const IdPair &k1, const IdPair &k2)
		{
			return k1.frame1 < k2.frame1 || (k1.frame1 == k2.frame1 && k1.frame2 < k2.frame2);
		}
	};

	// Give IdPair => MultiEdgeId
	struct MultiEdgeId
	{
		std::vector<int> edgeIds;	// multiple edges between two vertices
		int selectId;				// selected edges id between two vertices, decided by loop validate
		MultiEdgeId() :selectId(-1) {}
	};
}
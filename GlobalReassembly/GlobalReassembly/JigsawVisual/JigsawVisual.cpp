#include "JigsawVisual.h"


JigsawVisualNS::JigsawVisual::JigsawVisual(std::string root_dir, std::string pose_file, bool isLine)
{
	if (isLine)
		pose_.LoadFromPoseLine(pose_file);
	else
		pose_.LoadFromPose(pose_file);

	for (int i = 0; i < pose_.data_.size();++i)
	{
		assert(pose_.data_[i].frame1_ == pose_.data_[i].frame2_ && "PoseId should be the same");
		int fragmentId = pose_.data_[i].frame1_;
		//std::cout << "Load image " << fragmentId << "...";
		std::stringstream ss;
		char image_name[128];
		sprintf(image_name, "fragment_%04d.png", fragmentId + 1);				//use a special filename pattern
		ss << root_dir << image_name;
		cv::Mat img = cv::imread(ss.str());
		originalImgArr_.push_back(img);
		//std::cout << "Done!\n";
	}

	for (int i = 0; i < pose_.data_.size(); ++i)
	{
		int fragmentId = pose_.data_[i].frame1_;
		cropTranslation_.data_.push_back(FramedTransformation2d(fragmentId, fragmentId, 0.0, Eigen::Matrix3d::Identity()));
	}
}

JigsawVisualNS::JigsawVisual::JigsawVisual(const std::unordered_map<int, Eigen::Matrix3d>& pose_map,
	const std::vector<cv::Mat>& all_fragment_images)
{
	for(auto item:pose_map)
	{
		int fragment_id = item.first;
		Eigen::Matrix3d pose = item.second;
		pose_.data_.push_back(FramedTransformation2d(fragment_id, fragment_id, 0.0, pose));
		originalImgArr_.push_back(all_fragment_images[fragment_id]);
	}
	pose_.frame_num_ = pose_.data_.size();

	for (int i = 0; i < pose_.data_.size(); ++i)
	{
		int fragmentId = pose_.data_[i].frame1_;
		cropTranslation_.data_.push_back(FramedTransformation2d(fragmentId, fragmentId, 0.0, Eigen::Matrix3d::Identity()));
	}
}


void JigsawVisualNS::JigsawVisual::CropImages(uchar r, uchar g, uchar b)
{
	for (int i = 0; i < originalImgArr_.size(); ++i)
	{
		//std::cout << "Crop image " << i << "...";
		UV min_uv, max_uv;
		FindBoundingBox(originalImgArr_[i], r, g, b, min_uv, max_uv);
		cv::Rect bb(min_uv.v, min_uv.u, max_uv.v - min_uv.v+1, max_uv.u - min_uv.u+1);
		cv::Mat croppedRef(originalImgArr_[i], bb);
		cv::Mat croppedImage;
		croppedRef.copyTo(croppedImage);
		croppedImgArr_.push_back(croppedImage);

		// we should modify cropTranslation_, so that get correct visual result
		cropTranslation_.data_[i].transformation_(0, 2) += min_uv.u;
		cropTranslation_.data_[i].transformation_(1, 2) += min_uv.v;

		//std::cout << "Done!\n";
	}
}

void JigsawVisualNS::JigsawVisual::FindBoundingBox(const cv::Mat& img, uchar r, uchar g, uchar b, UV& out_min_uv, UV& out_max_uv)
{
	int min_u = 99999, min_v = 99999;
	int max_u = -99999, max_v = -99999;
	for (int u = 0; u < img.rows;++u)
	{
		for (int v = 0; v < img.cols; ++v)
		{
			cv::Vec3b intensity = img.at<cv::Vec3b>(u, v);
			uchar blue = intensity.val[0];
			uchar green = intensity.val[1];
			uchar red = intensity.val[2];
			if (r == red && g == green && b == blue)
				continue;
			if (min_u>u) min_u = u;
			if (min_v>v) min_v = v;
			if (max_u<u) max_u = u;
			if (max_v<v) max_v = v;
		}
	}
	out_min_uv.u = min_u;	out_min_uv.v = min_v;
	out_max_uv.u = max_u;	out_max_uv.v = max_v;
}

void JigsawVisualNS::JigsawVisual::AssembleAllImages(uchar r, uchar g, uchar b, Eigen::Matrix3d& out_offset)
{
	//std::cout << "Assembling...";
	// init
	std::vector<DummyImage> dummy_images;			// dummy image does not contain any background pixel
	for (auto& cv_image : croppedImgArr_)
		dummy_images.push_back(DummyImage(cv_image, r, g, b));

	int total_pixel = 0;
	std::vector<std::unordered_map<UV, int, hash_big, key_equal>> mapUV2IdArray(dummy_images.size());
	for(int ix = 0; ix<dummy_images.size();++ix)
	{
		for(int i=0;i<dummy_images[ix].pixels_.size();++i)
		{
			UV uv = dummy_images[ix].pixels_[i].uv;
			mapUV2IdArray[ix][uv] = i;
		}
		total_pixel += dummy_images[ix].pixels_.size();
	}

	// transform
	for (int i = 0; i < dummy_images.size(); ++i)
		dummy_images[i].Transform_f(cropTranslation_.data_[i].transformation_);		// first move to original position to eliminate cropped influence
	for (int i = 0; i < dummy_images.size();++i)
		dummy_images[i].Transform_f(pose_.data_[i].transformation_);		// then transform
	
	// assemble
	DummyImage big_img;
	big_img.pixels_.reserve(total_pixel);
	for (int ix = 0; ix<dummy_images.size(); ++ix)
	{
		auto& transformed_img = dummy_images[ix];

		UV min_uv, max_uv;
		transformed_img.BoundingBox_f(min_uv, max_uv);

		for (int u = min_uv.u; u<max_uv.u; ++u)
		{
			for (int v = min_uv.v; v<max_uv.v; ++v)
			{
				UV query_uv(u, v);
				Eigen::Matrix3d t = cropTranslation_.data_[ix].transformation_.inverse()*pose_.data_[ix].transformation_.inverse();
				query_uv.Transform(t);
				if(mapUV2IdArray[ix].find(query_uv) == mapUV2IdArray[ix].end())
					continue;
				auto pixel_index = mapUV2IdArray[ix][query_uv];
				Pixel new_pixel = dummy_images[ix].pixels_[pixel_index];
				new_pixel.uv.u = u;	new_pixel.uv.v = v;
				big_img.pixels_.push_back(new_pixel);
			}
		}
	}

	UV min_uv, max_uv;
	big_img.BoundingBox_i(min_uv, max_uv);
	int offset_u = -min_uv.u;
	int offset_v = -min_uv.v;
	Eigen::Matrix3d offset;
	offset <<
		1, 0, offset_u,
		0, 1, offset_v,
		0, 0, 1;
	out_offset = offset;
	big_img.Transform_i(offset);
	big_img.BoundingBox_i(min_uv, max_uv);
	// copy to opencv
	cv::Mat assembled_img(max_uv.u+1, max_uv.v+1, CV_8UC3);
	for (int u = 0; u < max_uv.u + 1;++u)
		for (int v = 0; v < max_uv.v + 1; ++v)
		{
			cv::Vec3b intensity;
			intensity.val[0] = 0;
			intensity.val[1] = 0;
			intensity.val[2] = 0;
			assembled_img.at<cv::Vec3b>(u, v) = intensity;
		}

	for (int i = 0; i < big_img.pixels_.size(); ++i)
	{
		int u = big_img.pixels_[i].uv.u;
		int v = big_img.pixels_[i].uv.v;
		cv::Vec3b intensity;
		intensity.val[0] = big_img.pixels_[i].b;
		intensity.val[1] = big_img.pixels_[i].g;
		intensity.val[2] = big_img.pixels_[i].r;
		assembled_img.at<cv::Vec3b>(u, v) = intensity;
	}
	assembledImg_ = assembled_img;

	//std::cout << "Done!\n";
}


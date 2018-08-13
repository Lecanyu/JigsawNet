#include "JigsawOpt2.h"

#include <string>
#include <set>
#include <iterator>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


std::vector<std::string> LoadDatasetList(std::string list_file)
{
	std::vector<std::string> data_list;

	std::ifstream input_file(list_file);
	std::string line;
	while (std::getline(input_file, line))
	{
		data_list.push_back(line);
	}

	return data_list;
}


int main(int argc, char** argv)
{
	if(argc!=2)
	{
		std::cout << "GlobalReassembly.exe dataset_list.txt" << std::endl;
		return -1;
	}

	std::vector<std::string> data_list = LoadDatasetList(argv[1]);
	int alignment_file_type = 1;			// 0: line style, 1: matrix style
	for (int i = 0; i < data_list.size(); i += 3)
	{
		std::cout << "Begin dataset root = " << data_list[i] << "...\n";
		std::string align_file = data_list[i + 1];
		std::string image_root_dir = data_list[i];
		std::string background_color = data_list[i + 2];
		std::string out_selected_transformation = image_root_dir;
		out_selected_transformation.append("selected_transformation.txt");

		std::istringstream iss(background_color);
		std::vector<std::string> results(std::istream_iterator < std::string > {iss}, std::istream_iterator<std::string>());
		uchar r = atoi(results[0].c_str());
		uchar g = atoi(results[1].c_str());
		uchar b = atoi(results[2].c_str());

		std::vector<CUDAImage> original_images;
		std::vector<uint8*> img_array;
		std::vector<int> fragment_rows;
		std::vector<int> fragment_cols;
		JigsawOpt2 JOpt(align_file, alignment_file_type);
		for (int i = 0; i < JOpt.pose_.frame_num_; ++i)
		{
			//std::cout << "Load image " << i << "...";
			std::stringstream ss;
			char image_name[128];
			sprintf(image_name, "fragment_%04d.png", i + 1);				//use a special filename pattern
			ss << image_root_dir << image_name;
			cv::Mat img = cv::imread(ss.str());
			CUDAImage di(img, r, g, b);
			di.fragmentIds_.push_back(i);
			original_images.push_back(di);

			uint8* img1_array = new uint8[img.rows*img.cols * 3];
			for (int i = 0; i<img.rows; ++i)
			{
				for (int j = 0; j<img.cols; ++j)
				{
					cv::Vec3b intensity;
					intensity = img.at<cv::Vec3b>(i, j);
					if (intensity.val[0] == b && intensity.val[1] == g && intensity.val[2] == r)
					{
						img1_array[i*img.cols * 3 + j * 3] = 0;
						img1_array[i*img.cols * 3 + j * 3 + 1] = 0;
						img1_array[i*img.cols * 3 + j * 3 + 2] = 0;
					}
					else
					{
						img1_array[i*img.cols * 3 + j * 3] = intensity.val[0];
						img1_array[i*img.cols * 3 + j * 3 + 1] = intensity.val[1];
						img1_array[i*img.cols * 3 + j * 3 + 2] = intensity.val[2];
					}
				}
			}
			img_array.push_back(img1_array);
			fragment_rows.push_back(img.rows);
			fragment_cols.push_back(img.cols);

			//std::cout << "Done!\n";
		}

		// Core Opt
		InitCUDA();
		std::vector<uint8*> gpu_fragments = InitializeGPUFragments(img_array, fragment_rows, fragment_cols);
		std::unordered_map<int, cv::Mat> final_reassembly = JOpt.OptWithInterSec(original_images, gpu_fragments, fragment_rows, fragment_cols);
		CloseCUDA();

		// output all of assembly
		int out_count = 0;
		for (auto& item : final_reassembly)
		{
			std::stringstream name;
			name << image_root_dir << "reassembled_result_" << out_count << ".png";
			cv::imwrite(name.str(), item.second);

			std::stringstream ss;
			int id = item.first;
			ss << image_root_dir << "pose_result_" << id << ".txt";
			std::string out_pose_file = ss.str();
			JOpt.pose_.SaveToFile(out_pose_file, JOpt.poseChunk_[id]);
			
			out_count++;
		}
		JOpt.OutputSelectEdge(out_selected_transformation);

		for (auto img : img_array)
			delete[] img;
	}
	return 0;
}

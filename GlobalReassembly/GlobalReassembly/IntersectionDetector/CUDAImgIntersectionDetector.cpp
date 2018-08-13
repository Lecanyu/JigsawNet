#include "CUDAImgIntersectionDetector.h"
#include "../JigsawVisual/JigsawVisual.h"

CUDAImage CUDAImgIntersectionDetector::DetectIntersection(const std::vector<CUDAImage>& images, const std::vector<int>& abstract_loop, const std::vector<int>& concrete_loop, MultiGraph& multi_graph)
{
	std::vector<int> vertexIdInLoop;
	std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> relative_alignment;
	for (int i = 0; i<concrete_loop.size(); ++i)
	{
		int edgeId = concrete_loop[i];
		int fragmentId;
		Eigen::Matrix3d trans;
		if (i == concrete_loop.size() - 1)
		{
			fragmentId = multi_graph[edgeId].frame2_;
			assert(fragmentId == abstract_loop[i]);
			trans = multi_graph[edgeId].transformation_;
		}
		else
		{
			if (multi_graph[edgeId].frame1_ == abstract_loop[i])
			{
				fragmentId = multi_graph[edgeId].frame1_;
				trans = multi_graph[edgeId].transformation_;
			}
			else
			{
				fragmentId = multi_graph[edgeId].frame2_;
				trans = multi_graph[edgeId].transformation_.inverse();
			}
		}
		vertexIdInLoop.push_back(fragmentId);
		relative_alignment.push_back(trans);
	}

	return DetectFromLoop(images, vertexIdInLoop, relative_alignment);
}

CUDAImage CUDAImgIntersectionDetector::DetectFromLoop(const std::vector<CUDAImage>& images, const std::vector<int>& vertexId_in_loop,
	const std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& transforms)
{
	std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> temp_poses;
	for (int i = 0; i < images.size(); ++i)
		temp_poses.push_back(Eigen::Matrix3d::Identity());

	// use the first fragment initialize temporary canvas
	CUDAImage temp_canvas = images[vertexId_in_loop[0]];

	// check intersection or not
	for (int i = 1; i<vertexId_in_loop.size(); ++i)
	{
		int prev_fragmentId = vertexId_in_loop[i - 1];
		int fragmentId = vertexId_in_loop[i];
		CUDAImage temp_fragment = images[fragmentId];
		Eigen::Matrix3d trans = temp_poses[prev_fragmentId] * transforms[i - 1];

		double overlap_ratio;
		int overlap_pixel_num;
		int offset_row, offset_col;
		CUDAImage fusion_img = Fusion(temp_canvas, temp_fragment, trans, overlap_ratio, offset_row, offset_col, overlap_pixel_num);
		
		if (overlap_ratio > Parameters::intersection_ratio_threshold_)
		{
			return CUDAImage();		// find intersection, return null image
		}
		// no intersection, update
		Eigen::Matrix3d offset_trans = Eigen::Matrix3d::Identity();
		offset_trans(0, 2) = offset_row;
		offset_trans(1, 2) = offset_col;
		for (int fid : temp_canvas.fragmentIds_)
		{
			temp_poses[fid] = offset_trans*temp_poses[fid];
		}
		for (int fid : temp_fragment.fragmentIds_)
		{
			temp_poses[fid] = offset_trans*trans*temp_poses[fid];
		}
		temp_canvas = fusion_img;
	}
	return temp_canvas;
}


std::unordered_map<int, cv::Mat> CUDAImgIntersectionDetector::UnionFindSetSelectNoIntersection(MultiGraph& multiGraph, const LoopClosure& loop_closure, const std::vector<cv::Mat>& all_fragment_images, std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>& all_fragment_poses, std::vector<int>& out_vertex2set)
{
	// init UPS data structure
	struct set_vertex
	{
		int set_id;
		std::vector<int> vertex_id;
	};
	std::vector<set_vertex> gather;			//from set to vertex
	gather.resize(multiGraph.vertexNum_);
	for (int i = 0; i < multiGraph.vertexNum_; ++i)
	{
		gather[i].set_id = i;
		gather[i].vertex_id.push_back(i);
	}
	std::vector<int> vertex2set;			//from vertex to set
	vertex2set.resize(multiGraph.vertexNum_);
	for (int i = 0; i < vertex2set.size(); ++i)
		vertex2set[i] = i;

	// start UPS...
	int big_chunk_vid = -1;
	for (int i = 0; i<multiGraph.edges_.size(); ++i)
	{
		if (multiGraph.edges_[i].select_)
		{
			int v1 = multiGraph.edges_[i].frame1_;
			int v2 = multiGraph.edges_[i].frame2_;
			big_chunk_vid = v1;
			//union
			if (vertex2set[v1] < vertex2set[v2])			// v2's set => v1's set
			{
				for (int t = 0; t < gather[vertex2set[v2]].vertex_id.size(); ++t)
				{
					gather[vertex2set[v1]].vertex_id.push_back(gather[vertex2set[v2]].vertex_id[t]);
				}
				for (int t = 0; t < gather[vertex2set[v2]].vertex_id.size(); ++t)
				{
					vertex2set[gather[vertex2set[v2]].vertex_id[t]] = vertex2set[v1];
				}

			}
			if (vertex2set[v1]>vertex2set[v2])
			{											// v1's set => v2's set
				for (int t = 0; t < gather[vertex2set[v1]].vertex_id.size(); ++t)
				{
					gather[vertex2set[v2]].vertex_id.push_back(gather[vertex2set[v1]].vertex_id[t]);
				}
				for (int t = 0; t < gather[vertex2set[v1]].vertex_id.size(); ++t)
				{
					vertex2set[gather[vertex2set[v1]].vertex_id[t]] = vertex2set[v2];
				}
			}
		}
	}
	bool connected = true;
	for (int i = 1; i < vertex2set.size(); ++i)
	{
		if (vertex2set[i] != vertex2set[i - 1])
		{
			connected = false;		// unconnected
			break;;
		}
	}

	Eigen::Matrix3d offset_transform;
	JigsawVisualNS::JigsawVisual visual(loop_closure.poses_, all_fragment_images);
	visual.CropImages(0, 0, 0);
	visual.AssembleAllImages(0, 0, 0, offset_transform);
	for(auto item:loop_closure.poses_)
	{
		int fragmentId = item.first;
		all_fragment_poses[fragmentId] = offset_transform*all_fragment_poses[fragmentId];
	}
	cv::Mat big_img = visual.assembledImg_;

	std::unordered_map<int, cv::Mat> final_reassembly;
	if (!connected)				// selected highest score
	{
		std::unordered_map<int, cv::Mat> setId2Img;
		for (int i = 0; i < vertex2set.size(); ++i)
		{
			if (vertex2set[i] == vertex2set[big_chunk_vid])
				setId2Img[vertex2set[i]] = big_img;
			else
			{
				setId2Img[vertex2set[i]] = all_fragment_images[i];
			}
		}

		std::vector<Edge, Eigen::aligned_allocator<Edge>> edges_temp = multiGraph.edges_;
		std::sort(edges_temp.begin(), edges_temp.end(), SortEdge());
		CUDAImgIntersectionDetector detector;

		for (int i = 0; i < edges_temp.size(); ++i)
		{
			int v1 = edges_temp[i].frame1_;
			int v2 = edges_temp[i].frame2_;
			if (vertex2set[v1] == vertex2set[v2])
				continue;

			int frame1, frame2;
			Eigen::Matrix3d transform;
			cv::Mat img1, img2;
			if (vertex2set[v1]<vertex2set[v2])
			{
				frame1 = v1;
				frame2 = v2;
				transform = edges_temp[i].transformation_;
				img1 = setId2Img[vertex2set[v1]];
				img2 = setId2Img[vertex2set[v2]];
			}
			else
			{
				frame1 = v2;
				frame2 = v1;
				transform = edges_temp[i].transformation_.inverse();
				img1 = setId2Img[vertex2set[v2]];
				img2 = setId2Img[vertex2set[v1]];
			}

			Eigen::Matrix3d pose1 = all_fragment_poses[frame1];
			Eigen::Matrix3d pose2 = all_fragment_poses[frame2];
			Eigen::Matrix3d trans = pose1*transform*pose2.inverse();
			double out_overlap_ratio = 0;
			int out_offset_row = 0, out_offset_col = 0, out_overlap_pixels = 0;

			cv::Mat fusion_img = detector.Fusion(img1, img2, trans, out_overlap_ratio, out_offset_row, out_offset_col, out_overlap_pixels);

			if (out_overlap_ratio>Parameters::intersection_ratio_threshold_)
				continue;

			int choose_id = edges_temp[i].idReflect_;
			multiGraph.edges_[choose_id].select_ = true;
			multiGraph.mapIdPair2EdgeIds_[IdPair(v1, v2)].selectId = choose_id;

			//union
			if (vertex2set[v1] < vertex2set[v2])		// v2's set => v1's set
			{
				// update image
				setId2Img[vertex2set[v1]] = fusion_img;
				// update pose
				Eigen::Matrix3d offset_trans = Eigen::Matrix3d::Identity();
				offset_trans(0, 2) = out_offset_row;
				offset_trans(1, 2) = out_offset_col;
				for (int t = 0; t < gather[vertex2set[v2]].vertex_id.size(); ++t)
				{
					int vid = gather[vertex2set[v2]].vertex_id[t];
					all_fragment_poses[vid] = offset_trans*trans*all_fragment_poses[vid];
				}
				for (int t = 0; t < gather[vertex2set[v1]].vertex_id.size(); ++t)
				{
					int vid = gather[vertex2set[v1]].vertex_id[t];
					all_fragment_poses[vid] = offset_trans*all_fragment_poses[vid];
				}

				// update Union-Set
				for (int t = 0; t < gather[vertex2set[v2]].vertex_id.size(); ++t)
				{
					gather[vertex2set[v1]].vertex_id.push_back(gather[vertex2set[v2]].vertex_id[t]);
				}
				for (int t = 0; t < gather[vertex2set[v2]].vertex_id.size(); ++t)
				{
					vertex2set[gather[vertex2set[v2]].vertex_id[t]] = vertex2set[v1];
				}
			}
			if (vertex2set[v1] > vertex2set[v2])		// v1's set => v2's set
			{
				// update image
				setId2Img[vertex2set[v2]] = fusion_img;

				// update pose
				Eigen::Matrix3d offset_trans = Eigen::Matrix3d::Identity();
				offset_trans(0, 2) = out_offset_row;
				offset_trans(1, 2) = out_offset_col;
				for (int t = 0; t < gather[vertex2set[v1]].vertex_id.size(); ++t)
				{
					int vid = gather[vertex2set[v1]].vertex_id[t];
					all_fragment_poses[vid] = offset_trans*trans*all_fragment_poses[vid];
				}
				for (int t = 0; t < gather[vertex2set[v2]].vertex_id.size(); ++t)
				{
					int vid = gather[vertex2set[v2]].vertex_id[t];
					all_fragment_poses[vid] = offset_trans*all_fragment_poses[vid];
				}

				// update Union-Set
				for (int t = 0; t < gather[vertex2set[v1]].vertex_id.size(); ++t)
				{
					gather[vertex2set[v2]].vertex_id.push_back(gather[vertex2set[v1]].vertex_id[t]);
				}
				for (int t = 0; t < gather[vertex2set[v1]].vertex_id.size(); ++t)
				{
					vertex2set[gather[vertex2set[v1]].vertex_id[t]] = vertex2set[v2];
				}
			}
		}
		if (!multiGraph.UnionFindSetJudgeAllLink())
		{
			std::cout << "After greedily select, some vertices still cannot be integrated. Check the log!\n";
			for (int i = 0; i < vertex2set.size(); ++i)
				std::cout << "vertex " << i << " belong set" << vertex2set[i] << "\n";
		}

		for (int i = 0; i < vertex2set.size(); ++i)
			final_reassembly[vertex2set[i]] = setId2Img[vertex2set[i]];
	}
	else
	{
		final_reassembly[0] = big_img;
		std::cout << "All vertices have been decided by Loop Search\n";
	}

	out_vertex2set = vertex2set;
	return final_reassembly;
}



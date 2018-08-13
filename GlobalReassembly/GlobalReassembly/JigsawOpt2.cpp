#include "JigsawOpt2.h"
#include <stack>
#include <set>
#include "JigsawVisual/JigsawVisual.h"


std::unordered_map<int, cv::Mat> JigsawOpt2::OptWithInterSec(const std::vector<CUDAImage>& original_images, const std::vector<uint8*>& gpu_fragments, const std::vector<int>& fragment_rows, const std::vector<int>& fragment_cols)
{
	// convert CUDAImage to opencv image
	for (auto& cuda_img : original_images)
		all_fragment_images.push_back(CUDAImage2CV(cuda_img));

	// Initialize all of smallest loop closures.
	std::cout << "Initialize all of smallest loop closures...";
	std::vector<LoopClosure> loop_closures;
	InitializeSmallestLoopClosure(loop_closures, original_images);
	std::cout << "Done!" << std::endl;
	// debug show
	//ShowLoopClosures(loop_closures);

	// From bottom to top, try to merge small loops into big one.
	std::cout << "The number of initial loop closures: "<< loop_closures.size()<< ", Begin bottom to top merging.\n";
	std::vector<std::vector<LoopClosure>> loop_closure_history;			// all loop closures iteration history.
	loop_closure_history.push_back(loop_closures);
	Bottom2TopMerge(loop_closures, loop_closure_history, gpu_fragments, fragment_rows, fragment_cols);
	std::cout << "The number of final loop closures: " << loop_closures.size() << "\n";


	int best_id = -1;
	double highest_score = -1;
	for (int i=0;i<loop_closures.size();++i)
	{
		if(loop_closures[i].probs_>highest_score)
		{
			highest_score = loop_closures[i].probs_;
			best_id = i;
		}
	}

	// From top to bottom, try to merge the maximum loop with the smaller loop closure history
	std::cout<<"Before top to bottom merging, the final loop closure contains "<< loop_closures[best_id].vertices_.size() << " fragments" << std::endl;
	std::cout << "Begin top to bottom merging..." << std::endl;
	loop_closure_history.pop_back();																	// delete the maximum loop closure
	for(int i=0;i<loop_closure_history.size();++i)
		std::sort(loop_closure_history[i].begin(), loop_closure_history[i].end(), SortLoopClosure());		// put higher score loop closure into front of vector.
	Top2BottomMerge(loop_closures[best_id], loop_closure_history, gpu_fragments, fragment_rows, fragment_cols);
	std::cout << "Finish top to bottom merging, the final loop closure contains " << loop_closures[best_id].vertices_.size() << " fragments" << std::endl;

	// modify the multi-graph
	for(LoopEdge edge:loop_closures[best_id].edges_)
	{
		IdRank idrank(edge.frame1_, edge.frame2_, edge.rank_);
		int edge_id = multiGraph_.mapIdRank2EdgeIds_[idrank];
		multiGraph_.edges_[edge_id].select_ = true;
		multiGraph_.mapIdPair2EdgeIds_[IdPair(edge.frame1_, edge.frame2_)].selectId = edge_id;
	}

	// prepare final result
	std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> all_fragment_poses(multiGraph_.vertexNum_, Eigen::Matrix3d::Identity());
	for(auto& pose: loop_closures[best_id].poses_)
	{
		int id = pose.first;
		Eigen::Matrix3d pose_mat = pose.second;
		all_fragment_poses[id] = pose_mat;
	}

	std::vector<int> out_vertex2set;
	CUDAImgIntersectionDetector intersection_detector;
	std::unordered_map<int, cv::Mat> final_reassembly = intersection_detector.UnionFindSetSelectNoIntersection(multiGraph_, loop_closures[best_id], all_fragment_images, all_fragment_poses, out_vertex2set);
	DFSPose(out_vertex2set);
	return final_reassembly;
}

void JigsawOpt2::InitializeSmallestLoopClosure(std::vector<LoopClosure>& loop_closures, const std::vector<CUDAImage>& original_images)
{
	CUDAImgIntersectionDetector intersection_detector;
	for (int loop_length = 3; loop_length <= Parameters::max_loop_length; ++loop_length)
	{
		// only generate loop length 3, 4.
		for (auto abstract_loop : multiGraph_.LoopGenerator(loop_length))
		{
			if (abstract_loop.size() <= 0)
				continue;
			std::vector<std::vector<int>> concrete_loops = GenerateValidableLoop(abstract_loop, Parameters::beam_width);

			double min_T_err = 9999.0, min_R_err = 9999.0;
			for (auto one_concrete_loop : concrete_loops)
			{
				// 1. check loop closure by calculating loop error
				double T_err = 0.0, R_err = 0.0;
				CalculateError2d(abstract_loop, one_concrete_loop, T_err, R_err);
				if (T_err< Parameters::T_err_threshold_ && R_err<Parameters::R_err_threshold_)
				{
					// 2. check intersection violation
					CUDAImage fusion = intersection_detector.DetectIntersection(original_images, abstract_loop, one_concrete_loop, multiGraph_);
					if (fusion.image_ != NULL)
					{
						// find a loop closure
						std::unordered_set<int> vertices;
						for (int i = 0; i < abstract_loop.size(); ++i)
							vertices.insert(abstract_loop[i]);
						std::unordered_set<LoopEdge, HashLoopEdge, LoopEdgeEq> edges;
						double score = 0.0;
						for (int i = 0; i<one_concrete_loop.size(); ++i)
						{
							int edgeId = one_concrete_loop[i];
							int frame1 = multiGraph_.edges_[edgeId].frame1_;
							int frame2 = multiGraph_.edges_[edgeId].frame2_;
							int rank = multiGraph_.edges_[edgeId].rank_;
							LoopEdge loop_edge(frame1, frame2, rank);
							edges.insert(loop_edge);
							score += multiGraph_.edges_[edgeId].score_;
						}
						std::unordered_map<int, Eigen::Matrix3d> poses;
						for (int i = 0; i<fusion.fragmentPoses_.size(); ++i)
						{
							int id = fusion.fragmentIds_[i];
							poses[id] = fusion.fragmentPoses_[i];
						}
						std::unordered_map<int, int> countPixels;
						for (int i = 0; i<fusion.countPixels_.size(); ++i)
						{
							int id = fusion.fragmentIds_[i];
							countPixels[id] = fusion.countPixels_[i];
						}
						double probs = score;
						LoopClosure loop_closure(vertices, edges, poses, countPixels, probs);
						loop_closures.push_back(loop_closure);
					}
				}
			}
		}
	}
}

void JigsawOpt2::Bottom2TopMerge(std::vector<LoopClosure>& loop_closures, std::vector<std::vector<LoopClosure>>& loop_closure_history, const std::vector<uint8*>& gpu_fragments, const std::vector<int>& fragment_rows, const std::vector<int>& fragment_cols)
{
	int iteration_count = 1;
	while (true)
	{
		// iteratively intgrate loop closures
		std::cout << "iterate " << iteration_count << ", try to integrate " << loop_closures.size() << " loop closures ...";
		std::vector<LoopClosure> temp_integrated_loop_closures;

		bool truncate_loop = false;
		for (int i = 0; i<loop_closures.size(); ++i)
		{
			if (truncate_loop)
				break;
			for (int j = i + 1; j<loop_closures.size(); ++j)
			{
				LoopEdge common_edge;
				const auto& edges1 = loop_closures[i].edges_;
				const auto& edges2 = loop_closures[j].edges_;
				for (LoopEdge e : edges2)
				{
					if (edges1.find(e) != edges1.end())
					{
						common_edge = e;
						break;
					}
				}
				if (common_edge.frame1_ != -1)		// find a common edge between two loop closures
				{
					auto& loop1 = loop_closures[i];
					auto& loop2 = loop_closures[j];
					// transform loop2 to loop1
					LoopClosure transformed_loop2 = loop2;
					Eigen::Matrix3d v1_in_loop1 = loop1.poses_[common_edge.frame1_];
					Eigen::Matrix3d v2_in_loop1 = loop1.poses_[common_edge.frame2_];
					Eigen::Matrix3d v1_in_loop2 = loop2.poses_[common_edge.frame1_];
					Eigen::Matrix3d v2_in_loop2 = loop2.poses_[common_edge.frame2_];
					Eigen::Matrix3d transform = v1_in_loop1*v1_in_loop2.inverse();

					Eigen::Matrix3d check_v2 = transform*v2_in_loop2;
					assert(IsSameMatrix(v2_in_loop1, check_v2) && "Align one pair of vertices should lead the other pair vertices to be aligned in the common edge");
					for (auto pose_pair : transformed_loop2.poses_)
					{
						transformed_loop2.poses_[pose_pair.first] = transform *transformed_loop2.poses_[pose_pair.first];
					}
					// validate the condition 2
					if (Condition2(loop1, transformed_loop2))
					{
						// validate the condition 1
						if (Condition1(loop1, transformed_loop2, gpu_fragments, fragment_rows, fragment_cols))
						{
							// integrate loop closures
							for (auto loop1_poses_pair : loop1.poses_)
								transformed_loop2.poses_[loop1_poses_pair.first] = loop1_poses_pair.second;
							for (auto loop1_vertex : loop1.vertices_)
								transformed_loop2.vertices_.insert(loop1_vertex);
							for (auto loop1_edge : loop1.edges_)
								transformed_loop2.edges_.insert(loop1_edge);
							for (auto pixels : loop1.countPixels_)
								transformed_loop2.countPixels_[pixels.first] = pixels.second;
							transformed_loop2.probs_ += loop1.probs_;

							if (temp_integrated_loop_closures.size() > Parameters::total_loop_num)
								truncate_loop = true;
							if (truncate_loop)
								break;
							bool isExisted = false;
							for (auto& lo : temp_integrated_loop_closures)
							{
								if (lo == transformed_loop2)
								{
									isExisted = true;
									break;
								}
							}
							if (!isExisted)
							{
								temp_integrated_loop_closures.push_back(transformed_loop2);
							}
						}
					}
				}
			}
		}

		iteration_count++;
		std::cout << "Done!" << std::endl;
		
		if (temp_integrated_loop_closures.size() == 0)
			break;
		else
		{
			loop_closures = temp_integrated_loop_closures;
			loop_closure_history.push_back(loop_closures);
		}
	}
}

void JigsawOpt2::Top2BottomMerge(LoopClosure& best_loop, std::vector<std::vector<LoopClosure>>& loop_closure_history, const std::vector<uint8*>& gpu_fragments, const std::vector<int>& fragment_rows, const std::vector<int>& fragment_cols)
{
	// from top to bottom
	for (int i = loop_closure_history.size() - 1; i >= 0; --i)
	{
		for (int j = 0; j<loop_closure_history[i].size(); ++j)
		{
			auto& loop1 = best_loop;
			auto& loop2 = loop_closure_history[i][j];
			if (Condition3(loop1, loop2))
			{
				LoopEdge common_edge;
				const auto& edges1 = loop1.edges_;
				const auto& edges2 = loop2.edges_;
				for (LoopEdge e : edges2)
				{
					if (edges1.find(e) != edges1.end())
					{
						common_edge = e;
						break;
					}
				}
				if (common_edge.frame1_ == -1)		// no common edge within loop closure, try to find isolated edge to connect them
				{
					bool finish_integrated = false;
					for (int vid1 : loop1.vertices_)
					{
						if (finish_integrated)
							break;
						for (int vid2 : loop2.vertices_)
						{
							if (finish_integrated)
								break;
							IdPair idp;
							bool exchange = false;
							if (vid1<vid2)
							{
								idp.frame1 = vid1;	idp.frame2 = vid2;
							}
							if (vid1>vid2)
							{
								idp.frame1 = vid2;	idp.frame2 = vid1;
								exchange = true;
							}
							MultiEdgeId multi_edges;
							if (multiGraph_.mapIdPair2EdgeIds_.find(idp) != multiGraph_.mapIdPair2EdgeIds_.end())
								multi_edges = multiGraph_.mapIdPair2EdgeIds_[idp];

							for (int eid : multi_edges.edgeIds)
							{
								common_edge.frame1_ = multiGraph_.edges_[eid].frame1_;
								common_edge.frame2_ = multiGraph_.edges_[eid].frame2_;
								common_edge.rank_ = multiGraph_.edges_[eid].rank_;

								if(multiGraph_.edges_[eid].score_<Parameters::top2bottom_greedy_score_threshold_)
									continue;

								// merge
								LoopClosure transformed_loop2 = loop2;
								Eigen::Matrix3d transform = Eigen::Matrix3d::Identity();
								if (exchange)
									transform = multiGraph_.edges_[eid].transformation_.inverse();
								else
									transform = multiGraph_.edges_[eid].transformation_;
								Eigen::Matrix3d loop2_transform_mat = loop1.poses_[vid1] * transform * loop2.poses_[vid2].inverse();
								for (auto pose_pair : transformed_loop2.poses_)
									transformed_loop2.poses_[pose_pair.first] = loop2_transform_mat * transformed_loop2.poses_[pose_pair.first];
								// validate the condition 1
								if (Condition1(loop1, transformed_loop2, gpu_fragments, fragment_rows, fragment_cols))
								{
									// integrate into loop closures1
									for (auto loop2_poses_pair : transformed_loop2.poses_)
										loop1.poses_[loop2_poses_pair.first] = loop2_poses_pair.second;
									for (auto loop2_vertex : transformed_loop2.vertices_)
										loop1.vertices_.insert(loop2_vertex);
									for (auto loop2_edge : transformed_loop2.edges_)
										loop1.edges_.insert(loop2_edge);
									for (auto pixels : transformed_loop2.countPixels_)
										loop1.countPixels_[pixels.first] = pixels.second;
									loop1.probs_ += transformed_loop2.probs_;
									LoopEdge loop_edge(common_edge.frame1_, common_edge.frame2_, common_edge.rank_);
									loop1.edges_.insert(loop_edge);

									finish_integrated = true;
									break;
								}
							}
						}
					}
				}
				else															// found common edge within loop closure, use this common edge to connect them
				{
					// transform loop2 to loop1
					LoopClosure transformed_loop2 = loop2;
					Eigen::Matrix3d v1_in_loop1 = loop1.poses_[common_edge.frame1_];
					Eigen::Matrix3d v2_in_loop1 = loop1.poses_[common_edge.frame2_];
					Eigen::Matrix3d v1_in_loop2 = loop2.poses_[common_edge.frame1_];
					Eigen::Matrix3d v2_in_loop2 = loop2.poses_[common_edge.frame2_];
					Eigen::Matrix3d transform = v1_in_loop1*v1_in_loop2.inverse();

					Eigen::Matrix3d check_v2 = transform*v2_in_loop2;
					assert(IsSameMatrix(v2_in_loop1, check_v2) && "Align one pair of vertices should lead the other pair vertices to be aligned in the common edge");
					for (auto pose_pair : transformed_loop2.poses_)
						transformed_loop2.poses_[pose_pair.first] = transform *transformed_loop2.poses_[pose_pair.first];
					if (Condition1(loop1, transformed_loop2, gpu_fragments, fragment_rows, fragment_cols))
					{
						// integrate loop closures
						for (auto loop2_poses_pair : transformed_loop2.poses_)
							loop1.poses_[loop2_poses_pair.first] = loop2_poses_pair.second;
						for (auto loop2_vertex : transformed_loop2.vertices_)
							loop1.vertices_.insert(loop2_vertex);
						for (auto loop2_edge : transformed_loop2.edges_)
							loop1.edges_.insert(loop2_edge);
						for (auto pixels : transformed_loop2.countPixels_)
							loop1.countPixels_[pixels.first] = pixels.second;
						loop1.probs_ += transformed_loop2.probs_;
					}
				}
			}
		}
	}
}

bool JigsawOpt2::Condition1(LoopClosure& loop1, LoopClosure& loop2, const std::vector<uint8*>& gpu_fragments, const std::vector<int>& fragment_rows, const std::vector<int>& fragment_cols)
{	
	std::vector<IdAndPose> cpu_poses1, cpu_poses2;
	for (int vid : loop1.vertices_)
	{
		if (loop2.vertices_.find(vid) == loop2.vertices_.end())
		{
			Eigen::Matrix3d pose = loop1.poses_[vid];
			double p[6];
			p[0] = pose(0, 0);	p[1] = pose(0, 1);	p[2] = pose(0, 2);
			p[3] = pose(1, 0);	p[4] = pose(1, 1);	p[5] = pose(1, 2);
			cpu_poses1.push_back(IdAndPose(vid, p));
		}
	}
	for (int vid : loop2.vertices_)
	{
		if (loop1.vertices_.find(vid) == loop1.vertices_.end())
		{
			Eigen::Matrix3d pose = loop2.poses_[vid];
			double p[6];
			p[0] = pose(0, 0);	p[1] = pose(0, 1);	p[2] = pose(0, 2);
			p[3] = pose(1, 0);	p[4] = pose(1, 1);	p[5] = pose(1, 2);
			cpu_poses2.push_back(IdAndPose(vid, p));
		}
	}
	int overlap_pixel_num = 0;
	std::vector<int> overlap_pixel_num_array;
	DetectBatchPosesIntersection(cpu_poses1, cpu_poses2, gpu_fragments, fragment_rows, fragment_cols, overlap_pixel_num, overlap_pixel_num_array);
	for(int i=0;i<overlap_pixel_num_array.size();++i)
	{
		if (overlap_pixel_num_array[i] > Parameters::overlapped_pixels_threshold_)
			return false;
	}
	return true;
}

bool JigsawOpt2::Condition2(LoopClosure& loop1, LoopClosure& loop2)
{
	bool result = true;
	for (int vid : loop2.vertices_)
	{
		if (loop1.vertices_.find(vid) != loop1.vertices_.end())
		{
			const auto& pose1 = loop1.poses_[vid];
			const auto& pose2 = loop2.poses_[vid];
			if(!IsSameMatrix(pose1, pose2))
			{
				result = false;
				break;
			}
		}
	}
	return result;
}

bool JigsawOpt2::Condition3(LoopClosure& loop1, LoopClosure& loop2)
{
	bool verticesConsistency = true;
	bool new_vertices = false;
	for (int vid : loop2.vertices_)
	{
		if (loop1.vertices_.find(vid) != loop1.vertices_.end())
		{
			const auto& pose1 = loop1.poses_[vid];
			const auto& pose2 = loop2.poses_[vid];
			if (!IsSameMatrix(pose1, pose2))
			{
				verticesConsistency = false;
				break;
			}
		}
		else
		{
			new_vertices = true;
		}
	}

	return (verticesConsistency && new_vertices);
}


void JigsawOpt2::CalculateError2d(const std::vector<int>& abstract_loop, const std::vector<int>& concrete_loop, double& out_translation_err, double& out_rotation_err)
{
	Eigen::Matrix3d err_mat = Eigen::Matrix3d::Identity();
	for (int i = 0; i<concrete_loop.size(); ++i)
	{
		int edgeId = concrete_loop[i];
		if (i == concrete_loop.size() - 1)
		{
			if (abstract_loop[i] == multiGraph_[edgeId].frame2_ && abstract_loop[0] == multiGraph_[edgeId].frame1_)
				err_mat *= multiGraph_[edgeId].transformation_.inverse();
			else if (abstract_loop[i] == multiGraph_[edgeId].frame1_ && abstract_loop[0] == multiGraph_[edgeId].frame2_)
				err_mat *= multiGraph_[edgeId].transformation_;
			else
				assert(false && "Unexpected id sequence");
		}
		else
		{
			if (abstract_loop[i] == multiGraph_[edgeId].frame1_ && abstract_loop[i + 1] == multiGraph_[edgeId].frame2_)
				err_mat *= multiGraph_[edgeId].transformation_;
			else if (abstract_loop[i] == multiGraph_[edgeId].frame2_ && abstract_loop[i + 1] == multiGraph_[edgeId].frame1_)
				err_mat *= multiGraph_[edgeId].transformation_.inverse();
			else
				assert(false && "Unexpected id sequence");
		}
	}

	out_translation_err = sqrt(err_mat(0, 2)*err_mat(0, 2) + err_mat(1, 2)*err_mat(1, 2));
	if (abs(err_mat(0.0) - 1) < 1e-5)
		err_mat(0.0) = 1.0;
	if (abs(err_mat(0, 0) + 1)<1e-5)
		err_mat(0.0) = -1.0;
	out_rotation_err = acos(err_mat(0, 0)) * 180 / 3.14159;
}

bool JigsawOpt2::IsSameMatrix(const Eigen::Matrix3d& mat1, const Eigen::Matrix3d& mat2)
{
	Eigen::Matrix3d err_mat = mat1*mat2.inverse();
	double t_err = sqrt(err_mat(0, 2)*err_mat(0, 2) + err_mat(1, 2)*err_mat(1, 2));
	if (abs(err_mat(0.0) - 1) < 1e-5)
		err_mat(0.0) = 1.0;
	if (abs(err_mat(0, 0) + 1)<1e-5)
		err_mat(0.0) = -1.0;
	double r_err = acos(err_mat(0, 0)) * 180 / 3.14159;

	if (t_err < Parameters::T_err_threshold_ && r_err < Parameters::R_err_threshold_)
		return true;
	else
		return false;
}


std::vector<std::vector<int>> JigsawOpt2::GenerateValidableLoop(std::vector<int> abstract_loop, const int beam_width)
{
	std::vector<int> loop_candidate;
	std::vector<std::vector<int>> out_result;
	RecurLoop(beam_width, abstract_loop, 0, loop_candidate, out_result);

	return out_result;
}

void JigsawOpt2::RecurLoop(const int beam_width,
	const std::vector<int> abstract_loop,
	int level,
	std::vector<int> loop_candidate,
	std::vector<std::vector<int>>& out_result)
{
	if (out_result.size()<beam_width)
	{
		if (level == abstract_loop.size())
			out_result.push_back(loop_candidate);
		else
		{
			int v1, v2;
			if (level == abstract_loop.size() - 1)
			{
				v1 = abstract_loop[level];	v2 = abstract_loop[0];
			}
			else
			{
				v1 = abstract_loop[level];	v2 = abstract_loop[level + 1];
			}
			auto multi_edges = multiGraph_(v1, v2);
			if (multi_edges.selectId >= 0)
			{
				loop_candidate.push_back(multi_edges.selectId);
				RecurLoop(beam_width, abstract_loop, level + 1, loop_candidate, out_result);
				loop_candidate.pop_back();
			}
			else
			{
				for (int i = 0; i<multi_edges.edgeIds.size(); ++i)
				{
					loop_candidate.push_back(multi_edges.edgeIds[i]);
					RecurLoop(beam_width, abstract_loop, level + 1, loop_candidate, out_result);
					loop_candidate.pop_back();
				}
			}
		}
	}
}


void JigsawOpt2::OutputSelectEdge(std::string filename)
{
	FILE * f = fopen(filename.c_str(), "w");
	fprintf(f, "# selected relative transformation \n\n");
	for (int i = 0; i < multiGraph_.edges_.size(); i++) {
		if (multiGraph_.edges_[i].select_)
		{
			Eigen::Matrix3d & trans = multiGraph_.edges_[i].transformation_;
			fprintf(f, "%d\t%d\t%lf\n", multiGraph_.edges_[i].frame1_, multiGraph_.edges_[i].frame2_, multiGraph_.edges_[i].score_);
			fprintf(f, "%.8f %.8f %.8f\n", trans(0, 0), trans(0, 1), trans(0, 2));
			fprintf(f, "%.8f %.8f %.8f\n", trans(1, 0), trans(1, 1), trans(1, 2));
			fprintf(f, "%.8f %.8f %.8f\n", trans(2, 0), trans(2, 1), trans(2, 2));
		}
	}
	fclose(f);
}

void JigsawOpt2::DFSPose(const std::vector<int>& vertex2set)
{
	std::set<int> setId;
	for (auto s : vertex2set)
		setId.insert(s);

	int frame_num = multiGraph_.vertexNum_;
	std::vector<bool> vertex_visit(frame_num, false);

	//DFS
	struct state {
		bool in_stack;
		int level;
		int start_vertex;

		Eigen::Matrix3d transformation_;
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		state() :in_stack(false), level(-1), start_vertex(0), transformation_(Eigen::Matrix3d::Identity()) {}
		state(bool ins, int le, int start, Eigen::Matrix3d& t) :in_stack(ins), level(le), start_vertex(start), transformation_(t) {}
	};
	std::vector<state, Eigen::aligned_allocator<state>> stack_vertex(frame_num);
	std::stack<int> DFS;

	for (int set_id : setId)
	{
		std::vector<int> pose_chunk;
		pose_chunk.push_back(set_id);

		vertex_visit[set_id] = true;
		DFS.push(set_id);
		Eigen::Matrix3d ft = Eigen::Matrix3d::Identity();
		stack_vertex[set_id] = state(true, 0, 0, ft);
		while (!DFS.empty())
		{
			int item = DFS.top();
			bool add_stack = false;
			for (int i = stack_vertex[item].start_vertex; i < frame_num; ++i)
			{
				const auto multi_edge = multiGraph_(item, i);
				if (multi_edge.selectId >= 0 && stack_vertex[i].in_stack == false)
				{
					DFS.push(i);
					pose_chunk.push_back(i);

					int inx = multi_edge.selectId;
					int last_level = stack_vertex[item].level;
					Eigen::Matrix3d t;
					if (item == multiGraph_[inx].frame1_)
						t = stack_vertex[item].transformation_*multiGraph_[inx].transformation_;
					else if (item == multiGraph_[inx].frame2_)
						t = stack_vertex[item].transformation_*multiGraph_[inx].transformation_.inverse();
					else
						assert(false && "Unexpected fragment id");
					pose_.data_[i].transformation_ = t;
					stack_vertex[i] = state(true, last_level + 1, 0, t);
					add_stack = true;
					stack_vertex[item].start_vertex = i + 1;
					vertex_visit[i] = true;
					break;
				}
			}
			if (!add_stack)
			{
				DFS.pop();
			}
		}
		poseChunk_[set_id] = pose_chunk;
	}
}


void JigsawOpt2::ShowLoopClosures(const std::vector<LoopClosure>& loop_closures)
{
	for (int i=0;i<loop_closures.size();++i)
	{
		Eigen::Matrix3d offset_transform;
		auto& loop_closure = loop_closures[i];
		JigsawVisualNS::JigsawVisual visual(loop_closure.poses_, all_fragment_images);
		visual.CropImages(0, 0, 0);
		visual.AssembleAllImages(0, 0, 0, offset_transform);
		cv::Mat resized;
		cv::resize(visual.assembledImg_, resized, cv::Size(500, 500));
		cv::imshow("1", resized);
		cv::waitKey();
	}
}

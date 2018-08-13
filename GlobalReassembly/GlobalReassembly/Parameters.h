#pragma once

/////////////  This class is used for define all of parameters except score system //////////////////////
#include <vector>
class Parameters
{
public:
	static const int max_loop_length;				// search maximum loop length
	static const int beam_width;					// maximum concrete loop candidate, e.g. there might be many possible loop candidate in 1,2,3,4. We use this threshold to truncate
	static const double T_err_threshold_;			// loop translation error, (pixel)
	static const double R_err_threshold_;			// loop rotation error, (degree)
	static const int overlapped_pixels_threshold_;		// the number of allowed overlapped pixels for non-overlapping judgement
	static const double intersection_ratio_threshold_;	// intersection ratio threshold, typically 0.1
	static const double top2bottom_greedy_score_threshold_;	// greedy select score threshold 

	static const int total_loop_num;				// if merged loop closure larger than this threshold, will truncate and exit attempt

	static std::vector<std::vector<int>> isomer_4;
	static std::vector<std::vector<int>> isomer_6;
};
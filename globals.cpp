#pragma once

#include "light_field_tracking.h"



std::vector<std::vector<CircleInf>> rows;
int n_rows;
int n_cols;
std::vector<CircleInf> rows_flat;
std::vector<int> rows_offsets;
int total_circles;
TargetState target;
const int ROI_WIDETH = 4;
const int ROI_HEIGHT = 4;
ROI focus_roi{0, 0, ROI_WIDETH, ROI_HEIGHT};
const int m = 2;





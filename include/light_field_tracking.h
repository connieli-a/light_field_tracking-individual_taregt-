// light_field_tracking.h: 标准系统包含文件的包含文件
// 或项目特定的包含文件。

#pragma once

#include <iostream>
#include <opencv2/core.hpp>
#include "Icamera.h"
#include "LightFieldProcess.cuh"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
////总控类
class LightFieldTracking
{
private:
	std::unique_ptr<ICamera> camera;
	LightFieldProcess lightFieldProcess;
	cv::Mat frame;//obtained frame from camera
	cv::Mat result;

public:
	LightFieldTracking(std::unique_ptr<ICamera> camera_, const int patchSize_, const int image_rows_, const int image_cols_)
		: camera(std::move(camera_)), lightFieldProcess(patchSize_, image_rows_, image_cols_){}
	~LightFieldTracking() {}

	bool acquire();
	void process(const cv::Mat& input);
	cv::Mat getImage() const;
};



struct CircleInf
{
	//data
	float x;
	float y;
	float radius;
	CircleInf() : x(0.0f), y(0.0f), radius(0.0f) {}
	CircleInf(float x_, float y_, float radius_) : x(x_), y(y_), radius(radius_) {}
};

struct Position3D
{
	float x = 0.0f;
	float y = 0.0f;
	float z_alpha = 0.0f;
};

struct TargetState
{
	Position3D current;
	Position3D previous;

	bool isValue = false; //whether the cuurent xy position is valid
	bool hasZPosition = false; //whether the current z position is valid
	bool hasZhistory = false; //whether the previous z position is valid
	int missedFrames = 0; //number of missed frames in xy plane
};

struct ROI
{
	int x; 
	int y;
	int width;
	int height;
};
struct focusPeak {
	int index;
	float alpha;
	float score;
};

//function declaration
void extract_rows(const std::vector<CircleInf>& circleList, const float y_tolerance);
void prepare_data();


// parameter
extern std::vector<std::vector<CircleInf>> rows;
extern int n_rows;
extern int n_cols;
extern std::vector<CircleInf> rows_flat;
extern std::vector<int> rows_offsets;
extern int total_circles;
extern TargetState target;
extern ROI focus_roi;
extern const int m;


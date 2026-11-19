#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <pylon/PylonIncludes.h>
#include <pylon/BaslerUniversalInstantCamera.h> 

using namespace GenApi;


class ICamera
{
public:
	virtual ~ICamera() = default;
	virtual bool grab(cv::Mat& image) = 0;
};

class BaslerCamera : public ICamera
{
public:
	BaslerCamera(const cv::Rect& roi);
	~BaslerCamera() override;
	bool grab(cv::Mat& image) override;
private:
	cv::Rect captured_roi;
	// Declared before the camera so the pylon runtime outlives it.
	Pylon::PylonAutoInitTerm pylon_runtime;
	Pylon::CBaslerUniversalInstantCamera camera_basler;
};

class ImageCamera : public ICamera
{
	private: std::string imagePath;
	public:
		ImageCamera(const std::string& path) : imagePath(path) {}
		~ImageCamera() {}
		bool grab(cv::Mat& image) override;

};

extern int rangex1;
extern int rangex2;
extern int rangey1;
extern int rangey2;


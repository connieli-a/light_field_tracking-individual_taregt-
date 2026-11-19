// light_field_tracking.cpp: 定义应用程序的入口点。
//

#include "light_field_tracking.h"
#include "gpu_test.cuh"


using namespace std;
namespace fs = std::filesystem;

int main()
{
	//preprocess the image and detect circles in the template image
	int Rmin = 4, Rmax = 12;
	float tolerance = 4;
	int patch_size = 18;
	int x1 = 12, y1 = 12, x2 = 351, y2 = 351;
	cv::Rect roi(x1, y1, x2 - x1, y2 - y1);

	cv::Mat image_template, image, image_gray;
	cv::Mat display;
	fs::path image_template_Path = fs::path(PROJECT_DIR)/ "data/Image__2026-07-25__22-13-36.bmp";
	fs::path image_Path = fs::path(PROJECT_DIR) / "data/Image__2026-07-25__22-10-43.bmp";
	image_template = cv::imread(image_template_Path.string(), cv::IMREAD_GRAYSCALE);

	
	
	if ( image_template.empty()) {
		throw std::runtime_error("Failed to load images.");
	}
	cv::Mat cropped_template = image_template(roi);
	if (image_template.type() != CV_8UC1) {
		cv::cvtColor(cropped_template, image_gray, cv::COLOR_BGR2GRAY);
	}
	else {
		image_gray = cropped_template.clone();//this way use the cloning, pay attention to the memory management.
	}
	cv::cvtColor(cropped_template, display, cv::COLOR_GRAY2BGR);
	vector<cv::Vec3f> circles;
	cv::HoughCircles(image_gray, circles, cv::HOUGH_GRADIENT, 1.1, 14, 100, 20, Rmin, Rmax);
	/*cout << "circle's number: "<<circles.size() << endl;*/


	//obtain the circle information and store it in a vector of CircleInf
	if(!circles.empty()){
		vector<CircleInf> circleList;
		for(int i = 0; i < circles.size(); i++){
			float x = std::round(circles[i][0] * 100) / 100.0f;
			float y = std::round(circles[i][1] * 100) / 100.0f;
			float radius = std::round(circles[i][2] * 100) / 100.0f; //ROUNDING TO 2 DECIMAL PLACES

			if (x - radius < 0 || x + radius > image_gray.cols || y - radius < 0 || y + radius > image_gray.rows) {
				continue; // Skip circles that are out of bounds

			}
			circleList.emplace_back(x, y, radius);
		}
		if(circleList.empty()){
			throw std::runtime_error("No valid circles detected in the template image.");
		}
		extract_rows(circleList, tolerance);
		int rowCount = rows.front().size();
		for(auto& row: rows){
			if(row.size() != rowCount){
				throw std::runtime_error("Detected rows have different number of circles.");
			}
		}
		prepare_data();
		
		//for (auto& row: rows) {
		//	cv::Mat rowDisplay = display.clone();
		//	for (auto& circle : row) {
		//		cv::circle(display, cv::Point(cvRound(circle.x), cvRound(circle.y)), circle.radius, cv::Scalar(0, 255, 0), 2); // 绿色圆
		//	}
		//	/*cv::imshow(
		//		"Detected Row " ,
		//		rowDisplay
		//	);*/
		//	/*const int key = cv::waitKey(1000);
		//	if (key== 27) {
		//		break;
		//	}*/
		//}
		//	cv::imshow("Detected Circles", display);
		//	cv::waitKey(0);
		// 
		// 
		//captured image
		/*auto camera = std::make_unique<ImageCamera>(image_Path.string());
		LightFieldTracking tracking(std::move(camera),patch_size, y2 - y1, x2 - x1);
		if (tracking.acquire()) {
			cv::Mat cropped_image = tracking.getImage()(roi);

			tracking.process(cropped_image);

		}*/

		//captured image by camera
		try {
			auto camera = std::make_unique<BaslerCamera>(roi);
			LightFieldTracking tracking(std::move(camera), patch_size, roi.height, roi.width);
			while (tracking.acquire()) {
				cv::Mat frame = tracking.getImage();
				tracking.process(frame);
				cv::imshow("Camera Image", frame);
				if (cv::waitKey(1) == 27) { // Press 'Esc' to exit
					break;
				}
			}
		}
		catch (const Pylon::GenericException& e) {
			std::cerr << "Camera error: " << e.GetDescription() << std::endl;
			return 1;
		}
		catch (const std::exception& e) {
			std::cerr << "Error during tracking: " << e.what() << std::endl;
			return 1;
		}
	}
	else
	{
		throw std::runtime_error("No circles detected in the template image.");
	}


	return 0;
}

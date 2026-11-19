#include "Icamera.h"
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>


bool ImageCamera::grab(cv::Mat& frame) {
	frame = cv::imread(imagePath);
	return !frame.empty();
}

BaslerCamera::BaslerCamera(const cv::Rect& roi) : captured_roi(roi) {
	if (captured_roi.width <= 0 || captured_roi.height <= 0 ||
		captured_roi.x < 0 || captured_roi.y < 0) {
		throw std::invalid_argument("Invalid camera ROI.");
	}
	camera_basler.Attach(Pylon::CTlFactory::GetInstance().CreateFirstDevice());
    camera_basler.Open();

    INodeMap& nodemap = camera_basler.GetNodeMap();
    //exposure time
    CFloatPtr exposureTime(nodemap.GetNode("ExposureTime"));
    if (IsWritable(exposureTime))
        exposureTime->SetValue(300.0);//µs 7000
    CIntegerPtr width(nodemap.GetNode("Width"));
    CIntegerPtr height(nodemap.GetNode("Height"));
    CIntegerPtr offsetx(nodemap.GetNode("OffsetX"));
    CIntegerPtr offsety(nodemap.GetNode("OffsetY"));
    //color camera
    CEnumerationPtr balancesector(nodemap.GetNode("BalanceWhiteAuto"));
    CEnumerationPtr balanceRatioSelector(nodemap.GetNode("BalanceRatioSelector"));
    CFloatPtr balanceRatio(nodemap.GetNode("BalanceRatio"));
    if (IsWritable(width)) width->SetValue(captured_roi.width);
    if (IsWritable(height)) height->SetValue(captured_roi.height);
    if (IsWritable(offsetx)) offsetx->SetValue(captured_roi.x);
    if (IsWritable(offsety)) offsety->SetValue(captured_roi.y);
    if (IsWritable(balancesector)) balancesector->FromString("Off");
    if (IsWritable(balanceRatioSelector)) balanceRatioSelector->FromString("Red");
    if (IsWritable(balanceRatio)) balanceRatio->SetValue(1.0);
    if (IsWritable(balanceRatioSelector)) balanceRatioSelector->FromString("Green");
    if (IsWritable(balanceRatio)) balanceRatio->SetValue(1.0);
    if (IsWritable(balanceRatioSelector)) balanceRatioSelector->FromString("Blue");
    if (IsWritable(balanceRatio)) balanceRatio->SetValue(1.5);
    camera_basler.StartGrabbing();
}

BaslerCamera::~BaslerCamera() {
    if (camera_basler.IsGrabbing()) {
        camera_basler.StopGrabbing();
    }
	camera_basler.Close();
}

bool BaslerCamera::grab(cv::Mat& image) {
	image.release();
    if (!camera_basler.IsGrabbing()) {
        return false;
    }

    Pylon::CGrabResultPtr ptrGrabResult;
    Pylon::CPylonImage pylonImage;
    Pylon::CImageFormatConverter converter;
    converter.OutputPixelFormat = Pylon::PixelType_BGR8packed;
    camera_basler.RetrieveResult(INFINITE, ptrGrabResult,
        Pylon::TimeoutHandling_Return);
    if (!ptrGrabResult || !ptrGrabResult->GrabSucceeded()) {
        if (ptrGrabResult) {
            std::cerr << "Camera grab failed: "
                << ptrGrabResult->GetErrorDescription() << std::endl;
        }
        return false;
    }

    converter.Convert(pylonImage, ptrGrabResult);
    if (pylonImage.GetWidth() != static_cast<uint32_t>(captured_roi.width) ||
        pylonImage.GetHeight() != static_cast<uint32_t>(captured_roi.height)) {
        throw std::runtime_error("Camera frame size does not match the calibrated ROI.");
    }
    
    cv::Mat frame(static_cast<int>(pylonImage.GetHeight()),
        static_cast<int>(pylonImage.GetWidth()), CV_8UC3,
        pylonImage.GetBuffer());
    image = frame.clone();
    return !image.empty();
}

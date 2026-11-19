#pragma once
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>



struct CircleInf;
struct TargetState;
struct focusPeak;
struct Vec3f {
	float x, y, z;

	__host__ __device__ Vec3f() : x(0), y(0), z(0) {}
	__host__ __device__ Vec3f(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}

	__host__ __device__ float& operator[](int i) { return ((&x)[i]); }
	__host__ __device__ const float& operator[](int i) const { return ((&x)[i]); }
};
class LightFieldProcess {
public:
	LightFieldProcess(const int patchSize_, const int image_rows_, const int image_cols_);
	~LightFieldProcess();
	void process(const cv::Mat& frame);
private:
	void cuda_preprocess(const cv::Mat& image_mla);
	cv::Point2f find_bead(const cv::Mat& image, float threshold_value, const TargetState& target, const float& maxDistance, bool useHistory);
	bool search_xy(const cv::Mat& central_image, TargetState& target);
	std::vector<focusPeak> check_peak(const std::vector<float>& scores, const std::vector<float>& alpha_range);
	bool get_zposition(const std::vector<float>& scores, const std::vector<float>& alpha_range, TargetState& target);

	cv::cuda::GpuMat d_img;
	cudaStream_t m_stream;
	// cudaGraphExec_t m_graphExec;

	CircleInf* d_rows_flat = nullptr;
	int* d_rows_offsets = nullptr;
	uchar3* d_image = nullptr; //each orginal frame
	size_t d_image_step = 0;
	Vec3f* d_imagefloat = nullptr;//transform_type
	Vec3f* d_images = nullptr;//save the data after transform
	Vec3f* d_volume = nullptr;


	float* d_brenner_scores = nullptr;
	float* d_alpha = nullptr;//depths of the num_plane
	Vec3f* d_volume_1 = nullptr;//generate display image
	float* d_alpha_1 = nullptr;
	Vec3f* d_central_SAI = nullptr;
	//optical parameter----------
	int patchSize;
	int image_rows;
	int image_cols;
	int patch_area = patchSize * patchSize;
	int num_depth_plane = 5;
	int mid_idx = patchSize / 2;

	std::vector<float> alpha_range;
	float start = 0.85;
	float end = 1.25;
	float step = static_cast<float>(end - start) / static_cast<float>(num_depth_plane - 1);

};


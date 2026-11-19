#include "LightFieldProcess.cuh"
#include "light_field_tracking.h"
#include <cmath>
#include <limits>
#include <algorithm>

LightFieldProcess::LightFieldProcess(const int patchSize_, const int image_rows_, const int image_cols_): patchSize(patchSize_), image_rows(image_rows_), image_cols(image_cols_) {
	if (num_depth_plane == 1)
		alpha_range.push_back(1.0);
	else {
		for (int i = 0; i < num_depth_plane; ++i)
		{
			alpha_range.push_back(start + step * i);
		}
	}


	// GPU memory allocation
	cudaStreamCreate(&m_stream);
	cudaMalloc(&d_rows_flat, sizeof(CircleInf) * total_circles);
	cudaMalloc(&d_rows_offsets, sizeof(int) * rows_offsets.size());
	cudaMalloc(&d_imagefloat, sizeof(Vec3f) * image_rows * image_cols);
	cudaMalloc(&d_images, sizeof(Vec3f) * n_rows * n_cols * patch_area);
	cudaMalloc(&d_volume, num_depth_plane * n_rows * n_cols * sizeof(Vec3f));
	cudaMalloc(&d_alpha, sizeof(float) * num_depth_plane);
	cudaMalloc(&d_brenner_scores, sizeof(float) * num_depth_plane);
	//cudaMalloc(&d_volume_1, n_rows * n_cols * sizeof(Vec3f));
	//cudaMalloc(&d_alpha_1, sizeof(float));
	cudaMalloc(&d_central_SAI, sizeof(Vec3f) * n_rows * n_cols);
	size_t width_bytes = image_cols * sizeof(uchar3);
	cudaMallocPitch(&d_image, &d_image_step, width_bytes, image_rows);

	cudaMemcpyAsync(d_rows_flat, rows_flat.data(),
		sizeof(CircleInf) * total_circles,
		cudaMemcpyHostToDevice, m_stream);

	cudaMemcpyAsync(d_rows_offsets, rows_offsets.data(),
		sizeof(int) * rows_offsets.size(),
		cudaMemcpyHostToDevice, m_stream);

	cudaMemcpyAsync(d_alpha, alpha_range.data(),
		sizeof(float) * num_depth_plane,
		cudaMemcpyHostToDevice, m_stream);
	/*d_img.create(cv::Size(image_cols, image_rows), CV_8UC1);*/
	
}

LightFieldProcess::~LightFieldProcess() {
	cudaFree(d_rows_flat);
	cudaFree(d_rows_offsets);
	cudaFree(d_imagefloat);
	cudaFree(d_images);
	cudaFree(d_volume);
	cudaFree(d_alpha);
	cudaFree(d_brenner_scores);
	//cudaFree(d_volume_1);
	//cudaFree(d_alpha_1);
	cudaFree(d_central_SAI);
	cudaFree(d_image);//original frame
	cudaStreamDestroy(m_stream);
}


//类型转换 type conversion
__global__ void copy_kernel(const uchar3* __restrict__ src, const int rows, int cols, const size_t step,
	Vec3f* __restrict__ dst)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= cols || y >= rows) return;

	const uchar3* row_ptr = (const uchar3*)((const char*)src + y * step);
	uchar3 p = row_ptr[x];

	int idx = y * cols + x;
	// int idx = y * cols + x;
	// uchar3 p = src[idx]; 
	dst[idx] = Vec3f(float(p.x) / 255.f, float(p.y) / 255.f, float(p.z) / 255.f);
}
//双线性插值bilinear interpolation
__device__ inline Vec3f bilinear_lookup(
	const Vec3f* img, int n_rows, int n_cols, float x, float y)
{
	if (!(x >= 0.f && x <= n_cols - 1 &&
		y >= 0.f && y <= n_rows - 1))
		return Vec3f(0.f, 0.f, 0.f);

	int ix = static_cast<int>(floorf(x));
	int iy = static_cast<int>(floorf(y));

	int ix1 = (ix + 1 < n_cols) ? ix + 1 : ix;
	int iy1 = (iy + 1 < n_rows) ? iy + 1 : iy;
	float dx = x - ix;
	float dy = y - iy;

	Vec3f c00 = img[iy * n_cols + ix];
	Vec3f c01 = img[iy * n_cols + ix1];
	Vec3f c10 = img[iy1 * n_cols + ix];
	Vec3f c11 = img[iy1 * n_cols + ix1];

	Vec3f val;
	val[0] = (1 - dx) * (1 - dy) * c00[0] + dx * (1 - dy) * c01[0]
		+ (1 - dx) * dy * c10[0] + dx * dy * c11[0];
	val[1] = (1 - dx) * (1 - dy) * c00[1] + dx * (1 - dy) * c01[1]
		+ (1 - dx) * dy * c10[1] + dx * dy * c11[1];
	val[2] = (1 - dx) * (1 - dy) * c00[2] + dx * (1 - dy) * c01[2]
		+ (1 - dx) * dy * c10[2] + dx * dy * c11[2];
	return val;
}
__global__ void transform_kernel(Vec3f* d_img, const int image_rows, const int image_cols,
	const CircleInf* __restrict__ rows_flat, const int* __restrict__ rows_offsets,
	const int total_circles, const int n_rows, const int n_cols, const int patch_size,
	Vec3f* __restrict__ out) {
	if (!d_img || !rows_flat || !rows_offsets) return;
	int patch_area = patch_size * patch_size;
	int total = n_rows * n_cols * patch_area;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= total) return;

	// if(tid == 0){
	//     printf("kernel running!\n");
	// }

	float half = patch_size * 0.5f;
	// 还原 (uv, s, t)
	int st_idx = tid % (n_rows * n_cols); 
	int u = (tid / (n_rows * n_cols)) / patch_size;
	int v = (tid / (n_rows * n_cols)) % patch_size;

	int s = st_idx / n_cols;
	int t = st_idx % n_cols;


	// 这一行的起止（可处理“每行列数不同”的情况）
	int start = rows_offsets[s];
	// int end   = (s + 1 < n_rows) ? rows_offsets[s + 1] : total_circles;
	// int row_len = end - start;

	// // 若该 (u,v) 在这一行不存在，直接填零
	// if (v >= row_len) {
	//     out[3 * tid + 0] = 0.f;
	//     out[3 * tid + 1] = 0.f;
	//     out[3 * tid + 2] = 0.f;
	//     return;
	// }

	CircleInf ci = rows_flat[start + t];
	int idx_out = (u * patch_size + v) * n_rows * n_cols + s * n_cols + t;
	//if (!ci.valid) {
	//	out[idx_out] = Vec3f(0, 0, 0);
	//	return;
	//}

	// 找到对应的 (x, y) 坐标
	float x = ci.x + (v - half);
	float y = ci.y + (u - half);

	Vec3f bgr = bilinear_lookup(d_img, image_rows, image_cols, x, y);
	out[idx_out] = bgr;
}


__global__ void shift_and_sum_kernel(const Vec3f* d_images,  // [patch_area][n_rows][n_cols]
	Vec3f* d_volume,        // [num_depth_plane][n_rows][n_cols]
	int n_rows, int n_cols, int patch_size,
	const float* d_alpha, const int num_depth_plane)
{
	int s = blockIdx.y * blockDim.y + threadIdx.y;//row
	int t = blockIdx.x * blockDim.x + threadIdx.x;//col
	int z = blockIdx.z;  // 当前深度平面

	if (s >= n_rows || t >= n_cols || z >= num_depth_plane) return;
	int mid_idx = patch_size * 0.5f;
	int patch_area = patch_size * patch_size;
	Vec3f sum = { 0, 0, 0 };
	float alpha = d_alpha[z];
	float factor;
	if (alpha == 0)
		factor = -1;
	else
		factor = 1 - 1 / alpha;
	float radius = mid_idx * 0.75f;
	int count = 0;
	for (int idx = 0; idx < patch_area; idx++) {

		int h = idx / patch_size;
		int w = idx % patch_size;
		float r2 = (h - mid_idx) * (h - mid_idx) + (w - mid_idx) * (w - mid_idx);
		if (r2 <= radius * radius) {
			// disparity compute
			count++;
			float dx = (w - mid_idx) * factor;
			float dy = (h - mid_idx) * factor;

			// 原始像素位置（一个一个views检查）
			int img_base_idx = idx * n_rows * n_cols;

			// 浮点位移，使用双线性插值
			float uu = (float)s + dy;
			float vv = (float)t + dx;
			// uu = fminf(fmaxf(uu, 0.f), n_rows - 1.001f);
			// vv = fminf(fmaxf(vv, 0.f), n_cols - 1.001f);
			Vec3f val = bilinear_lookup(d_images + img_base_idx, n_rows, n_cols, vv, uu);

			sum[0] += val[0];
			sum[1] += val[1];
			sum[2] += val[2];
		}
	}

	// 平均
	sum[0] /= count;
	sum[1] /= count;
	sum[2] /= count;


	// 输出到 refocused volume
	int out_idx = z * n_rows * n_cols + s * n_cols + t;
	d_volume[out_idx] = sum;

}

__global__ void brenner_kernel(const Vec3f* d_volume, float* d_brenner_scores, int m, int n_rows, int n_cols, const int num_depth_plane, int roi_x, int roi_y, int roi_width, int roi_height) {
	int local_u = blockIdx.y * blockDim.y + threadIdx.y;
	int local_v = blockIdx.x * blockDim.x + threadIdx.x;
	int z = blockIdx.z;  // 当前深度平面

	
	if (local_u >= roi_height || local_v >= roi_width || z >= num_depth_plane) return;
	float brenner_val = 0.0f;

	int u = roi_y + local_u;
	int v = roi_x + local_v;

	if (u >= n_rows || v >= n_cols )
		return;

	int idx = z * n_rows * n_cols + u * n_cols + v;
	Vec3f color = d_volume[idx];
	float center = (color.x * 0.114f + color.y * 0.587f + color.z * 0.299f) * 255.0f;
	if (local_u + m < roi_height && u + m < n_rows) {
		Vec3f down_color = d_volume[z * n_rows * n_cols + (u + m) * n_cols + v];
		float down = (down_color.x * 0.114f + down_color.y * 0.587f + down_color.z * 0.299f)* 255.0f;
		brenner_val += (center - down) * (center - down);
	}
	if ( local_u - m >= 0) {
		Vec3f up_color = d_volume[z * n_rows * n_cols + (u - m) * n_cols + v];
		float up = (up_color.x * 0.114f + up_color.y * 0.587f + up_color.z * 0.299f) * 255.0f;
		brenner_val += (center - up) * (center - up);
	}
	if (v + m < n_cols && local_v + m < roi_width) {
		Vec3f right_color = d_volume[z * n_rows * n_cols + u * n_cols + (v + m)];
		float right = (right_color.x * 0.114f + right_color.y * 0.587f + right_color.z * 0.299f) * 255.0f;
		brenner_val += (center - right) * (center - right);
	}
	if (local_v - m >= 0) {
		Vec3f left_color = d_volume[z * n_rows * n_cols + u * n_cols + (v - m)];
		float left = (left_color.x * 0.114f + left_color.y * 0.587f + left_color.z * 0.299f) * 255.0f;
		brenner_val += (center - left) * (center - left);
	}
	atomicAdd(&d_brenner_scores[z], brenner_val);

}
void LightFieldProcess::process(const cv::Mat& frame) {
	
	 //  配置 kernel ?度并 launch
	int threads = 256;
	int total_threads = n_rows * n_cols * patch_area;
	int transform_blocks = (total_threads + threads - 1) / threads;
	
	cuda_preprocess(frame);

	// 在同一 stream 里执行 kernel
	transform_kernel <<<transform_blocks, threads, 0, m_stream >>> (
		d_imagefloat, image_rows, image_cols,
		d_rows_flat, d_rows_offsets,
		total_circles, n_rows, n_cols, patchSize,
		d_images
		);
	cudaError_t err_transform = cudaGetLastError();
	if (err_transform != cudaSuccess) printf("transform_Kernel error: %s\n", cudaGetErrorString(err_transform));
	cudaDeviceSynchronize();
	
	//generate the central subaperture image
	 std::vector<cv::Vec3f> central_image(n_cols * n_rows);
	 int center_u = patchSize  / 2;
	 int center_v = patchSize  / 2;
	 int central_idx = center_u * patchSize + center_v;
	 cudaMemcpyAsync(central_image.data(), d_images + central_idx * (n_cols * n_rows), central_image.size() * sizeof(Vec3f), cudaMemcpyDeviceToHost, m_stream);
	
	 cudaStreamSynchronize(m_stream);
	 cv::Mat central_image_mat(n_rows, n_cols, CV_32FC3, central_image.data());
	 if(!search_xy(central_image_mat, target)){
		 std::cout << "Target not found." << std::endl;
		 return;
	 }

	 //cv::Mat certral_image_mat_big;
	 //cv::resize(central_image_mat, certral_image_mat_big, cv::Size(n_cols * 10, n_rows * 10), 0, 0, cv::INTER_NEAREST);

	 //cv::Point2f bead_position(cvRound(target.current.x * 10), cvRound(target.current.y * 10));
	 //cv::drawMarker(certral_image_mat_big, bead_position, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 20, 2);
	 //cv::imshow("Central Subaperture Image", certral_image_mat_big);
	 //cv::waitKey(0);

	 dim3 block(16, 16);
	 dim3 grid((n_cols + 15) / 16, (n_rows + 15) / 16, num_depth_plane);

	 shift_and_sum_kernel <<<grid, block, 0, m_stream >>> (d_images, d_volume,
		 n_rows, n_cols, patchSize,
		 d_alpha, num_depth_plane);
	 cudaError_t err_shift = cudaGetLastError();
	 if (err_shift != cudaSuccess) printf("shift_Kernel error: %s\n", cudaGetErrorString(err_shift));
	

	 std::vector<cv::Mat> volume_image(num_depth_plane);
	 for (int d = 0; d < num_depth_plane; d++) {
		 volume_image[d] = cv::Mat(n_rows, n_cols, CV_32FC3);
		 cudaMemcpyAsync(volume_image[d].data, d_volume + d * n_rows * n_cols, n_rows * n_cols * sizeof(Vec3f), cudaMemcpyDeviceToHost, m_stream);
	}
	 
	 focus_roi.x = cvRound(target.current.x) - focus_roi.width / 2;
	 focus_roi.y = cvRound(target.current.y) - focus_roi.height / 2;

	 if (focus_roi.x < 0) focus_roi.x = 0;
	 else if (focus_roi.x + focus_roi.width > n_cols) focus_roi.x = n_cols - focus_roi.width;
	 if (focus_roi.y < 0) focus_roi.y = 0;
	 else if (focus_roi.y + focus_roi.height > n_rows) focus_roi.y = n_rows - focus_roi.height;
	 CV_Assert(n_cols >= focus_roi.width && n_rows >= focus_roi.height);

	 dim3 block_brenner(16, 16);
	 dim3 grid_brenner((focus_roi.width + 15) / 16, (focus_roi.height + 15) / 16, num_depth_plane);
	 cudaMemsetAsync(d_brenner_scores, 0, num_depth_plane * sizeof(float), m_stream);//
	 brenner_kernel<<<grid_brenner, block_brenner, 0, m_stream >>> (d_volume, d_brenner_scores, m, n_rows, n_cols, num_depth_plane, focus_roi.x, focus_roi.y, focus_roi.width, focus_roi.height);
	 cudaError_t err_brenner = cudaGetLastError();
	 if (err_brenner != cudaSuccess) printf("brenner_Kernel error: %s\n", cudaGetErrorString(err_brenner));



	// std::vector<cv::Mat> volume_image_show(num_depth_plane);
	// std::vector<cv::Mat> volume_image_show_big(num_depth_plane);
	// for (int i = 0; i < num_depth_plane; i++) {
	//	 volume_image_show[i].convertTo(volume_image_show_big, CV_8UC3, 255.0);
	//	 cv::resize(volume_image[i], volume_image_show[i], cv::Size(n_cols * 10, n_rows * 10), 0, 0, cv::INTER_NEAREST);
	//	 cv::rectangle(volume_image_show[i], cv::Rect(focus_roi.x * 10, focus_roi.y * 10, focus_roi.width * 10, focus_roi.height * 10), cv::Scalar(0, 0, 255), 2);
	//	 cv::imshow("vloume_images_show", volume_image_show[i]);
	//	 cv::waitKey(1000);
	//}
	//
	
	 std::vector<float> brenner_scores(num_depth_plane);
	 cudaMemcpyAsync(brenner_scores.data(), d_brenner_scores, num_depth_plane * sizeof(float), cudaMemcpyDeviceToHost, m_stream);
	 cudaStreamSynchronize(m_stream);

	 for(int i = 0; i < num_depth_plane; i++){
		 std::cout << "Depth plane " << i << " Brenner score: " << brenner_scores[i] << std::endl;
	 }
	 if (get_zposition(brenner_scores, alpha_range, target)) {
		//control the stage to target.current(x,y,z) position
	 };

	
	

	
	
	
	
	
	
	
	return;
}
void LightFieldProcess::cuda_preprocess(const cv::Mat& image_mla) {
	
	CV_Assert(image_mla.type() == CV_8UC3);
	CV_Assert(image_mla.rows == image_rows &&
		image_mla.cols == image_cols);
	/*d_img.upload(image_mla);
	cudaError_t err_upload = cudaGetLastError();
	if (err_upload != cudaSuccess) printf("copy_Kernel error: %s\n", cudaGetErrorString(err_upload));
	uchar3* d_img_ptr = d_img.ptr<uchar3>();*/
	// size_t size_bytes = image_rows * image_cols * sizeof(uchar3);
	// cudaMemcpyAsync(d_img_ptr, image_mla.ptr<uchar3>(), size_bytes, cudaMemcpyHostToDevice, m_stream);
	cudaError_t error = cudaMemcpy2DAsync(
		d_image,
		d_image_step,
		image_mla.data,
		image_mla.step,
		static_cast<size_t>(image_mla.cols) * sizeof(uchar3),
		image_mla.rows,
		cudaMemcpyHostToDevice,
		m_stream);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy2DAsync error: %s\n", cudaGetErrorString(error));
		return;
	}

	dim3 block_copy(16, 16);
	dim3 grid_copy((image_cols + 15) / 16, (image_rows + 15) / 16);
	//  分配输出 device 内存（用 float*，回传时直接拷回到 Vec3f 数组）

	copy_kernel << <grid_copy, block_copy, 0, m_stream >> > (d_image, image_rows, image_cols, d_image_step, d_imagefloat);
	cudaDeviceSynchronize();
	cudaError_t err_copy = cudaGetLastError();
	if (err_copy != cudaSuccess) printf("copy_Kernel error: %s\n", cudaGetErrorString(err_copy));
}

bool LightFieldProcess::search_xy(const cv::Mat& central_image, TargetState& target) {
	constexpr int maxMissedFrames = 3;
	const float threshold_value = 100.0f / 255.0f;
	const float maxDistance = 3.0f;

	// 必须先判断，再重置当前帧的有效标志
	const bool useHistory =
		target.isValue ||
		(target.missedFrames > 0 &&
			target.missedFrames < maxMissedFrames);

	target.isValue = false;
	target.hasZPosition = false;

	// 全局重捕获可能换了目标，不能沿用旧目标的 Z 历史
	if (!useHistory) {
		target.hasZhistory = false;
	}

	cv::Point2f bead(-1.0f, -1.0f);

	if (!central_image.empty()) {
		bead = find_bead(
			central_image, threshold_value,
			target, maxDistance, useHistory);
	}

	if (bead.x < 0.0f || bead.y < 0.0f|| bead.x > central_image.cols || bead.y > central_image.rows) {
		if (useHistory) {
			++target.missedFrames;

			if (target.missedFrames >= maxMissedFrames) {
				target.hasZhistory = false;
			}
		}
		else {
			// 全局搜索失败，不产生有效位置历史
			target.missedFrames = 0;
		}

		return false;
	}

	if (useHistory) {
		target.previous = target.current;
	}

	target.current.x = bead.x;
	target.current.y = bead.y;
	target.isValue = true;
	target.missedFrames = 0;

	return true;
	
	//target.hasZPosition = false;//reset the z position for each frame

	//if (central_image.empty()) {
	//	target.isValue = false;
	//	target.hasZhistory = false;
	//	target.hasZPosition = false;
	//	return false;
	//}
	//
	//cv::Point2f bead;
	//float threshold_value = 100.0f /255.0f;
	//float maxDistance = 3.0f;
	//if (!target.isValue) {
	//	//first frame->find the brightest bead
	//	bead = find_bead(central_image, threshold_value, target, maxDistance);
	//	if (bead.x < 0.0f || bead.y < 0.0f)
	//	{
	//		target.isValue = false;
	//		target.missedFrames++;
	//		target.hasZhistory = false;
	//		target.hasZPosition = false;
	//		return false;
	//	}
	//	target.current.x = bead.x;
	//	target.current.y = bead.y;
	//	target.missedFrames = 0;
	//	target.isValue = true;
	//}
	//else {
	//	//give the current value to previous and find the (x,y) near to the previous position
	//	target.previous = target.current;
	//	bead = find_bead(central_image, threshold_value, target, maxDistance);
	//	if (bead.x < 0.0f || bead.y < 0.0f)
	//	{
	//		target.isValue = false;
	//		target.missedFrames++;
	//		target.hasZhistory = false;
	//		target.hasZPosition = false;
	//		return false;
	//	}
	//	target.current.x = bead.x;
	//	target.current.y = bead.y;
	//	target.missedFrames = 0;
	//	target.isValue = true;
	//}
	//
	//return true;
}

cv::Point2f LightFieldProcess::find_bead(const cv::Mat& image, float threshold_value, const TargetState& target, const float& maxDistance, bool useHistory) {
	CV_Assert(image.type() == CV_32FC3);
	cv::Mat green;
	cv::extractChannel(image, green, 1);
	
	cv::Mat mask;
	cv::compare(green, threshold_value, mask, cv::CMP_GT);

	cv::Mat labels;
	cv::Mat stats;
	cv::Mat centroids;
	
	int numLabels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
	if (numLabels <= 1) {
		return cv::Point2f(-1.0f, -1.0f);
	}
	/*cv::imshow("Mask", mask);
	cv::waitKey(0);*/
	if (!useHistory) {
		// First frame or lost the target, find the brightest bead
		int brightestLabel = -1;
		float maxIntensity = -1.0f;

		for (int label = 1; label < numLabels; label++) {
			float sumIntensity = 0.0f;
			for (int y = 0; y < green.rows; y++) {
				for (int x = 0; x < green.cols; x++) {
					if (labels.at<int>(y, x) == label) {
						sumIntensity += green.at<float>(y, x);
					}
				}
			}
			if (sumIntensity > maxIntensity) {
				maxIntensity = sumIntensity;
				brightestLabel = label;
			}
		}

		float centerX = static_cast<float>(centroids.at<double>(brightestLabel, 0));
		float centerY = static_cast<float>(centroids.at<double>(brightestLabel, 1));

		return cv::Point2f(centerX, centerY);
	}
	else {
		// Subsequent frames, find the bead closest to the current position
		float bead_x = target.current.x;
		float bead_y = target.current.y;
		float minDistance = std::numeric_limits<float>::max();
		int minLabel = -1;
		for (int label = 1; label < numLabels; label++) {
			float dx = bead_x - static_cast<float>(centroids.at<double>(label, 0));
			float dy = bead_y - static_cast<float>(centroids.at<double>(label, 1));
			float sumDistance = dx * dx + dy * dy;
			if (sumDistance < minDistance) {
				minDistance = sumDistance;
				minLabel = label;
			}
		}
		if (minDistance > maxDistance * maxDistance) {
			return cv::Point2f(-1.0f, -1.0f);
		}
		float centerX = static_cast<float>(centroids.at<double>(minLabel, 0));
		float centerY = static_cast<float>(centroids.at<double>(minLabel, 1));

		return cv::Point2f(centerX, centerY);

	}
}

bool LightFieldProcess::get_zposition(const std::vector<float>& scores, const std::vector<float>& alpha_range, TargetState& target) {
	target.hasZPosition = false;
	std::vector<focusPeak> peaks = check_peak(scores, alpha_range);
	if (peaks.empty()) {
		std::cout << "No valid peaks found in the scores." << std::endl;
		return false;
	}
	std::sort(peaks.begin(), peaks.end(), [](const focusPeak& a, const focusPeak& b) {
		return a.score > b.score;
		});
	if(peaks.size() > 3) {
		peaks.resize(3);
	}
	std::sort(
		peaks.begin(),
		peaks.end(),
		[](const focusPeak& a, const focusPeak& b) {
			return a.alpha < b.alpha;
		}
	);
	float selected_z;
	if (peaks.size() == 1) {
		// only one peak, and set it as the current Z position
		selected_z = peaks[0].alpha;
	}
	else {
		// have two or three peaks
		if(!target.hasZhistory) {
			// if no previous Z position, choose the first peak
			selected_z = peaks.front().alpha;
		}
		else {
			// have many peaks, and choose the one closest to the previous Z position
			const float last_valid_z = target.current.z_alpha;
			const auto nearest_peak = std::min_element(
				peaks.begin(),
				peaks.end(),
				[last_valid_z](const focusPeak& a,
					const focusPeak& b) {
						return std::abs(a.alpha - last_valid_z) <
							std::abs(b.alpha - last_valid_z);
				}
			);
			selected_z = nearest_peak->alpha;
		}
	}
	if (target.hasZhistory) {
		target.previous.z_alpha = target.current.z_alpha;
	}

	target.current.z_alpha = selected_z;
	target.hasZPosition = true;
	target.hasZhistory = true;

	return true;
}

std::vector<focusPeak> LightFieldProcess::check_peak(const std::vector<float>& scores, const std::vector<float>& alpha_range) {
	std::vector<focusPeak> peaks;
	if(scores.size() < 3  || scores.size() != alpha_range.size()) {
		std::cout << "Scores size is too small or does not match alpha_range size, no valid peaks." << std::endl;
		return peaks;
	}

	int count = scores.size();
	std::vector<float> smooth_scores = scores;
	for(int i = 1; i < count - 1; ++i) {
		smooth_scores[i] =
			0.25f * scores[i - 1] +
			0.50f * scores[i] +
			0.25f * scores[i + 1];
	}
	const auto minmax = std::minmax_element(
		smooth_scores.begin(),
		smooth_scores.end()
	);

	const float min_score = *minmax.first;
	const float max_score = *minmax.second;
	const float score_range = max_score - min_score;

	if (score_range <= 1e-6f) {
		std::cout << "Score range is too small, no valid peaks." << std::endl;
		return peaks;
	}

	// 峰至少需要达到整个分数范围的 20%
	const float minimum_height =
		min_score + 0.20f * score_range;

	// 峰相对于左右低谷的突出度至少为 10%
	const float minimum_prominence =
		0.10f * score_range;

	// 2. 查找内部局部最大值
	for (int i = 1; i < count - 1; ++i) {
		const float left = smooth_scores[i - 1];
		const float center = smooth_scores[i];
		const float right = smooth_scores[i + 1];

		if (!(center > left && center >= right)) {
			continue;
		}

		if (center < minimum_height) {
			continue;
		}

		// 查找该峰左右两侧的最低值
		const float left_min = *std::min_element(
			smooth_scores.begin(),
			smooth_scores.begin() + i
		);

		const float right_min = *std::min_element(
			smooth_scores.begin() + i + 1,
			smooth_scores.end()
		);

		const float prominence =
			center - std::max(left_min, right_min);

		if (prominence < minimum_prominence) {
			continue;
		}

		/*
		 * 3. 使用相邻三个点进行二次曲线拟合：
		 *
		 * y = ax² + bx + c
		 *
		 * 这里支持 alpha_range 的间隔不完全相等。
		 */
		const double x1 = alpha_range[i - 1];
		const double x2 = alpha_range[i];
		const double x3 = alpha_range[i + 1];

		const double y1 = smooth_scores[i - 1];
		const double y2 = smooth_scores[i];
		const double y3 = smooth_scores[i + 1];

		const double denominator1 = (x1 - x2) * (x1 - x3);
		const double denominator2 = (x2 - x1) * (x2 - x3);
		const double denominator3 = (x3 - x1) * (x3 - x2);

		double fitted_alpha = x2;
		double fitted_score = y2;

		if (std::abs(denominator1) > 1e-12 &&
			std::abs(denominator2) > 1e-12 &&
			std::abs(denominator3) > 1e-12) {

			const double a =
				y1 / denominator1 +
				y2 / denominator2 +
				y3 / denominator3;

			const double b =
				-y1 * (x2 + x3) / denominator1
				- y2 * (x1 + x3) / denominator2
				- y3 * (x1 + x2) / denominator3;

			const double c =
				y1 * x2 * x3 / denominator1 +
				y2 * x1 * x3 / denominator2 +
				y3 * x1 * x2 / denominator3;

			// a < 0 才是向下开口的峰
			if (a < -1e-12) {
				const double vertex = -b / (2.0 * a);

				// 拟合结果必须位于相邻两个深度平面之间
				if (vertex >= x1 && vertex <= x3) {
					fitted_alpha = vertex;
					fitted_score =
						a * vertex * vertex +
						b * vertex +
						c;
				}
			}
		}

		peaks.push_back({
			i,
			static_cast<float>(fitted_alpha),
			static_cast<float>(fitted_score)
			});
	}
	
	return peaks;
}

#ifndef FACE_TRANSFORM_HPP
#define FACE_TRANSFORM_HPP

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

class FaceTransform
{
	public:
		cv::Point2f getCenter(const int left, const int top, const int right, const int bottom);
		float getSize(const int left, const int top, const int right, const int bottom);
		cv::Mat getTransformMatrix(const cv::Point2f center, const float size);
		cv::Mat crop(const cv::Mat& image, const cv::Mat& transform_Matrix);
		cv::Mat transformBack(cv::Mat& transMat, const std::vector<float>& landmarks);
		void drawLandmarks(cv::Mat & image, const cv::Mat & landmarks);
};

#endif
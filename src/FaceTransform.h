/*
 * This file is part of project link.developers/ld-node-facial-landmark-detector-2.
 * It is copyrighted by the contributors recorded in the version control history of the file,
 * available from its original location https://gitlab.com/link.developers.beta/ld-node-facial-landmark-detector-2.
 *
 * SPDX-License-Identifier: MPL-2.0
 */
 
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

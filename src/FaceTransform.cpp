/*
 * This file is part of project link.developers/ld-node-facial-landmark-detector-2.
 * It is copyrighted by the contributors recorded in the version control history of the file,
 * available from its original location https://gitlab.com/link.developers.beta/ld-node-facial-landmark-detector-2.
 *
 * SPDX-License-Identifier: MPL-2.0
 */
 
#include "FaceTransform.h"
#include <algorithm>

cv::Point2f FaceTransform::getCenter(const int left, const int top, const int right, const int bottom) 
{
	int old_size = (right - left + bottom - top) / 2.0f;
	cv::Point2f center;
	center.x = right - (right - left) / 2.f;
	center.y = bottom - (bottom - top) / 2.f + old_size * 0.14f;
	return center;
}

cv::Mat FaceTransform::getTransformMatrix(const cv::Point2f center, const float size) 
{
	std::vector<cv::Point2f> srcPts;
	std::vector<cv::Point2f> dstPts;
	cv::Mat transMat(2, 3, CV_32F);
	srcPts.push_back(cv::Point2f(center.x - size / 2, center.y - size / 2));
	srcPts.push_back(cv::Point2f(center.x - size / 2, center.y + size / 2));
	srcPts.push_back(cv::Point2f(center.x + size / 2, center.y - size / 2));
	dstPts.push_back(cv::Point2f(0.0, 0.0));
	dstPts.push_back(cv::Point2f(0.0, 255.0));
	dstPts.push_back(cv::Point2f(255.0, 0.0));
	transMat = cv::getAffineTransform(srcPts, dstPts);
	return transMat;
}

float FaceTransform::getSize(const int left, const int top, const int right, const int bottom) 
{
	float old_size = (right - left + bottom - top) / 2.0f;
	return old_size*1.58f;
}

cv::Mat FaceTransform::crop(const cv::Mat& image, const cv::Mat& transform_Matrix)
{
	cv::Mat roi;
	cv::warpAffine(image, roi, transform_Matrix, cv::Size(256, 256));
	return roi;
}

void FaceTransform::drawLandmarks(cv::Mat& image, const cv::Mat& landmarks) 
{
	for (int i = 0; i < landmarks.rows; i++) 
	{
		if (landmarks.at<int>(i,0) < 0 || landmarks.at<int>(i, 0) >= image.cols || 
		    landmarks.at<int>(i, 1) < 0 || landmarks.at<int>(i, 1) >= image.rows)
		{
			continue;		
		}

		cv::Point onePoint(landmarks.at<int>(i, 0), landmarks.at<int>(i, 1));
		cv::circle(image, onePoint, 1, cv::Scalar(255, 0, 255), 2);
	}
}

cv::Mat FaceTransform::transformBack(cv::Mat& trans_Matrix, const std::vector<float>& landmarks) 
{
	cv::Mat origin_landmarks;
	cv::Mat third_row = cv::Mat::zeros(1, 3, CV_64F);
	third_row.at<double>(0,2) = 1;
	trans_Matrix.push_back(third_row);
	cv::Mat inv_transMat = trans_Matrix.inv();
	
	for (int i = 0; i < landmarks.size(); i = i + 3) 
	{
		cv::Mat origin_landmark(1, 3, CV_32SC1);
		origin_landmark.at<int>(0,2) = int(landmarks[i+2] / trans_Matrix.at<double>(cv::Point(0, 0)));
		cv::Mat old_pt = cv::Mat::zeros(3, 1, CV_64F);
		old_pt.at<double>(0, 0) = landmarks[i];
		old_pt.at<double>(1, 0) = landmarks[i+1];
		old_pt.at<double>(2, 0) = 1.0f;
		cv::Mat origin_pt = inv_transMat * old_pt;
		origin_landmark.at<int>(0, 0) = int(origin_pt.at<double>(0, 0));
		origin_landmark.at<int>(0, 1) = int(origin_pt.at<double>(1, 0));
		origin_landmarks.push_back(origin_landmark);
	}
	
	return origin_landmarks;
}

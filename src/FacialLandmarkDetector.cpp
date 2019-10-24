/*
 * This file is part of project link.developers/ld-node-facial-landmark-detector-2.
 * It is copyrighted by the contributors recorded in the version control history of the file,
 * available from its original location https://gitlab.com/link.developers.beta/ld-node-facial-landmark-detector-2.
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#include <opencv2/highgui.hpp>
#include <link_dev/Interfaces/OpenCvToImage.h> 

#include "FacialLandmarkDetector.h"
#include "FaceTransform.h"
#include "LandmarkPredictor.h"

void link_dev::Services::FacialLandmarkDetector::Load_uv_indices(const std::string & filePath) 
{	
	std::ifstream ifs(filePath);
	
	if (!ifs) 
	{
		std::cerr << "File not found or failed to open : " << filePath << std::endl;
	}
	std::string s;
	
	while (ifs >> s) 
	{
		uint val = static_cast<uint>(std::stof(s));
		m_uv_kpt_indices.push_back(val);
	}
	
	if (m_uv_kpt_indices.size() != (2 * 68)) 
	{
		std::cerr << "Invalid number of UV values. Must be 2 * 68, but got " << m_uv_kpt_indices.size() << std::endl;
	}
}

int link_dev::Services::FacialLandmarkDetector::Run() 
{
	/*Initialize the landmarkPredictor class with input and output layer and the path
	   to the (protobuf format) model file.*/
	m_predictor.loadModel(m_pathToModel, "Placeholder", "resfcn256/Conv2d_transpose_16/Sigmoid");
	std::cout << "Model is loaded." << std::endl;
	
	Load_uv_indices(m_pathToUVIndices);
	std::cout << "UV-DATA is loaded." << std::endl;

	m_inputPin.addOnDataCallback("l2demand:/image_with_bounding_boxes", 
	[&](const ImageWithBoundingBoxT& imageWithBB)
	{
		HandleNewFrame(imageWithBB.imageWithFace, imageWithBB.boxes);
	});

	while(m_signalHandler.receiveSignal() != LINK2_SIGNAL_INTERRUPT);
	
	std::cout << "Ending the Run() function." << std::endl;
	return 0;
}

void link_dev::Services::FacialLandmarkDetector::HandleNewFrame(
	                        const std::unique_ptr<link_dev::ImageT>& frame, 
                            const std::vector<std::unique_ptr<BoundingBoxT>>& boundingBoxes) 
{
	cv::Mat allFacesLandmarks; /*Landmarks are stored here*/
	std::vector<cv::Rect> faceLocations; /*Bounding boxes (in opencv format)for all faces 
	                                       in the current frame will be stored here.*/
	cv::Mat imageFrame = link_dev::Interfaces::ImageToOpenCV(*frame); /*Local copy of the current
	                                                                    frame*/
	std::vector<BoundingBoxT> imageBoundingBoxes; /*Local copy of bounding boxes in BoundingBoxT
	                                                format.*/

	/*Create a copy of incoming bounding boxes in a local vector*/
	for(auto& obj : boundingBoxes)
	{
		imageBoundingBoxes.push_back(*obj);
	}

	/*Convert a vector of Bounding Boxes to a vector of cv::Rect*/
	for(std::vector<BoundingBoxT>::iterator iter = imageBoundingBoxes.begin(); 
	    iter != imageBoundingBoxes.end(); ++iter)
	{
		faceLocations.push_back(cv::Rect(iter->left, iter->top, /*x, y cordinates of top left*/
		                                 iter->right - iter->left, iter->bottom - iter->top));
										/*width, height*/		
	}

	if (imageFrame.channels() == 1) 
	{
		cv::cvtColor(imageFrame, imageFrame, CV_GRAY2BGR);
	}

	if (faceLocations.size() > 0)
	{
		/*Iterate over each "face" found in the current frame.*/
		for (const cv::Rect& face : faceLocations) 
		{
			int left = face.x;
			int top = face.y;
			int right = face.x + face.width;
			int bottom = face.y + face.height;
			
			/*Create a crop of the face*/
			cv::Point2f center = m_faceTransFormer.getCenter(left, top, right, bottom);
			float size = m_faceTransFormer.getSize(left, top, right, bottom);
			cv::Mat transMat = m_faceTransFormer.getTransformMatrix(center, size);
			cv::Mat croppedFace = m_faceTransFormer.crop(imageFrame, transMat);
			
			/*Create input tensor using the norm*/
			cv::Mat norm_face;
			croppedFace.convertTo(norm_face, CV_32FC3, 1.0 / 255, 0);
			Tensor inpTensor(DT_FLOAT, TensorShape({ 1,256,256,3 }));
			float * inpPt = inpTensor.flat<float>().data();
			cv::Mat inpImg(256, 256, CV_32FC3, inpPt);
			norm_face.convertTo(inpImg, CV_32FC3);

			/*Hard for me to guess why this has been done exactly in this way.
			  What has been done is clear, but the why eludes me.*/
			Tensor outTensor = m_predictor.predict(inpTensor);
			std::vector<float> landmarks = m_predictor.getLandmarks(outTensor, m_uv_kpt_indices);
			cv::Mat origin_landmarks = m_faceTransFormer.transformBack(transMat, landmarks);
			if (m_visualize)
			{
				/*TODO Explain what happens here*/
				m_faceTransFormer.drawLandmarks(imageFrame, origin_landmarks);
			}
			else
			{
				allFacesLandmarks.push_back(origin_landmarks);
			}
		}
	}
	
	if(m_visualize) 
	{
		m_outputPin.push(link_dev::Interfaces::ImageFromOpenCV(imageFrame, 
		                                                       link_dev::Format::Format_RGB_U8),
													           "l2offer:/imagesWithLandmarks");
	}
	else 
	{
		m_outputPin.push(link_dev::Interfaces::ImageFromOpenCV(allFacesLandmarks,
			                                                   link_dev::Format::Format_GRAY_U8),
													           "l2offer:/imagesWithLandmarks");
	}
}


#pragma once

/*
#include <link_dev/Data/Imaging/Image2D.pb.h>
#include <link_dev/Data/Math/List_Tuple_UInt64_UInt64.pb.h>
#include <link_dev/Data/Math/GenericMatrix3D.pb.h>
#include <link_dev/Data/Imaging/Image2D_Plus_Points.pb.h>
*/
#include <iostream>
#include <DRAIVE/Link2/NodeDiscovery.hpp>
#include <DRAIVE/Link2/NodeResources.hpp>
#include <DRAIVE/Link2/SignalHandler.hpp>
#include <DRAIVE/Link2/OutputPin.hpp>
#include <DRAIVE/Link2/InputPin.hpp>

#include "ImageWithBB_generated.h"

#include "FaceTransform.h"
#include "LandmarkPredictor.h"

namespace link_dev 
{
	namespace Services 
	{
		class FacialLandmarkDetector
		{
			DRAIVE::Link2::SignalHandler m_signalHandler;
     	 	DRAIVE::Link2::NodeResources m_nodeResources;
      		DRAIVE::Link2::NodeDiscovery m_nodeDiscovery;
      		DRAIVE::Link2::OutputPin m_outputPin;
			DRAIVE::Link2::InputPin m_inputPin;

			std::vector<uint> m_uv_kpt_indices;
			FaceTransform m_faceTransFormer;
			LandmarkPredictor m_predictor;

			bool m_visualize = false;
			std::string m_pathToModel = "";
			std::string m_pathToUVIndices = "";

			void HandleNewFrame(const std::unique_ptr<link_dev::ImageT>& frame, 
                                const std::vector<std::unique_ptr<BoundingBoxT>>& boundingBoxes);
			void Load_uv_indices(const std::string & filePath);
			
		public:
			FacialLandmarkDetector(DRAIVE::Link2::SignalHandler signalHandler,
                   				   DRAIVE::Link2::NodeResources nodeResources,
                   				   DRAIVE::Link2::NodeDiscovery nodeDiscovery,
                                   DRAIVE::Link2::OutputPin outputPin,
						           DRAIVE::Link2::InputPin inputPin,
								   bool visualise,
								   std::string pathToUVData,
								   std::string pathToModel) :
								   m_signalHandler(signalHandler),
                   				   m_nodeResources(nodeResources),
                                   m_nodeDiscovery(nodeDiscovery),
                   				   m_outputPin(outputPin),
						           m_inputPin(inputPin),
								   m_visualize(visualise),
								   m_pathToUVIndices(pathToUVData),
								   m_pathToModel(pathToModel) 
			{}					   

			int Run();
	
		};
	}

}

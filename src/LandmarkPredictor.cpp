#include "LandmarkPredictor.h"

bool LandmarkPredictor::loadModel(const std::string& graph_filename, 
                                  const std::string& inp_layer, 
								  const std::string& out_layer) 
{
	tensorflow::GraphDef graph_def;
	Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_filename, &graph_def);
	if (!load_graph_status.ok()) 
	{
		std::cerr << tensorflow::errors::NotFound("Failed to load compute graph at '", graph_filename, "'");
		return false;
	}

	session.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
	Status session_create_status = session->Create(graph_def);
	if (!session_create_status.ok()) 
	{
		std::cerr << session_create_status;
		return false;
	}
	
	input_layer = inp_layer;
	output_layer = out_layer;
	
	return true;
}

Tensor LandmarkPredictor::predict(const Tensor & input_tensor) 
{
	std::vector<Tensor> output_tensors;
	Status run_status = session->Run({ { input_layer, input_tensor } },
	                                 { output_layer }, {}, 
									 &output_tensors);
	if (!run_status.ok()) 
	{
		std::cerr << "Running model failed: " << run_status;
	}

	return output_tensors[0];
}

std::vector<float> LandmarkPredictor::getLandmarks(const Tensor& result, 
                                                   const std::vector<uint>& uv_kpt_indices) 
{
	std::vector<float> landmarks;
	TTypes<float, 4>::ConstTensor tensor = result.tensor<float, 4>();
	
	for (int i = 0; i < (uv_kpt_indices.size() / 2); i++) 
	{
		uint x_idx = uv_kpt_indices[i];
		uint y_idx = uv_kpt_indices[i + uv_kpt_indices.size() / 2];
		auto real_x = tensor(0, y_idx, x_idx, 0) * 256 * 1.1;
		auto real_y = tensor(0, y_idx, x_idx, 1) * 256 * 1.1;
		auto real_z = tensor(0, y_idx, x_idx, 2) * 256 * 1.1;
		landmarks.push_back(real_x);
		landmarks.push_back(real_y);
		landmarks.push_back(real_z);
	}
	
	return landmarks;
}



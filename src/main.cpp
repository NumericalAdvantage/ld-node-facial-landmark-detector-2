/*
 * This file is part of project link.developers/ld-node-facial-landmark-detector-2.
 * It is copyrighted by the contributors recorded in the version control history of the file,
 * available from its original location https://gitlab.com/link.developers.beta/ld-node-facial-landmark-detector-2.
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#include "FacialLandmarkDetector.h"

int main(int argc, char** argv)
{
    try 
    {
        DRAIVE::Link2::NodeResources nodeResources { "l2spec:/link_dev/ld-node-facial-landmark-detector-2", 
                                                      argc, argv };
        DRAIVE::Link2::NodeDiscovery nodeDiscovery { nodeResources };

        DRAIVE::Link2::ConfigurationNode rootNode = nodeResources.getUserConfiguration();
        DRAIVE::Link2::OutputPin outputPin{nodeDiscovery, nodeResources, "imagesWithLandmarks"};
        DRAIVE::Link2::InputPin inputPin{nodeDiscovery, nodeResources, "imagesWithBoundingBox"};

        DRAIVE::Link2::SignalHandler signalHandler {};
        signalHandler.setReceiveSignalTimeout(-1);
        
        link_dev::Services::FacialLandmarkDetector fLDetector(signalHandler, nodeResources, 
		                                                      nodeDiscovery, outputPin, inputPin,
                                                              rootNode.getBoolean("Visualize"),
                                                              rootNode.getString("PathToUVData"),
                                                              rootNode.getString("PathToModel"));

        return fLDetector.Run();
    } 
    catch (const std::exception& e) 
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}

/*
References:
[1]  https://github.com/YadiraF/PRNet
[2]  https://github.com/lighttransport/prnet-infer/blob/master/README.md
[3]  https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f
[4]  https://medium.com/coinmonks/tensorflow-graphs-and-sessions-c7fa116209db 
[5]  https://danijar.com/what-is-a-tensorflow-session/
*/

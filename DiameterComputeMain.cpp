// ROS Includes
#include <ros/ros.h>
#include <rosbag/bag.h>           
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/CompressedImage.h>
#include <std_msgs/Time.h>
#include <image_proc/processor.h>
#include <camera_calibration_parsers/parse.h>
// Realsense SDK Include (Only for rs_intrinsic*)
#include <rs.hpp>
// Boost Includes
#include <boost/foreach.hpp>
#include <boost/progress.hpp>
#include <boost/format.hpp>
// OpenCV Includes
#include <opencv2/highgui/highgui.hpp> // for cv::imwrite
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/ocl.hpp>
//stdlib Includesf
#include <stdio.h>
#include <mutex>
#include <thread>
#include <queue>
#include <signal.h>
#include <numeric> //for std::accumulate
#include <unistd.h>
//DepthCamProcess Includes
#include "Compute.h"
#include "Defines.h"
#include "Descriptors.h"
#include "RecordingTriggerROI.h"
#include <realsense_fia_dbh/diameter_compute_pointcloud_sync.h>
#include <realsense_fia_dbh/depth_segnet_sync.h>
#include "UserIOStateMachine.h"
#include "NormalMapGradientCompute.h"

namespace DepthCamProcessing
{
    //Use Compute Namespace for convenience
    using namespace Compute;

    namespace
    {
        //----------------------------------------------------------
        //  Anonymous Structs / Typedefs / Enums
        //----------------------------------------------------------
        enum FIA_DiameterComputationStatus
        {
            COMPUTING_DIAMETER,     //There is a tree in the frame, and we are computing it's diameter
            STANDBY                 //We are not computing diameter right now
        };

        //----------------------------------------------------------
        //  Anonymous Variables
        //----------------------------------------------------------

        //Controls main program loop. Set to false by custom SIGINT callback routine.
        static bool run = true;
        //DBH Computation
        static std::vector<float> DBHList;
        static std::vector<float> taperList;
        static std::vector<float> sixFTlist;
        static std::vector<float> lowBranchHeightList;
        static FIA_DiameterComputationStatus diameterComputationStatus = STANDBY;
        static int treeID = 0;
        //Create a flag that indicates whether we have received at least ONE CameraInfo message
        //At least one message is required to do processing on the depth image to update the 'rs2_camera_intrinsics' variable
        //Without a valid 'rs2_camera_intrinsics' measuring distance on the depth image will not work.
        //When a CameraInfo message is received, this is set to 'true'
        static bool cameraInfoMessageReceived                               = false;
        static Descriptors::RS2ParameterDescriptor rs2_descriptor           = {0};
        static Descriptors::AlgorithmDescriptor algorithm_descriptor        = {0};
        static RecordingTriggerROI triggerROI = RecordingTriggerROI(
		    0.0,
		    0.0,
	        0.0,
		    0.0,
		    0.0, 0.0, 
		    0.0
        );
        //Intrinsic camera parameters. Used to comptue DBH
        static rs2_intrinsics camera_intrinsics = {0};

        //Timeout duration since last message received of five seconds
        static const ros::Duration timeout_duration(3.0);
        //Mutex - Used for publisher
        static std::mutex alignLocalPointcloudMutex;
        static std::vector<realsense_fia_dbh::diameter_compute_pointcloud_sync> alignLocalPointcloudMessageQueue;

        // TEMPORARY - REMOVE THIS AT EARLIEST POSSIBLE TIME
        // Replace this variable with a parameter from alg.desc
        // static const bool neuralNetworkInterfacingEnabled = true;  //replaced with algdesc.NEURAL_NET_INTERFACING
        // Data for interfacing with neural network output
        static bool isWaitingForNeuralNetworkOutput = false;
        // Neural net output image
        static cv::Mat neuralNetworkOutputSegmentedImage;
        // Neural net input image
        static cv::Mat neuralNetworkInputDepthImage;
        // Incremented every time a neural network input is received
        // Used for indentifying when a new neural network output has been received
        static bool receivedNeuralNetworkOutput = false;

        //Keep track of the 1) the last depth_segnet_sync message sent and 2) the most recent neural network output received
        //These two form a corresponding pair of input/output pair
        //Idea: Keep track of the number of input depth images sent out and the number of output segmented images received
        //We will have to rearrange the main function to account for this
        //After sending an inputDepthImage, we must increment the variable and WAIT for the neural network output to be received
        //We know that we have a corresponding input/output pair if their counters are the same value. The inputDepthImageSentCtr is incremented first,
        //and the outputSegmentedImageCtr is incremented after some time has passed and we receive a neural network output 
        
        //We use a mutex to ensure that we safely read/write outputSegmentedImageReceivedCtr
        std::mutex segmentedImageReceivedCtrMutex;
        static unsigned int inputDepthImageSentCtr = 0;
        static unsigned int outputSegmentedImageReceivedCtr = 0;
        static cv::Mat inputDepthImage;
        static cv::Mat outputSegmentedImage;

        //variables to keep track of tree seperation
        int framesWithoutTree = 0;
        int treeCounter = 0;

        //Lowest branch detection
        std::vector<int> tNums;
        std::vector<cv::Point> tCols;

        //Trunk straightness
        std::vector<float> areasComputed = std::vector<float>{};
        int prevBin = 0;

        //Flag to initialize ROI bounds
        int initROI = 1;

        //----------------------------------------------------------
        //  Anonymous Functions
        //----------------------------------------------------------

        //Interrupt handler that replaces default SIGINT.
        void RS_DBH_SIGINT_Handler(int signal_id)
        {
            ROS_INFO("Shutting down...");
            ros::shutdown();
            run = false;
        }

        //If the video ends while we are recording a DBH, this DBH will not get published. This function handles this edge case.
        void PublishFinalDBHAfterRecordingEnd(void)
        {
            if(diameterComputationStatus == COMPUTING_DIAMETER)
            {
                //Go back to idle / standby for trunk
                diameterComputationStatus = STANDBY;
                //print the average DBH (cm) in terminal
                float diameter = GetDBHWithoutOutliers(DBHList);
                if(diameter <= 4.0f)
                {
                    DBHList.clear();
                    return;
                }
                else
                {
                    ROS_INFO("Tree %d has DBH %lf", ++treeID, diameter );
                    //Clear the DBHList
                    DBHList.clear();
                }
            }
        }
        //UserIO::UserIOVars userIOVars = { 0 };

        //Function to rotate image (more complex than I though it would be) 

        ros::Publisher depthSegnetSyncPublisher;  //move this up to anon vars

        void DepthSegNetSyncMessagePublish(const sensor_msgs::Image::ConstPtr depthImagePtr, const ros::Publisher & depthSegnetSyncPublisher)
        {
            realsense_fia_dbh::depth_segnet_sync depthSegnetSyncMsg;
            static int depthSegnetSyncCounter = 0;
            depthSegnetSyncMsg.depthImage = *depthImagePtr;
            depthSegnetSyncMsg.header.stamp = ros::Time::now();
            depthSegnetSyncMsg.header.seq = depthSegnetSyncCounter++;
            depthSegnetSyncMsg.header.frame_id = "/realsense_fia/depth_segnet_sync";
            depthSegnetSyncPublisher.publish (depthSegnetSyncMsg);
            //Track the previously sent 
            //lastSentDepthSegnetSyncMessage = depthSegnetSyncMsg;
        }

        struct pixelCoords {
            int x;  //cols
            int y;  //rows
            bool gotIntersect;
        };
        typedef struct pixelCoords pixelCoords_t;

        //This function takes in a segmented image and finds the approximate trunk intersection
        //It searches the segmented image (class map) horizontally row by row from the bottom up
        //The first trunk pixel encountered is assumed to be the approximate trunk intersection
        pixelCoords_t ApproxGroundTrunkIntersectSegment(cv::Mat cvMatImg) {
            //printf("calculating approx trunk intersect\n");
            int breakpoint = cvMatImg.rows/20; //if a row is more than 5% trunk, then that's the approx intersect
            int x = -1; //target x and y coords of approx intersect
            int y = -1; //note that x corresponds to columns and y corresponds to rows
            bool gotIntersect = false;
            
            cv::Mat classMapReadable = cvMatImg.clone();
            for (int i = 0; i < classMapReadable.rows; i++) {
                for (int j = 0; j < classMapReadable.cols; j++) {
                    classMapReadable.at<uchar>(i,j) = classMapReadable.at<uchar>(i,j)*25;
                }
            }
            for(int i = cvMatImg.rows - 1; i > cvMatImg.rows/2; i--) {
                int trunkPixels = 0;
                int leftMostTrunk = -1;
                bool gotLeftMost = false;
                int rightMostTrunk = -1;
                for (int j = 0; j < cvMatImg.cols; j++) {
                    if (cvMatImg.at<uchar>(i, j) == 2) { //0: background, 1: ground, 2: trunk, 3: branch
                        trunkPixels++;
                        if (!gotLeftMost) {
                            //printf("making sure this only get run once for every i\n");
                            leftMostTrunk = j;
                            gotLeftMost = true;
                        }
                        rightMostTrunk = j;
                    }
                }
                //printf("trunkPixels: %d\n", trunkPixels);
                if (trunkPixels > breakpoint) {
                    // printf("trunk detected\n");
                    // printf("i: %d\n", i);
                    // printf("left most trunk: %d\n", leftMostTrunk);
                    // printf("right most trunk: %d\n", rightMostTrunk);
                    y = i;
                    x = (leftMostTrunk + rightMostTrunk)/2;
                    break;
                }
            }

            if (x == -1) {
                assert(y == -1);
            } else {
                assert(x >= 0 && x < cvMatImg.cols);
                assert(y >= 0 && y < cvMatImg.rows);
                gotIntersect = true;
            }

            // printf("returning\n");
            // printf("x (cols): %d\n", x);
            // printf("y (rows): %d\n", y);
            // printf("gotintersect: %s\n", gotIntersect?"true":"false");
            return {x, y, gotIntersect};
        }

        //This function takes in a depth image and the coordinates of where the trunk intersect is approximately
        //It then does a normal map analysis of the surrounding area, divinding the area of interest into grids
        //Grid dominantness analysis is then performed to locate the grid with the trunk, trunk assumed to be grid center
        pixelCoords_t ExactGroundTrunkIntersectNormal(cv::Mat depthImage, pixelCoords_t approxIntersect) {
            //printf("calculate exact trunk intersect\n");
            // cv::imshow("exact_depth_image", depthImage);
            // cv::waitKey(1000);
            //setting bound for region of interest
            //upperbound: minus 16 and round down to 16 % 0
            //lowerbound: 0
            //leftbound: minus 16 and round down to 16 % 0
            //rightbound: add 16 and round up to 16 % 0
            //0 STARTS FROM THE TOP OF THE IMAGE and INCREASES DOWNWARD
            //  THUS the UPPER bound will have a LOWER value than the LOWER bound
            int ROIUpperBound = approxIntersect.y - 16;
            ROIUpperBound = ROIUpperBound - (ROIUpperBound % 16);
            assert(ROIUpperBound % 16 == 0);
            int ROILowerBound = depthImage.rows - 1;
            int ROILeftBound = approxIntersect.x - 16;
            ROILeftBound = ROILeftBound - (ROILeftBound % 16);
            assert(ROILeftBound % 16 == 0);
            int ROIRightBound = approxIntersect.x + 16;
            ROIRightBound = ROIRightBound + (16 - (ROIRightBound % 16));
            assert(ROIRightBound % 16 == 0);

            //if out of bounds, set bound to be image edges
            if (ROIUpperBound < 0) {
                ROIUpperBound = 0;
            }
            if (ROILeftBound < 0) {
                ROILeftBound = 0;
            }
            if (ROIRightBound > depthImage.cols - 1) {
                ROIRightBound = depthImage.cols - 1;
            }

            // printf("ROIUpperBound: %d\n", ROIUpperBound);
            // printf("ROILowerBound: %d\n", ROIUpperBound);
            // printf("ROILeftBound: %d\n", ROIUpperBound);
            // printf("ROIRightBound: %d\n", ROIUpperBound);

            //step 1: generating normal map
            //printf("generating normal map\n");
            cv::Mat depthImageFloatNorm; 	
			cv::Mat surfaceNormalImage = cv::Mat(depthImage.rows, depthImage.cols, CV_16UC3, cv::Scalar(0, 0, 0));
            depthImage.convertTo(depthImageFloatNorm, CV_32FC1, 1.0f);
            for (int i = ROIUpperBound + 1; i < ROILowerBound - 1; i++) {
                //getting row ptrs, also why do the row ptrs have to floats?
                float* rowAbovePtr = depthImageFloatNorm.ptr<float>(i-1);
                float* rowCurrentPtr = depthImageFloatNorm.ptr<float>(i);
                float* rowBelowPtr = depthImageFloatNorm.ptr<float>(i+1);
                //computing normals
                for (int j = ROILeftBound + 1; j < ROIRightBound - 1; j++) {
                    //getting row pointers
                    float above = rowAbovePtr[j];
                    float below = rowBelowPtr[j];
                    float left = rowCurrentPtr[j-1];
                    float right = rowCurrentPtr[j+1];

                    //calculating surface normals
                    if (above == 0.0f || below == 0.0f || left == 0.0f || right == 0.0f) {
                        surfaceNormalImage.at<cv::Vec3w>(i, j) = cv::Vec3w(0, 0, 0);
                        continue;
                    }
                    float dzdx = (above - below) / 2.0f;
                    float dzdy = (right - left) / 2.0f;
                    //float magnitude = sqrtf(dzdx*dzdx + dzdy*dzdy + 1.0f); //wait why + 1f*1f at the end?

                    // printf("i: %d, j: %d\n", i, j);
                    // printf("\tabs(dzdx): %f\n", abs(dzdx));
                    // printf("\tabs(dzdy): %f\n", abs(dzdy));

                    //converting to unit vectors
                    if (abs(dzdx) > abs(dzdy) && abs(dzdx) > 1.0f) surfaceNormalImage.at<cv::Vec3w>(i, j) = cv::Vec3w(0, 0, 0);
                    else if (abs(dzdy) > abs(dzdx) && abs(dzdy) > 1.0f) surfaceNormalImage.at<cv::Vec3w>(i, j) = cv::Vec3w(0, 1, 0);
                    else if (1.0f > abs(dzdy) && 1.0f > abs(dzdx)) surfaceNormalImage.at<cv::Vec3w>(i, j) = cv::Vec3w(0, 0, 1);
                }
            }

            //step2 find grid with greatest z component
            //printf("finding grid with greatest z-sum\n");
            //sliding window params
            const int WINDOW_HEIGHT = 8;
            const int WINDOW_WIDTH = 8;
            const int WINDOW_STRIDE = 8;

            //grids are identified by their top left pixel
            int gridRowWithGreatestZ = -1;
            int gridColWithGreatestZ = -1;
            ushort gridGreatestZSum = 0;

            //no overbounding checks cause the ROI is supposedly supposed to be
            //  divisible by the sliding window
            //iterate over grid rows
            for (int i = ROIUpperBound; i < ROILowerBound; i+=WINDOW_STRIDE) {
                //iterate over grid cols
                for (int j = ROILeftBound; j < ROIRightBound; j+=WINDOW_STRIDE) {
                    ushort currentGridZSum = 0;
                    //iterate over rows within grids
                    for (int m = i; m < i + WINDOW_STRIDE; m++) {
                        //iterate over cols within grids
                        for (int n = j; n < j + WINDOW_STRIDE; n++) {
                            //dear god send help it's 4 different for loops nested in each other
                            //printf("surfacenormalImage z comp: %f\n:", surfaceNormalImage.at<cv::Vec3f>(m, n)[2]);
                            currentGridZSum += surfaceNormalImage.at<cv::Vec3w>(m, n)[2];
                        }
                    }
                    if (currentGridZSum > gridGreatestZSum) {
                        //printf("\t new greatest z sum found\n");
                        gridRowWithGreatestZ = i;
                        gridColWithGreatestZ = j;
                        gridGreatestZSum = currentGridZSum;
                    }
                }
            }
            // printf("z-sum outcomes\n");
            // printf("\tgridRowWithGreatestZ: %d\n", gridRowWithGreatestZ);
            // printf("\tgridColWithGreatestZ: %d\n", gridColWithGreatestZ);
            // printf("\tgridGreatestZSum: %f\n", gridGreatestZSum);

            //step 3: with identified col, find grid with where it transitioned from x dominant to yz dominant
            int verticalGridCounter = 0;
            int targetGridRow = -1;
            int targetGridCol = gridColWithGreatestZ;
            bool currWindowYZDom = false; //true denotes yz-dominantness, false denotes x-dominantness
            bool lastWindowYZDom = false; //if the bottom grid (first one looked at) is yz-dom, then it's the intersect
            //iterate through different grid rows of the identified column, bottom up
            for (int i = ROILowerBound - 64; i >= ROIUpperBound; i-=WINDOW_STRIDE) { //-64 to account for black frames
                //determining XYZ sums
                ushort currentGridXSum = 0;
                ushort currentGridYSum = 0;
                ushort currentGridZSum = 0;
                //iterate over the rows of the grid
                for (int m = i; m < i + WINDOW_STRIDE; m++) {
                    //iterate over the collumns of the grid
                    for (int n = gridColWithGreatestZ; n < gridColWithGreatestZ + WINDOW_STRIDE; n++) {
                        currentGridXSum += surfaceNormalImage.at<cv::Vec3w>(m, n)[0];
                        currentGridYSum += surfaceNormalImage.at<cv::Vec3w>(m, n)[1];
                        currentGridZSum += surfaceNormalImage.at<cv::Vec3w>(m, n)[2];
                    }
                }
                //determining grid dominantness
                lastWindowYZDom = currWindowYZDom;
                if (currentGridYSum + currentGridZSum > currentGridXSum) {
                    currWindowYZDom = true;
                }
                // printf("vertical grid %d\n", verticalGridCounter);
                // printf("\tXsum: %d\n", currentGridXSum);
                // printf("\tYsum: %d\n", currentGridYSum);
                // printf("\tZsum: %d\n", currentGridZSum);
                // printf("\t%s dominant\n", currWindowYZDom?"YZ":"X");
                verticalGridCounter++;
                //determining if correct transition
                if (lastWindowYZDom  == false && currWindowYZDom == true) {
                    targetGridRow = i;
                    break;
                }
            }
            
            //output checking
            if (targetGridCol == -1 || targetGridRow == -1) {
                return {-1, -1, false};
            }

            if (targetGridCol < 0 || targetGridCol + 4 >= depthImage.cols) {
                printf("targetGridCol: %d\n", targetGridCol);
                assert(false);
            }

            if (targetGridRow - 4 < 0 || targetGridRow >= depthImage.rows) {
                printf("targetGridRow: %d\n", targetGridRow);
                assert(false);
            }

            return {targetGridCol + 4, targetGridRow - 4, true}; // +/- 4 cause we want the center of the grid
        }

        //This function takes in a depth image and its corresponding segmented image (generated by the segnet)
        //then computes the dbh of the tree in the image and inserts it into DBHList
        //If no such tree exist, nothing happens, and framesWithoutTrees is incremented
        //Once 5 consecutive frames without trees, a final dbh is computed from the DBHList and the list is cleared
        void ComputeDBHFromDepthAndSegmentedFrames(cv::Mat depthImage, cv::Mat segmentedImage,
            const rs2_intrinsics& intrin, 
            const Descriptors::RS2ParameterDescriptor& rs2desc, 
            const Descriptors::AlgorithmDescriptor& algdesc)
        {

            float dbh = 0;
            //printf("framesWithoutTree: %d\n", framesWithoutTree);
            //compute approx intersect intersect
            pixelCoords_t approxIntersect = ApproxGroundTrunkIntersectSegment(segmentedImage);
            if (!approxIntersect.gotIntersect) {
                // printf("approx no intersect, returning\n");
                framesWithoutTree++;
                if (framesWithoutTree >= 5) {
                    float treeDBH = GetDBHWithoutOutliers(DBHList);
                    if (treeDBH != 0) {
                        printf("Tree %d has dbh %f\n", treeCounter, treeDBH);
                        DBHList.clear();
                        framesWithoutTree = 0;
                        treeCounter++;
                    }
                    framesWithoutTree = 0;
                }
                return;
            } else {
            }

            //compute exact intersect
            pixelCoords_t exactIntersect = ExactGroundTrunkIntersectNormal(depthImage, approxIntersect);
            if (!exactIntersect.gotIntersect) {
                dbh = ComputeDBHFromRow(depthImage, depthImage.rows/2, &intrin, rs2desc);
            } else {
                dbh = ComputeDBHFromGroundIntersect(depthImage, exactIntersect.y, exactIntersect.x, &intrin, rs2desc);
            }

            if (dbh == 0.0f || isnan(dbh)) {
                //printf("Invalid dbh\n");
                return;
            }
            //printf("%f\n", dbh);
            DBHList.push_back(dbh);
            framesWithoutTree = 0;
        }

        void PublishDepthImageToNeuralNet(
            const sensor_msgs::Image::ConstPtr& depthimgPtr,
            cv::Mat depthImage,
            const rs2_intrinsics& intrin, 
            const Descriptors::RS2ParameterDescriptor& rs2desc, 
            const Descriptors::AlgorithmDescriptor& algdesc,
            RecordingTriggerROI trigger)
        {
            // If we are already waiting for a neural network output, we should forego any computations
            if( isWaitingForNeuralNetworkOutput )
            {
                return;
            }
            else
            {
                // Apply depth cutoff
                // Show Depth cutoff on the image, Useful for debugging and visualization
                for(int i = 0; i < depthImage.rows; i++)
                {
                    unsigned short* rowPtr = depthImage.ptr<unsigned short>(i);

                    for(int k = 0; k < depthImage.cols; k++)
                    {
                        if(rowPtr[k] > algdesc.ALG_SEGMENT_MAX_DEPTH_CLAMP || rowPtr[k] < algdesc.ALG_SEGMENT_MIN_DEPTH_CLAMP) rowPtr[k] = 0;
                    }
                }
                // Transpose image as necessary
                if( rs2desc.RS2_VERTICAL_ORIENTATION )
                {
                    cv::transpose(depthImage, depthImage);
                }

                //received depth image from arguments, sending to neural network topic
                cv_bridge::CvImage cvImageMessage;
                cvImageMessage.header       = depthimgPtr->header;
                cvImageMessage.encoding     = sensor_msgs::image_encodings::TYPE_16UC1;
                cvImageMessage.image        = depthImage;

                // Use global variable to update the Header
                // Idea: The header of the image sent to realsense_fia_neural_net is the same header we will receive
                // Use the received header to check if the incoming neural net output is proper
                // We don't want to just assume the incoming neural net output is correct
                // Data for interfacing with neural network output
                // Assign depth image
                neuralNetworkInputDepthImage = depthImage;
                // Publish the message, wait for the neural network node to process it
                DepthSegNetSyncMessagePublish(cvImageMessage.toImageMsg(), depthSegnetSyncPublisher);
                // Set flag here to indicate we have sent a depth image and are awaiting a neural network output
                isWaitingForNeuralNetworkOutput = true;
            }
        }

        //Process a depth frame, attempt to compute a DBH
        void ComputeDBHFromDepthFrame(
            const sensor_msgs::Image::ConstPtr& depthimgPtr,
            ros::Time timestampImageRecorded,
            cv::Mat depthImage, 
            const rs2_intrinsics& intrin, 
            const Descriptors::RS2ParameterDescriptor& rs2desc, 
            const Descriptors::AlgorithmDescriptor& algdesc,
            RecordingTriggerROI trigger   
        )
        {
            //Set the window name            
            const char* window_name ="realsense_fia: depth";

            //CRSA - close range scanning algorithm (basically calculate dbh from middle row)
            if(algdesc.ALG_CLOSERANGE_VS_GROUNDTRUTH == 0){  //REPLACE 0s AND 1s WITH ENUMS or CONSTS
                // Detect trunk through classifier
                if(algdesc.ALG_ROI_VS_CASCADE == 1)
                {
                    //Format to color
                    depthImage.convertTo(depthImage, CV_8U, 255.0 / 4096.0);    // Change its format to be able to display
                    cvtColor(depthImage, depthImage, CV_GRAY2RGB);              // Change its format again to see bounding boxes
    
                    //Rotate (comment out if rotation not needed)
                    cv::transpose(depthImage, depthImage);

                    //Detect trunks
                    std::vector<cv::Rect> trunks;

                    if(trunks.size() != 0)
                    {
                        ROS_INFO("Recording");
                        //Set status to 1 on detection of tree
                        diameterComputationStatus = COMPUTING_DIAMETER;

                        //Draw rectangles on the detected trunks
                        for(int i = 0; i < trunks.size(); i++)
                        {

                            cv::rectangle(depthImage, cv::Point(trunks[i].x,trunks[i].y), cv::Point(trunks[i].x + trunks[i].width, trunks[i].y + trunks[i].height), cv::Scalar(255, 0, 255));
                            std::cout << i << "," << trunks[i];
                        }

                        //filter code to make sure a tree is actually inside the bounding box
                        //count the number of white and black points in the rectangle
                        for(int i = 0; i < trunks.size(); i++){
                            int dark = 0; //number of dark points;
                            int light = 0; //number of light points

                            //extract x,y values
                            for(int j = trunks[i].x; j <= (trunks[i].x + trunks[i].width); j++){
                                for(int k = trunks[i].y; k <= (trunks[i].y + trunks[i].height); k++){
                                    //get the color of a point
                                    int pixel = (int)depthImage.at<uchar>(k,j);

                                    //check if it is a light or dark color
                                    if(pixel > 128){
                                        light++;
                                    }
                                    else{
                                        dark++;
                                    }

                                }
                            }
                            //check the ratio of dark and light pixels
                            //if ar least one third of the pixels are light, we assume that there is a trunk
                            if(light / dark >= 0.5){
                                diameterComputationStatus = COMPUTING_DIAMETER;
                            }
                            else{
                                diameterComputationStatus = STANDBY;
                            }
                        }

                    }
                    else if(diameterComputationStatus == COMPUTING_DIAMETER)
                    {
                        //Publish it
                        //Go back to idle / standby for trunk
                        diameterComputationStatus = STANDBY;

                        //Compute Diameter
                        float diameter = GetDBHWithoutOutliers(DBHList);
                        //Ignore diameters less than 4.0 cm
                        if(diameter <= 4.0f)
                        {
                            DBHList.clear();
                            return;
                        }
                        else
                        {
                            realsense_fia_dbh::diameter_compute_pointcloud_sync alignPointcloudMessageToPublish;
                            alignPointcloudMessageToPublish.header.stamp    = ros::Time::now();
                            alignPointcloudMessageToPublish.header.seq      = treeID;
                            alignPointcloudMessageToPublish.header.frame_id = "/realsense_fia/align_pointcloud";
                            alignPointcloudMessageToPublish.treeID          = treeID;
                            alignPointcloudMessageToPublish.dbh             = diameter;
                            alignLocalPointcloudMessageQueue.push_back( alignPointcloudMessageToPublish );

                            ROS_INFO("Tree %d has DBH %lf", ++treeID, diameter );
                            //Clear the DBHList
                            DBHList.clear();
                        }
                    }
                }
                // Detect trunk through static ROI
                else if(algdesc.ALG_ROI_VS_CASCADE == 0){
                    if (trigger.ShouldRecord(depthImage, algdesc))
                    {
                        //ROS_INFO("Recording");
                        //Set status to 1 on detection of tree
                        diameterComputationStatus = COMPUTING_DIAMETER;

                        //Rotate image whenever triggered ROI
                        if (rs2desc.RS2_VERTICAL_ORIENTATION)
                        {
                            cv::transpose(depthImage, depthImage);
                        }

                        //Grab list of segments in the middle of the image
                        auto middleRowSegmentList = UnboundedDepthImageSegmentGenerate(depthImage, depthImage.rows / 2, algdesc);
                        //Make sure we actually found segments
                        if (middleRowSegmentList.size() > 0)
                        {
                            //Compute diameter in cm from radius in m
                            float dbh = RadiusOfSegment(middleRowSegmentList[0], depthImage, &intrin, rs2desc, algdesc) * 2.0f * 100.0f;
                            //Increment DBH sum (in centimeters)
                            DBHList.push_back(dbh);
                        }
                    }
                    else if (diameterComputationStatus == COMPUTING_DIAMETER)
                    {
                        //Publish it
                        //Go back to idle / standby for trunk
                        diameterComputationStatus = STANDBY;
                        //Compute Diameter
                        float diameter = GetDBHWithoutOutliers(DBHList);
                        //Ignore diameters less than 4.0 cm
                        if(diameter <= 4.0f)
                        {
                            DBHList.clear();
                            return;
                        }
                        else
                        {
                            realsense_fia_dbh::diameter_compute_pointcloud_sync alignPointcloudMessageToPublish;
                            alignPointcloudMessageToPublish.header.stamp    = ros::Time::now();
                            alignPointcloudMessageToPublish.header.seq      = treeID;
                            alignPointcloudMessageToPublish.header.frame_id = "/realsense_fia/align_pointcloud";
                            alignPointcloudMessageToPublish.treeID          = treeID;
                            alignPointcloudMessageToPublish.dbh             = diameter;
                            alignLocalPointcloudMessageQueue.push_back( alignPointcloudMessageToPublish ); 

                            ROS_INFO("Tree %d has DBH %lf", ++treeID, diameter );
                            //Clear the DBHList
                            DBHList.clear();
                        }
                    }
                }
            }
            else if(algdesc.ALG_CLOSERANGE_VS_GROUNDTRUTH == 1){

                if (trigger.ShouldRecord(depthImage, algdesc))
                {
                    //ROS_INFO("Recording");
                    //Set status to 1 on detection of tree
                    diameterComputationStatus = COMPUTING_DIAMETER;

                    //Rotate image whenever triggered ROI
                    //Temporary change but good idea? Just look to see if there are more columns than rows to transpose the image
                    if (rs2desc.RS2_VERTICAL_ORIENTATION)
                    {
                        cv::transpose(depthImage, depthImage);
                    }

                    //Publish a message to indicate we expect a neural network output from this input
                    cv_bridge::CvImage cvImageMessage;
                    cvImageMessage.header = depthimgPtr->header;
                    cvImageMessage.encoding = sensor_msgs::image_encodings::TYPE_16UC1;
                    cvImageMessage.image = depthImage;
                    DepthSegNetSyncMessagePublish(cvImageMessage.toImageMsg(), depthSegnetSyncPublisher);

                    int groundProfileRow = -1;
                    int groundProfileCol = -1;
                    cv::Mat normalMap = GenerateNormalMapFromDepthImage(depthImage, groundProfileCol, groundProfileRow, algdesc);

                    //!!! EXPERIMENTAL NORMAL_MAP_GRADIENT_SOFTWARE !!!
                    //Ensure we have found a profile ground profile
                    if (groundProfileRow != -1 && groundProfileCol != -1) 
				    {
                        float DBHReported = 0.0f;
                        float taperReported = 0.0f;
                        float lowBranchHeightReported = 0.0f;
                        // Compute taper, DBH
					    DBHReported = ComputeDBHGroundTruth(depthImage, groundProfileRow, groundProfileCol, &intrin, rs2desc, algdesc);
                        // The breastRow parameter is set to zero here - for testing purposes we can set it to this
						// In this future this should be changed to the breast height computation
						ComputeTaper(depthImage, 0, &intrin, rs2desc, taperReported, algdesc);
                        if(DBHReported != 0.0f)
                            DBHList.push_back(DBHReported);
                        if(taperReported != 0.0f)
                            taperList.push_back(taperReported);
                    }
                    
                    cv::imshow("realsense_fia: normal", normalMap);
                    //depthImage.convertTo(depthImage, CV_8U, 255.0 / 4096.0);
                    //cv::imshow("realsense_fia: depth", depthImage);
                }
                else if (diameterComputationStatus == COMPUTING_DIAMETER)
                {
                    //Publish it
                    //Go back to idle / standby for trunk
                    diameterComputationStatus = STANDBY;
                    //Compute Diameter
                    float diameter = GetDBHWithoutOutliers(DBHList);
                    float avg_taper = GetDBHWithoutOutliers(taperList);
                    //Ignore diameters less than 4.0 cm
                    if(diameter <= 4.0f)
                    {
                        DBHList.clear();
                        taperList.clear();
                        sixFTlist.clear();
                        lowBranchHeightList.clear();
                        return;
                    }
                    else
                    {
                        realsense_fia_dbh::diameter_compute_pointcloud_sync alignPointcloudMessageToPublish;
                        alignPointcloudMessageToPublish.header.stamp    = ros::Time::now();
                        alignPointcloudMessageToPublish.header.seq      = treeID;
                        alignPointcloudMessageToPublish.header.frame_id = "/realsense_fia/align_pointcloud";
                        alignPointcloudMessageToPublish.treeID          = treeID;
                        alignPointcloudMessageToPublish.dbh             = diameter;
                        alignLocalPointcloudMessageQueue.push_back( alignPointcloudMessageToPublish ); 

                        ROS_INFO("Tree %d has DBH %lf, taper is %lf cm/m.", treeID++, diameter, avg_taper);
                        //Clear the DBHList
                        DBHList.clear();
                        taperList.clear();
                        sixFTlist.clear();
                        lowBranchHeightList.clear();
                    }
                }
            }
            
            //Trigger activated?
            //Display for fun. 
            if( algdesc.ALG_DISPLAY_WINDOW )
            {
                depthImage.convertTo(depthImage, CV_8U, 255.0 / 4096.0);
                //cv::equalizeHist(depthImage, depthImage);
                cv::imshow("realsense_fia: depth", depthImage);
                cv::waitKey(1);
                //To control the frame rate with the Enter button, use below (comment the one above)
                //while(cv::waitKey(1) != 13);
            }
        }

        //We received a CameraInfo Message from ROS - Use these message to update camera parameters for depth -> distance deprojection function
        //TODO: Requires Implementation
        void CameraInfoMsgProc(const sensor_msgs::CameraInfo::ConstPtr& camera_info, rs2_intrinsics& intrin)
        {
            //Update 'intrin'
            intrin.height       = camera_info->height;      // Height of the image in pixels
            intrin.width        = camera_info->width;        // Width of the image in pixels
            intrin.fx           = camera_info->K[0];            // Focal length of the image plane, as a multiple of pixel width    
            intrin.fy           = camera_info->K[4];            // Focal length of the image plane, as a multiple of pixel height
            intrin.ppx          = camera_info->K[2];           // Horizontal coordinate of the principal point of the image, as a pixel offset from the left edge
            intrin.ppy          = camera_info->K[5];           // Vertical coordinate of the principal point of the image, as a pixel offset from the top edge
            intrin.coeffs[5]    = {0};                   // Distortion coefficients

            // A simple model of radial and tangential distortion (AKA plumb_bob model used in CameraInfo)
            // https://calib.io/blogs/knowledge-base/camera-calibration-101
            intrin.model = RS2_DISTORTION_INVERSE_BROWN_CONRADY;

            return;
        }

        //A color image message was received. Currently, nothing is done with this information - displayed for debugging purpose.
        void ColorImgMsgProc(const sensor_msgs::Image::ConstPtr& color_img_ptr)
        {
            //Make static cv image ptr - we don't need to instantiate every time. Also, this gives us access to the previous image from the previous call if we wish to use it.
            static cv_bridge::CvImagePtr cv_img_ptr;
            //Attempt to convert
            try
            {
                cv_img_ptr = cv_bridge::toCvCopy(color_img_ptr, sensor_msgs::image_encodings::TYPE_8UC3);
            }
            catch (cv_bridge::Exception& e)
            {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }
        }

        //When a depth image is published to the depth image topic, we should process it with this function
        void DepthImgMsgProc(
            const sensor_msgs::Image::ConstPtr& depth_img_ptr,
            const rs2_intrinsics& intrin, 
            const Descriptors::RS2ParameterDescriptor& rs2desc, 
            const Descriptors::AlgorithmDescriptor& algdesc,
            RecordingTriggerROI trigger
        )
        {
            //Make static cv image ptr - we don't need to instantiate every time. Also, this gives us access to the previous image from the previous call if we wish to use it.
            static cv_bridge::CvImagePtr cv_img_ptr;
            //Attempt to convert
            try
            {
                cv_img_ptr = cv_bridge::toCvCopy(depth_img_ptr, sensor_msgs::image_encodings::TYPE_16UC1);
            }
            catch (cv_bridge::Exception& e)
            {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }

            if(initROI)
            {
                
                //Set Trigger ROI values based on algdesc
                triggerROI = RecordingTriggerROI(
		        algdesc.ALG_ROI_LEFT_COL_BOUND,
		        algdesc.ALG_ROI_RIGHT_COL_BOUND,
		        algdesc.ALG_ROI_UPPER_ROW_BOUND,
		        algdesc.ALG_ROI_LOWER_ROW_BOUND,
		        cv_img_ptr->image.cols, cv_img_ptr->image.rows, 
		        algdesc.ALG_ROI_PERCENT_VALID_AREA_THRESHOLD
                );
                
                //std::cout << "Width: " << cv_img_ptr->image.cols << " Height: " << cv_img_ptr->image.rows << std::endl;
                initROI = 0;
            }

            //Attempt to compute a DBH ==========
            //The contents of this function are analgous to the main loop of the legacy code in 'detect_tree_main.cpp'
            //PublishDepthImageToNeuralNet(depth_img_ptr, cv_img_ptr->image, intrin, rs2desc, algdesc, trigger);            
            //ComputeDBHFromDepthFrame(depth_img_ptr, depth_img_ptr->header.stamp, cv_img_ptr->image, intrin, rs2desc, algdesc, trigger);
            if (algdesc.NEURAL_NET_INTERFACING) {
                PublishDepthImageToNeuralNet(depth_img_ptr, cv_img_ptr->image, intrin, rs2desc, algdesc, trigger);
            } else {
                ComputeDBHFromDepthFrame(depth_img_ptr, depth_img_ptr->header.stamp, cv_img_ptr->image, intrin, rs2desc, algdesc, trigger);
            }




            //Attempt to compute trunk straightness ==========      //DELETE THIS AND MOVE TO COMPUTE.CPP
            /*cv::Mat depthImage = cv_img_ptr->image.clone();
            for(int i = 0; i < depthImage.rows; i++)
            {
                unsigned short* rowPtr = depthImage.ptr<unsigned short>(i);

                for(int k = 0; k < depthImage.cols; k++)
                {
                    if(rowPtr[k] > algdesc.ALG_SEGMENT_MAX_DEPTH_CLAMP || rowPtr[k] < algdesc.ALG_SEGMENT_MIN_DEPTH_CLAMP) rowPtr[k] = 0;
                }
            }

            int bin = 0;
            std::vector<cv::Point3i> centerLine = FindTrunkCenterLine(depthImage, rs2desc, &bin);
            areasComputed.push_back(computeStraightness_m4(centerLine));
            if(bin != 0 & prevBin == 0){
                //Find the mean and variance for a single tree
                float sum = std::accumulate(areasComputed.begin(), areasComputed.end(), 0);
                float mean = sum / areasComputed.size();

                float sumOfSquares = 0;
                for(int i = 0; i < areasComputed.size(); i++) sumOfSquares += (areasComputed[i] - mean) * (areasComputed[i] - mean);

                float var = sumOfSquares / areasComputed.size();
                float stdDev = sqrt(var);

                
                std::ofstream myfile;
                myfile.open("straightnessTest.txt", std::ofstream::app);
                myfile << mean << "\t" << var << "\t" << stdDev << "\n";
                myfile.close();
                

                areasComputed.clear();
            }
            prevBin = bin;
            
            //Attempt to locate the lowest branch of a tree ==========
            if(algdesc.ALG_LOW_BRANCH_DET)      //DELETE THIS AND MOVE TO COMPUTE.CPP
            {
                cv::Mat depthImage = cv_img_ptr->image;
                cv::Mat rawDepthImage = depthImage.clone();
                //Add depth cutoff to image  (6000 == 6 meters, 1 == 0.001 meters)
                for(int i = 0; i < depthImage.rows; i++)
                {
                    unsigned short* rowPtr = depthImage.ptr<unsigned short>(i);
                    for(int k = 0; k < depthImage.cols; k++)
                    {
                        if(rowPtr[k] > 6000 || rowPtr[k] < 1) rowPtr[k] = 0;
                    }
                }
                
            //     //Format to color
            //     depthImage.convertTo(depthImage, CV_8U, 255.0 / 4096.0);    // Change its format to be able to display
            //     cvtColor(depthImage, depthImage, CV_GRAY2RGB);              // Change its format again to see bounding boxes

                //Rotate (comment out if you don't need to rotate the footage)
                cv::transpose(depthImage, depthImage);
                cv::transpose(rawDepthImage, rawDepthImage);

            //     //Get an idea as to where the trunks can be before locating lowest branch
            //     std::vector<int> definiteTrunkCols = locateTrunk(depthImage);
                
            //     //Locate the lowest branch independent of each trunk
            //     float lowBranchHeightReported = 0.0f;
            //     int groundProfileRow = 0;
            //     int currentFrameGroundRow = FindEdgeOfGroundRow(cv_img_ptr->image, algdesc, &groundProfileRow);
            //     ComputeLowestBranchHeight(depthImage, definiteTrunkCols, tNums, tCols, currentFrameGroundRow, &intrin, rs2desc, lowBranchHeightReported, algdesc, rawDepthImage);
            // }
            }*/
        }

        //----------------------------------------------------------
        //  Subscriber callback functions
        //----------------------------------------------------------
        void AlignedDepthSubscriberCallback(const sensor_msgs::Image::ConstPtr aligned_depth_img_ptr )
        {
            if( cameraInfoMessageReceived )  DepthImgMsgProc(aligned_depth_img_ptr, camera_intrinsics, rs2_descriptor, algorithm_descriptor, triggerROI);
        }

        void CameraInfoSubscriberCallback(const sensor_msgs::CameraInfo::ConstPtr camera_info_msg_ptr)
        {
            //We only need one camera info, we assume it does not change and we only need to update it once.
            //If we receive another camera_info message after the first one, we should ignore it and return.
            if(cameraInfoMessageReceived) return;

            //Populate comaera_intrinsics with the contents of the camera_info message
            CameraInfoMsgProc(camera_info_msg_ptr, camera_intrinsics);
            //Update the flag that indicates we have received at least ONE camera info message
            cameraInfoMessageReceived = true;
        }

        //Function to receive segmented image, convert it to cv::Mat, and push it onto the queue
        void NeuralNetworkOutputSubscriberCallback(const sensor_msgs::Image::ConstPtr segmented_image_ptr)
        {
            //uses toCvCopy instead of toCvShare so we don't have to worry about modifying it
            cv_bridge::CvImagePtr cv_img_ptr = cv_bridge::toCvCopy(segmented_image_ptr, sensor_msgs::image_encodings::TYPE_8UC1);
            neuralNetworkOutputSegmentedImage = (*cv_img_ptr).image;
            // Increment neural network ID
            receivedNeuralNetworkOutput = true;
        }
    }

    //Bag Process Main
    int DiameterComputeMain(
        int argc,
        char** argv, 
        const Descriptors::RS2ParameterDescriptor& rs2desc, 
        const Descriptors::AlgorithmDescriptor& algdesc
        )
    {
        // UserIO::userIOVars.MouseX = 0;
	    // UserIO::userIOVars.MouseY = 0;
        //Output Debug Messages based on descriptors
        _DEBUG_MSG(algdesc.ALG_OPTION_WARNING && algdesc.ALG_PAINT_ROWS || algdesc.ALG_PUT_TEXT, "ALGDESC: PAINTING OPTIONS ENABLED");
        _DEBUG_MSG(algdesc.ALG_DISPLAY_WINDOW, "ALGDESC: DISPLAYING WINDOW");

        //Update descriptors
        rs2_descriptor          = rs2desc;
        algorithm_descriptor    = algdesc;

        //Lowest branch vars
        tNums.push_back(1);
        tCols.push_back(cv::Point(0,0));

        //Invoke init() before creation of node handle
        ros::init(argc, argv, "realsense_fia");
        //Obtain a nodehandle
        ros::NodeHandle realsense_fia_node;
        //Init time (if we need it)
        ros::Time::init();
        ros::Time::waitForValid();

        ROS_INFO("Starting realsense_fia_dbh. Enter CTRL + C to exit.");

        //Use ros::rate
        ros::Rate publishRate(100);

        //Override default SIGINT handler callback routine
        signal(SIGINT, RS_DBH_SIGINT_Handler);

        //Subscribe to camera info, depth image, and color image topics.
        ros::Subscriber alignedDepthSubscriber = realsense_fia_node.subscribe(TOPIC_STRING_ALIGNED_DEPTH_IMG, 0,AlignedDepthSubscriberCallback);
        ros::Subscriber cameraInfoSubscriber   = realsense_fia_node.subscribe(TOPIC_STRING_CAMERA_INFO, 0, CameraInfoSubscriberCallback);
        //Publisher to advertise times in a bag when the pointcloud should be saved
        ros::Publisher alignLocalPointcloudFlagPublisher = realsense_fia_node.advertise<realsense_fia_dbh::diameter_compute_pointcloud_sync>("/realsense_fia/align_pointcloud_flag", 0);
        depthSegnetSyncPublisher = realsense_fia_node.advertise<realsense_fia_dbh::depth_segnet_sync>("/realsense_fia/depth_segnet_sync", 0);
        //Subscriber to neural network output topic. Listen for the neural network output corresponding to the depth-segnet_sync message sent by us
            //maximum 10 messsages in the buffer before messages are dropped (10 chosen arbitrarily)
        ros::Subscriber neuralNetworkOutputSubscriber = realsense_fia_node.subscribe("/realsense_fia_neural_net/segmented_image", 0, NeuralNetworkOutputSubscriberCallback);

        //Continue to get messages unless we haven't received a message in 'timeout_duration' seconds
        int frameCounter = 0;
        while(run)
        {
            //Publish a timestamp if the queue is not empty. Be sure to use a mutex to lock the queue so we don't have any data races.
            std::lock_guard<std::mutex> locker(alignLocalPointcloudMutex);
            //alignLocalPointcloudMutex.lock();
            if(!alignLocalPointcloudMessageQueue.empty())
            {
                alignLocalPointcloudFlagPublisher.publish( alignLocalPointcloudMessageQueue.back() );
                alignLocalPointcloudMessageQueue.pop_back();
            }
            // alignLocalPointcloudMutex.unlock();

            // Handle neural network data
            // If we are waiting for neural network output, check to see if we have received anything on the neural network output topic
            // Two nested if statements - the outer is for checking
            // if( neuralNetworkInterfacingEnabled && isWaitingForNeuralNetworkOutput)
            if( algdesc.NEURAL_NET_INTERFACING && isWaitingForNeuralNetworkOutput)
            {
                //If this is true, then we have a corresponding neural network input and output pair
                if(receivedNeuralNetworkOutput)
                {
                    // Deactivate receivedneuralNetworkOutput flag
                    receivedNeuralNetworkOutput = false;
                    // Deactivate isWaitingForNeuralNetworkOutput flag
                    isWaitingForNeuralNetworkOutput = false;
                    // Multiply classmap by constant to make classes more visible
                    cv::Mat neuralNetworkOutputSegmentedImageViewable = 80*neuralNetworkOutputSegmentedImage;
                    cv::Mat neuralNetworkInputDepthImageViewable;
                    neuralNetworkInputDepthImage.convertTo(neuralNetworkInputDepthImageViewable, CV_8U, 255.0 / 4095.0);
                    //cv::imshow("Neural network output", neuralNetworkOutputSegmentedImageViewable);
                    //cv::imshow("Neural network input", neuralNetworkInputDepthImage);

                    // At this point, we can perform any necessary computations here
                    // printf("frame: %d\n", frameCounter);
                    ComputeDBHFromDepthAndSegmentedFrames(neuralNetworkInputDepthImage, neuralNetworkOutputSegmentedImage, camera_intrinsics, rs2desc, algdesc);
                    cv::imshow("Neural network input", neuralNetworkInputDepthImageViewable);
                    frameCounter++;
                }
            }
            // Invoke waitKey to update window callbacks
            cv::waitKey(1);
            //Publish anything we have
            ros::spinOnce(); 
            //Wait a minute!
            publishRate.sleep(); 
        }

        //Destroy windows if there are any
        if( algdesc.ALG_DISPLAY_WINDOW )
        {
            cv::destroyAllWindows();
            cv::waitKey(1);
        }
        //If we don't receive any more messages, publish the remaining DBH that was being computed.
        PublishFinalDBHAfterRecordingEnd();
        //Since the above function call doesnt work with the current implementation of segnet dbh compute
        //I'll just do a quick and dirty hack here
        if (DBHList.size() != 0) {
            printf("Remaining DBH in list, computing\n");
            float treeDBH = GetDBHWithoutOutliers(DBHList);
            if (treeDBH != 0) {
                printf("Tree %d has dbh %f\n", treeCounter, treeDBH);
                DBHList.clear();
                framesWithoutTree = 0;
                treeCounter++;
            }
            framesWithoutTree = 0;
        }

        printf("processed %d frames\n", frameCounter);

//calling function for straightness.
	//------------------START---------------------------------------
            //std::cout << "1";
            cv::Mat NNImage = cv::imread("/home/user/Desktop/Fake_image/Fake image/23.png", cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
		//check if read correctly
		if (NNImage.empty()) {
            		std::cout << "Could not open or find the image"<<std::endl;
        	}
        	else {


			//get centerline
			//get colored image
			cv::Mat coloredMap;
			//coloredMap = NNImage.clone();
			NNImage.copyTo(coloredMap);
			std::vector<std::tuple<int,int>> trunkVec;
			//showing the colored classmap with centerline marked
			std::vector<std::tuple<int,int>> centerline = FindTrunkCenterLine_V2(NNImage, coloredMap, trunkVec);
			std::cout << "the number of centers is: " <<centerline.size()<<std::endl;
			//for(int i =0; i<sizeof(centerline)
			
			//get and print the coordinate of the farthest point
			std::tuple<int,int> farthestPoint = FindFarthestPoint(centerline);
			std::cout << "farthest point (x,y):" <<"("<< std::get<1>(farthestPoint) << "," << std::get<0>(farthestPoint) <<")" << std::endl;
			
			//Get the MG and its row information
			std::pair <float,int> MGandRow = FindMG(centerline);
			int row = std::get<1>(MGandRow);
			std::cout << "MG is : " <<std::get<0> (MGandRow) << std::endl;
			std::cout << "MG row is : " <<std::get<1> (MGandRow) << std::endl;

			//Find DMG
			
			//int trunkStartPos = findDMG(NNImage, coloredMap, MGandRow);
			//int DMG = abs(std::get<0>(farthestPoint) - trunkStartPos) * 2;
			int DMG = findDMG(NNImage, coloredMap, MGandRow);
			std::cout << "DMG is : " <<DMG<< std::endl;
			//Get straightness
			float striaght_d = trunkStraightness_new(MGandRow, DMG);
			std::cout << "Straightness is : " <<striaght_d<< std::endl;
			cv::imshow("centerline & AB-line (classmap)", coloredMap);
			cv::waitKey(1);
		}
	//------------------END---------------------------------------
        return EXIT_SUCCESS;
    }
}

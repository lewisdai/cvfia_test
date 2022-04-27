#include "Compute.h"
#include "Defines.h"
#include "Descriptors.h"
#include "UserIOStateMachine.h"
#include <rsutil.h>
#include <array>
#include <climits>
#include <numeric>
#include <algorithm>
#include <thread>
#include <iomanip>
#include <sstream>

#define ABS(N) ((N<0)?(-N):(N))
#define SQRT_2_F sqrt(2.0f)

namespace DepthCamProcessing
{
	namespace Compute
	{

		//Portion initialize
		ComputeVars computeVars = { 0.5 };
		// Follow this link to understand what's going on
		// https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/box-whisker-plots/a/identifying-outliers-iqr-rule
		float GetDBHWithoutOutliers(std::vector<float> dbhList)
		{
			//SCheck the size of dbhList
			if (dbhList.size() == 0) return 0.0f;

			std::sort(dbhList.begin(), dbhList.end());
			float Q1 = dbhList[dbhList.size() / 4];
			float Q3 = dbhList[3 * (dbhList.size() / 4)];
			float IQR = Q3 - Q1;
			float dbhSum = 0.0f;
			float dbhCount = 0.0f;
			for (float dbh : dbhList) {
				if (dbh < Q3 + 1.5 * IQR && dbh > Q1 - 1.5 * IQR) {
					dbhSum += dbh;
					dbhCount += 1.0f;
				}
			}
			if(dbhCount <= 0.0f) return 0.0f;
			
			return dbhSum / dbhCount;
		}

		float GetLowBranchHeightWithoutOutliers(std::vector<float> lowBranchHeightList)
		{
			//SCheck the size of height list
			if (lowBranchHeightList.size() == 0) return 0.0f;

			std::sort(lowBranchHeightList.begin(), lowBranchHeightList.end());
			float Q1 = lowBranchHeightList[lowBranchHeightList.size() / 4];
			float Q3 = lowBranchHeightList[3 * (lowBranchHeightList.size() / 4)];
			float IQR = Q3 - Q1;
			float heightSum = 0.0f;
			float heightCount = 0.0f;
			for (float height : lowBranchHeightList) {
				if (height < Q3 + 1.5 * IQR && height > Q1 - 1.5 * IQR && height < 3.5) {
					heightSum += height;
					heightCount += 1.0f;
				}
			}
			if(heightCount <= 0.0f) return 0.0f;
			
			return heightSum / heightCount;
		}

		//Catalog all segments encountered in a row. Return in a std::vector<...>
		std::vector<Segment> UnboundedDepthImageSegmentGenerate(cv::Mat depthImage, int rowIdx, Descriptors::AlgorithmDescriptor algdesc)
		{
			//List of segments 
			std::vector<Segment> rowGeneratedSegmentList = std::vector<Segment>();

			//Acquire read-only row
			ushrt* rowPtr = depthImage.ptr<ushrt>(rowIdx);

			//Flag that indicates whether we are recording
			//a segment
			bool segmentAcquireToggle = false;

			//Current segment information (start, end)
			int colEnd = -1;
			int colStart = -1;

			//Now go backwards through the columns, and detect segments
			for (int k = 0; k < depthImage.cols; k++)
			{
				//If pixel is out of bounds, make it black
				//Determine if we are moving into a region of valid depths from a non-valid region => 'offToOn
				//Determine if we are moving into a region of non-valid depths from a valid region => 'onToOff'
				bool offToOn = (rowPtr[k] > algdesc.ALG_SEGMENT_MIN_DEPTH_CLAMP) && (rowPtr[k] < algdesc.ALG_SEGMENT_MAX_DEPTH_CLAMP);
				bool onToOff = (rowPtr[k] < algdesc.ALG_SEGMENT_MIN_DEPTH_CLAMP) || (rowPtr[k] > algdesc.ALG_SEGMENT_MAX_DEPTH_CLAMP);

				//ON -> OFF
				if (segmentAcquireToggle == true && onToOff)
				{
					//Add to list if the segment is at least MIN_SEGMENT_WIDTH
					int segmentWidth = colEnd - colStart + 1;

					//Note: subtracting 1 from the cols or rows field of depthImage is done because
					//the static_cast result is used as an index
					//if row 720 is used as an index in an image with 720 rows, the index is off by 1
					//(not zero based)
					//In order to avoid this error we subtract one, so if it is multiplied by 1.0f it will only be 719
					//Is this segment valid?
					//EDIT: SEGMENTS THE ENTIRE IMAGE
					//SAMPLE AS YOU NEED ACCORDINGLY
					if (segmentWidth >= algdesc.ALG_SEGMENT_MIN_REJECT_WIDTH &&
						segmentWidth <= algdesc.ALG_SEGMENT_MAX_REJECT_WIDTH
						)
					{
						//Push back segment
						rowGeneratedSegmentList.push_back(Segment(rowIdx, colStart, colEnd));
					}

					//We are no longer recording a segment
					segmentAcquireToggle = false;
				}
				//OFF -> ON
				else if (segmentAcquireToggle == false && offToOn)
				{
					//Set column bounds
					colStart = k;
					colEnd = k;
					//Recording segment currently
					segmentAcquireToggle = true;
				}

				//Update the end of the segment if we are recording
				if (segmentAcquireToggle == true)
				{
					//Update enFd of column
					colEnd = k;
					//Paint rows enabled?
				}
			}

			//Return all segments generated @ row index 'row'
			return rowGeneratedSegmentList;
		}//End Unbounded...Generate(){}	

		void isThisGroundLine(int rownum, ushrt* rowPtr, int maxCols, ushrt maxDepth, ushrt minDepth, int UIground, int *outputRow){
			//Current valid pixels (within depth cutoff)
			int currentRowValidPixelCount = 0;
			//Current invalid pixels (outside depth cutoff)
			int currentRowInvalidPixelCount = 0;

			for (int j = 0; j < maxCols; j++)
			{
				//Increment validPixelCount if the pixel is within the depth cutoff
				if (rowPtr[j] < minDepth || rowPtr[j] > maxDepth)
				{
					//printf("min: %d.\n", minDepth);
					//printf("max: %d.\n", maxDepth);
					//printf("pixel is: %d.\n", rowPtr[j]);
					currentRowInvalidPixelCount++;
				}
			}

			if (rownum > *outputRow && (((float)currentRowInvalidPixelCount / maxCols) >= (computeVars.Portion)))
			{
				//Store edge of Ground Row
				*outputRow = rownum;
			}
		}

		int FindGroundProfileT(cv::Mat depthImage, Descriptors::AlgorithmDescriptor algdesc)
		{
			//Return variable
			int groundProfileRow = -1;
			int cols = depthImage.cols;

			//Iterate over rows starting from the bottom of the image
			for (int i = depthImage.rows - 1; i >= 0; i-=4)
			{
				//Obtain row pointer
				ushrt* rowPtr0 = depthImage.ptr<ushrt>(i);
				ushrt* rowPtr1 = depthImage.ptr<ushrt>(i-1);
				ushrt* rowPtr2 = depthImage.ptr<ushrt>(i-2);
				ushrt* rowPtr3 = depthImage.ptr<ushrt>(i-3);

				//threading here
				std::thread t0(&isThisGroundLine, i  , rowPtr0, cols, algdesc.ALG_SEGMENT_MAX_DEPTH_CLAMP, algdesc.ALG_SEGMENT_MIN_DEPTH_CLAMP, algdesc.ALG_UIGROUND, &groundProfileRow);
				std::thread t1(&isThisGroundLine, i-1, rowPtr1, cols, algdesc.ALG_SEGMENT_MAX_DEPTH_CLAMP, algdesc.ALG_SEGMENT_MIN_DEPTH_CLAMP, algdesc.ALG_UIGROUND, &groundProfileRow);
				std::thread t2(&isThisGroundLine, i-2, rowPtr2, cols, algdesc.ALG_SEGMENT_MAX_DEPTH_CLAMP, algdesc.ALG_SEGMENT_MIN_DEPTH_CLAMP, algdesc.ALG_UIGROUND, &groundProfileRow);
				std::thread t3(&isThisGroundLine, i-3, rowPtr3, cols, algdesc.ALG_SEGMENT_MAX_DEPTH_CLAMP, algdesc.ALG_SEGMENT_MIN_DEPTH_CLAMP, algdesc.ALG_UIGROUND, &groundProfileRow);

				t0.join();
				t1.join();
				t2.join();
				t3.join();

				//finish
				if(groundProfileRow != -1)
					break;
			}

			//Return it
			return groundProfileRow;
		}//End FindGroundProfile(...)


		//New + Improved Robust ground truth detection
		int FindEdgeOfGroundRow(cv::Mat depthImage, Descriptors::AlgorithmDescriptor algdesc, int* groundProfileRow)
		{
			//Compute the profile of the ground
			*groundProfileRow = FindGroundProfileT(depthImage, algdesc);
			//add the default minimum shift of ground line
			int edgeOfGroundRow = *groundProfileRow + 30;
			//initiate the variables
			int status = 0;
			int jj = 0;
			float sum = 0;
			int left = 0;
			int right = 0;
			int count = 0;

			//loop throught the row of 30-20 rows above found edge
			for (int ii = edgeOfGroundRow - 60; ii < edgeOfGroundRow - 50; ii++) {
				//Grab a row pointer
				ushrt* rowPtr = depthImage.ptr<ushrt>(ii);
				//loop through every pixels
				for (int jj = 0.25f*depthImage.cols; jj < 0.75f*depthImage.cols; jj++) {
					if (rowPtr[jj] > algdesc.ALG_SEGMENT_MIN_DEPTH_CLAMP && rowPtr[jj] < algdesc.ALG_SEGMENT_MAX_DEPTH_CLAMP) {
						//first valid is left bound
						if (status == 0) {
							status = 1;
							left = jj;
						}
						//current valid is right bound
						else if (status == 1) {
							right = jj;
						}
						sum += rowPtr[jj];
						count++;
					}
				}
			}
			//sum becomes average depth
			sum /= count;

			//find grnd row, go to bottom
			for (int i = edgeOfGroundRow; i < depthImage.rows; i++) {
				float sum_of_grnd = 0;
				//sum of 4 rows
				for (int j = i; j < i + 4; j++) {
					//Grab a row pointer
					ushrt* rowPtr = depthImage.ptr<ushrt>(j);
					for (int k = left; k < right; k++) {
						sum_of_grnd += rowPtr[k];
					}
				}
				//average depth of the 4 rows
				sum_of_grnd /= (4 * (right - left));
				if (sum_of_grnd < sum * 0.91) {
					edgeOfGroundRow = i;
					break;
				}
			}

			//Return it!
			return edgeOfGroundRow;
		}

		//Show the depth cutoff on the image
		//Mostly for debug purposes
		void ShowDepthCutoff(cv::Mat depthImage, Descriptors::AlgorithmDescriptor algdesc)
		{
			//Remove
			for (int i = 0; i < depthImage.rows; i++)
			{
				//Obtain a row pointer
				ushrt* rowPtr = depthImage.ptr<ushrt>(i);

				//Paint the the columns of the row according to the depth cutoff
				for (int j = 0; j < depthImage.cols; j++)
				{
					if (rowPtr[j] < algdesc.ALG_SEGMENT_MIN_DEPTH_CLAMP || rowPtr[j] > algdesc.ALG_SEGMENT_MAX_DEPTH_CLAMP) rowPtr[j] = 0;
				}
			}
		}

		//In: segments
		//Out: Slope (broken into components) dx, dy
        bool ComputeTrunkSlope(cv::Mat depthImage, const std::vector<Segment>& segments, float& dx, float& dy, float& dz, Descriptors::RS2ParameterDescriptor rs2desc)
		{
			//dx => change in column
			//dy => change in row
			//Zero slope components
			dx = 0.0f;
			dy = 0.0f;
            dz = 0.0f;

			//Make sure there are at least two elements
			if (segments.size() < 2) return false;

			//Variables to determine slope
			float currentRow = 0.0f;
			float previousRow = 0.0f;
			float currentStartColumn = 0.0f;
			float previousStartColumn = 0.0f;
            float previousDepth = 0.0f;

			//Grab an iterator for the vector of segments
			auto iterator = segments.cbegin();

			//Read each segment from the list (assumes that the segments describe one uniform tree trunk
			//Assign previous row, start column
			previousRow = static_cast<float>((*iterator).row);
			previousStartColumn = static_cast<float>((*iterator).colStart);
            ushrt* rowPtr = depthImage.ptr<ushrt>((*iterator).row);
            previousDepth = static_cast<float>(rowPtr[(*iterator).colStart]) * rs2desc.RS2_DEPTH_UNITS;
			//Increment iterator
			iterator++;

			//Go through the rest of the vector (1st element used to determine previousRow, Column
			for (iterator; iterator != segments.cend(); iterator++)
			{
				//Compute column difference, row difference
				dx += static_cast<float>((*iterator).colStart) - previousStartColumn;
				dy += static_cast<float>((*iterator).row) - previousRow;
                rowPtr = depthImage.ptr<ushrt>((*iterator).row);
                dz += static_cast<float>(rowPtr[(*iterator).colStart]) * rs2desc.RS2_DEPTH_UNITS - previousDepth;

				//Update previous
				previousRow = static_cast<float>((*iterator).row);
				previousStartColumn = static_cast<float>((*iterator).colStart);
                previousDepth = static_cast<float>(rowPtr[(*iterator).colStart]) * rs2desc.RS2_DEPTH_UNITS;
			}

			return true;
		}

		//Assumes a segment is a semi-circular cross section of an object
		//This method is more consistent than measuring the distance between the end points
		//This method is INCORRECT!
		float RadiusOfSegment(const Segment& segment, cv::Mat depthImage, const rs2_intrinsics* intrin, Descriptors::RS2ParameterDescriptor rs2desc, 
			Descriptors::AlgorithmDescriptor algdesc)
		{
			//Get row pointer
			ushrt* rowPtr = depthImage.ptr<ushrt>(segment.row);
			//Store indices of segment start and end points
			int D = segment.colEnd;
			int E = segment.colStart;

			//Distance to middle of segment
			ushrt depthCD = rowPtr[D];
			ushrt depthCE = rowPtr[E];

			//Convert to meters 
			float CD = static_cast<float>(depthCD) * rs2desc.RS2_DEPTH_UNITS;
			float CE = static_cast<float>(depthCE) * rs2desc.RS2_DEPTH_UNITS;

			//Average the depth of the end points 'a'
			float a = (CD + CE) / 2.0f;

			//use 'a' instead of CD, CE for endpoint depths
			//The method assumes that the endpoints are at the same depth. an average will reasonably meet
			//that assumption - 'a' is used here for that reason
			//Refer to diagram
			float b = DistBetweenPixels(segment.row, E, segment.row, D, a, a, intrin) / 2.0f;
		
			//Compute the radius
			float r = (a*b) / sqrt(a*a - b*b);

			//Return the radius
			return r;
		}

		//Computes distance between 2 pixel coords given their depth (and camera intrinsics)
		float DistBetweenPixels(int row1, int col1, int row2, int col2, float depth1, float depth2, const rs2_intrinsics* intrin)
		{
			//Return the real-world coordinates of the endpoints
			//pass in depth, row, and column of start and endpoint
			float startPos[3];
			float endPos[3];
			float startPt[2];
			float endPt[2];

			//Assign start, end point appropriately
			startPt[0]	= static_cast<float>(row1);
			startPt[1]	= static_cast<float>(col1);
			endPt[0]	= static_cast<float>(row2);
			endPt[1]	= static_cast<float>(col2);
		
			//Where the magic happens
			rs2_deproject_pixel_to_point(startPos, intrin, startPt, depth1);
			rs2_deproject_pixel_to_point(endPos, intrin, endPt, depth2);

			//Compute the distances between start and end
			float distX = endPos[0] - startPos[0];
			float distY = endPos[1] - startPos[1];
			float distZ = endPos[2] - startPos[2];

			//Compute Euclidean distance
			float dist = pow(distX, 2.0f) + pow(distY, 2.0f) + pow(distZ, 2.0f);
			dist = sqrt(dist);

			//Grab the distance
			return dist;
		}

		//Computes the DBH from a certain row identified as the breast height row (DBHRow)
		float ComputeDBHFromRow(cv::Mat depthImage, int DBHRow, const rs2_intrinsics* intrin, const Descriptors::RS2ParameterDescriptor& rs2desc) {
			int leftMostTrunk = 0;
			int rightMostTrunk = 0;
			bool gotLeftMost = false;
			for (int i = 0; i < depthImage.cols; i++) {
				if (depthImage.at<ushrt>(DBHRow, i) != 0) {  //NOTE: fix this, the background is NOT always 0
					if (!gotLeftMost) {
						leftMostTrunk = i;
						gotLeftMost = true;
					}
					rightMostTrunk = i;
				}
			}
			// if (leftMostTrunk == 0 || rightMostTrunk == 0) {
			// 	printf("edges not detected\n");
			// }
			if (leftMostTrunk < 24 || rightMostTrunk > depthImage.cols - 8) { //disregard cases where trunk possible clipped
				//printf("edge clip detected\n");
				return 0;
			}
			if (rightMostTrunk - leftMostTrunk <= 0) {
				//printf("invalid trunk edges\n");
				return 0;
			}

			// printf("left: %d\t", leftMostTrunk);
			// printf("right: %d\n", rightMostTrunk);
			// printf("left val: %d\t right val: %d\t",
			// 	depthImage.at<ushrt>(DBHRow, leftMostTrunk), depthImage.at<ushrt>(DBHRow, leftMostTrunk));

			cv::Mat depthImageViewable;

			depthImage.convertTo(depthImageViewable, CV_8U, 255.0 / 4095.0);
			cv::Mat depthImageColor;
			cv::cvtColor(depthImageViewable, depthImageColor, cv::COLOR_GRAY2BGR);
			depthImageColor.at<cv::Vec3b>(DBHRow, leftMostTrunk) = cv::Vec3b(0, 255, 0);
			depthImageColor.at<cv::Vec3b>(DBHRow, rightMostTrunk) = cv::Vec3b(0, 255, 0);
			depthImageColor.at<cv::Vec3b>(depthImageViewable.rows/2, depthImageViewable.cols/2) = cv::Vec3b(0, 0, 255);
			cv::imshow("depthImageColor", depthImageColor);

			// printf("row: %d\n", DBHRow);
			// printf("left most: %d\n", leftMostTrunk);
			// printf("right most: %d\n", rightMostTrunk);
			// printf("leftMostTrunk depth (no rs2): %f\n", static_cast<float> (depthImage.at<ushrt>(DBHRow, leftMostTrunk)));
			// printf("rightMostTrunk depth (no rs2): %f\n", static_cast<float> (depthImage.at<ushrt>(DBHRow, rightMostTrunk)));
			// printf("rs2_depth_units: %f\n", rs2desc.RS2_DEPTH_UNITS);
			// printf("leftMostTrunk depth (with rs2): %f\n", rs2desc.RS2_DEPTH_UNITS * static_cast<float> (depthImage.at<ushrt>(DBHRow, leftMostTrunk)));
			// printf("rightMostTrunk depth (with rs2): %f\n", rs2desc.RS2_DEPTH_UNITS * static_cast<float> (depthImage.at<ushrt>(DBHRow, rightMostTrunk)));

			//refer to diagram and explanation https://academic.oup.com/jof/article/118/4/355/5811312
			float CD = rs2desc.RS2_DEPTH_UNITS * static_cast<float> (depthImage.at<ushrt>(DBHRow, leftMostTrunk));
			float CE = rs2desc.RS2_DEPTH_UNITS * static_cast<float> (depthImage.at<ushrt>(DBHRow, rightMostTrunk));
			float DE = DistBetweenPixels(DBHRow, leftMostTrunk,
							DBHRow, rightMostTrunk,
							CD,
							CE,
							intrin);
			// printf("CD: %f\n", CD);
			// printf("CE: %f\n", CE);
			// printf("DE: %f\n", DE);
			float a = (CD + CE)/2; //CD and CE should theoretically be equal, taking the average just in case
			float b = DE/2;
			float dbh = (2*a*b)/sqrt(a*a - b*b); //multiply by 2 diameter = 2*radius
			return dbh * 100; //conversion from meters to centimeters
		}

		//Computes the DBH given the exact trunk ground intersection
		float ComputeDBHFromGroundIntersect(
			cv::Mat depthImage,
			int intersectRow,
			int intersectCol,
			const rs2_intrinsics* intrin,
			const Descriptors::RS2ParameterDescriptor& rs2desc)
		{
			//printf("intersectRow: %d\n", intersectRow);

			float groundIntersectPixels[2] = {intersectRow, intersectCol};
			float groundIntersectIRL[3] = {0, 0, 0};
			rs2_deproject_pixel_to_point(groundIntersectIRL, intrin, groundIntersectPixels,
				rs2desc.RS2_DEPTH_UNITS * depthImage.at<ushrt>(intersectRow, intersectCol));

			// printf("groundintersectIRL x: %f\n", groundIntersectIRL[0]);
			// printf("groundintersectIRL y: %f\n", groundIntersectIRL[1]);
			// printf("groundintersectIRL z: %f\n", groundIntersectIRL[2]);
			
			float DBHIRL[3] = {groundIntersectIRL[0], groundIntersectIRL[1] - 1.3f, groundIntersectIRL[2]};
			float DBHPixels[2] = {0, 0};
			rs2_project_point_to_pixel(DBHPixels, intrin, DBHIRL);

			int DBHRow = DBHPixels[1];

			//printf("DBHRow: %d\n", DBHRow);

			if (DBHRow < 0 || DBHRow >= depthImage.rows) {
				//printf("Invalid DBH row\n");
				//printf("from middle\n");
				DBHRow = depthImage.rows/2;
			}
			else {
				//printf("from 1.4m\n");
			}

			return ComputeDBHFromRow(depthImage, DBHRow, intrin, rs2desc);
		}

		//Take a look
		//MODIFIED Jan 29 2021
		//Added argument that indicates what column we should inspect to compute the distance
		float ComputeDBHGroundTruth(
			cv::Mat depthImage,
			int groundRow,
			int groundCol,
			const rs2_intrinsics* intrin,
			const Descriptors::RS2ParameterDescriptor& rs2desc,
			Descriptors::AlgorithmDescriptor algdesc)
		{
			//Set DBH Reported to 0.0f
			float diameterAtBreastHeight = 0.0f;
		
			//Return the real-world coordinates of the ground intersection point
			float groundIntersectionPoint3D[3];
			float breastHeightPixelCoords[2];
			float groundIntersectionPixelCoords[2] = {static_cast<float>(groundCol), static_cast<float>(groundRow)};
			ushrt* groundIntersectionRowPtr = depthImage.ptr<ushrt>(groundRow);
			float groundIntersectionDepth = rs2desc.RS2_DEPTH_UNITS * static_cast<float>(groundIntersectionRowPtr[groundCol]);
			int breastHeightRow;
			//Where the magic happens
			rs2_deproject_pixel_to_point(groundIntersectionPoint3D, intrin, groundIntersectionPixelCoords, groundIntersectionDepth);
			//Idea - measure 1.3 meters above groundIntersectionPoint in real-world coordinates
			//Note: All real-world coordinates are in meters
			groundIntersectionPoint3D[1] -= 1.4;
			//Convert model from INVERSE_BROWN_CONRADY to BROWN_CONRADY temporarily
			//Create local copy of intrinsics structure
			//Deproject ground row point to 3D
			rs2_project_point_to_pixel(breastHeightPixelCoords, intrin, groundIntersectionPoint3D);
			//Extract row from 'breastHeightPixelCoords'
			breastHeightRow = breastHeightPixelCoords[1];
			//If breastHeightRow is not valid (within bounds of depthImage, then we should return a DBH of 0.0)
			if( breastHeightRow < 0 || breastHeightRow >= depthImage.rows)
			{
				//std::cout << "Invalid breast height row from rs2_project_point_to_pixel\n";
				return 0.0f;
			}
			//Paint!
			//Uncomment if you want to see the painted breast height row
			//BEWARE: This may or may not interfere with other computations
			/*
    		for(int k = 0; k < depthImage.rows; k++)
			{
				auto row_ptr = depthImage.ptr<ushrt>(k);

				for(int z = 0; z < depthImage.cols; z++)
				{
					if(z == groundCol) row_ptr[z] = 65535;
					if(k == breastHeightRow) row_ptr[z] = 65535;
					if(k == groundRow) row_ptr[z] = 65535;
				}
			}
			*/
			//Generate a list of segments (std::vector<Segment>)
			auto segmentList = UnboundedDepthImageSegmentGenerate(depthImage, breastHeightRow, algdesc);
			//Keep track of the predicted trunk cross section, or segment
			//If we do not found a segment, initialize such that it is 0,0,0 (this will not cause a segfault)
			Segment predictedTrunkCrossSection = Segment(0,0,0);
			//Keep track of distanceo of the closest section's midpoint
			int closestDistanceFromColumnOfSegmentToTrunkCenterColumn = 10000;
			//Sift through the list of segments whose midpoint is closest to 'groundCol'
			for( Segment s : segmentList)
			{
				//Compute midpoint of segment
				int midpoint = (s.colStart + s.colEnd) / 2;
				//Determine if the midpoint is closest to the trunk
				//Note we assume the trunk is also centered on 'groundCol'
				int distanceToTrunkCol = abs( midpoint - groundCol );
				//Choose the segment with the closest midpoint to to groundCol
				if( distanceToTrunkCol < closestDistanceFromColumnOfSegmentToTrunkCenterColumn )
				{
					closestDistanceFromColumnOfSegmentToTrunkCenterColumn = distanceToTrunkCol;
					predictedTrunkCrossSection = s;
				}
			}
			//Compute DBH
			diameterAtBreastHeight = RadiusOfSegment(predictedTrunkCrossSection, depthImage, intrin, rs2desc, algdesc) * 2.0f * 100.0f;
			//Return diameter at breast height
			return diameterAtBreastHeight;
		}		

		void ComputeTaper(cv::Mat depthImage,
			int breastRow,
			const rs2_intrinsics* intrin,
			const Descriptors::RS2ParameterDescriptor& rs2desc,
			float& taperReported,
			Descriptors::AlgorithmDescriptor algdesc)
		{
			//printf("breast row is %d.\n", breastRow);
			const float BREAST_HEIGHT_ACCEPTABLE_ERROR = 0.04f;
			if(!breastRow){breastRow = depthImage.rows / 2;}
			//Set Taper Reported to 0.0f
			taperReported = 0.0f;

			float dUpper = 0.0f;
			float dLower = 0.0f;

			int segmentMidPoint1 = 0;
			int segmentMidPoint2 = 0;

			ushrt* grndRowPtr = depthImage.ptr<ushrt>(breastRow);

			Segment Usegment;
			Segment Lsegment;

			ushrt* segRowPtr1;
			ushrt* segRowPtr2;

			std::vector<DepthCamProcessing::Segment> upperSegments;
			std::vector<DepthCamProcessing::Segment> lowerSegments;

			for(int i = 0; i < depthImage.rows / 2; i++)
			{
				//Generate segments from the current row
				upperSegments = UnboundedDepthImageSegmentGenerate(depthImage, breastRow-i, algdesc);

				//Check if ANY segments were encountered
				if (upperSegments.size() == 0)
				{
					continue;
				}

				//Grab the 1st segment encountered
				Usegment = upperSegments[0];

				//Compute distance between this segment and the ground truth
				segmentMidPoint1 = (Usegment.colEnd + Usegment.colStart) / 2;

				//Obtain row pointers
				segRowPtr1 = depthImage.ptr<ushrt>(Usegment.row);

				//Compute distance
				float heightToBreast = DistBetweenPixels(
					Usegment.row,
					segmentMidPoint1,
					breastRow,
					segmentMidPoint1,
					segRowPtr1[segmentMidPoint1] * rs2desc.RS2_DEPTH_UNITS,
					grndRowPtr[segmentMidPoint1] * rs2desc.RS2_DEPTH_UNITS,
					intrin);

				//Choose this segment to determine the diameter since it is 1.3m above the groundTruthRow
				if (abs(heightToBreast - 0.7f) <= BREAST_HEIGHT_ACCEPTABLE_ERROR)
				{
					//Compute DBH
					dUpper = RadiusOfSegment(Usegment, depthImage, intrin, rs2desc, algdesc) * 2.0f * 100.0f;
					//printf("rUpper is %f.\n", rUpper);
					//printf("Upper row is %d.\n", Usegment.row);
					break;
				}
			}

			for(int i = 0; i < depthImage.rows / 2; i++)
			{
				//Generate segments from the current row
				lowerSegments = UnboundedDepthImageSegmentGenerate(depthImage, breastRow+i, algdesc);

				//Check if ANY segments were encountered
				if (lowerSegments.size() == 0)
				{
					continue;
				}

				//Grab the 1st segment encountered
				Lsegment = lowerSegments[0];

				//Compute distance between this segment and the ground truth
				segmentMidPoint2 = (Lsegment.colEnd + Lsegment.colStart) / 2;

				//Obtain row pointers
				segRowPtr2 = depthImage.ptr<ushrt>(Lsegment.row);

				//Compute distance
				float heightToBreast = DistBetweenPixels(
					Lsegment.row,
					segmentMidPoint2,
					breastRow,
					segmentMidPoint2,
					segRowPtr2[segmentMidPoint2] * rs2desc.RS2_DEPTH_UNITS,
					grndRowPtr[segmentMidPoint2] * rs2desc.RS2_DEPTH_UNITS,
					intrin);

				//Choose this segment to determine the diameter since it is 1.3m above the groundTruthRow
				if (abs(heightToBreast - 0.3f) <= BREAST_HEIGHT_ACCEPTABLE_ERROR)
				{
					//Compute DBH
					dLower = RadiusOfSegment(Lsegment, depthImage, intrin, rs2desc, algdesc) * 2.0f * 100.0f;
					//printf("rLower is %f.\n", rLower);
					//printf("Lower row is %d.\n", Lsegment.row);
					break;
				}
			}
			float heightTaper = 0.0f;
			if(dUpper && dLower)
				heightTaper = DistBetweenPixels(
					Usegment.row,
					segmentMidPoint1,
					Lsegment.row,
					segmentMidPoint2,
					segRowPtr1[segmentMidPoint1] * rs2desc.RS2_DEPTH_UNITS,
					segRowPtr2[segmentMidPoint2] * rs2desc.RS2_DEPTH_UNITS,
					intrin);
				//printf("%f\n",heightTaper);
				taperReported = ABS((dLower - dUpper)) / heightTaper; // 50cm
				paintRow(segRowPtr1, Usegment.colStart, Usegment.colEnd);
				paintRow(segRowPtr2, Lsegment.colStart, Lsegment.colEnd);
			return;
		}


		// ------- START OF LOWEST BRANCH DETECTION -------
		// Color lower bound of 90
		//Function to rotate image 

		//Attempt to detect trunks in the image
		// std::vector<int> locateTrunk(cv::Mat depthImage)
		// {
		// 	//First get an idea as to what columns the trunk could be in
		// 	std::vector<int> trunkCheck;
		// 	//Check center, 20 pixels above and below center
		// 	cv::Vec3b* ptr = depthImage.ptr<cv::Vec3b>(depthImage.rows / 2);
		// 	cv::Vec3b* ptrUp = depthImage.ptr<cv::Vec3b>((depthImage.rows / 2) - 20);
		// 	cv::Vec3b* ptrDown = depthImage.ptr<cv::Vec3b>((depthImage.rows / 2) - 20);
		// 	for(int c = 0; c <= depthImage.cols; c++)
		// 	{
		// 		if((ptr[c][0] > 90) && (ptrUp[c][0] > 90) && (ptrDown[c][0] > 90))
		// 		{
		// 		trunkCheck.push_back(c);
		// 		}
		// 	}

		// 	//Rotate image 90 degrees clock-wise to work with columns instead of rows
		// 	//Now find those trunks assuming they are made of relatively straight (horizontal) white lines
		// 	cv::Mat rotatedImg = RotateImg(depthImage, -90.0);

		// 	//Find the columns with highest pixel values
		// 	std::vector<int> highCol;
		// 	for(int r = 0; r < rotatedImg.rows; r++)
		// 	{
		// 		int pixTot = 0;
		// 		cv::Vec3b* ptr = rotatedImg.ptr<cv::Vec3b>(r);
		// 		for(int c = 0; c < rotatedImg.cols; c++)
		// 		{
		// 			pixTot += ptr[c][0];
		// 		}

		// 		//If the total pixel value is high enough
		// 		if(pixTot > (90 * (depthImage.rows-1)))
		// 		{
		// 			highCol.push_back(r);
		// 		}
		// 	}

		// 	std::vector<int> definiteTrunkCols;
		// 	//Select the columns with a definite trunk found
		// 	for(int i = 0; i < highCol.size(); i++)
		// 	{
		// 		for(int j = 0; j < trunkCheck.size(); j++)
		// 		{
		// 			if (highCol[i] == trunkCheck[j])
		// 			{
		// 				definiteTrunkCols.push_back(highCol[i]);
		// 			}
		// 		}  
		// 	}
			
		// 	return definiteTrunkCols;
		// }

		//Map different column numbers to unique trunks
		std::map<int,std::vector<int>> mapTrunks(std::vector<int> definiteTrunkCols)
		{
			int totNum;                                       //Current tree number
			std::map<int,std::vector<int>> trunkDictionary;   //Output dictionary
			std::vector<int> uniqueTrunkColumns;              //Temporary vector
			if(definiteTrunkCols.size() == 0) totNum = 0;
			else
			{
				totNum = 1;
				for(int i = 1; i < definiteTrunkCols.size(); i++)
				{
					int distanceCheck = definiteTrunkCols[i] - definiteTrunkCols[i - 1];
					if((distanceCheck < 20) && (i < definiteTrunkCols.size() - 1))
					{
						//Insert columns to the corresponding trunk key 
						uniqueTrunkColumns.push_back(definiteTrunkCols[i-1]);
					}
					else
					{
						//Assign Dictionary to respective column set
						//If we're looking at the last element in definiteTrunkCols
						if(i == (definiteTrunkCols.size() - 1))uniqueTrunkColumns.push_back(definiteTrunkCols[i]);
						
						//Generate the key and value for the dictionary
						trunkDictionary.insert(std::make_pair(totNum, uniqueTrunkColumns));

						//Clear vector to reuse
						uniqueTrunkColumns.clear();

						//Move to next trunk
						totNum += 1;
					}
				}
			}
		return trunkDictionary;
		}
		
		//Locate the lowest branch
		int ComputeLowestBranchHeight(cv::Mat depthImage, 
			std::vector<int> definiteTrunkCols, 
			std::vector<int>& tNums, 
			std::vector<cv::Point>& tCols,
			int groundRow,
			const rs2_intrinsics* intrin,
			const Descriptors::RS2ParameterDescriptor& rs2desc,
			float& lowBranchHeightReported,
			Descriptors::AlgorithmDescriptor algdesc,
			cv::Mat rawDepthImage
			)
		{

			//Obtain trunk Dicitonary 
			std::map<int,std::vector<int>> trunkDict;
			trunkDict = mapTrunks(definiteTrunkCols);

			std::map<int,std::vector<int>>::iterator it = trunkDict.begin();
			//it->first = local tree ID
			//it->second = Vector holding column numbers for where a trunk is located
			//Vectors used in the enumeration of trunks
			std::vector<cv::Point> checkCols;     // Vector holding the location of rightmost trunk in the frame
			checkCols.push_back(cv::Point(0,0));  
			std::vector<cv::Point> colHolder;     // Vector holding all trunk locations(identified by columns) for a frame

			int firstBranchRow;   //Row locating the first branch

			//Find the lowest branch for each trunk independently
			while(it != trunkDict.end())
			{
				std::vector<int> tempRowPixValueHolder;
				int bottomStop = depthImage.rows;
				int trunkWidth = 0;

				//Check for any empty vectors in dictionary (causes seg fault if empty)
				if(it->second.size() > 0)
				{	
					trunkWidth = it->second[it->second.size() - 1] - it->second[0];

					//Downward sweep from center to find ground level
					for(int r = (depthImage.rows/2); r < depthImage.rows; r++)
					{
						cv::Vec3b* ptr = depthImage.ptr<cv::Vec3b>(r);
						int rowCheck = 0;
						for(int c = 0; c < depthImage.cols; c++)
						{
							if(ptr[c][0] > 90) rowCheck += 1;
						}

						if(rowCheck > trunkWidth + 150)
						{
							bottomStop = r;
							break;
						}
					}
					
					//Upward sweep from ground level to first branch
					firstBranchRow = 0;   //Row locating the first branch
					std::queue<int> holder;   //Holds the pixel values for given columns of a single row
					int extraWidth = 28;      //To take more pixels into account
					int extraHeight = 100;    //To prevent any undergrowth from messing everything up :/

					std::queue<int> holderPix;
					for(int r = bottomStop - extraHeight; r >= 0; r--)
					{
						cv::Vec3b* ptr = depthImage.ptr<cv::Vec3b>(r);
						//Method1
						int var1 = 0;
						int var2 = 0;
						int diff = 0;
						for(int c = it->second[0] - extraWidth; c < (it->second[it->second.size() - 1] + extraWidth); c++)
						{
							if(ptr[c][0] > 90 && c < depthImage.cols)
							{
								holder.push(c);
								holderPix.push(ptr[c][0]);
							} 
						}

						//Check if a branch is present (pattern of white-black-white) in row
						while (!holder.empty())
						{
							var1 = holder.front();
							holder.pop();
							if(!holder.empty())
							{
								var2 = holder.front();

								diff = var2 - var1;
								if(diff > 10)
								{ 
									//std::cout << diff << std::endl;
									firstBranchRow = r;
									break;
								}
							}
						}

						//If the first branch has been found, leave the loop
						if(firstBranchRow != 0) break;
					}

					//Color the trunk until the first branch
					int extra = 30;
					/*
					for(int r = bottomStop; r >= firstBranchRow; r--)
					{
						cv::Vec3b* ptr = depthImage.ptr<cv::Vec3b>(r);
						for(int c = 0; c < depthImage.cols; c++)
						{
							if(ptr[c][0] > 90)
							{
								if(c < ((it->second[it->second.size() - 1]) + extra) && c > (it->second[0] - extra))
								{
									ptr[c] = cv::Vec3b(0, 255, 0); //BGR 
								}
							}
						}
					}
					*/
					//std::cout << "holder " << it->first << ": " << it->second[it->second.size() - 1] << " " << it->second[0] << std::endl;
					colHolder.push_back(cv::Point(it->second[it->second.size() - 1], it->second[0]));

					int trunkMidCol = (colHolder[colHolder.size() - 1].x + colHolder[colHolder.size() - 1].y) / 2;
					if(firstBranchRow > 5)
					{
						//Obtain row pointers
						ushrt* segRowPtr = rawDepthImage.ptr<ushrt>(firstBranchRow);
						ushrt* grndRowPtr = rawDepthImage.ptr<ushrt>(bottomStop);

						//Turn the row into the real-world height
						lowBranchHeightReported = DistBetweenPixels(
							firstBranchRow,
							trunkMidCol,
							groundRow,
							trunkMidCol,
							segRowPtr[trunkMidCol] * rs2desc.RS2_DEPTH_UNITS,
							grndRowPtr[trunkMidCol] * rs2desc.RS2_DEPTH_UNITS,
							intrin);
						/*
						std::cout << "firstBranchRow: " << firstBranchRow << std::endl;
						std::cout << "trunkMidCol: " << trunkMidCol << std::endl;
						std::cout << "groundRow: " << groundRow << std::endl;
						std::cout << "segRowPtr: " << segRowPtr[trunkMidCol] * rs2desc.RS2_DEPTH_UNITS << std::endl;
						std::cout << "grndRowPtr: " << grndRowPtr[trunkMidCol] * rs2desc.RS2_DEPTH_UNITS << std::endl;
						std::cout << "-----" << std::endl;
						*/
					}
					else lowBranchHeightReported = 0.0;
					std::ostringstream height;
					height << std::fixed << std::setprecision(2) << lowBranchHeightReported;
					std::string heightWithPrec = height.str();

					// Display the height in the depth image
					cv::putText(depthImage,                                   				// img
						heightWithPrec,     			         							// text
						cv::Point(trunkMidCol - 120, int(depthImage.rows/2)),				// org
						cv::FONT_HERSHEY_DUPLEX,                              				// fontFace
						1,                                                   				// fontScale
						cv::Scalar(0,0,255),                                  				// color (B,G,R)
						2,                                                    				// thickness
						cv::LINE_8                                            				// lineType
					);
				}
				it++;
			} 
			//Enumerate the trunks
			//x = right bound
			//y = left bound
			if(it->first!= 0)
			{
				it--;
				checkCols[0] = (cv::Point(it->second[it->second.size() - 1], it->second[0])); 
			}
			//Initialize the comparison vector tCols
			//This vector serves like memory of the rightmoust trunk in the previous frame
			//When the previous and the current rightmost tree of the frame have very close colmn values,
			//no enumaration update is made. However, when the rightmost tree leaves the frame, the count is increased
			if(tCols[0] == cv::Point(0,0)) tCols[0] = checkCols[0];
			for(int j = 0; j < colHolder.size(); j += 1)
			{
				// X coordinate for the number placement  
				int trunkMidCol = (colHolder[colHolder.size() - 1 - j].x + colHolder[colHolder.size() - 1 - j].y) / 2;
			
				// Draw the number in the depth image
				cv::putText(depthImage,                                   // img
					std::to_string(tNums[0] + j),                         // text
					cv::Point(trunkMidCol + 30, int(depthImage.rows/2)),  // org
					cv::FONT_HERSHEY_DUPLEX,                              // fontFace
					2,                                                    // fontScale
					cv::Scalar(255,0,0),                                  // color (B,G,R)
					2,                                                    // thickness
					cv::LINE_8                                            // lineType
				);

			}
			if( tCols[0] != cv::Point(0,0) && checkCols[0] != cv::Point(0,0) &&
				checkCols[0].x < tCols[0].x - 150 && checkCols[0].y < tCols[0].y + 150 && 
				colHolder[0].y <= 280 && colHolder[0].x >= 100)
			{
				tNums[0] = (tNums[tNums.size() - 1] + 1);
			}
			// Update the comparison vector tCols
			if(checkCols[0] != cv::Point(0,0)) tCols[0] = checkCols[0];  
			
			checkCols.clear();
			checkCols.push_back(cv::Point(0,0));
			colHolder.clear();
			
			// View results (press ENTER to view next image)
			cv::imshow("Lowest Branch", depthImage);
			//Use this waitkey to step frame by frame using the ENTER button
			while(cv::waitKey(1) != 13);
			//cv::waitKey(1);
			return 0;
		}
	
		double sixFeetVolume(float taper, double sixFTdiameter){
			double groundDiameter = sixFTdiameter + taper * 1.83;
			double volume = 3.14159 * 1.83 / 3 * (sixFTdiameter * sixFTdiameter + groundDiameter * groundDiameter + groundDiameter * sixFTdiameter) / 40000;
			return volume;
		}

		void paintRow(ushrt* rowPtr, int left, int right){
			for(int i = left; i < right; i++){
				rowPtr[i] = 65535;
			}
		}
		//Output looks like: [[x_p,y_p,z_n],...,[x_p,y_p,z_n],[x_p,y_p,z_n]]
		//x_p,y_p correspond to the 2D frame Pixel position in a frame for the centerline
		//z_n correspond to the Z-component of the surface Normal image for the [x_p,y_p] pixel coordinate
		std::vector<cv::Point3i> FindTrunkCenterLine(cv::Mat depthImage, const Descriptors::RS2ParameterDescriptor& rs2desc, int *bin)
		{
			if (rs2desc.RS2_VERTICAL_ORIENTATION)
            {
                cv::transpose(depthImage, depthImage);
			}
			//-------------- Start of code snippet snatched from NormalMapGradientCompute.h --------------//
			//Mats
			cv::Mat depthImageFloatNorm; 	
			cv::Mat surfaceNormalImage 		= cv::Mat(depthImage.rows, depthImage.cols, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));
			//Convert depth image to normalized 32 bit floating point image [0,65535]
			depthImage.convertTo(depthImageFloatNorm, CV_32FC1, 1.0f); 
			//Iterate over pixel rows. Note this task can be easily parallelized in the future
			for (int k = 1; k < depthImageFloatNorm.rows-1; k++)
			{
				//Obtain pointer to a pixel row
				float* rowPtrAbove	= depthImageFloatNorm.ptr<float>(k-1);
				float* rowPtr		= depthImageFloatNorm.ptr<float>(k);
				float* rowPtrBelow  = depthImageFloatNorm.ptr<float>(k+1);
				
				//Iterate over columns
				for (int j = 1; j < depthImageFloatNorm.cols-1; j++)
				{
					//Points used to compute normal
					float above = rowPtrAbove[j];
					float below	= rowPtrBelow[j];
					float left	= rowPtr[j-1];
					float right	= rowPtr[j+1];

					//Perform a check to ensure that we are ignoring depths of 0 meters and depths that are very far from each other
					//Note: This methods ASSUMES that a depth cutoff filter has already been applied.
					if(above == 0.0f || below == 0.0f || left == 0.0f || right == 0.0f)
					{
						//Assign to normal map
						surfaceNormalImage.at<cv::Vec3f>(k,j) = cv::Vec3f(0.0f);
						continue;	
					}
					
					//Compute surface normals
					//Use method in this post: https://stackoverflow.com/questions/34644101/calculate-surface-normals-from-depth-image-using-neighboring-pixels-cross-produc
					float dzdx = (above - below) / (2.0f);
					float dzdy = (right - left) / (2.0f);
					float mag = sqrtf(dzdx*dzdx + dzdy*dzdy + 1.0f*1.0f);

					surfaceNormalImage.at<cv::Vec3f>(k,j) =  cv::Vec3f( dzdx/mag, dzdy/mag, 1.0/mag );
				}
			}
			//-------------- End of code snippet snatched from NormalMapGradientCompute.h --------------//

			//Vector where each entry is a vector of vectors representing the position and normal values of each pixel 
			//making part of the center line of the trunk
			std::vector<cv::Point3i> centerLine = std::vector<cv::Point3i>();
			
			cv::Mat normalMap;
			surfaceNormalImage.convertTo(normalMap, CV_8UC3, 255);
			cv::Mat centerLineDepthImage;
			depthImage.convertTo(centerLineDepthImage, CV_8U, 255.0 / 4096.0); 

			int binSize = normalMap.cols / 16;
			int frequencyBin[16] = {0};
			//Buffer to be added for a wider window view
			int buffer = 30;

			//Begin by filtering out the X and Y components from the normal image
			for (int r = 0; r < normalMap.rows; r++)
			{	
				cv::Vec3b *ptr = normalMap.ptr<cv::Vec3b>(r);
				for (int c = 0; c < normalMap.cols; c++)
				{
					// Remove the X and Y component, and low Z components
					if (ptr[c][0] > 0 || ptr[c][1] > 0)
					{
						ptr[c] = cv::Vec3b(0, 0, 0); //BGR
					}
					// Count how many valid Z component pixels are left
					if(c < (3 * normalMap.cols / 4) && ptr[c][2] > 0) 
					{
						frequencyBin[int(c / binSize)] += 1; 
					}            
				}
			}
			//cv::imshow("Suface Nomal", surfaceNormalImage);
			//cv::imshow("Suface Nomal Z-Component", normalMap);

			//Find the bin holding the highest value
			int N = sizeof(frequencyBin) / sizeof(int);
			int mostFrequentBinIndex = std::distance(frequencyBin, std::max_element(frequencyBin, frequencyBin + N));
			//Left most column of the most frequent bin
			int col = mostFrequentBinIndex * binSize;
			//New frequency bin to find the most common colors in that bin
			int colorBinSize = 256 / 32;
			int colorFrequencyBin[32] = {0};
			//Crop everything out except content inside the mostFrequentBin
            for (int r = 0; r < normalMap.rows; r++) 
			{
				cv::Vec3b *ptr1 = normalMap.ptr<cv::Vec3b>(r);		
				uchar *ptr2 = centerLineDepthImage.ptr<uchar>(r);	//Only 1 channel, not 3 like RGB
				for (int c = 0; c < normalMap.cols; c++)
				{
					if((c > col + binSize + buffer) || (c < col - buffer))
					{
						ptr1[c] = cv::Vec3b(0, 0, 0); 	//BGR of SNA
						ptr2[c] = 0; 					//Depth
					}
					//Find the most common colors in that bin, but disregard black (background cutoff color)
					else if(ptr2[c] != 0)
					{
						colorFrequencyBin[int(ptr2[c] / colorBinSize)] += 1;
					}
				}				 
			}

			*bin = col;
			//cv::imshow("Depth Image: Filter #1", centerLineDepthImage);
			
			//Find the bin holding the highest color value
			N = sizeof(colorFrequencyBin) / sizeof(int);
			mostFrequentBinIndex = std::distance(colorFrequencyBin, std::max_element(colorFrequencyBin, colorFrequencyBin + N));
			//Most frequnt color bin
			int color = mostFrequentBinIndex * colorBinSize;
			int colorBuffer = 25;
			//Filter out all colors except the one in the mostFrequentBin
            for (int r = 0; r < centerLineDepthImage.rows; r++) 
			{
				uchar *ptr1 = centerLineDepthImage.ptr<uchar>(r);
				for (int c = 0; c < centerLineDepthImage.cols; c++)
				{
					if(ptr1[c] < color - colorBuffer || ptr1[c] > color + colorBinSize + colorBuffer)
					{
						ptr1[c] = 0;
					}
				}				 
			}

			//cv::imshow("Depth Image: Filter #2", centerLineDepthImage);

			//Attempt to remove branches so that a better centerline can be obtained
			//References used:
			//https://docs.opencv.org/3.4.0/d9/d61/tutorial_py_morphological_ops.html
			//https://www.tutorialspoint.com/opencv/opencv_morphological_operations.htm
			int morph_size = 2;
			cv::Mat element = getStructuringElement( cv::MORPH_RECT, cv::Size( 2*morph_size+1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );
			
			cv::morphologyEx( centerLineDepthImage, centerLineDepthImage, cv::MORPH_OPEN, element );
			cv::morphologyEx( centerLineDepthImage, centerLineDepthImage, cv::MORPH_CLOSE, element );

			cv::erode(centerLineDepthImage, centerLineDepthImage, element);
			cv::erode(centerLineDepthImage, centerLineDepthImage, element);
			cv::dilate(centerLineDepthImage, centerLineDepthImage, element);
			cv::dilate(centerLineDepthImage, centerLineDepthImage, element);

			//cv::imshow("Depth Image: Morphological Operation", centerLineDepthImage);

			//Obtain the outline of the trunk with some of the branches erroded away
			//Followed the following example:
			//https://riptutorial.com/opencv/example/23501/canny-algorithm---cplusplus
			cv::Mat depthImgBlurred, depthImgCanny;
			int lowThreshold = 0;
			const int max_lowThreshold = 100;
			const int ratio = 3;
			const int kernel_size = 3;
			cv::GaussianBlur(centerLineDepthImage, depthImgBlurred, cv::Size(5, 5), 1.5); 
			cv::Canny(depthImgBlurred, depthImgCanny, 100, 200);

			//cv::imshow("Depth Image: outline", depthImgCanny);

			//Now that we only display the outline, let's find it's center line using the depth image.
			//This was used instead of the normal image because in the imshow of only the z component,
			//gaps can be seen for the outline of the trunk, providing inacurate results for a center line
			int rowPixelTotal, cnt;
			cv::Point3i pixelData;
			surfaceNormalImage.convertTo(surfaceNormalImage, CV_8UC3, 255);
			std::vector<int> tempCenterHolder = std::vector<int>();
			std::vector<int> tempRowHolder = std::vector<int>();
			
			cv::morphologyEx( normalMap, normalMap, cv::MORPH_CLOSE, element );

			int outlierRange = 40;
			int diffMax = 80;
			for (int r = 0; r < (normalMap.rows * 17) / 20 ; r++) 
			{
				cv::Vec3b *ptr0 = surfaceNormalImage.ptr<cv::Vec3b>(r);
				cv::Vec3b *ptr1 = normalMap.ptr<cv::Vec3b>(r);
				uchar *ptr2 = depthImgCanny.ptr<uchar>(r);	//Only 1 channel, not 3 like RGB

				rowPixelTotal = 0;
				cnt = 0;
				for (int c = 0; c < depthImgCanny.cols; c++)
				{
					if(ptr2[c] == 255) {
						rowPixelTotal += c;
						cnt += 1;
						tempRowHolder.push_back(c);
					}
				}
				//If this row is below the base of trunk, don't utilize it
				if(!tempRowHolder.empty() && (tempRowHolder.back() - tempRowHolder.front() > (binSize + 2*buffer - 2))) {
					cnt = 0;
				}
				tempRowHolder.clear(); 
				//Obtain the center pixel of a row and color it in the normal image
				//center == x-axis && r == y-axis
				if(cnt != 0) {
					int center = int(rowPixelTotal / cnt);
					//Check for outliers if there are enough 'outlierRange' points to use as reference
					//This serves to decrease any noise in the centerline when branches are present
					//Because noise is removed, gaps in the centerline appear
					if(tempCenterHolder.size() > outlierRange) {
						int diff = 0;
						//Calculate the distance difference between pixels
						for(int i = tempCenterHolder.size() - outlierRange - 1; i < tempCenterHolder.size(); i++) {
							diff += abs(center - tempCenterHolder[i]);
						}
						//The lower the difference, the lower the noise, the straighter the centerline.
						if(diff < diffMax) {
							pixelData = cv::Point3i(center,r,255);
							centerLine.push_back(pixelData);
						}
						//If difference is to big:
						//Fill in with 0s as place holders representing the 'gaps' in the centerline
						else {	
							pixelData = cv::Point3i(0,r,0);
							centerLine.push_back(pixelData);
						}
						//Fill the vector used to eliminate outliers
						tempCenterHolder.push_back(center);
					}
					//If there are not enough 'outlierRange' points
					else {
						//Push contents to output vector
						pixelData = cv::Point3i(0,r,0);
						centerLine.push_back(pixelData);
						//Fill the vector used to eliminate outliers
						tempCenterHolder.push_back(center);
					}
					//std::cout << centerLine[centerLine.size() - 1] <<std::endl;
				}
				else {
					pixelData = cv::Point3i(0,r,0);
					centerLine.push_back(pixelData);
				}
			}
			
			// Not the best approach :/
			//Now we have a centerline containing gaps after the noise created by branches was removed. Let's fix these gaps
			//Rows == y-axis
			/*
			cv::Point2i gapStart = cv::Point2i(0,0);
			cv::Point2i gapEnd = cv::Point2i(0,0);
			for (int r = int(centerLine.size() * 3 / 34); r < int(centerLine.size() * 13 / 17); r++) {
				cv::Vec3b *ptr0 = surfaceNormalImage.ptr<cv::Vec3b>(r);
				//Step 1: Find the start of the gap. When found, save the starting row index and center position of the gap
				if(gapStart.x == 0 && centerLine[r].x == 0) {
					gapStart = cv::Point2i(centerLine[r-1].x, r-1);
				}
				//Step 2: Find the end of the gap
				if(gapEnd.x == 0 && gapStart.x != 0 && centerLine[r].x != 0) {
					gapEnd = cv::Point2i(centerLine[r].x, r); 
					//Step 3: Now begin filling it up from start to end using y=mx + b (x is unknown)
					if((gapEnd.x == gapStart.x) || (gapStart.y == gapEnd.y)) 
					{
						for(int i = gapStart.y; i < gapEnd.y; i++) {
							centerLine[i].x = gapStart.x;
							centerLine[i].z = ptr0[gapStart.x][2];
						}
					}
					else
					{
						int slope = (gapEnd.y- gapStart.y) / (gapEnd.x- gapStart.x);
						int b = (-slope * gapStart.x) +  gapStart.y;
						for(int i = gapStart.y; i < gapEnd.y; i++) {
							if(slope != 0) {
								centerLine[i].x = int((i-b)/slope);
								centerLine[i].z = ptr0[int((i-b)/slope)][2];
							} 
							else {
								centerLine[i].x = gapStart.x;
								centerLine[i].z = ptr0[gapStart.x][2];
							}
						}
					}

					//Step 4: Reset to find the next gap
					gapStart = cv::Point2i(0,0);
					gapEnd = cv::Point2i(0,0);
				}
			}
			*/
		
			for (int r = 0; r < centerLine.size(); r++) 
			{
				cv::Vec3b *ptr1 = normalMap.ptr<cv::Vec3b>(r);
				if (centerLine[r].x != 0) ptr1[centerLine[r].x] = cv::Vec3b(255, 255, 0); //BGR
			}

			cv::Point2i start = cv::Point2i(0,0);
			cv::Point2i end = cv::Point2i(0,0);
			
			for(int i = 0; i < centerLine.size()-1 ; i++) {
				if(start.x == 0 && centerLine[i].z == 255) start = cv::Point2i(centerLine[i].x, centerLine[i].y);
				if(centerLine[i].z == 255) end = cv::Point2i(centerLine[i].x, centerLine[i].y);
			}
			cv::line(normalMap,start,end,cv::Scalar(255,255,255),1);

			cv::imshow("realsense_fia: normal-z", normalMap);
			return centerLine;
		}
		
		//Compute the straightness of a trunk by finding the shortest distance between a line and a point in a 3D plane
		float computeStraightness_m1(std::vector<cv::Point3i> centerLine)
		{
			//Begin by chosing the edge points so that a straight line can be drawn
			cv::Point3i A = cv::Point3i(0,0,0);
			cv::Point3i B = cv::Point3i(0,0,0);
			for(int i = 0; i < centerLine.size()-1 ; i++) {
				if(A.z == 0 && centerLine[i].z == 255) A = cv::Point3i(centerLine[i].x, centerLine[i].y, centerLine[i].z);
				if(centerLine[i].z == 255) B = cv::Point3i(centerLine[i].x, centerLine[i].y, centerLine[i].z);
			}
			//Find the point that is located farthest away from the line created by points A and B
			//This point represents the largest curvature of the trunk
			//Reference: https://www.geeksforgeeks.org/shortest-distance-between-a-line-and-a-point-in-a-3-d-plane/
			float largestDist = 0;
			cv::Point3i C = cv::Point3i(0,0,0);
			for(int i = 0; i < centerLine.size()-1 ; i++) {
				if(centerLine[i].z) {
					C = cv::Point3i(centerLine[i].x, centerLine[i].y, centerLine[i].z);
					cv::Point3i AB = B - A;
    				cv::Point3i AC = C - A;
    				cv::Point3i ABdotAC = cv::Point3i(AB.y*AC.z-AB.z*AC.y, AB.z*AC.x-AB.x*AC.z, AB.x*AC.y-AB.y*AC.x);
					float area = sqrt(ABdotAC.x*ABdotAC.x + ABdotAC.y*ABdotAC.y + ABdotAC.z*ABdotAC.z);
					float CD = area / sqrt(AB.x*AB.x + AB.y*AB.y + AB.z*AB.z);
    				if( CD > largestDist) largestDist = CD;
				}
			}
			//std::cout<<largestDist<<std::endl;
			return largestDist;
		}

		//Compute the straightness of a trunk by finding the area between a line and curve (centerline) in a 3D plane
		//Separates areas between intersections of the centerline and line AB
		std::vector<float> computeStraightness_m2(std::vector<cv::Point3i> centerLine)
		{
			//Output areas
			std::vector<float> computedAreas = std::vector<float>();
			//Begin by chosing the edge points so that a straight line can be drawn
			cv::Point3i A = cv::Point3i(0,0,0);
			cv::Point3i B = cv::Point3i(0,0,0);
			for(int i = 0; i < centerLine.size()-1 ; i++) {
				if(A.z == 0 && centerLine[i].z == 255) A = cv::Point3i(centerLine[i].x, centerLine[i].y, centerLine[i].z);
				if(centerLine[i].z == 255) B = cv::Point3i(centerLine[i].x, centerLine[i].y, centerLine[i].z);
			}

			//Parametrics equations of AB
			//x(t) = A.x + (B.x-A.x) * t;
			//y(t) = A.y + (B.y-A.y) * t;
			//z(t) = A.z + (B.z-A.z) * t;

			//Find areas between intersections of line AB and the inputted centerLine
			//(x,y) are used as the intersection reference, z is disregarded for area use
			//pos 1 == centerline is right of line AB
			//pos 0 == centerline is left of line AB
			int pos = 0;
			int prevPos = 0;
			float area = 0;
			float mag = 0;
			float prevMag = 0;
			for(int y = A.y + 1; y < B.y; y++)
			{
				//x coordinate of line AB using the cartesian form of the parametric equations 
				float ABx = centerLine[y].x;
				if((B.y-A.y) != 0) {
					ABx = A.x + ((float(B.x-A.x)/float(B.y-A.y)) * (y - A.y));
				}
				//If centerline is to the left of line AB
				if(centerLine[y].x - ABx < 0) pos = 0;
				//Else if centerline is to the right of line AB
				else if(centerLine[y].x - ABx > 0) pos = 1;
				//If it's our first comparison, fresh start
				if(y == A.y + 1) prevPos = pos;
				
				//Compute the area (sum of magnitudes between line AB and the centerline)
				float ABz = centerLine[y].z;
				if(B.y-A.y) ABz = A.z + (B.z-A.z) * ((y - A.y)/B.y-A.y);
				if(centerLine[y].z != 0) {
					mag = sqrt((ABx-centerLine[y].x)*(ABx-centerLine[y].x) +\
					 (y-centerLine[y].y)*(y-centerLine[y].y) +\
					 (ABz-centerLine[y].z)*(ABz-centerLine[y].z));
					area += mag;
					prevMag = mag;
				}
				else area += prevMag;
				//If centerline moves from one side of line AB to the other
				if(pos != prevPos) {
					prevPos = pos;
					computedAreas.push_back(area);
					area = 0;
				}
				//Same point in centerline and line AB
				else if(centerLine[y].x == ABx) {
					computedAreas.push_back(area);
					area = 0;
				}
			}

			return computedAreas;
		}

		//Compute the straightness of a trunk by finding the area between a line and curve (centerline) in a 3D plane
		//Separates areas into k sections, so that we can relate these k sections between trees
		std::vector<float> computeStraightness_m3(std::vector<cv::Point3i> centerLine)
		{
			//Output areas
			int k = 10;
			std::vector<float> computedAreas(k, 0);
			//Begin by chosing the edge points so that a straight line can be drawn
			cv::Point3i A = cv::Point3i(0,0,0);
			cv::Point3i B = cv::Point3i(0,0,0);
			for(int i = 0; i < centerLine.size()-1 ; i++) {
				if(A.z == 0 && centerLine[i].z == 255) A = cv::Point3i(centerLine[i].x, centerLine[i].y, centerLine[i].z);
				if(centerLine[i].z == 255) B = cv::Point3i(centerLine[i].x, centerLine[i].y, centerLine[i].z);
			}

			//Divide the trunk into k sections and find areas within these sections between line AB and the inputted centerLine
			float mag = 0;
			float prevMag = 0;
			for(int y = A.y + 1; y < B.y; y++)
			{
				//x coordinate of line AB using the cartesian form of the parametric equations 
				float ABx = centerLine[y].x;
				if((B.y-A.y) != 0) {
					ABx = A.x + ((float(B.x-A.x)/float(B.y-A.y)) * (y - A.y));
				}
				
				//Compute the area (sum of magnitudes between line AB and the centerline)
				float ABz = centerLine[y].z;
				if(B.y-A.y) ABz = A.z + (B.z-A.z) * ((y - A.y)/B.y-A.y);
				//Choose the section this area belongs too by scaling 'y' into a range [a,b] = [0,k] using:
				//y_normalized = ((b-a) * ((y - y_min) / (y_max - y_min))) + a
				int section = k * (float(y - (A.y + 1)) / float(B.y - (A.y + 1)));
				//Add area to chosen trunk section
				if(centerLine[y].z != 0) {
					mag = sqrt((ABx-centerLine[y].x)*(ABx-centerLine[y].x) +\
					 (y-centerLine[y].y)*(y-centerLine[y].y) +\
					 (ABz-centerLine[y].z)*(ABz-centerLine[y].z));
					computedAreas[section] += mag;
					prevMag = mag;
				}
				else computedAreas[section] += prevMag;
			}

			for(int i = 0; i < computedAreas.size(); i++)
			{
				std::cout<<computedAreas[i]<<std::endl;
			}
			return computedAreas;
		}

		//Compute the straightness of a trunk by finding the total area between a line and curve (centerline) in a 3D plane
		float computeStraightness_m4(std::vector<cv::Point3i> centerLine)
		{
			//Output area
			float computedArea = 0;
			//Begin by chosing the edge points so that a straight line can be drawn
			cv::Point3i A = cv::Point3i(0,0,0);
			cv::Point3i B = cv::Point3i(0,0,0);
			for(int i = 0; i < centerLine.size()-1 ; i++) {
				if(A.z == 0 && centerLine[i].z == 255) A = cv::Point3i(centerLine[i].x, centerLine[i].y, centerLine[i].z);
				if(centerLine[i].z == 255) B = cv::Point3i(centerLine[i].x, centerLine[i].y, centerLine[i].z);
			}

			//If line AB is to short, no accurate straigtness can be computed
			if(abs(A.y-B.y) < (0.35 * centerLine.size())) return computedArea;

			//Divide the trunk into k sections and find areas within these sections between line AB and the inputted centerLine
			float mag = 0;
			float prevMag = 0;
			for(int y = A.y + 1; y < B.y; y++)
			{
				//x coordinate of line AB using the cartesian form of the parametric equations 
				float ABx = centerLine[y].x;
				if((B.y-A.y) != 0) {
					ABx = A.x + ((float(B.x-A.x)/float(B.y-A.y)) * (y - A.y));
				}
				//Compute the area (sum of magnitudes between line AB and the centerline)
				float ABz = centerLine[y].z;
				if(B.y-A.y) ABz = A.z + (B.z-A.z) * ((y - A.y)/B.y-A.y);
				//Add area to chosen trunk section
				if(centerLine[y].z != 0) {
					mag = sqrt((ABx-centerLine[y].x)*(ABx-centerLine[y].x) +\
					 (y-centerLine[y].y)*(y-centerLine[y].y) +\
					 (ABz-centerLine[y].z)*(ABz-centerLine[y].z));
					computedArea += mag;
					prevMag = mag;
				}
				else computedArea += prevMag;
			}
			return computedArea;
		}
// Helper function color cv::Mat that represents neural network output
        
        cv::Mat ColorizeDepthSegNetClassmap(cv::Mat depthSegnetClassmap)
        {
            // Init. return mat
            cv::Mat colorizedClassmap(depthSegnetClassmap.rows, depthSegnetClassmap.cols, CV_8UC3);
            // Color map
            // 0 = background = black
            // 1 = ground = red
            // 2 = trunk = green
            // 3 = branch = blue
            const cv::Vec3b classIndexToColorDictionary[] = {cv::Vec3b(0,0,0), cv::Vec3b(255,0,0), cv::Vec3b(0,255,0), cv::Vec3b(0,0,255)};

            unsigned int pixelRowIdx = 0;
            unsigned int pixelColIdx = 0;

            //Ensure depthSegnetClassmap is contiguous
	//is this line necessary??
            //torch::Tensor depthSegnetClassmapContiguous = depthSegnetClassmap.contiguous();
            //  Iterate over tensor
            for( pixelRowIdx = 0; pixelRowIdx < depthSegnetClassmap.rows; pixelRowIdx++)
            {
                for(pixelColIdx = 0; pixelColIdx < depthSegnetClassmap.cols; pixelColIdx++)
                {
                    // Check value of *k to determine how to colorize
                    unsigned char predictedClassIndex = depthSegnetClassmap.at<uchar>(pixelRowIdx, pixelColIdx);
//std::count << +predictedClassIndex << std::endl;

                    colorizedClassmap.at<cv::Vec3b>(cv::Point(pixelColIdx, pixelRowIdx)) = classIndexToColorDictionary[(int)predictedClassIndex];

			//colorizedClassmap.at<cv::Vec3b>(cv::Point(pixelColIdx, pixelRowIdx)) = classIndexToColorDictionary[depthSegnetClassmap.at<uchar>(pixelRowIdx, pixelColIdx)];
                }
            }

            // Return colorized classmap in proper colorspace
            cv::cvtColor(colorizedClassmap,colorizedClassmap, cv::COLOR_BGR2RGB);
            return colorizedClassmap;
        }

//find trunk centerline version 2,return a vector of 2-d tuples.(all of the mid-point of trunk)
		std::vector<std::tuple<int,int>>  FindTrunkCenterLine_V2(cv::Mat NNImage, cv::Mat& coloredMap, std::vector<std::tuple<int,int>> trunkVec)
		{
			//define output
			std::vector<std::tuple<int,int>> centerline;
			int trunkStart = -1;
			int trunkEnd = -1;
			//int Trunk = 2;
			
            //int column = 0;


			//check if images are empty
			if (NNImage.empty() || coloredMap.empty()) {
            			std::cout << "Could not open or find the image";
				return centerline;
        		}



			//color the classmap:
			//0 = background = black
			//1 = ground = red
			// 2 = trunk = green
			// 3 = branch = blue
			coloredMap = ColorizeDepthSegNetClassmap(NNImage);
			cv::imshow("classmap", coloredMap);
			cv::waitKey(1);

			int height = NNImage.rows;
			int width = NNImage.cols;
			unsigned int i = 0;
			unsigned int j = 0;

			//check the size of the image
			if (height == 0 || width == 0){
				return centerline;			
			}
			//Iterate Rows From Bottom Up
			for (i = height-1; i > 0; i--)
			{
				//Define Trunk Bounds
				trunkStart = -1;
				trunkEnd = -1;
				//Iterate Over Column Values
				for (j = 0; j < width; j++)
				{
					//Get Pixel Value
					unsigned char pixel = NNImage.at<uchar>(i,j);

					//Detect Trunk
					if (trunkStart == -1 && trunkEnd == -1 && (int)pixel == 2)
					{
						trunkStart = j;
						trunkEnd = j;

					}
					else if (trunkStart != -1 && pixel == 2)
					{
						trunkEnd = j;
					}
				}

				if (trunkStart != -1 && trunkEnd != -1)
				{
                                centerline.push_back(std::make_tuple(i, (trunkStart+trunkEnd)/2));
				trunkVec.push_back(std::make_tuple(trunkStart, trunkEnd));


				unsigned int mid = trunkStart/2+trunkEnd/2;

				coloredMap.at<cv::Vec3b>(cv::Point(mid,i)) = cv::Vec3b(255,255,255);
				coloredMap.at<cv::Vec3b>(cv::Point(mid+1,i)) = cv::Vec3b(255,255,255);
				coloredMap.at<cv::Vec3b>(cv::Point(mid-1,i)) = cv::Vec3b(255,255,255);
				coloredMap.at<cv::Vec3b>(cv::Point(mid+2,i)) = cv::Vec3b(255,255,255);
				coloredMap.at<cv::Vec3b>(cv::Point(mid-2,i)) = cv::Vec3b(255,255,255);
				}

			}
			//show the image after marking the centerline
			cv::imshow("centerline", coloredMap);
			cv::waitKey(1);
			//std::cout << "The number of centers should be : " << centerline.size() << std::endl;
			return centerline;
		}
//get farthest point, lied on centerline, that has the longest distance to AB-line.
std::tuple<int,int> FindFarthestPoint(std::vector<std::tuple<int,int>> centerline)
{
			std::tuple<int,int> A; // Define point A (Highest point)
			std::tuple<int,int> B; // Define point B (Lowest  point)
			std::tuple<int,int> farthestPoint;
			if (centerline.size() >= 3) {
				A = centerline[centerline.size() - 1];
				B = centerline[0];

				// Find the pixel location of point A and B in the images
				int A_x = std::get<1>(A);
				int A_y = std::get<0>(A);
				int B_x = std::get<1>(B);
				int B_y = std::get<0>(B);

				//find the AB line eqution: y=kx+b where x = column, y = row.
				//find k
				float k = (A_y - B_y)/(A_x - B_x);

				//find b 
				float b = A_y - k * A_x;

				//initialize the distance & wanted return values
				int row = 0;
				int column = 0;
				float distance  = 0.0;
				//std::tuple<int,int> farthestPoint;
				float tmp_distance = 0.0;

				//we do not iterate the start and the end tuple because they are definitly on AB line,
				// which makes the tmp_distance = 0.
				for (int i = 1; i < centerline.size() - 1; i++)
				{
					//get the distance between centerline and AB line at same row
					column = std::get<1>(centerline[i]);
					row = std::get<0>(centerline[i]); 
					tmp_distance = abs(column - (row - b)/k);
					if (tmp_distance > distance)
					{
						distance = tmp_distance;
						farthestPoint = centerline[i];
					}
					
				}
				//return farthestPoint;
			}
			else{
				std::cout << "Not enough centerline pixels" << std::endl;
				farthestPoint = centerline[1];	
			}
			return farthestPoint;
		}
//get farthest point, lied on centerline, that has the longest distance to AB-line.
std::pair <float,int> FindMG(std::vector<std::tuple<int,int>> centerline)
{
			std::tuple<int,int> A; // Define point A (Highest point)
			std::tuple<int,int> B; // Define point B (Lowest  point)
			if (centerline.size() >= 3) {
				A = centerline[centerline.size() - 1];
				B = centerline[0];
			}else{
				printf("The number of Centerline pixel value not enough.\n");
				std::pair <float,int> MGandRow (-1, -1);
				return MGandRow;
			}
			// Find the pixel location of point A and B in the images
			int A_x = std::get<1>(A);
			int A_y = std::get<0>(A);
			int B_x = std::get<1>(B);
			int B_y = std::get<0>(B);

			//find the AB line eqution: y=kx+b where x = column, y = row.
			//find k
			float k = (A_y - B_y)/(A_x - B_x);

			//find b 
			float b = A_y - k * A_x;

			//initialize the distance & wanted return values
			int row = 0;
			int column = 0;
			float MG  = 0.0;
			std::tuple<int,int> farthestPoint;
			float tmp_MG = 0.0;

			//we do not iterate the start and the end tuple because they are definitly on AB line,
			// which makes the tmp_distance = 0.
			for (int i = 1; i < centerline.size() - 1; i++)
			{
				//get the distance between centerline and AB line at same row
				column = std::get<1>(centerline[i]);
				row = std::get<0>(centerline[i]); 
				tmp_MG = abs(column - (row - b)/k);
				if (tmp_MG > MG)
				{
					MG = tmp_MG;
					farthestPoint = centerline[i];
				}	
			}

			std::pair <float,int> MGandRow (MG, std::get<0>(farthestPoint));
			return MGandRow;		
}
int findDMG(cv::Mat NMImage, cv::Mat& coloredMap, std::pair <float,int>MGandRow)
{
	int tarRow = MGandRow.second;
	int trunkStart = -1;
	int trunkEnd = -1;

	//check if images are empty
	if (NMImage.empty() || coloredMap.empty()) {
        std::cout << "Could not open or find the image";
		return -1;
    }

	//color the classmap:
	//0 = background = black
	//1 = ground = red
	// 2 = trunk = green
	// 3 = branch = blue
	coloredMap = ColorizeDepthSegNetClassmap(NMImage);
	cv::imshow("classmap", coloredMap);
	cv::waitKey(1);

	int height = NMImage.rows;
	int width = NMImage.cols;
	unsigned int i = 0;
	unsigned int j = 0;

	//check the size of the image
	if (height == 0 || width == 0){
		return -1;			
	}

	//Find the DMG at tarRow
	for(i = 0; i < width; i++){
		unsigned char pixel = NMImage.at<uchar>(tarRow,i);
		//Detect Trunk
		//std::cout << "The trunkstart is: " <<trunkStart << std::endl;
		//std::cout << "The trunkend is: " <<trunkEnd << std::endl;
		if (trunkStart == -1 && trunkEnd == -1 && (int)pixel == 2)
		{
			trunkStart = i;
			trunkEnd = i;
		}
		else if(trunkStart != -1 && pixel == 2){
			trunkEnd = i;
		}
	}
	int DMG = abs(trunkEnd - trunkStart);
	return DMG;
}

float trunkStraightness_new(std::pair <float,int> MgandRow, int DMG)
{
	float MG = MgandRow.first;
	float straightness = 1 - MG/DMG;
	return straightness;
}
	}
}

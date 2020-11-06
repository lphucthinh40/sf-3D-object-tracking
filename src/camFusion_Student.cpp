#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 1, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 1, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "Result";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
bool cmpMatch(const cv::DMatch &m1, const cv::DMatch &m2)
{
    return m1.distance < m2.distance;
}

void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches, cv::Mat* visImg)
{   
    // double sumDistance = 0.;
    std::vector<cv::DMatch> tempKptMatches;
    std::vector<double> distance;

    const float SINK_FACTOR = 0.15;
    const float MATCH_PERCENT = 0.5;
    const float UPPER_DISTANCE_LIMIT = 1.4;

    cv::Rect smallerROI;
    smallerROI.x = boundingBox.roi.x + SINK_FACTOR * boundingBox.roi.width / 2.0;
    smallerROI.y = boundingBox.roi.y + SINK_FACTOR * boundingBox.roi.height / 2.0;
    smallerROI.width = boundingBox.roi.width * (1 - SINK_FACTOR);
    smallerROI.height = boundingBox.roi.height * (1 - SINK_FACTOR);
    
    // find matches that belongs to the current bbox
    for ( cv::DMatch match : kptMatches ) {
        if (smallerROI.contains(kptsCurr[match.trainIdx].pt)) {
            tempKptMatches.push_back(match);
        }
    }

    // sort based on DMatch distance
    sort(tempKptMatches.begin(), tempKptMatches.end(), cmpMatch);

    // percentage of matches to be kept (keeping only good matches)
    int numGoodMatches = tempKptMatches.size() * MATCH_PERCENT;
    // cout << "nGoodMatch: " << numGoodMatches << endl;

    // find distance from current keypoint to itself on the previous frame
    for ( cv::DMatch match : tempKptMatches ){
        double d = cv::norm(kptsCurr[match.trainIdx].pt - kptsPrev[match.queryIdx].pt);
        distance.push_back(d);
    }
    // Using median
    // std::sort(distance.begin(), distance.end());
    // int mid = std::floor(distance.size()/2.0);
    // double median = (distance.size()%2==0)? (distance[mid]+distance[mid-1])/2.0 : distance[mid];

    // Using mean
    double mean = std::accumulate(distance.begin(), distance.end(), 0.0) / distance.size();
    double limit = mean * UPPER_DISTANCE_LIMIT; 

    // collect qualified matches for this bbox
    for ( int i=0; i<numGoodMatches; ++i ){
        if (distance[i]<=limit) {
            boundingBox.kptMatches.push_back(tempKptMatches[i]);
        }
    }
    // std::cout << "kptMatch size: " << boundingBox.kptMatches.size() << std::endl;
    if (visImg != nullptr) {
        cv::RNG rng(boundingBox.boxID);
        // cv::Scalar color = cv::Scalar(rng.uniform(10,200), rng.uniform(10, 200), rng.uniform(10, 200));
        cv::Scalar color = cv::Scalar(0,255,0);
        cv::Rect currROI = boundingBox.roi;
        cv::rectangle(*visImg, cv::Point(currROI.x, currROI.y), cv::Point(currROI.x + currROI.width, currROI.y + currROI.height), color, 2);

        for (cv::DMatch match : boundingBox.kptMatches)
        {   
            cv::circle(*visImg, kptsCurr[match.trainIdx].pt, 1, color, -1);
        }

        string windowName = "Result";
        cv::namedWindow(windowName, 4);
        cv::imshow(windowName, *visImg);
        cout << "Press key to continue" << endl;
        cv::waitKey(0);
    }

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{   const double MAX_DISTANCE = 120.0;
    const double MIN_DISTANCE = 80.0;

    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    int n_unchanged = 0;
    
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    {   // outer kpt. loop
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        {  // inner kpt.-loop
            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
            // std::cout << "Curr: " << distCurr << " | Prev: " << distPrev << std::endl;
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= MIN_DISTANCE && distCurr <= MAX_DISTANCE)
            { // avoid division by zero
                if (visImg != nullptr){
                    cv::line(*visImg, kpOuterCurr.pt, kpInnerCurr.pt, cv::Scalar(0, 255, 0));
                }
                double distRatio = distCurr / distPrev;
                if (distRatio <= 1.0)
                    n_unchanged++;
                else
                distRatios.push_back(distRatio); // collect only distance that has been increased
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = (distRatios.size() % 2 == 0) ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence
    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    // std::cout << "TTC: " << TTC << std::endl;

    if (visImg != nullptr){
        string windowName = "Result";
        cv::namedWindow(windowName, 4);
        cv::imshow(windowName, *visImg);
        cout << "Press key to continue" << endl;
        cv::waitKey(0);
    }
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{   
    std::vector<double> prevLidarX;
    std::vector<double> currLidarX;
    for (LidarPoint lp: lidarPointsPrev)
        prevLidarX.push_back(lp.x);
    for (LidarPoint lp: lidarPointsCurr)
        currLidarX.push_back(lp.x);

    std::sort(prevLidarX.begin(), prevLidarX.end());
    std::sort(currLidarX.begin(), currLidarX.end());

    int max_sample = std::min(currLidarX.size(), prevLidarX.size());
    
    // int n_sample = 0.05 * currLidarX.size();
    // n_sample = (n_sample == 0)? 1 : n_sample;
    const int n_sample = 1; // number of sample to calculate TTC

    double prev_minX = 0, curr_minX = 0;
    for (int i=0; i<n_sample; ++i)
    {   prev_minX += prevLidarX[i];
        curr_minX += currLidarX[i];
    }
    prev_minX /= static_cast<double>(n_sample);
    curr_minX /= static_cast<double>(n_sample);

    // std::cout << curr_minX << " | " << prev_minX << std::endl;
    double dT = 1 / frameRate;
    TTC = curr_minX * dT / (prev_minX - curr_minX);
}

void mapKeypoint2ROI(std::vector<cv::KeyPoint> &keypoints, std::vector<BoundingBox> &boundingBoxes, std::map<int,int> &kpt2box)
{   
    // map keypoint to bounding box, keypoint will be assigned to the first bbox that contains it
    // this assume bboxes are sorted in descending order of confidence
    for (int i=0; i<keypoints.size(); ++i)
    {  
        vector<int> boxIndexes; // indexes of all bounding boxes which enclose the current Lidar point
        for (int j = 0; j < boundingBoxes.size(); ++j)
        {   if (boundingBoxes[j].roi.contains(keypoints[i].pt))
                boxIndexes.push_back(j);
        }
        if (boxIndexes.size()==1) {
            boundingBoxes[boxIndexes[0]].keypoints.push_back(keypoints[i]);
            kpt2box[i] = boxIndexes[0];
        }
    }
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame, bool bVis)
{   
    int height = currFrame.boundingBoxes.size();
    int width = prevFrame.boundingBoxes.size();
    vector<vector<int>> matchList(height, vector<int>(width, 0));

    // count number of matches for each current bbox and prev bbox
    for (cv::DMatch match : matches)
    {
        int currBoxIdx = currFrame.keypoint2Box[match.trainIdx];
        int prevBoxIdx = prevFrame.keypoint2Box[match.queryIdx]; 
        matchList[currBoxIdx][prevBoxIdx]++;
    }

    // for each bounding box in current frame, find the best matched bounding box in prev frame
    int bestBoxIdx;
    vector<bool> bPicked(width, false);
    for (int i=0; i<height; i++) 
    {
        bestBoxIdx = std::distance(matchList[i].begin(), std::max_element(matchList[i].begin(), matchList[i].end()));
        // cout << "match boxes (" << i << "," << bestBoxIdx << ") count: " << matchList[i][bestBoxIdx] << endl;
        if (matchList[i][bestBoxIdx]>0 && bPicked[bestBoxIdx]==false) {
            bbBestMatches[prevFrame.boundingBoxes[bestBoxIdx].boxID] = currFrame.boundingBoxes[i].boxID;
            bPicked[bestBoxIdx] = true;
        }
    }

    // display matches bounding boxes
    if (bVis) {
        cv::Mat visCurrImg = currFrame.cameraImg.clone();
        cv::Mat visPrevImg = prevFrame.cameraImg.clone();

        std::map<int, int>::iterator it;
        int n = 0;
        for (it = bbBestMatches.begin(); it != bbBestMatches.end(); ++it)
        {   cv::RNG rng(currFrame.boundingBoxes[it->second].boxID);
            cv::Scalar color = cv::Scalar(rng.uniform(10,200), rng.uniform(10, 200), rng.uniform(10, 200));
            cv::Rect currROI = currFrame.boundingBoxes[it->second].roi;
            cv::Rect prevROI = prevFrame.boundingBoxes[it->first].roi;
            cv::rectangle(visCurrImg, cv::Point(currROI.x, currROI.y), cv::Point(currROI.x + currROI.width, currROI.y + currROI.height), color, 2);
            cv::rectangle(visPrevImg, cv::Point(prevROI.x, prevROI.y), cv::Point(prevROI.x + prevROI.width, prevROI.y + prevROI.height), color, 2);
        }

        cv::Mat combinedImg;
        cv::vconcat(visPrevImg, visCurrImg, combinedImg);
        string windowName = "Result";
        cv::namedWindow(windowName, 4);
        cv::imshow(windowName, combinedImg);
        cout << "Press key to continue" << endl;
        cv::waitKey(0);
    }
}

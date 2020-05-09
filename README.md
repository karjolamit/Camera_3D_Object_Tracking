# Camera_3D_Object_Tracking (Collision Avoidance System- CAS)
## MP.1 Match 3D Objects
Implemented the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides an output of the matched regions of interest IDs (i.e. the boxID property). The matches are the ones with the highest number of keypoint correspondences. Use of ``` std::multimap<int,int> ``` is done to track the pair of bounding box IDs. Further, keypoints per box are determined to find out the best matches between the frames. Following code represents the method explained:

```
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::multimap<int, int> mmap {};
    int maxPrevBoxID = 0;
    for (auto match : matches) {
        cv::KeyPoint prevKp = prevFrame.keypoints[match.queryIdx];
        cv::KeyPoint currKp = currFrame.keypoints[match.trainIdx];
        int prevBoxID = -1;
        int currBoxID = -1;
        for (auto bbox : prevFrame.boundingBoxes) {
            if (bbox.roi.contains(prevKp.pt)) prevBoxID = bbox.boxID;
        }
        for (auto bbox : currFrame.boundingBoxes) {
            if (bbox.roi.contains(currKp.pt)) currBoxID = bbox.boxID;
        }
        mmap.insert({currBoxID, prevBoxID});
        maxPrevBoxID = std::max(maxPrevBoxID, prevBoxID);
    }
    vector<int> currFrameBoxIDs {};
    for (auto box : currFrame.boundingBoxes) currFrameBoxIDs.push_back(box.boxID);
      for (int k : currFrameBoxIDs) {
        auto rangePrevBoxIDs = mmap.equal_range(k);
        std::vector<int> counts(maxPrevBoxID + 1, 0);
        for (auto it = rangePrevBoxIDs.first; it != rangePrevBoxIDs.second; ++it) {
            if (-1 != (*it).second) counts[(*it).second] += 1;
        }
        int modeIndex = std::distance(counts.begin(), std::max_element(counts.begin(), counts.end()));
        bbBestMatches.insert({modeIndex, k});
    }
}
```

## MP.2 Compute Lidar-based TTC
Computed the time-to-collision (TTC) in seconds for all matched 3D objects using only ```Lidar measurements``` from the matched bounding boxes between current and previous frame. Following images show the Lidar measurements and equations used to calculate TTC.

![Lidar_TTC_Image](https://github.com/karjolamit/Camera_3D_Object_Tracking/blob/master/Lidar_TTC_Image.png)

   ![Lidar_TTC_Equations](https://github.com/karjolamit/Camera_3D_Object_Tracking/blob/master/Lidar_TTC_Equations.png)

Figure Reference: Udacity

For calculating the TTC, it is necessary to find the distance to the closest Lidar point in the path of driving (ego lane). Also, to reduce the impact of erroneous points (outliers), minimum distance point in direction of driving is calculated using ``` sortLidarPointsX ``` function which outputs the lidar points in increasing order. Further, using constant velocity model, TTC is computed as follows:

``` TTC = (1.0/framerate)*currXMean/(prevXMean - currXMean) ```

Following code represents TTC calculation:

```
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    sortLidarPointsX(lidarPointsPrev);
    sortLidarPointsX(lidarPointsCurr);
    double d0 = lidarPointsPrev[lidarPointsPrev.size()/2].x;
    double d1 = lidarPointsCurr[lidarPointsCurr.size()/2].x;
    TTC = d1 * (1.0 / frameRate) / (d0 - d1);
}

```

## MP.3 Associate Keypoint Correspondences with Bounding Boxes
Prepared the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition are added to a vector in the respective bounding box. This is achieved by function ``` clusterKptMatchesWithROI ```, that loops through every matched keypoint pair in an image and determines all the keypoints within one bounding box. Further it checks if these points lie in the current frame Region of interest and then associate it with the corresponding bounding box in the current frame. Following code represents the evaluation:

```
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    for (cv::DMatch match : kptMatches) {
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt)) {
            boundingBox.kptMatches.push_back(match);
        }
    }
}

```
## MP.4 Compute Camera-based TTC
Computed the time-to-collision (TTC) in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame. This is implemented in the function ``` computeTTCCamera ```. Following images show the Camera measurements and equations used to calculate TTC.

![Camera_TTC_Image](https://github.com/karjolamit/Camera_3D_Object_Tracking/blob/master/Camera_TTC_Image.png)

![Camera_TTC_Equations](https://github.com/karjolamit/Camera_3D_Object_Tracking/blob/master/Camera_TTC_Equations.png) 

Figure Reference: Udacity

Following code shows implementation. The distance ratios on keypoints matched between frames is used to find the rate of scale change within an image. This rate of scale change can be used to estimate the TTC using equation, ``` TTC = (-1.0 / frameRate) / (1 - medianDistRatio) ```

```
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    vector<double> distRatios;
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) {
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx); 
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx); 
        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) {
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx); 
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);  
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
            double minDist = 100.0;  
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }
    if (distRatios.size() == 0)
    {
        TTC = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    std::sort(distRatios.begin(), distRatios.end());
    double medianDistRatio = distRatios[distRatios.size() / 2];
    TTC = (-1.0 / frameRate) / (1 - medianDistRatio);
}

```

## MP.5 Performance Evaluation 1
## MP.6 Performance Evaluation 2

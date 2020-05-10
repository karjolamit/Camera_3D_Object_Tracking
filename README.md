# Track an Object in 3D Space (Collision Avoidance System- CAS)
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
   
<p align="center">
  <img width="460" height="300" src="https://github.com/karjolamit/Camera_3D_Object_Tracking/blob/master/Lidar_TTC_Image.png">
</p>

<p align="center">
  <img width="460" height="300" src="https://github.com/karjolamit/Camera_3D_Object_Tracking/blob/master/Lidar_TTC_Equations.png">
</p>

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

<p align="center">
  <img width="460" height="300" src="https://github.com/karjolamit/Camera_3D_Object_Tracking/blob/master/Camera_TTC_Image.png">
</p>

<p align="center">
  <img width="460" height="300" src="https://github.com/karjolamit/Camera_3D_Object_Tracking/blob/master/Camera_TTC_Equations.png">
</p>

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
Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.

<p align="center">
  <img width="460" height="300" src="https://github.com/karjolamit/Camera_3D_Object_Tracking/blob/master/MP_5.PNG">
</p>

From the above figure, it is clearly seen that all the vehicles are approaching an intersection with Red traffic light ON, meaning all vehciles will STOP. Also, the tail lights of all preceeding vehicle justify this fact. As per this observation, the TTC for Lidar must decrease than its previous instance in the frame. However, observing the values in following tables (MP.6), it can be inferred that there are some issues with the lidar measurements at certain instances. These implausible outcomes may be due to the presence of additional points (outliers) in the preceeding vehicle bounding box. The most out of the way measurement is the value > 16 seconds. This problem can be resolved by tuning variables like shrink factor and maxKeyPoints.

## MP.6 Performance Evaluation 2
Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

Based on the system response for different detector-descriptor configurations, following are the top 3 estimates:

1. FAST + BRIEF: Based on Time efficiency

| Detector Type | Descriptor Type | Lidar TTC | Camera TTC |
| ------------- | --------------- | --------- | ---------- |
| FAST | BRIEF | 12.515600 | 11.750258 |
| FAST | BRIEF | 12.614245 | 11.758331 |
| FAST | BRIEF | 14.091013 | 13.955042 |
| FAST | BRIEF | 16.689386 | 13.130921 |
| FAST | BRIEF | 15.008233 | 14.829752 |
| FAST | BRIEF | 12.678716 | 13.594156 |
| FAST | BRIEF | 14.984351 | 13.277444 |
| FAST | BRIEF | 13.124118 | 12.774939 |
| FAST | BRIEF | 13.024118 | 12.800778 |
| FAST | BRIEF | 11.174641 | 13.009249 |
| FAST | BRIEF | 12.808601 | 11.870213 |
| FAST | BRIEF | 8.959780 | 11.525555 |
| FAST | BRIEF | 9.964399 | 11.706558 |
| FAST | BRIEF | 9.598630 | 10.645161 |
| FAST | BRIEF | 8.573525 | 11.531169 |
| FAST | BRIEF | 9.516170 | 10.323035 |
| FAST | BRIEF | 9.546581 | 9.894525 |
| FAST | BRIEF | 8.398803 | 11.508075 |

Graph:

<p align="center">
  <img width="460" height="300" src="https://github.com/karjolamit/Camera_3D_Object_Tracking/blob/master/TTC_1.png">
</p>

The averagre TTC is ~12 seconds for both; Lidar and Camera.  

2. BRISK + BRIEF: Based on measurement accuracy efficiency

| Detector Type | Descriptor Type | Lidar TTC | Camera TTC |
| ------------- | --------------- | --------- | ---------- |
| BRISK | BRIEF | 12.515600 | 12.844751 | 
| BRISK | BRIEF | 12.614245 | 16.861843 |
| BRISK | BRIEF | 14.091013 | 11.667223 |
| BRISK | BRIEF | 16.689386 | 20.572493 |
| BRISK | BRIEF | 15.908233 | 19.705172 |
| BRISK | BRIEF | 12.678716 | 17.564209 |
| BRISK | BRIEF | 11.984351 | 15.580851 |
| BRISK | BRIEF | 13.124118 | 18.745060 |
| BRISK | BRIEF | 13.024118 | 15.458500 |
| BRISK | BRIEF | 11.174641 | 11.480950 |
| BRISK | BRIEF | 12.808601 | 13.363625 |
| BRISK | BRIEF | 8.959780 | 14.307072 |
| BRISK | BRIEF | 9.964390 | 12.326637 |
| BRISK | BRIEF | 9.598630 | 10.945685 |
| BRISK | BRIEF | 8.573525 | 11.628937 |
| BRISK | BRIEF | 9.516170 | 12.173885 |
| BRISK | BRIEF | 9.546581 | 11.379207 |
| BRISK | BRIEF | 8.398803 | 10.702170 |

Graph:

<p align="center">
  <img width="460" height="300" src="https://github.com/karjolamit/Camera_3D_Object_Tracking/blob/master/TTC_2.PNG">
</p>

The averagre TTC is ~12 seconds and ~14 seconds for Lidar and Camera respectively.

3. BRISK + BRISK: Based on combined efficiency (time and measurement accuracy)

| Detector Type | Descriptor Type | Lidar TTC | Camera TTC |
| ------------- | --------------- | --------- | ---------- |
| BRISK | BRISK | 12.515600 | 13.408608 | 
| BRISK | BRISK | 12.614245 | 21.527645 |
| BRISK | BRISK | 14.091013 | 12.625001 |
| BRISK | BRISK | 16.689386 | 15.203748 |
| BRISK | BRISK | 15.908233 | 27.667923 |
| BRISK | BRISK | 12.678716 | 18.311445 |
| BRISK | BRISK | 11.984351 | 17.142694 |
| BRISK | BRISK | 13.124118 | 16.099065 |
| BRISK | BRISK | 13.024118 | 14.801027 |
| BRISK | BRISK | 11.174641 | 13.929688 |
| BRISK | BRISK | 12.808601 | 13.135876 |
| BRISK | BRISK | 8.959780 | 11.340908 |
| BRISK | BRISK | 9.964390 | 11.856502 |
| BRISK | BRISK | 9.598630 | 12.397777 |
| BRISK | BRISK | 8.573525 | 12.708913 |
| BRISK | BRISK | 9.516170 | 11.503045 |
| BRISK | BRISK | 9.546581 | 9.293792 |
| BRISK | BRISK | 8.398803 | 10.775890 |

Graph:

<p align="center">
  <img width="460" height="300" src="https://github.com/karjolamit/Camera_3D_Object_Tracking/blob/master/TTC_3.PNG">
</p>

The averagre TTC is ~12 seconds and ~15 seconds for Lidar and Camera respectively.

From above tables, Camera TTC for BRISK+BRISK are as high as 21.527645 and 27.667923 & for BRISK+BRIEF, highest value estimated is 20.572493. The reason for this may be same as explained above for Lidar TTC. Additionally, the plots show the trend of how both the TTC varies throughout all 18 instances of the image frame.  

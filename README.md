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


## MP.3 Associate Keypoint Correspondences with Bounding Boxes
## MP.4 Compute Camera-based TTC
## MP.5 Performance Evaluation 1
## MP.6 Performance Evaluation 2

#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorCategory, std::string matcherType, std::string selectorType)
{
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    // Brute force calculation
    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType;

        // for SIFT method
        if (descriptorCategory.compare("DES_HOG") == 0)
        {
            normType = cv::NORM_L2;
        }

        // for all other binary descriptors
        else if (descriptorCategory.compare("DES_BINARY") == 0)
        {
            normType = cv::NORM_HAMMING;
        }
        
        else {
            throw invalid_argument(descriptorCategory + " is not a valid descriptorCategory");
        }

        matcher = cv::BFMatcher::create(normType, crossCheck);
    }

    // FLANN matching method
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // for SIFT method
        if (descriptorCategory.compare("DES_HOG") == 0)
        {
            matcher = cv::FlannBasedMatcher::create();
        }

        // for all other binary descriptorTypes
        else if (descriptorCategory.compare("DES_BINARY") == 0)
        {
            const cv::Ptr<cv::flann::IndexParams>& indexParams = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
            matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams);
        }

        else {
            throw invalid_argument(descriptorCategory + " is not a valid descriptorCategory");
        }
    }

    else {
        throw invalid_argument(matcherType + " is not a valid matcherType");
    }

    // Nearest neighbor matching (best match) selector
    if (selectorType.compare("SEL_NN") == 0)
    {
        matcher->match(descSource, descRef, matches);
    }
    
    // k nearest neighbors (k=2) selector
    else if (selectorType.compare("SEL_KNN") == 0)
    {
        int k = 2;
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, k);
        
        // Filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it : knn_matches) {
            // The returned knn_matches vector contains few nested vectors with size less than 2
            if ( 2 == it.size() && (it[0].distance < minDescDistRatio * it[1].distance) ) {
                matches.push_back(it[0]);
            }
        }
    }

    else {
        throw invalid_argument(selectorType + " is not a valid selectorType");
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // Select appropriate descriptor, using default values for now
    cv::Ptr<cv::DescriptorExtractor> extractor;
    
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        // when specified descriptorType is unsupported
        throw invalid_argument(descriptorType + " is not a valid descriptorType");
    }

    extractor->compute(img, keypoints, descriptors);
}

// Shi-Thomasi keypoint detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  average blocksize for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max no of keypoints

    double qualityLevel = 0.01; 
    double k = 0.04;

    // Applying corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in an image by adapting the Harris detector developed in a previous exercise
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // blockSize neighborhood 
    int apertureSize = 3;  // Aperture parameter for the Sobel operator
    assert(1 == apertureSize % 2);  // Aperture size check to be odd
    int minResponse = 100; // Minimum value for a corner in the scaled response matrix
    double k = 0.04;       // Harris parameter equation

    // Non-maximum suppression (NMS) settings
    double maxOverlap = 0.0;  // Maximum overlap between two features in %

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    if (bVis)
    {
        // Visualize the results
        string windowName = "Harris Corner Detector Response Matrix";
        cv::namedWindow(windowName);
        cv::imshow(windowName, dst_norm_scaled);
        cv::waitKey(0);
    }

    // Apply non-maximum suppression (NMS)
    for (size_t j = 0; j < dst_norm.rows; j++) {
        for (size_t i = 0; i < dst_norm.cols; i++) {
            int response = (int)dst_norm.at<float>(j, i);

            // Apply the minimum threshold for Harris cornerness response
            if (response < minResponse) continue;

            // Otherwise create a tentative new keypoint
            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt = cv::Point2f(i, j);
            newKeyPoint.size = 2 * apertureSize;
            newKeyPoint.response = response;

            // Performing non-maximum suppression (NMS) in local neighbourhood around the new keypoint
            bool bOverlap = false;
            // Loop over all existing keypoints
            for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
                double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                // Test if overlap exceeds the maximum permissible percentage
                if (kptOverlap > maxOverlap) {
                    bOverlap = true;
                    // If overlapping, test if new response is the local maximum
                    if (newKeyPoint.response > (*it).response) {
                        *it = newKeyPoint;  // Replace the old keypoint
                        break;  // Exit for loop
                    }
                }
            }

            // If above response threshold and not overlapping any other keypoint
            if (!bOverlap) {
                keypoints.push_back(newKeyPoint);  // Add to keypoints list
            }
        }
    }

    if (bVis)
    {
        // Visualize the keypoints
        string windowName = "Harris corner detection results";
        cv::namedWindow(windowName);
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in an image using more recent methods available in OpenCV
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    if (detectorType.compare("FAST") == 0) {
        auto fast = cv::FastFeatureDetector::create();
        fast->detect(img, keypoints);
    }
    else if (detectorType.compare("BRISK") == 0) {
        auto brisk = cv::BRISK::create();
        brisk->detect(img, keypoints);
    }
    else if (detectorType.compare("ORB") == 0) {
        auto orb = cv::ORB::create();
        orb->detect(img, keypoints);
    }
    else if (detectorType.compare("AKAZE") == 0) {
        auto akaze = cv::AKAZE::create();
        akaze->detect(img, keypoints);
    }
    else if (detectorType.compare("SIFT") == 0) {
        auto sift = cv::xfeatures2d::SIFT::create();
        sift->detect(img, keypoints);
    }
    else {
        // when specified detectorType is unsupported
        throw invalid_argument(detectorType + " is not a valid detectorType");
    }

    if (bVis)
    {
        // Visualize the keypoints
        string windowName = detectorType + " keypoint detection results";
        cv::namedWindow(windowName);
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

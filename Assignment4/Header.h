#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <cmath>

using namespace cv;
using namespace std;
class Assignment {
public:

	vector<Mat> stabilizationMatrices;
	Assignment() {};
	Mat core1(Mat img1, Mat img2);
	Mat core2(Mat img1, Mat img2);
	void core3(Mat img1, Mat img2);
	void compleation();
	void videoStabilization(vector<Mat>& frames);
	void generate1DGaussian(double mean, double stddev, int size, vector<double>& gaussian);
	vector<Mat> loadImages(int numFrames, const string& prefix);
	void exportImages(const vector<Mat>& images, const string& prefix, const string& outputFolder);
	Mat estimateHomography(const std::vector<KeyPoint>& keypoints_1, const std::vector<KeyPoint>& keypoints_2,
	const std::vector<DMatch>& matches, double epsilon, int numIterations);
	void minimumCrop(vector<Mat>& frames);
	void detectKeypoints(Mat img1, Mat img2, vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2, Mat& descriptors_1, Mat& descriptors_2);
	void matchFeatures(Mat descriptors_1, Mat descriptors_2, vector<DMatch>& matches);
	void filterMatches(vector<DMatch>& matches, vector<DMatch>& good_matches);
	void drawKeypoints(Mat img1, Mat img2, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, Mat& img_keypoints_1, Mat& img_keypoints_2);
	void drawMatches(Mat img_keypoints_1, Mat img_keypoints_2, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<DMatch> good_matches, Mat& img_matches);
	void generateCumulativeMatrices(vector<Mat>& frames, vector<Mat>& cumulativeMatrices);
};
//include open cv
#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "Header.h"
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;


int main()
{
	Assignment A = Assignment();
	Mat img1 = imread("C:/Users/kahum/source/repos/Assignment4/Assignment4/res/Frame039.jpg");
	Mat img2 = imread("C:/Users/kahum/source/repos/Assignment4/Assignment4/res/Frame041.jpg");

	A.core1(img1, img2);
	waitKey(0);
	A.core2(img1, img2);
	waitKey(0);
	A.core3(img1, img2);
	waitKey(0);
	A.compleation();
	cout << "done" << endl;
	waitKey(0);
}

void Assignment::detectKeypoints(Mat img1, Mat img2, vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2, Mat& descriptors_1, Mat& descriptors_2) {
	Ptr<SIFT> sift = SIFT::create();
	sift->detectAndCompute(img1, noArray(), keypoints_1, descriptors_1);
	sift->detectAndCompute(img2, noArray(), keypoints_2, descriptors_2);
}

// Perform feature matching
void Assignment::matchFeatures(Mat descriptors_1, Mat descriptors_2, vector<DMatch>& matches) {
	BFMatcher matcher(NORM_L2, true);
	matcher.match(descriptors_1, descriptors_2, matches);
}

// Filter the matches based on distance
void Assignment::filterMatches(vector<DMatch>& matches, vector<DMatch>& good_matches) {
	double max_dist = 400;
	for (int i = 0; i < matches.size(); i++) {
		if (matches[i].distance < max_dist) {
			good_matches.push_back(matches[i]);
		}
	}
}

// Draw keypoints on the images
void drawKeypoints(Mat img1, Mat img2, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, Mat& img_keypoints_1, Mat& img_keypoints_2) {
	drawKeypoints(img1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
}

// Draw lines between matched keypoints
void Assignment::drawMatches(Mat img_keypoints_1, Mat img_keypoints_2, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<DMatch> good_matches, Mat& img_matches) {
	vconcat(img_keypoints_1, img_keypoints_2, img_matches);

	vector<Point2f> matchedPoints1, matchedPoints2;
	for (int i = 0; i < good_matches.size(); i++) {
		Point2f pt1 = keypoints_1[good_matches[i].queryIdx].pt;
		Point2f pt2 = keypoints_2[good_matches[i].trainIdx].pt;
		matchedPoints1.push_back(pt1);
		matchedPoints2.push_back(pt2);

		line(img_matches, pt1, Point2f(pt2.x, pt2.y + img_keypoints_1.rows), Scalar(0, 255, 0), 1);
	}
}

// Find the homography matrix
Mat findHomography(vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<DMatch> good_matches) {
	vector<Point2f> matchedPoints1, matchedPoints2;
	for (int i = 0; i < good_matches.size(); i++) {
		Point2f pt1 = keypoints_1[good_matches[i].queryIdx].pt;
		Point2f pt2 = keypoints_2[good_matches[i].trainIdx].pt;
		matchedPoints1.push_back(pt1);
		matchedPoints2.push_back(pt2);
	}

	Mat homography = findHomography(matchedPoints1, matchedPoints2, RANSAC);

	return homography;
}

// Display the result
void displayResult(Mat img_matches) {
	namedWindow("Matches", WINDOW_NORMAL);
	imshow("Matches", img_matches);
}

Mat Assignment::core1(Mat img1, Mat img2)
{
	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;

	detectKeypoints(img1, img2, keypoints_1, keypoints_2, descriptors_1, descriptors_2);

	vector<DMatch> matches;
	matchFeatures(descriptors_1, descriptors_2, matches);

	vector<DMatch> good_matches;
	filterMatches(matches, good_matches);

	Mat img_keypoints_1, img_keypoints_2;
	drawKeypoints(img1, img2, keypoints_1, keypoints_2, img_keypoints_1, img_keypoints_2);

	Mat img_matches;
	drawMatches(img_keypoints_1, img_keypoints_2, keypoints_1, keypoints_2, good_matches, img_matches);

	Mat homography = findHomography(keypoints_1, keypoints_2, good_matches);

	displayResult(img_matches);

	return homography;
}

Mat Assignment::core2(Mat img1, Mat img2) {

	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;

	detectKeypoints(img1, img2, keypoints_1, keypoints_2, descriptors_1, descriptors_2);

	vector<DMatch> matches;
	matchFeatures(descriptors_1, descriptors_2, matches);

	double epsilon = 3.0; // Adjust this threshold as needed
	int numIterations = 100;

	Mat homography = estimateHomography(keypoints_1, keypoints_2, matches, epsilon, numIterations);

	Mat concatenatedImg;
	vconcat(img1, img2, concatenatedImg);


	drawInliersOutliers(concatenatedImg, keypoints_1, keypoints_2, matches, homography, epsilon);

	displayResult(concatenatedImg);

	return homography;
}

// Draw lines between inlier pairs and outlier pairs
void drawInliersOutliers(Mat concatenatedImg, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<DMatch> matches, Mat homography, double epsilon) {
	for (const DMatch& match : matches) {
		Point2f pt1 = keypoints_1[match.queryIdx].pt;
		Point2f pt2 = keypoints_2[match.trainIdx].pt;

		Point2f pt1Concatenated(pt1.x, pt1.y);
		Point2f pt2Concatenated(pt2.x, pt2.y + concatenatedImg.rows);

		Mat pt1Homogeneous = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
		Mat tranformedPt1Homogeneous = homography * pt1Homogeneous;
		Point2f transformedPt1(tranformedPt1Homogeneous.at<double>(0, 0) / tranformedPt1Homogeneous.at<double>(2, 0),
			tranformedPt1Homogeneous.at<double>(1, 0) / tranformedPt1Homogeneous.at<double>(2, 0));

		double error = norm(transformedPt1 - pt2);

		if (error < epsilon) {
			line(concatenatedImg, pt1Concatenated, pt2Concatenated, Scalar(0, 255, 0), 1);
		}
		else {
			line(concatenatedImg, pt1Concatenated, pt2Concatenated, Scalar(0, 0, 255), 1);
		}
	}
}


// Select four random feature pairs
void selectRandomPairs(const vector<KeyPoint>& keypoints_1, const vector<KeyPoint>& keypoints_2,
	const vector<DMatch>& matches, vector<Point2f>& srcPts, vector<Point2f>& dstPts, vector<DMatch>& randomMatches) {
	for (int i = 0; i < 4; i++) {
		int randomIdx = theRNG().uniform(0, matches.size());
		randomMatches.push_back(matches[randomIdx]);

		srcPts.push_back(keypoints_1[randomMatches[i].queryIdx].pt);
		dstPts.push_back(keypoints_2[randomMatches[i].trainIdx].pt);
	}
}

// Count the number of inliers
int countInliers(const vector<KeyPoint>& keypoints_1, const vector<KeyPoint>& keypoints_2,
	const vector<DMatch>& matches, Mat homography, double epsilon) {
	int numInliers = 0;

	for (const DMatch& match : matches) {
		Point2f srcPt = keypoints_1[match.queryIdx].pt;
		Point2f dstPt = keypoints_2[match.trainIdx].pt;

		Mat srcPtHomogeneous = (Mat_<double>(3, 1) << srcPt.x, srcPt.y, 1.0);
		Mat transformedPtHomogeneous = homography * srcPtHomogeneous;
		Point2f transformedPt(transformedPtHomogeneous.at<double>(0, 0) / transformedPtHomogeneous.at<double>(2, 0),
			transformedPtHomogeneous.at<double>(1, 0) / transformedPtHomogeneous.at<double>(2, 0));

		double error = norm(transformedPt - dstPt);

		if (error < epsilon) {
			numInliers++;
		}
	}

	return numInliers;
}

// Find the inlier feature pairs
void findInliers(const vector<KeyPoint>& keypoints_1, const vector<KeyPoint>& keypoints_2,
	const vector<DMatch>& matches, Mat bestHomography, double epsilon, vector<Point2f>& inlierSrcPts, vector<Point2f>& inlierDstPts) {
	for (const DMatch& match : matches) {
		Point2f srcPt = keypoints_1[match.queryIdx].pt;
		Point2f dstPt = keypoints_2[match.trainIdx].pt;

		Mat srcPtHomogeneous = (Mat_<double>(3, 1) << srcPt.x, srcPt.y, 1.0);
		Mat transformedPtHomogeneous = bestHomography * srcPtHomogeneous;
		Point2f transformedPt(transformedPtHomogeneous.at<double>(0, 0) / transformedPtHomogeneous.at<double>(2, 0),
			transformedPtHomogeneous.at<double>(1, 0) / transformedPtHomogeneous.at<double>(2, 0));

		double error = norm(transformedPt - dstPt);

		if (error < epsilon) {
			inlierSrcPts.push_back(srcPt);
			inlierDstPts.push_back(dstPt);
		}
	}
}

Mat Assignment::estimateHomography(const vector<KeyPoint>& keypoints_1, const vector<KeyPoint>& keypoints_2,
	const vector<DMatch>& matches, double epsilon, int numIterations) {
	Mat bestHomography;
	int maxInliers = 0;

	for (int iteration = 0; iteration < numIterations; iteration++) {
		vector<Point2f> srcPts;
		vector<Point2f> dstPts;
		vector<DMatch> randomMatches;

		selectRandomPairs(keypoints_1, keypoints_2, matches, srcPts, dstPts, randomMatches);

		Mat homography = findHomography(srcPts, dstPts, 0);

		int numInliers = countInliers(keypoints_1, keypoints_2, matches, homography, epsilon);

		if (numInliers > maxInliers) {
			maxInliers = numInliers;
			bestHomography = homography;
		}
	}

	vector<Point2f> inlierSrcPts;
	vector<Point2f> inlierDstPts;

	findInliers(keypoints_1, keypoints_2, matches, bestHomography, epsilon, inlierSrcPts, inlierDstPts);

	bestHomography = findHomography(inlierSrcPts, inlierDstPts, RANSAC);

	return bestHomography;
}


void convertToGrayscale(Mat img1, Mat img2, Mat& gray1, Mat& gray2) {
	cvtColor(img1, gray1, COLOR_BGR2GRAY);
	cvtColor(img2, gray2, COLOR_BGR2GRAY);
}

// Detect keypoints and compute descriptors using ORB
void detectAndComputeORB(Mat gray1, Mat gray2, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, Mat& descriptors1, Mat& descriptors2) {
	Ptr<ORB> sift = ORB::create();
	sift->detectAndCompute(gray1, noArray(), keypoints1, descriptors1);
	sift->detectAndCompute(gray2, noArray(), keypoints2, descriptors2);
}

// Find the matched points using Brute-Force Matcher
void findMatchedPoints(Mat descriptors1, Mat descriptors2, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, vector<Point2f>& points1, vector<Point2f>& points2) {
	BFMatcher matcher(NORM_HAMMING);
	vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	for (size_t i = 0; i < matches.size(); i++)
	{
		points1.push_back(keypoints1[matches[i].queryIdx].pt);
		points2.push_back(keypoints2[matches[i].trainIdx].pt);
	}
}

// Add 100px padding to each image
void addPadding(Mat img1, Mat img2, Mat& padded1, Mat& padded2) {
	copyMakeBorder(img1, padded1, 100, 100, 100, 100, BORDER_CONSTANT, Scalar(0, 0, 0));
	copyMakeBorder(img2, padded2, 100, 100, 100, 100, BORDER_CONSTANT, Scalar(0, 0, 0));
}

// Warp the first image using the homography matrix
void warpImage(Mat padded1, Mat H, Mat& warped1) {
	warpPerspective(padded1, warped1, H, padded1.size());
}

// Add the two images together
void addImages(Mat warped1, Mat padded2, Mat& stitched) {
	stitched = Mat::zeros(warped1.size(), CV_8UC3);
	warped1.copyTo(stitched);
	for (int y = 0; y < padded2.rows; y++)
	{
		for (int x = 0; x < padded2.cols; x++)
		{
			if (padded2.at<Vec3b>(y, x) != Vec3b(0, 0, 0))
			{
				stitched.at<Vec3b>(y, x) = padded2.at<Vec3b>(y, x);
			}
		}
	}
}

void Assignment::core3(Mat img1, Mat img2) {

	Mat gray1, gray2;
	convertToGrayscale(img1, img2, gray1, gray2);

	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	detectAndComputeORB(gray1, gray2, keypoints1, keypoints2, descriptors1, descriptors2);

	vector<Point2f> points1, points2;
	findMatchedPoints(descriptors1, descriptors2, keypoints1, keypoints2, points1, points2);

	Mat padded1, padded2;
	addPadding(img1, img2, padded1, padded2);

	Mat H = findHomography(points1, points2, RANSAC);

	Mat warped1;
	warpImage(padded1, H, warped1);

	Mat stitched;
	addImages(warped1, padded2, stitched);

	displayResult(stitched);
}

void Assignment::compleation() {
	//load frames
	vector<Mat> images = loadImages(102, "C:/Users/kahum/source/repos/Assignment4/Assignment4/res/Frame");
	videoStabilization(images);
	exportImages(images, "Stable", "C:/Users/kahum/source/repos/Assignment4/Assignment4/outImgs");
	waitKey(0);

}

vector<Mat> Assignment::loadImages(int numFrames, const string& prefix) {
	vector<Mat> images;
	for (int i = 0; i < numFrames; i++) {
		ostringstream frameNumber;
		frameNumber << setfill('0') << setw(3) << i;
		string filename = prefix + frameNumber.str() + ".jpg";
		Mat image = imread(filename);
		images.push_back(image);
	}

	return images;
}


void Assignment::exportImages(const vector<Mat>& images, const string& prefix, const string& outputFolder) {
	for (int i = 0; i < images.size(); i++) {
		ostringstream frameNumber;
		frameNumber << setw(3) << setfill('0') << i;
		string filename = outputFolder + "/" + prefix + frameNumber.str() + ".png";
		imwrite(filename, images[i]);
	}
}


void Assignment::generate1DGaussian(double mean, double stddev, int size, vector<double>& gaussian)
{
	// Create the 1D Gaussian kernel
	gaussian.resize(size);
	Mat kernel = getGaussianKernel(size, stddev, CV_64F);
	for (int i = 0; i < size; ++i)
	{
		gaussian[i] = kernel.at<double>(i);
	}
}

void generateDifferences(vector<Mat>& frames, vector<Mat>& differences) {
	int numFrames = frames.size();
	differences.resize(numFrames - 1);
	for (int i = 0; i < numFrames - 1; ++i)
	{
		differences[i] = frames[i + 1] - frames[i];
	}
}

// Generate vector of cumulative matrices
void Assignment::generateCumulativeMatrices(vector<Mat>& frames, vector<Mat>& cumulativeMatrices) {
	int numFrames = frames.size();
	cumulativeMatrices.resize(numFrames);
	cumulativeMatrices[0] = Mat::eye(3, 3, CV_64F);
	for (int i = 1; i < numFrames; ++i)
	{
		Mat prevToCurrent;
		Mat transformation = core1(frames[i], frames[i - 1]);
		prevToCurrent = transformation;  // Update to store the transformation matrix
		cumulativeMatrices[i] = cumulativeMatrices[i - 1] * prevToCurrent;
	}
}

// Create vector of cumulative matrices with Gaussian smoothing
void smoothCumulativeMatrices(vector<Mat>& cumulativeMatrices, vector<double>& gaussian, int filterSize, vector<Mat>& smoothedMatrices) {
	int numFrames = cumulativeMatrices.size();
	smoothedMatrices.resize(numFrames);
	for (int i = 0; i < numFrames; ++i)
	{
		smoothedMatrices[i] = Mat::zeros(3, 3, CV_64F); //g transformation matrix
		for (int j = -filterSize / 2; j <= filterSize / 2; ++j)
		{
			int index = i + j;
			if (index >= 0 && index < numFrames)
			{
				double weight = gaussian[j + filterSize / 2];
				smoothedMatrices[i] += weight * cumulativeMatrices[index];
			}
		}
	}
}

// Create vector of stabilization matrices by getting movement and subtracting smooth
void generateStabilizationMatrices(vector<Mat>& smoothedMatrices, vector<Mat>& cumulativeMatrices, vector<Mat>& stabilizationMatrices) {
	int numFrames = smoothedMatrices.size();
	stabilizationMatrices.resize(numFrames);
	for (int i = 0; i < numFrames; ++i)
	{
		stabilizationMatrices[i] = smoothedMatrices[i].inv() * cumulativeMatrices[i];
	}
}

// Apply stabilization transforms to each frame
void applyStabilization(vector<Mat>& frames, vector<Mat>& stabilizationMatrices) {
	int numFrames = frames.size();
	for (int i = 0; i < numFrames; ++i)
	{
		Mat stabilizedFrame;
		warpPerspective(frames[i], stabilizedFrame, stabilizationMatrices[i], frames[i].size());
		frames[i] = stabilizedFrame;
	}
}

void Assignment::videoStabilization(vector<Mat>& frames)
{
	vector<Mat> differences;
	generateDifferences(frames, differences);

	vector<Mat> cumulativeMatrices;
	generateCumulativeMatrices(frames, cumulativeMatrices);

	double mean = 0.;
	double stddev = 5;
	int filterSize = 9;  // Window size for the Gaussian filter
	vector<double> gaussian;
	generate1DGaussian(mean, stddev, filterSize, gaussian);

	vector<Mat> smoothedMatrices;
	smoothCumulativeMatrices(cumulativeMatrices, gaussian, filterSize, smoothedMatrices);

	vector<Mat> stabilizationMatrices;
	generateStabilizationMatrices(smoothedMatrices, cumulativeMatrices, stabilizationMatrices);

	applyStabilization(frames, stabilizationMatrices);

	minimumCrop(frames);
}

void Assignment::minimumCrop(vector<Mat>& frames)
{
	int numFrames = frames.size();
	Mat cumulativeMask = Mat::ones(frames[0].size(), CV_8U);
	for (int i = 0; i < numFrames; ++i)
	{
		Mat mask = Mat::ones(frames[i].size(), CV_8U);
		warpPerspective(mask, mask, stabilizationMatrices[i], frames[i].size());
		cumulativeMask &= mask;
	}

	// Find the bounding box of the non-zero pixels in the cumulative mask
	vector<Point> points;
	findNonZero(cumulativeMask, points);
	Rect boundingBox = boundingRect(points);

	// Crop all frames to the bounding box
	for (int i = 0; i < numFrames; ++i)
	{
		frames[i] = frames[i](boundingBox);
	}
	
	//export
	exportImages(frames, "Crop", "C:/Users/kahum/source/repos/Assignment4/Assignment4/croped");
}




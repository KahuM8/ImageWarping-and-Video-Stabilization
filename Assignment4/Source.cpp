//include open cv
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "Header.h"
#include <iostream>


using namespace cv;
using namespace std;


int main()
{
	Assignment A = Assignment();
	A.core1();
	A.core2();
	A.core3();
}

void Assignment::core1()
{
	//read images
	Mat img1 = imread("C:/Users/kahum/source/repos/Assignment4/Assignment4/res/Frame039.jpg");
	Mat img2 = imread("C:/Users/kahum/source/repos/Assignment4/Assignment4/res/Frame041.jpg");

	// Create a SIFT object and detect keypoints
	Ptr<SIFT> sift = SIFT::create();
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;

	sift->detectAndCompute(img1, noArray(), keypoints_1, descriptors_1);
	sift->detectAndCompute(img2, noArray(), keypoints_2, descriptors_2);

	// Perform feature matching
	BFMatcher matcher(NORM_L2, true);
	std::vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	// Filter the matches based on distance
	double max_dist = 500;
	std::vector<DMatch> good_matches;

	for (int i = 0; i < matches.size(); i++) {
		if (matches[i].distance < max_dist) {
			good_matches.push_back(matches[i]);
		}
	}

	// Draw keypoints on the images
	Mat img_keypoints_1, img_keypoints_2;
	drawKeypoints(img1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	// Draw lines between matched keypoints
	Mat img_matches;
	vconcat(img_keypoints_1, img_keypoints_2, img_matches);

	for (int i = 0; i < good_matches.size(); i++) {
		Point2f pt1 = keypoints_1[good_matches[i].queryIdx].pt;
		Point2f pt2 = keypoints_2[good_matches[i].trainIdx].pt;

		line(img_matches, pt1, Point2f(pt2.x, pt2.y + img1.rows), Scalar(0, 255, 0), 1);
	}

	// Display the result
	namedWindow("Matches", WINDOW_NORMAL);
	imshow("Matches", img_matches);
	waitKey(0);

}


void Assignment::core2() {

	Mat img1 = imread("C:/Users/kahum/source/repos/Assignment4/Assignment4/res/Frame039.jpg");
	Mat img2 = imread("C:/Users/kahum/source/repos/Assignment4/Assignment4/res/Frame041.jpg");


	Ptr<SIFT> sift = SIFT::create();
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;

	sift->detectAndCompute(img1, noArray(), keypoints_1, descriptors_1);
	sift->detectAndCompute(img2, noArray(), keypoints_2, descriptors_2);

	// Perform feature matching
	BFMatcher matcher(NORM_L2, true);
	std::vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	// Set RANSAC parameters
	double epsilon = 3.0; // Adjust this threshold as needed
	int numIterations = 100;

	// Estimate homography using RANSAC
	Mat homography = estimateHomography(keypoints_1, keypoints_2, matches, epsilon, numIterations);

	// Concatenate the two images
	Mat concatenatedImg;
	vconcat(img1, img2, concatenatedImg);

	// Draw lines between inlier pairs and outlier pairs
	for (const DMatch& match : matches) {
		Point2f pt1 = keypoints_1[match.queryIdx].pt;
		Point2f pt2 = keypoints_2[match.trainIdx].pt;

		Point2f pt1Concatenated(pt1.x, pt1.y);
		Point2f pt2Concatenated(pt2.x, pt2.y + img1.rows);

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

	// Display the result
	imshow("Matches", concatenatedImg);
	waitKey(0);
}


Mat Assignment::estimateHomography(const std::vector<KeyPoint>& keypoints_1, const std::vector<KeyPoint>& keypoints_2,
	const std::vector<DMatch>& matches, double epsilon, int numIterations) {
	Mat bestHomography;
	int maxInliers = 0;

	for (int iteration = 0; iteration < numIterations; iteration++) {
		// Select four random feature pairs
		std::vector<Point2f> srcPts;
		std::vector<Point2f> dstPts;
		std::vector<DMatch> randomMatches;

		for (int i = 0; i < 4; i++) {
			int randomIdx = theRNG().uniform(0, matches.size());
			randomMatches.push_back(matches[randomIdx]);

			srcPts.push_back(keypoints_1[randomMatches[i].queryIdx].pt);
			dstPts.push_back(keypoints_2[randomMatches[i].trainIdx].pt);
		}

		// Compute the homography transform H for the selected pairs
		Mat homography = findHomography(srcPts, dstPts, 0);

		// Count the number of inliers
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

		// Update the best homography if the current iteration has more inliers
		if (numInliers > maxInliers) {
			maxInliers = numInliers;
			bestHomography = homography;
		}
	}

	// Re-compute homography estimation on the largest set of inliers
	std::vector<Point2f> inlierSrcPts;
	std::vector<Point2f> inlierDstPts;

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

	// Re-compute homography estimation on the largest set of inliers
	Mat refinedHomography = findHomography(inlierSrcPts, inlierDstPts, 0);

	return refinedHomography;
}


void  Assignment::core3() {

	// Load the images
	Mat img1 = imread("C:/Users/kahum/source/repos/Assignment4/Assignment4/res/Frame039.jpg");
	Mat img2 = imread("C:/Users/kahum/source/repos/Assignment4/Assignment4/res/Frame041.jpg");


	// Convert the images to grayscale
	Mat gray1, gray2;
	cvtColor(img1, gray1, COLOR_BGR2GRAY);
	cvtColor(img2, gray2, COLOR_BGR2GRAY);

	// Detect keypoints and compute descriptors using ORB
	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	Ptr<ORB> sift = ORB::create();
	sift->detectAndCompute(gray1, noArray(), keypoints1, descriptors1);
	sift->detectAndCompute(gray2, noArray(), keypoints2, descriptors2);

	// Match the descriptors using Brute-Force Matcher
	BFMatcher matcher(NORM_HAMMING);
	vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	// Find the homography matrix using RANSAC
	vector<Point2f> points1, points2;
	for (size_t i = 0; i < matches.size(); i++)
	{
		points1.push_back(keypoints1[matches[i].queryIdx].pt);
		points2.push_back(keypoints2[matches[i].trainIdx].pt);
	}


	//add 100px padding to each image
	Mat padded1, padded2;
	copyMakeBorder(img1, padded1, 100, 100, 100, 100, BORDER_CONSTANT, Scalar(0, 0, 0));
	copyMakeBorder(img2, padded2, 100, 100, 100, 100, BORDER_CONSTANT, Scalar(0, 0, 0));

	//find homography
	Mat H = findHomography(points1, points2, RANSAC);
	

	Mat warped1;
	warpPerspective(padded1, warped1, H, padded1.size());
	
	//add the two images together
	Mat stitched = Mat::zeros(padded1.size(), CV_8UC3);
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

	
	
	//show the result
	imshow("Stitched", stitched);
	waitKey(0);
	destroyAllWindows();
}
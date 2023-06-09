#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;
class Assignment {
public:
    Assignment() {};
    void core1();
    void core2();
	void core3();
    void compleation();
    vector<Mat> loadImages(int numFrames, const string& prefix);
    void exportImages(const vector<Mat>& images, const string& prefix, const string& outputFolder);
    Mat estimateHomography(const std::vector<KeyPoint>& keypoints_1, const std::vector<KeyPoint>& keypoints_2,
        const std::vector<DMatch>& matches, double epsilon, int numIterations);
};
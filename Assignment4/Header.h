#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
class Assignment {
public:
    Assignment() {};
    void core1();
    void core2();
	void core3();
    void compleation();
    Mat estimateHomography(const std::vector<KeyPoint>& keypoints_1, const std::vector<KeyPoint>& keypoints_2,
        const std::vector<DMatch>& matches, double epsilon, int numIterations);
};
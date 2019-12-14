#include <fstream>
#include <sstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;



string prototxt = R"(model/face/face-detection-adas-0001.xml)";
string model = R"(model/face/face-detection-adas-0001.bin)";


int main(int argc, char*atgv[])
{
    cout << "start " << endl;

    Mat image;
    image = cv::imread("4.jpg", IMREAD_UNCHANGED);
    if (image.empty())
    {
        cout << "read image failed!!" << endl;
        return -1;
    }
    cout << "read image done" << endl;

    Net net;
    

    try {
        net = readNet(prototxt, model);
    }
    catch (cv::Exception &ee) {        
        if (net.empty())
        {
            cout << "read net failed!!," << ee.msg << endl;
            return -1;
        }
    }
    cout << "read net done" << endl;
    

    //net.setPreferableBackend(DNN_BACKEND_OPENCV);
    //net.setPreferableTarget(DNN_TARGET_CPU);
    


    //Mat inputblob = blobFromImage(image, 1.0, Size(384, 672), Scalar(127.5, 127.5, 127.5), true, false);
    //Mat inputblob = blobFromImage(image,1.0, Size(384, 672));
    Mat inputblob = blobFromImage(image);
    net.setInput(inputblob);
    Mat detection = net.forward("detection_out");
    //cout << detection << endl;
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    cout << "detectionMat:" << detectionMat.rows << "," << detectionMat.cols << endl;
    //cout << detectionMat << endl;

    float *detectOut = (float*)detectionMat.data;
    for (size_t i = 0; i < detectionMat.total(); i += 7)
    {
        int imageid = (int)detectOut[i];
        int label = (int)detectOut[i+1];
        float  score = detectOut[i + 2];
        
        if (score > 0.5)
        {
            printf("score  = %f\n", score);

            int xmin = detectOut[i + 3] * image.cols;
            int ymin = detectOut[i + 4] * image.rows;

            int xmax = detectOut[i + 5] * image.cols;
            int ymax = detectOut[i + 6] * image.rows;

            printf("rect from [%d %d] to [%d %d]\n", xmin, ymin, xmax, ymax);
            cv::rectangle(image, Point(xmin, ymin), Point(xmax, ymax), Scalar(255, 0, 0));

        }
    }

    cv::imwrite("output.jpg", image);
    
    return 0;
}
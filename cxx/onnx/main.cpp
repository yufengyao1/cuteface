#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include "opencv2/imgproc/types_c.h"
using namespace cv;
float* point_line(float* point51, float* point1, float* point31) {
    float x1 = point1[0];
    float y1 = point1[1];
    float x2 = point31[0];
    float y2 = point31[1];
    float x3 = point51[0];
    float y3 = point51[1];
    float k1 = (y2 - y1) * 1.0 / (x2 - x1);
    float b1 = y1 * 1.0 - x1 * k1 * 1.0;
    float k2 = -1.0 / k1;
    float b2 = y3 * 1.0 - x3 * k2 * 1.0;
    float x = (b2 - b1) * 1.0 / (k1 - k2);
    float y = k1 * x * 1.0 + b1 * 1.0;
    return new float[2]{ x,y };
}
float point_point(float* point_1, float* point_2) {
    float x1 = point_1[0];
    float y1 = point_1[1];
    float x2 = point_2[0];
    float y2 = point_2[1];
    float distance = pow((pow((x1 - x2), 2) + pow((y1 - y2), 2)), 0.5);
    return distance;
}
int* get_faceangle(std::vector<float*> keypoints){ //非solvepnp面部姿态估计
    float* point1 = keypoints[1];
    float* point31 = keypoints[31];
    float* point51 = keypoints[51];
    float* crossover51 = point_line(point51, point1, point31);
    float yaw_mean = point_point(point1, point31) / 2;
    float yaw_right = point_point(point1, crossover51);
    float yaw = (yaw_mean - yaw_right) / yaw_mean;
    yaw = yaw*71.58+0.7037;
    float pitch_dis = point_point(point51, crossover51);
    if (point51[1] < crossover51[1]) {
        pitch_dis = -pitch_dis;
    }
    float pitch = 1.497 * pitch_dis + 18.97;
    float roll_tan = abs((keypoints[60][1] - keypoints[72][1]) / (keypoints[60][0] - keypoints[72][0]));
    float roll = atan(roll_tan);
    roll = 1.0 * 180 * roll / 3.1415926;
    if (keypoints[60][1] > keypoints[72][1]) {
        roll = -roll;
    }
    return new int[3]{(int)yaw,(int)pitch,(int)roll};
}
std::vector<float *> get_anchors(int h, int w) {
    int min_sizes[3][2] = { {16,32},{64,128},{256,512} };
    int steps[3] = { 8,16,32 };
    bool clip = false;
    int feature_maps[3][2];
    int a = sizeof(steps);
    int b = sizeof(steps[0]);
    for (int i = 0; i < sizeof(steps)/sizeof(steps[0]); i++) {
        int tmp[2];
        feature_maps[i][0] = ceil(1.0*h / steps[i]);
        feature_maps[i][1] = ceil(1.0*w / steps[i]);
    }
    std::vector<float *> anchors;
    for (int index = 0; index < 3;index++) {
        int ii = feature_maps[index][0];
        int jj = feature_maps[index][1];
        for (int i = 0; i < ii; i++) {
            for (int j = 0; j < jj; j++) {
                for (int k = 0; k < 2; k++) {
                    float *tmp=new float[4];
                    tmp[2] = 1.0*min_sizes[index][k] / w;
                    tmp[3] = 1.0*min_sizes[index][k] / h;
                    tmp[0] = 1.0*(j + 0.5) * steps[index] / w;
                    tmp[1] = 1.0*(i + 0.5) * steps[index] / h;
                    anchors.push_back(tmp);
                }
            }
        }
    }
    return anchors;

}
std::vector<float*> decode_np(float* loc, std::vector<float*> priors, float variances[2] ) {
    std::vector<float*> boxes;
    for (int i = 0; i < priors.size(); i++) {
        float* tmp = new float[4];
        float v_0 = priors[i][0] + loc[i * 4] * variances[0] * priors[i][2];
        float v_1 = priors[i][1] + loc[i * 4+1] * variances[0] * priors[i][3];
        float v_2 = priors[i][2]* exp(loc[i*4+2] * variances[1]);
        float v_3 = priors[i][3] * exp(loc[i * 4 + 3] * variances[1]);
        v_0 -= v_2 / 2;
        v_1 -= v_3 / 2;
        v_2 += v_0;
        v_3 += v_1;
        tmp[0] = v_0;
        tmp[1] = v_1;
        tmp[2] = v_2;
        tmp[3] = v_3;
        boxes.push_back(tmp);
    }
    return boxes;
}
std::vector<float*> decode_landm(float* pre, std::vector<float*>priors, float* variances) {
    std::vector<float*> boxes;
    for (int i = 0; i < priors.size(); i++) {
        float* tmp = new float[10];
        tmp[0] = priors[i][0] + pre[i * 10 + 0] * variances[0] * priors[i][2];
        tmp[1] = priors[i][1] + pre[i * 10 + 1] * variances[0] * priors[i][3];
        tmp[2] = priors[i][0] + pre[i * 10 + 2] * variances[0] * priors[i][2];
        tmp[3] = priors[i][1] + pre[i * 10 + 3] * variances[0] * priors[i][3];
        tmp[4] = priors[i][0] + pre[i * 10 + 4] * variances[0] * priors[i][2];
        tmp[5] = priors[i][1] + pre[i * 10 + 5] * variances[0] * priors[i][3];
        tmp[6] = priors[i][0] + pre[i * 10 + 6] * variances[0] * priors[i][2];
        tmp[7] = priors[i][1] + pre[i * 10 + 7] * variances[0] * priors[i][3];
        tmp[8] = priors[i][0] + pre[i * 10 + 8] * variances[0] * priors[i][2];
        tmp[9] = priors[i][1] + pre[i * 10 + 9] * variances[0] * priors[i][3];
        boxes.push_back(tmp);
    }
    return boxes;
}
void get_iou(float* bestbox, std::vector<float*>* boxes, float nms_thres) {
    float b1_x1 = bestbox[0];
    float b1_y1 = bestbox[1];
    float b1_x2 = bestbox[2];
    float b1_y2 = bestbox[3];
    std::vector<float> inter_rect_x1;
    std::vector<float> inter_rect_x2;
    std::vector<float> inter_rect_y1;
    std::vector<float> inter_rect_y2;
    int a = (*boxes).size();
    for (int i = 0; i < (*boxes).size(); i++) {
        if ((*boxes)[i][0] < b1_x1) {
            inter_rect_x1.push_back(b1_x1);
        }
        else {
            inter_rect_x1.push_back((*boxes)[i][0]);
        }

        if ((*boxes)[i][1] < b1_y1) {
            inter_rect_y1.push_back(b1_y1);
        }
        else {
            inter_rect_y1.push_back((*boxes)[i][1]);
        }

        if ((*boxes)[i][2] > b1_x2) {
            float tmp = b1_x2 - inter_rect_x1[i];
            tmp = tmp > 0 ? tmp : 0;
            inter_rect_x2.push_back(tmp);
        }
        else {
            float tmp = (*boxes)[i][2] - inter_rect_x1[i];
            tmp = tmp > 0 ? tmp : 0;
            inter_rect_x2.push_back(tmp);
        }

        if ((*boxes)[i][3] > b1_y2) {
            float tmp = b1_y2 - inter_rect_y1[i];
            tmp = tmp > 0 ? tmp : 0;
            inter_rect_y2.push_back(tmp);
        }
        else {
            float tmp = (*boxes)[i][3] - inter_rect_y1[i];
            tmp = tmp > 0 ? tmp : 0;
            inter_rect_y2.push_back(tmp);
        }
    }
    float area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1);
    for (int i = (*boxes).size() - 1; i >= 0; i--) {
        float iou = inter_rect_x2[i]*inter_rect_y2[i]/(((*boxes)[i][2] - (*boxes)[i][0]) * ((*boxes)[i][3] - (*boxes)[i][1]) + area_b1 - inter_rect_x2[i] * inter_rect_y2[i]);
        iou = iou > 1e-6 ? iou : 1e-6;
        if (iou >= nms_thres) (*boxes).erase((*boxes).begin() + i); //删除不合格
    }
}
std::vector<float*> non_max_suppression(std::vector<float*> boxes, float conf_thres, float nms_thres) {
    std::vector<float*> bestboxes;
    for (int i = boxes.size() - 1; i >= 0; i--) {
        if (boxes[i][4] <= conf_thres) {
            boxes.erase(boxes.begin() + i);
        }
    }
    while (true) {
        if (boxes.size() == 0) break;
        float max_val = 0;
        int max_index = 0;
        for (int i = 0; i < boxes.size(); i++) { //寻找最大矩形框
            if (boxes[i][4] > max_val) {
                max_val = boxes[i][3];
                max_index = i;
            }
        }
        bestboxes.push_back(boxes[max_index]); //添加最好的框
        if (boxes.size() == 1) break;
        boxes.erase(boxes.begin() + max_index);//删除最大
        //求iou,删除重复
        for (int i = 0; i < boxes.size(); i++) { //根据最大矩形框滤除重复
            get_iou(bestboxes[bestboxes.size() - 1], &boxes, nms_thres);
        }
    }
    return bestboxes;
}
std::vector<float*> get_face(Ort::Session* session, Mat dst) {
    int h = dst.rows;
    int w = dst.cols;
    std::vector<const char*> input_node_names = { "input" };
    std::vector<const char*> output_node_names = { "output","514" ,"513" };
    std::vector<int64_t> input_node_dims = { 1,3,h,w }; //input尺寸数组
    size_t input_tensor_size = 3 * w * h; //总尺寸
    std::vector<float> img_data(input_tensor_size); //图像数据数组
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if (c == 0) {
                    img_data[c * w * h + i * w + j] = (dst.ptr<uchar>(i)[j * 3 + c]) - 104;
                }
                if (c == 1) {
                    img_data[c * w * h + i * w + j] = (dst.ptr<uchar>(i)[j * 3 + c]) - 117;
                }
                if (c == 2) {
                    img_data[c * w * h + i * w + j] = (dst.ptr<uchar>(i)[j * 3 + c]) - 123;
                }
            }
        }
    }
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, img_data.data(), input_tensor_size, input_node_dims.data(), 4);
    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor));
    //double timeStart = (double)getTickCount();
    auto output_tensors = (*session).Run(Ort::RunOptions{ nullptr }, input_node_names.data(), ort_inputs.data(), input_node_names.size(), output_node_names.data(), output_node_names.size());
    //double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
    float* loc = output_tensors[0].GetTensorMutableData<float>();
    float* conf = output_tensors[1].GetTensorMutableData<float>();
    float* landms = output_tensors[2].GetTensorMutableData<float>();

    std::vector<float*> anchors = get_anchors(h, w); //计算anchor
    float variances[2] = { 0.1,0.2 };
    std::vector<float*> boxes = decode_np(loc, anchors, variances);
    std::vector<float*> landmarks = decode_landm(landms, anchors, variances);
    std::vector<float*> boxes_conf_landms;
    for (int i = 0; i < landmarks.size(); i++) {
        float* tmp = new float[15];
        tmp[0] = boxes[i][0];
        tmp[1] = boxes[i][1];
        tmp[2] = boxes[i][2];
        tmp[3] = boxes[i][3];
        tmp[4] = conf[i * 2 + 1];
        tmp[5] = landmarks[i][0];
        tmp[6] = landmarks[i][1];
        tmp[7] = landmarks[i][2];
        tmp[8] = landmarks[i][3];
        tmp[9] = landmarks[i][4];
        tmp[10] = landmarks[i][5];
        tmp[11] = landmarks[i][6];
        tmp[12] = landmarks[i][7];
        tmp[13] = landmarks[i][8];
        tmp[14] = landmarks[i][9];
        boxes_conf_landms.push_back(tmp);
    }
    std::vector<float*> faces = non_max_suppression(boxes_conf_landms, 0.5, 0.3);
    return faces;
}
std::vector<float*> get_facepoints98(Ort::Session* session, Mat img,float rect[4]) {
    std::vector<float*> result;
    try {
        float tmp_x = rect[0] > 0 ? rect[0] : 0;
        float tmp_y = rect[1] > 0 ? rect[1] : 0;
        float tmp_x2 = rect[2] > img.cols ? img.cols : rect[2];
        float tmp_y2 = rect[3] > img.rows ? img.rows : rect[3];
        int pad[4] = { 0,0,0,0 };
        if (rect[0] < 0) {
            pad[0] = -1 * rect[0];
        }
        if (rect[1] < 0) {
            pad[1] = -1 * rect[1];
        }
        if (rect[2] > img.cols) {
            pad[2] = rect[2] - img.cols;
        }
        if (rect[3] > img.rows) {
            pad[3] = rect[3] - img.rows;
        }
        Mat frame_face = img(Range(tmp_y, tmp_y2), Range(tmp_x, tmp_x2));
        int pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
        int w = frame_face.cols + pad[0] + pad[2];
        int h = frame_face.rows + pad[1] + pad[3];
        if (w > h) { //计算上下左右需要填充的像素尺寸，将图像填充成正方形
            if ((w - h) % 2 == 0) {
                pad_top = (w - h) / 2;
                pad_bottom = pad_top;
            }
            else {
                pad_bottom = ceil((w - h) / 2);
                pad_top = w - h - pad_bottom;
            }
        }
        else if (w < h) {
            if ((h - w) % 2 == 0) {
                pad_left = (h - w) / 2;
                pad_right = pad_left;
            }
            else {
                pad_right = ceil((h - w) / 2);
                pad_left = h - w - pad_right;
            }
        }
        cv::copyMakeBorder(frame_face, frame_face, pad_top + pad[1], pad_bottom + pad[3], pad_left + pad[0], pad_right + pad[2], cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));  // 图像边缘扩展
        int max_size = frame_face.rows; //加边以后的尺寸w=h
        cv::resize(frame_face, frame_face, cv::Size(112, 112), 0, 0, cv::INTER_CUBIC); //resize 到112 * 112
        std::vector<int64_t> input_node_dims = { 1,3,112,112 }; //input尺寸数组
        size_t input_tensor_size = 3 * 112 * 112; //总尺寸
        std::vector<float> img_data(input_tensor_size); //图像数据数组
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < 112; i++) {
                for (int j = 0; j < 112; j++) {
                    if (c == 0) {
                        img_data[c * 112 * 112 + i * 112 + j] = 1.0 * (frame_face.ptr<uchar>(i)[j * 3 + c]) / 255;
                    }
                    if (c == 1) {
                        img_data[c * 112 * 112 + i * 112 + j] = 1.0 * (frame_face.ptr<uchar>(i)[j * 3 + c]) / 255;
                    }
                    if (c == 2) {
                        img_data[c * 112 * 112 + i * 112 + j] = 1.0 * (frame_face.ptr<uchar>(i)[j * 3 + c]) / 255;
                    }
                }
            }
        }
        std::vector<const char*> input_node_names = { "input" };
        std::vector<const char*> output_node_names = { "output" };
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, img_data.data(), input_tensor_size, input_node_dims.data(), 4);
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(input_tensor));
        auto output_tensors = (*session).Run(Ort::RunOptions{ nullptr }, input_node_names.data(), ort_inputs.data(), input_node_names.size(), output_node_names.data(), output_node_names.size());
        float* pred = output_tensors[0].GetTensorMutableData<float>();
        for (int i = 0; i < 98; i++) {
            result.push_back(new float[2]{ tmp_x + pred[i * 2] * max_size - pad_left - pad[0] ,tmp_y + pred[i * 2 + 1] * max_size - pad_top - pad[1] });
        }
    }
    catch(const char* msg){
    }
    return result;
    
}
int main()
{
    //加载图像
    Mat img = imread("C:/Users/yufen/Desktop/onnx/onnx/1.jpg");
    Mat dst(img.rows, img.cols, CV_8UC3);
    cvtColor(img, dst, CV_BGR2RGB);

    size_t img_data_size = img.rows * img.cols * 3; //total size
    const wchar_t* face_model_path = L"face.onnx"; //onnx file
    const wchar_t* keypoints_model_path = L"facepoints98.onnx"; //onnx file
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_face_detect");//onnx env
    Ort::Env env_keypoints(ORT_LOGGING_LEVEL_WARNING, "onnx_keypoints_detect");//onnx env
    Ort::SessionOptions session_options; //onnx options
    session_options.SetIntraOpNumThreads(1); //onnx thread nums
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); 
    Ort::Session* session_face=new Ort::Session(env, face_model_path, session_options); //加载人脸检测模型
    Ort::Session* session_keypoints = new Ort::Session(env_keypoints, keypoints_model_path, session_options); //加载关键点检测模型
    
    std::vector<float*> faces= get_face(session_face, dst);
    for (int i = 0; i < faces.size(); i++) {
        faces[i][0] *= dst.cols;
        faces[i][1] *= dst.rows;
        faces[i][2] *= dst.cols;
        faces[i][3] *= dst.rows;
        float rect[4] = { faces[i][0],faces[i][1] ,faces[i][2] ,faces[i][3] };
        auto keypoints=get_facepoints98(session_keypoints, dst, rect);
        for (int j = 0; j < keypoints.size(); j++) {
            circle(img, Point(keypoints[j][0], keypoints[j][1]), 1, Scalar(0, 255, 0),-1);
        }
        int* angles = get_faceangle(keypoints);
        cv::putText(img, "yaw:" + std::to_string(angles[0]), cv::Point(10, 20), cv::FONT_HERSHEY_PLAIN, 1,cv::Scalar(0,255,0), 2);
        cv::putText(img, "pitch:" + std::to_string(angles[1]), cv::Point(10, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(img, "roll:" + std::to_string(angles[2]), cv::Point(10, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 2);

    }
    imshow("test", img);
    waitKey();
    std::cout << "successful!\n";
}


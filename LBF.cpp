//
//  LBF.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#include "LBF.h"
#include "LBFRegressor.h"
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
using namespace std;
using namespace cv;

// parameters
Params global_params;


typedef dlib::object_detector<dlib::scan_fhog_pyramid<dlib::pyramid_down<6> > > frontal_face_detector;

string modelPath ="./../../model_69/";
string dataPath = "./../../Datasets/";
string dlib_face_detector = "front_face.dat";
string cascadeName = "haarcascade_frontalface_alt.xml";

void InitializeGlobalParam();
void PrintHelp();

void test_dlib_face_detect() {
    cv::Mat gray = cv::imread("img/1.jpg", 0);
    frontal_face_detector detector;
    dlib::deserialize(dlib_face_detector) >> detector;
    dlib::array2d<unsigned char> img;
    dlib::cv_image<unsigned char> *pimg = new dlib::cv_image<unsigned char>(gray);
    assign_image(img, *pimg);
    delete pimg;
    std::vector<dlib::rectangle> dets;
    dets  = detector(img);
    for (int i = 0; i < dets.size(); i++) {
        cv::Rect rect = cv::Rect(cv::Point(dets[i].left(), dets[i].top()), cv::Point(dets[i].right(), dets[i].bottom()) );
        cv::rectangle(gray, rect, cv::Scalar(255));
    }
    cv::imshow("dst", gray);
    cv::waitKey();
}

int main( int argc, const char** argv ){
    test_dlib_face_detect();
    return 0;
    if (argc > 1 && strcmp(argv[1],"TrainModel")==0){
        InitializeGlobalParam();
    }
    else {
        ReadGlobalParamFromFile(modelPath+"LBF.model");
    }
    
    // main process
    if (argc==1){
        PrintHelp();
    }
    else if(strcmp(argv[1],"TrainModel")==0){
        vector<string> trainDataName;
     // you need to modify this section according to your training dataset
        trainDataName.push_back("afw");
        trainDataName.push_back("helen");
        trainDataName.push_back("lfpw");
        TrainModel(trainDataName);
    }
    else if (strcmp(argv[1], "TestModel")==0){
        vector<string> testDataName;
     // you need to modify this section according to your training dataset
        testDataName.push_back("ibug");
     //   testDataName.push_back("helen");
        double MRSE = TestModel(testDataName);
        
    }
    else if (strcmp(argv[1], "Demo")==0){
        if (argc == 2){
            return FaceDetectionAndAlignment("");
        }
        else if(argc ==3){
            return FaceDetectionAndAlignment(argv[2]);
        }
    }
    else {
        PrintHelp();
    }
    return 0;
}

// set the parameters when training models.
void InitializeGlobalParam(){
    global_params.bagging_overlap = 0.4;
    global_params.max_numtrees = 10;
    global_params.max_depth = 5;
    global_params.landmark_num = 68;
    global_params.initial_num = 5;
    
    global_params.max_numstage = 7;
    double m_max_radio_radius[10] = {0.4,0.3,0.2,0.15, 0.12, 0.10, 0.08, 0.06, 0.06,0.05};
    double m_max_numfeats[10] = {500, 500, 500, 300, 300, 200, 200,200,100,100};
    for (int i=0;i<10;i++){
        global_params.max_radio_radius[i] = m_max_radio_radius[i];
    }
    for (int i=0;i<10;i++){
        global_params.max_numfeats[i] = m_max_numfeats[i];
    }
    global_params.max_numthreshs = 500;
}

void ReadGlobalParamFromFile(string path){
    cout << "Loading GlobalParam..." << endl;
    ifstream fin;
    fin.open(path);
    fin >> global_params.bagging_overlap;
    fin >> global_params.max_numtrees;
    fin >> global_params.max_depth;
    fin >> global_params.max_numthreshs;
    fin >> global_params.landmark_num;
    fin >> global_params.initial_num;
    fin >> global_params.max_numstage;
    
    for (int i = 0; i< global_params.max_numstage; i++){
        fin >> global_params.max_radio_radius[i];
    }
    
    for (int i = 0; i < global_params.max_numstage; i++){
        fin >> global_params.max_numfeats[i];
    }
    cout << "Loading GlobalParam end"<<endl;
    fin.close();
}
void PrintHelp(){
    cout << "Useage:"<<endl;
    cout << "1. train your own model:    LBF.out  TrainModel "<<endl;
    cout << "2. test model on dataset:   LBF.out  TestModel"<<endl;
    cout << "3. test model via a camera: LBF.out  Demo "<<endl;
    cout << "4. test model on a pic:     LBF.out  Demo xx.jpg"<<endl;
    cout << "5. test model on pic set:   LBF.out  Demo Img_Path.txt"<<endl;
    cout << endl;
}

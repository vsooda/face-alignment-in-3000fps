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

// parameters
Params global_params;


typedef dlib::object_detector<dlib::scan_fhog_pyramid<dlib::pyramid_down<6> > > frontal_face_detector;

std::string modelPath ="../models/";
std::string dataPath = "./../../Datasets/";
std::string dlib_face_detector = "../data/front_face.dat";
std::string cascadeName = "../data/haarcascade_frontalface_alt.xml";

void InitializeGlobalParam(int landmark_num = 68);
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
//    test_dlib_face_detect();
//    return 0;
    if (argc > 1 && ( strcmp(argv[1],"TrainModel")==0 || strcmp(argv[1], "traintxt") == 0)){
        int landmark_num = 68;
        if (argc > 3) {
            landmark_num = atoi(argv[3]);
        }
        InitializeGlobalParam(landmark_num);
    }
    else {
        ReadGlobalParamFromFile(modelPath+"LBF.model");
    }
    
    // main process
    if (argc==1){
        PrintHelp();
    }
    else if(strcmp(argv[1],"TrainModel")==0){
        std::vector<std::string> trainDataName;
     // you need to modify this section according to your training dataset
        trainDataName.push_back("afw");
        trainDataName.push_back("helen");
        trainDataName.push_back("lfpw");
        TrainModel(trainDataName);
    }
    else if (strcmp(argv[1], "TestModel")==0){
        //test text: ./LBF.out TestModel ~/data/lfpw/lfpw_train.txt
        std::vector<std::string> testDataName;
        //testDataName.push_back("ibug");
        testDataName.push_back(argv[2]);
        double MRSE = TestModel(testDataName);
        
    }
    else if (strcmp(argv[1], "traintxt") == 0) {
        //train text format: ./LBF.out traintxt ~/data/lfpw/lfpw_train.txt
        std::string annotateName(argv[2]);
        std::cout << "annotationName: " << annotateName << std::endl;;
        TrainSelfModel(annotateName);
    }
    else if (strcmp(argv[1], "Demo")==0){
        if (argc == 2){
            return FaceDetectionAndAlignment("");
        }
        else if(argc ==3){
            return FaceDetectionAndAlignment(argv[2]);
        } else {
            dlibDetectAndDraw(argc, argv);
        }
    }
    else {
        PrintHelp();
    }
    return 0;
}

// set the parameters when training models.
void InitializeGlobalParam(int landmark_num){
    global_params.bagging_overlap = 0.4;
    global_params.max_numtrees = 10;
    global_params.max_depth = 5;
    global_params.landmark_num = landmark_num;
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

void ReadGlobalParamFromFile(std::string path){
    std::cout << "Loading GlobalParam... "  << path << std::endl;
    std::ifstream fin;
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
    std::cout << "Loading GlobalParam end"<< std::endl;
    fin.close();
}
void PrintHelp(){
    std::cout << "1. train your own model:    LBF.out  TrainModel "<< std::endl;
    std::cout << "2. test model on dataset:   LBF.out  TestModel"<<std::endl;
    std::cout << "3. test model via a camera: LBF.out  Demo "<<std::endl;
    std::cout << "4. test model on a pic:     LBF.out  Demo xx.jpg"<<std::endl;
    std::cout << "5. test model on pic set:   LBF.out  Demo Img_Path.txt"<<std::endl;
    std::cout << std::endl;
}

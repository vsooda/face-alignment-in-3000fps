//
//  TrainDemo.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//
#include "LBFRegressor.h"
using namespace std;
using namespace cv;
void LoadCofwTrainData(vector<Mat_<uchar> >& images,
                       vector<Mat_<double> >& ground_truth_shapes,
                       vector<BoundingBox>& bounding_boxs);
void TrainModel(vector<string> trainDataName){
    vector<Mat_<uchar> > images;
    vector<Mat_<double> > ground_truth_shapes;
    vector<BoundingBox> bounding_boxs;

    for(int i=0;i<trainDataName.size();i++){
        string path;
        if(trainDataName[i]=="helen"||trainDataName[i]=="lfpw")
            path = dataPath + trainDataName[i] + "/trainset/Path_Images.txt";
        else
            path = dataPath + trainDataName[i] + "/Path_Images.txt";

       // LoadData(path, images, ground_truth_shapes, bounding_boxs);
          LoadOpencvBbxData(path, images, ground_truth_shapes, bounding_boxs);
    }

    LBFRegressor regressor;
    regressor.Train(images,ground_truth_shapes,bounding_boxs);
    regressor.Save(modelPath+"LBF.model");
}

void TrainSelfModel(std::string annotatenName) {
    std::vector<cv::Mat_<uchar> > images;
    std::vector<cv::Mat_<double> > ground_truth_shapes;
    std::vector<BoundingBox> bounding_boxes;

    loadSelfDataFromText(annotatenName, images, ground_truth_shapes, bounding_boxes);
    std::cout << images.size() << " " << ground_truth_shapes.size() << " " << bounding_boxes.size() << std::endl;
    LBFRegressor regressor;
    regressor.Train(images,ground_truth_shapes,bounding_boxes);
    regressor.Save(modelPath+"LBF.model");
}


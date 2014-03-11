// needed for finger tracking
//          if (kUseGestures)
//          {
//              if (sf.skeletons(i).joints(j).type() == 9)
//              {
//                  mapIdSkel[sf.skeletons(i).skeleton_id()].joints[9].image_x = sf.skeletons(i).joints(j).image_x();
//                  mapIdSkel[sf.skeletons(i).skeleton_id()].joints[9].image_y = sf.skeletons(i).joints(j).image_y();
//              }

//              if (sf.skeletons(i).joints(j).type() == 15)
//              {
//                  mapIdSkel[sf.skeletons(i).skeleton_id()].joints[15].image_x = sf.skeletons(i).joints(j).image_x();
//                  mapIdSkel[sf.skeletons(i).skeleton_id()].joints[15].image_y = sf.skeletons(i).joints(j).image_y();
//              }

//              if (sf.skeletons(i).joints(j).type() == 7)
//              {
//                  mapIdSkel[sf.skeletons(i).skeleton_id()].joints[7].image_x = sf.skeletons(i).joints(j).image_x();
//                  mapIdSkel[sf.skeletons(i).skeleton_id()].joints[7].image_y = sf.skeletons(i).joints(j).image_y();
//              }
//          }

/*
        RemoteKinect::DepthMap dm;

        if (depth_socket->recv(dm))
        {
            for (int y = 0; y < 480; y++)
            {
                for (int x = 0; x < 640; x++)
                {
                    float val = dm.depths(y * 640 + x);
                    depth_pixels[640 * (479 - y) + x] = val;
                }
            }
        }

        int YRES = 480;
        int XRES = 640;
        int ROIOFFSET = 70;
        cv::Mat depthRaw(YRES, XRES, CV_16UC1);
        cv::Mat depthShow(YRES, XRES, CV_8UC1);
        uint16_t depth_c[640 * 480];

        for (int l = 0; l < 640 * 480; l++)
            depth_c[l] = depth_pixels[l];

        memcpy(depthRaw.data, depth_c, XRES * YRES * 2);
        depthRaw.convertTo(depthShow, CV_8U, DEPTH_SCALE_FACTOR);

        // for every skeleton

        for (int i = 0; i < sf.skeletons_size(); i++)
        {
            //-----------------------------------------------------------------------------
            cv::Rect roi;
            // Distances in 2D
            // We want 70 px around center of the hand when ~1.5 m away
            // Distance from elbow to wrist is 330 mm, hand is 170 mm, we want 130 mm around the center
            // 13/33 = 0.4
            osg::Vec3 realworldHandToElbow(mapIdSkel[sf.skeletons(i).skeleton_id()].joints[9].position.x() - mapIdSkel[sf.skeletons(i).skeleton_id()].joints[7].position.x(),
                                           mapIdSkel[sf.skeletons(i).skeleton_id()].joints[9].position.y() - mapIdSkel[sf.skeletons(i).skeleton_id()].joints[7].position.y(),
                                           0);
            osg::Vec3 projectiveHandToElbow(mapIdSkel[sf.skeletons(i).skeleton_id()].joints[9].image_x - mapIdSkel[sf.skeletons(i).skeleton_id()].joints[7].image_x,
                                            mapIdSkel[sf.skeletons(i).skeleton_id()].joints[9].image_y - mapIdSkel[sf.skeletons(i).skeleton_id()].joints[7].image_y,
                                            0);
            // x=330/realworld // because of lack of Z coordinate
            // projective*330/realworld*0.4
            //    printf("%g \n",projectiveHandToElbow.length()*0.5*330/realworldHandToElbow.length());
            ROIOFFSET = projectiveHandToElbow.length() * 0.5 * 330 / realworldHandToElbow.length();

            if (ROIOFFSET < 10 || ROIOFFSET > 80) ROIOFFSET = 70;

            roi.width = ROIOFFSET * 2;
            roi.height = ROIOFFSET * 2;
            int handDepth = mapIdSkel[sf.skeletons(i).skeleton_id()].joints[9].position.z() * (DEPTH_SCALE_FACTOR);
            double handx = mapIdSkel[sf.skeletons(i).skeleton_id()].joints[9].image_x;
            double handy = mapIdSkel[sf.skeletons(i).skeleton_id()].joints[9].image_y;

            if (!handApproachingDisplayPerimeter(handx, handy, ROIOFFSET))
            {
                roi.x = handx - ROIOFFSET;
                roi.y = handy - ROIOFFSET;
            }
            else handDepth = -1;

            if (handDepth != -1)
            {
                cv::Mat handCpy(depthShow, roi);
                mapIdSkel[sf.skeletons(i).skeleton_id()].checkHandOpen(9, handCpy.clone(), handDepth, ROIOFFSET);
            }

            //-----------------------------------------------------------------------------
            handDepth = mapIdSkel[sf.skeletons(i).skeleton_id()].joints[15].position.z() * (DEPTH_SCALE_FACTOR);
            handx = mapIdSkel[sf.skeletons(i).skeleton_id()].joints[15].image_x;
            handy = mapIdSkel[sf.skeletons(i).skeleton_id()].joints[15].image_y;

            if (!handApproachingDisplayPerimeter(handx, handy, ROIOFFSET))
            {
                roi.x = handx - ROIOFFSET;
                roi.y = handy - ROIOFFSET;
            }
            else handDepth = -1;

            if (handDepth != -1)
            {   cv::Mat handCpy2(depthShow, roi);
                mapIdSkel[sf.skeletons(i).skeleton_id()].checkHandOpen(15, handCpy2.clone(), handDepth, ROIOFFSET);
            }

            //-----------------------------------------------------------------------------
        }
*/







////...................compare head IR & kinect...................................
////..............................................................................
////Compare HandMat and HeadMat to Skeleton
//    bool ktest3 = false;
//
//    if (ktest3)
//    {
//        //Find Sensors' Origin and Rotation
//        //IR
//        Vec3 iOrigin;
//        Quat iOriginRot;
//        //Kinect
//        Vec3 kOrigin;
//        Quat kOriginRot;
//        //Kinect rotation comes in Upside down--will need to rotate all positions  according to origin and apply same offset so that it matches IR sensor.
//        //position[i][0]=0
//        //-------------------------------------------
//        double kinectX = position[0][0] * 1000 + 141;
//        double kinectY = -(position[0][2] * 1000 - 2000);
//        double kinectZ = position[0][1] * 1000 + 517;
//
//        //std::vector<TrackerBase::TrackedBody *> * tbList = (std::vector<TrackerBase::TrackedBody *> *) userdata;
//        //TrackerBase::TrackedBody * tb = tbList->at(0);
//        //tb->x = kinectX;
//        //tb->y = kinectY;
//        //tb->z = kinectZ;
//        //-------------------------------------------
//        //Get Kinect Head
//        if (position[0][0] != 0)
//        {
//            cout << "Head Kinect Output-Position: " << kinectX << "," << kinectY << "," << kinectZ<<"\n";
//// << "\nHead Kinect Output-Rotation:" << orientation[0][0] << "," << orientation[0][1] << "," << orientation[0][2] << "\n";
//        }
//
//        //Get IR Head
//        int heads = PluginHelper::getNumHeads();
//
//        if (heads > 0)
//        {
//            Matrix headIR = PluginHelper::getHeadMat();
//            Vec3 headIrTrans = headIR.getTrans();
//            Quat headIrRot = headIR.getRotate();
//
//            cout << "Head IR Output-Position: " << headIrTrans.x() << "," << headIrTrans.y() << "," << headIrTrans.z() << "\n";
////Head IR Output-Rotation:" << headIrRot.x() << "," << headIrRot.y() << "," << headIrRot.z() << "\n";
//            //Get Difference between Kinect and IR
//            if ((position[0][0] != 0) && (heads > 0) && false)
//            {
//                double xdif, ydif, zdif, rxdif, rydif, rzdif;
//                xdif = ydif = zdif = rxdif = rydif = rzdif = 0.0;
//                xdif = headIrTrans.x() - kinectX;
//                ydif = headIrTrans.y() - kinectY;
//                zdif = headIrTrans.z() - kinectZ;
//
//                //rxdif = headIrRot.x() - orientation[0][0];
//                if (xdif > 250 || ydif > 250 || zdif > 250)
//                    cout << "diff: " << xdif << " " << ydif << " " << zdif << "\n";
//            }
//        }
//    }
//..............................................................................

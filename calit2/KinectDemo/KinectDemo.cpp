#include "KinectDemo.h"

using namespace osg;
using namespace std;
using namespace cvr;

CVRPLUGIN(KinectDemo)
KinectDemo* KinectDemo::_kinectDemo = NULL;

KinectDemo::KinectDemo() {}

KinectDemo::~KinectDemo() {}

KinectDemo* KinectDemo::instance()
{
    if (!_kinectDemo)
        _kinectDemo = new KinectDemo();

    return _kinectDemo;
}

bool KinectDemo::init()
{
    _root = new osg::MatrixTransform();
    buttonDown = false;
    rightButtonDown = false;
    masterKinect = 1;
    masterKinectServer = 2;
    max_users = 12;
    wandLockedToSkeleton = false;
    useKinect = true;
    _firstRun = true;
    calibCount = 0;
    kShowPCloud = ConfigManager::getBool("Plugin.KinectDemo.KinectDefaultOn.ShowPCloud");
    // kShowPCloud = false;
    useKColor = ConfigManager::getBool("Plugin.KinectDemo.KinectDefaultOn.ShowColor");
    userColor = false;
    kShowDepth = false;
    kShowColor = false;
    kMoveWithCam = false;
    kFreezeCloud = false;
    std::cerr << "KinectDemo init\n";
    //Menu Setup:
    _avMenu = new SubMenu("Kinect Demo", "Kinect Demo");
    _avMenu->setCallback(this);
    _bookmarkLoc = new MenuButton("Save Location");
    _bookmarkLoc->setCallback(this);
    _testInteract = new MenuButton("Send Interact");
    _testInteract->setCallback(this);
    _kinectOn = new MenuCheckbox("Use Kinect", useKinect);  //new
    _kinectOn->setCallback(this);
    _avMenu->addItem(_kinectOn);
    _kShowColor = new MenuCheckbox("Show Color Map", kShowColor);  //new
    _kShowColor->setCallback(this);
    _kShowDepth = new MenuCheckbox("Show Depth Map", kShowDepth);
    _kShowDepth->setCallback(this);
    _kMoveWithCam = new MenuCheckbox("Move skeleton with camera", kMoveWithCam);
    _kMoveWithCam->setCallback(this);
    _kFreezeCloud = new MenuCheckbox("Move skeleton with camera", kFreezeCloud);
    _kFreezeCloud->setCallback(this);
    colorfps = 100;
    _kColorFPS = new MenuRangeValue("1/FPS for camera/depth", 1, 100, colorfps);
    _kColorFPS->setCallback(this);
    _kShowPCloud = new MenuCheckbox("Show Point Cloud", kShowPCloud);  //new
    _kShowPCloud->setCallback(this);
    _kShowInfoPanel = new MenuCheckbox("Show Info Panel", kShowInfoPanel);  //new
    _kShowInfoPanel->setCallback(this);
    _infoPanel = new TabbedDialogPanel(500, 20, 4, "Info Panel", "Plugin.KinectDemo.InfoPanel");
    _infoPanel->addTextTab("Info", "");
    _infoPanel->setVisible(false);
    _infoPanel->setActiveTab("Info");
    _kColorOn = new MenuCheckbox("Show Real Colors on Point Cloud", useKColor);  //new
    _kColorOn->setCallback(this);
    _kUserColorOn = new MenuCheckbox("User coloring", userColor);  //new
    _kUserColorOn->setCallback(this);
    _avMenu->addItem(_kShowColor);
    _avMenu->addItem(_kShowDepth);
    _avMenu->addItem(_kShowPCloud);
    _avMenu->addItem(_kColorOn);
    _avMenu->addItem(_kUserColorOn);
    _avMenu->addItem(_kColorFPS);
    _avMenu->addItem(_bookmarkLoc);
    _avMenu->addItem(_testInteract);
    _avMenu->addItem(_kMoveWithCam);
    _avMenu->addItem(_kFreezeCloud);
    _avMenu->addItem(_kShowInfoPanel);
    MenuSystem::instance()->addMenuItem(_avMenu);
    _switchMasterSkeleton = new MenuButton("Switch Master Skeleton");
    _switchMasterSkeleton->setCallback(this);
    _avMenu->addItem(_switchMasterSkeleton);
    _devMenu = new SubMenu("Dev options", "Dev options");
    _devMenu->setCallback(this);
    _devFixXY = new MenuCheckbox("Fix XY", fixXY);
    _devIgnoreZeros = new MenuCheckbox("Ignore 0s", ignoreZeros);
    _devFilterBackground = new MenuCheckbox("Filter background", filterBackground);
    _devAssignPointsToSkeletons = new MenuCheckbox("Assign points to skeletons", assignPointsToSkeletons);
    _devClassifyPoints = new MenuCheckbox("Classify points", classifyPoints);
    _devDenoise = new MenuCheckbox("Denoise (mode)", denoise);
    _devFixXY->setCallback(this);
    _devIgnoreZeros->setCallback(this);
    _devFilterBackground->setCallback(this);
    _devAssignPointsToSkeletons->setCallback(this);
    _devClassifyPoints->setCallback(this);
    _devDenoise->setCallback(this);
    _devMenu->addItem(_devFixXY);
    _devMenu->addItem(_devIgnoreZeros);
    _devMenu->addItem(_devFilterBackground);
    _devMenu->addItem(_devAssignPointsToSkeletons);
    _devMenu->addItem(_devClassifyPoints);
    _devMenu->addItem(_devDenoise);
    _avMenu->addItem(_devMenu);
    _calibrateMenu = new SubMenu("Calibrate", "Calibrate");
    _calibrateMenu->setCallback(this);
    _calibrateIrMenu = new SubMenu("Calibrate from IR", "Calibrate from IR");
    _calibrateIrMenu->setCallback(this);
    _calibrateRefMenu = new SubMenu("Calibrate from Ref", "Calibrate from Ref");
    _calibrateRefMenu->setCallback(this);
    _toggleCalibrate = new MenuCheckbox("Start Calibration", false);
    _toggleCalibrate->setCallback(this);
    _toggleRefCalibrate = new MenuCheckbox("Start Calibration", false);
    _toggleRefCalibrate->setCallback(this);
    _skeletonCalibrate = new MenuButton("Skeleton Calibrate");
    _skeletonCalibrate->setCallback(this);
    _kinectPC1 = new MenuCheckbox("Kinect PC1", false);
    _kinectPC1->setCallback(this);
    _kinectPC2 = new MenuCheckbox("Kinect PC2", false);
    _kinectPC2->setCallback(this);
    _kinectPC3 = new MenuCheckbox("Kinect PC3", false);
    _kinectPC3->setCallback(this);
    _kinectPC4 = new MenuCheckbox("Kinect PC4", false);
    _kinectPC4->setCallback(this);
    _kinectRef = new MenuCheckbox("Kinect Ref", false);
    _kinectRef->setCallback(this);
    _kinectTransformed = new MenuCheckbox("Kinect Transformed", false);
    _kinectTransformed->setCallback(this);
    _kinectIrTransformed = new MenuCheckbox("Kinect Transformed", false);
    _kinectIrTransformed->setCallback(this);
    _kinectCreateSelectPoint = new MenuCheckbox("Create Select Point", false);
    _kinectCreateSelectPoint->setCallback(this);
    _showRefPoints = new MenuCheckbox("Show Ref Points", false);
    _showRefPoints->setCallback(this);
    _showIRPoints = new MenuCheckbox("Show IR Points", false);
    _showIRPoints->setCallback(this);
    _kinectCreateIrPoint = new MenuCheckbox("Create Select Point", false);
    _kinectCreateIrPoint->setCallback(this);
    _eraseAllSelectPoints = new MenuButton("Erase All Select Points");
    _eraseAllSelectPoints->setCallback(this);
    _triangulateKinect = new MenuButton("Triangulate Kinect");
    _triangulateKinect->setCallback(this);
    _triangulateIrKinect = new MenuButton("Triangulate Kinect");
    _triangulateIrKinect->setCallback(this);
    _toggleButton0 = new MenuCheckbox("Button0", false);
    _toggleButton0->setCallback(this);
    _toggleNavigation = new MenuCheckbox("Toggle Navigation", false);
    _toggleNavigation->setCallback(this);
    _calibrateMenu->addItem(_calibrateIrMenu);
    _calibrateIrMenu->addItem(_toggleCalibrate);
    _calibrateIrMenu->addItem(_skeletonCalibrate);
    _calibrateIrMenu->addItem(_kinectCreateIrPoint);
    //_calibrateIrMenu->addItem(_eraseAllSelectPoints);
    _calibrateIrMenu->addItem(_triangulateIrKinect);
    _calibrateIrMenu->addItem(_kinectPC1);
    _calibrateIrMenu->addItem(_kinectPC2);
    _calibrateIrMenu->addItem(_kinectPC3);
    _calibrateIrMenu->addItem(_kinectPC4);
    _calibrateIrMenu->addItem(_kinectIrTransformed);
    _calibrateMenu->addItem(_calibrateRefMenu);
    _calibrateRefMenu->addItem(_toggleRefCalibrate);
    _calibrateRefMenu->addItem(_kinectCreateSelectPoint);
    _calibrateRefMenu->addItem(_eraseAllSelectPoints);
    _calibrateRefMenu->addItem(_triangulateKinect);
    _calibrateRefMenu->addItem(_kinectRef);
    _calibrateRefMenu->addItem(_kinectTransformed);
    _calibrateMenu->addItem(_showRefPoints);
    _calibrateMenu->addItem(_showIRPoints);
    _calibrateMenu->addItem(_toggleNavigation);
    _avMenu->addItem(_calibrateMenu);
    loadScreensMenu();
    SceneManager::instance()->getObjectsRoot()->addChild(_root);

    if (useKinect) kinectInit();

    std::cerr << "KinectDemo initialized\n";
    return true;
}

void KinectDemo::menuCallback(MenuItem* menuItem)
{
    if (menuItem == _kinectOn)
    {
        if (useKinect) kinectOff();
        else kinectInit();

        useKinect = !useKinect;
    }

    if (menuItem == _kColorFPS)
    {
        colorfps = _kColorFPS->getValue();
    }

    if (menuItem == _kMoveWithCam)
    {
        if (_kMoveWithCam->getValue())
        {
            moveWithCamOn();
            kMoveWithCam = true;
        }
        else
        {
            moveWithCamOff();
            kMoveWithCam = false;
        }
    }

    if (menuItem == _kFreezeCloud)
    {
        if (_kFreezeCloud->getValue())
        {
            kFreezeCloud = true;
        }
        else
        {
            kFreezeCloud = false;
        }
    }

    if (menuItem == _kShowPCloud)
    {
        if (_kShowPCloud->getValue())
        {
            kShowPCloud = true;
            cloudOn();
        }
        else
        {
            kShowPCloud = false;
            cloudOff();
        }
    }

    if (menuItem == _kShowDepth)
    {
        if (_kShowDepth->getValue())
        {
            depthOn();
            kShowDepth = true;
        }
        else
        {
            depthOff();
            kShowDepth = false;
        }
    }

    if (menuItem == _kShowColor)
    {
        if (_kShowColor->getValue())
        {
            colorOn();
            kShowColor = true;
        }
        else
        {
            colorOff();
            kShowColor = false;
        }
    }

    if (menuItem == _switchMasterSkeleton)
    {
        masterKinect++;
    }

    if (menuItem == _kColorOn)
    {
        if (_kColorOn->getValue())
        {
            for (int i = 0; i < kinects->size(); i++)
            {
                kinects->at(i)->cm->useKColor = true;
            }
        }
        else
        {
            for (int i = 0; i < kinects->size(); i++)
            {
                kinects->at(i)->cm->useKColor = false;
            }
        }
    }

    if (menuItem == _kUserColorOn)
    {
        if (_kUserColorOn->getValue())
        {
            for (int i = 0; i < kinects->size(); i++)
            {
                kinects->at(i)->cm->userColor = true;
            }
        }
        else
        {
            for (int i = 0; i < kinects->size(); i++)
            {
                kinects->at(i)->cm->userColor = false;
            }
        }
    }

    if (menuItem == _bookmarkLoc)
    {
        Matrixd camMat = PluginHelper::getObjectMatrix();
        float cscale = PluginHelper::getObjectScale();
        Vec3 camTrans = camMat.getTrans();
        Quat camQuad = camMat.getRotate();
        cerr << "Saved camera position: " << endl;
        cerr << cscale << ", " << (camTrans.x() / cscale) << ", " << (camTrans.y() / cscale) << ", " << (camTrans.z() / cscale) << ", " << camQuad.x() << ", " << camQuad.y() << ", " << camQuad.z() << ", " << camQuad.w() << endl;
        bool savePointCloud = true;
        //        if (savePointCloud/* && cloud_socket*/)
        //        {
        //            ExportPointCloud();
        //        }
    }

    if (menuItem == _skeletonCalibrate)
    {
        CalibrateKinect* calibrateTool = new CalibrateKinect();
        Matrix headMat = PluginHelper::getHeadMat(0);
        Matrix handMat = PluginHelper::getHandMat(0);

        if (kinects != NULL)
        {
            std::vector<std::map<int, Skeleton>* > skeletonSensorsArray;

            for (int kinect_id = 0; kinect_id < kinects->size(); kinect_id++)
            {
                std::map<int, Skeleton>* skel_map = kinects->at(kinect_id)->skeletonGetMap();

                if (skel_map != NULL)
                {
                    skeletonSensorsArray.push_back(skel_map);
                }
            }

            calibrateTool->calibrateFromSkeletons(headMat, handMat, skeletonSensorsArray);
        }
    }

    if (menuItem == _kinectPC1)
    {
        if (kinects->size() > 0)
        {
            string filename = cvr::ConfigManager::getEntry("Plugin.KinectDemo.3DModelFolder");
            CalibrateKinect* calibrateTool = new CalibrateKinect();

            if (_kinectCloud1.group == NULL && _kinectPC1->getValue())
            {
                filename.append("kinectCalibrateIr_K0_P1.ply");
                _kinectCloud1 = calibrateTool->loadKinectCloud(filename);

                if (_kinectCloud1.group != NULL)
                {
                    if (_kinectCloud1.group->getNumChildren() > 0)
                    {
                        kinects->at(0)->switchNode->addChild(_kinectCloud1.group);
                    }
                }
            }
            else if (_kinectPC1->getValue())
            {
                if (_kinectCloud1.group->getNumChildren() > 0)
                {
                    kinects->at(0)->switchNode->addChild(_kinectCloud1.group);
                }
            }
            else
            {
                kinects->at(0)->switchNode->removeChild(_kinectCloud1.group);
            }
        }
    }

    if (menuItem == _kinectPC2)
    {
        if (kinects->size() > 0)
        {
            string filename = cvr::ConfigManager::getEntry("Plugin.KinectDemo.3DModelFolder");
            CalibrateKinect* calibrateTool = new CalibrateKinect();

            if (_kinectCloud2.group == NULL && _kinectPC2->getValue())
            {
                filename.append("kinectCalibrateIr_K0_P2.ply");
                _kinectCloud2 = calibrateTool->loadKinectCloud(filename);

                if (_kinectCloud2.group != NULL)
                {
                    if (_kinectCloud2.group->getNumChildren() > 0)
                    {
                        kinects->at(0)->switchNode->addChild(_kinectCloud2.group);
                    }
                }
            }
            else if (_kinectPC1->getValue())
            {
                if (_kinectCloud2.group->getNumChildren() > 0)
                {
                    kinects->at(0)->switchNode->addChild(_kinectCloud2.group);
                }
            }
            else
            {
                kinects->at(0)->switchNode->removeChild(_kinectCloud2.group);
            }
        }
    }

    if (menuItem == _kinectPC3)
    {
        if (kinects->size() > 0)
        {
            string filename = cvr::ConfigManager::getEntry("Plugin.KinectDemo.3DModelFolder");
            CalibrateKinect* calibrateTool = new CalibrateKinect();

            if (_kinectCloud3.group == NULL && _kinectPC3->getValue())
            {
                filename.append("kinectCalibrateIr_K0_P3.ply");
                _kinectCloud3 = calibrateTool->loadKinectCloud(filename);

                if (_kinectCloud3.group != NULL)
                {
                    if (_kinectCloud3.group->getNumChildren() > 0)
                    {
                        kinects->at(0)->switchNode->addChild(_kinectCloud3.group);
                    }
                }
            }
            else if (_kinectPC3->getValue())
            {
                if (_kinectCloud3.group->getNumChildren() > 0)
                {
                    kinects->at(0)->switchNode->addChild(_kinectCloud3.group);
                }
            }
            else
            {
                kinects->at(0)->switchNode->removeChild(_kinectCloud3.group);
            }
        }
    }

    if (menuItem == _kinectPC4)
    {
        if (kinects->size() > 0)
        {
            string filename = cvr::ConfigManager::getEntry("Plugin.KinectDemo.3DModelFolder");
            CalibrateKinect* calibrateTool = new CalibrateKinect();

            if (_kinectCloud4.group == NULL && _kinectPC4->getValue())
            {
                filename.append("kinectCalibrateIr_K0_P4.ply");
                _kinectCloud4 = calibrateTool->loadKinectCloud(filename);

                if (_kinectCloud4.group != NULL)
                {
                    if (_kinectCloud4.group->getNumChildren() > 0)
                    {
                        kinects->at(0)->switchNode->addChild(_kinectCloud4.group);
                    }
                }
            }
            else if (_kinectPC4->getValue())
            {
                if (_kinectCloud4.group->getNumChildren() > 0)
                {
                    kinects->at(0)->switchNode->addChild(_kinectCloud4.group);
                }
            }
            else
            {
                kinects->at(0)->switchNode->removeChild(_kinectCloud4.group);
            }
        }
    }

    if (menuItem == _kinectRef)
    {
        if (kinects->size() > 0)
        {
            string filename = cvr::ConfigManager::getEntry("Plugin.KinectDemo.3DModelFolder");
            CalibrateKinect* calibrateTool = new CalibrateKinect();

            if (_kinectCloudRef.group == NULL && _kinectRef->getValue())
            {
                filename.append("kinectCalibrateRef_K0_P1.ply");
                _kinectCloudRef = calibrateTool->loadKinectCloud(filename);

                if (_kinectCloudRef.group != NULL)
                {
                    if (_kinectCloudRef.group->getNumChildren() > 0)
                    {
                        kinects->at(0)->switchNode->addChild(_kinectCloudRef.group);
                    }
                }
            }
            else if (_kinectRef->getValue())
            {
                if (_kinectCloudRef.group->getNumChildren() > 0)
                {
                    kinects->at(0)->switchNode->addChild(_kinectCloudRef.group);
                }
            }
            else
            {
                kinects->at(0)->switchNode->removeChild(_kinectCloudRef.group);
            }
        }
    }

    if (menuItem == _kinectTransformed)
    {
        if (kinects->size() > 0)
        {
            string filename = cvr::ConfigManager::getEntry("Plugin.KinectDemo.3DModelFolder");
            CalibrateKinect* calibrateTool = new CalibrateKinect();

            if (_kinectCloudTransformed.group == NULL && _kinectTransformed->getValue())
            {
                filename.append("kinectCalibrate_transform.ply");
                _kinectCloudTransformed = calibrateTool->loadKinectCloud(filename);

                if (_kinectCloudTransformed.group != NULL)
                {
                    if (_kinectCloudTransformed.group->getNumChildren() > 0)
                    {
                        kinects->at(0)->switchNode->addChild(_kinectCloudTransformed.group);
                    }
                }
            }
            else if (_kinectTransformed->getValue())
            {
                if (_kinectCloudTransformed.group->getNumChildren() > 0)
                {
                    kinects->at(0)->switchNode->addChild(_kinectCloudTransformed.group);
                }
            }
            else
            {
                kinects->at(0)->switchNode->removeChild(_kinectCloudTransformed.group);
            }
        }
    }

    if (menuItem == _kinectIrTransformed)
    {
        if (kinects->size() > 0)
        {
            string filename = cvr::ConfigManager::getEntry("Plugin.KinectDemo.3DModelFolder");
            CalibrateKinect* calibrateTool = new CalibrateKinect();

            if (_kinectCloudIrTransformed.group == NULL && _kinectIrTransformed->getValue())
            {
                filename.append("kinectCalibrateIr_transform.ply");
                _kinectCloudIrTransformed = calibrateTool->loadKinectCloud(filename);

                if (_kinectCloudIrTransformed.group != NULL)
                {
                    if (_kinectCloudIrTransformed.group->getNumChildren() > 0)
                    {
                        kinects->at(0)->switchNode->addChild(_kinectCloudIrTransformed.group);
                    }
                }
            }
            else if (_kinectIrTransformed->getValue())
            {
                if (_kinectCloudIrTransformed.group->getNumChildren() > 0)
                {
                    kinects->at(0)->switchNode->addChild(_kinectCloudIrTransformed.group);
                }
            }
            else
            {
                kinects->at(0)->switchNode->removeChild(_kinectCloudIrTransformed.group);
            }
        }
    }

    if (menuItem == _eraseAllSelectPoints)
    {
        if (selectPoints.size() > 0)
        {
            int count = selectPoints.size() - 1;

            for (int i = count; i > -1; i--)
            {
                //Erase Each Geode
                kinects->at(0)->switchNode->removeChild(selectPoints[i]);
                selectPoints.pop_back();
            }
        }
    }

    if (menuItem == _triangulateKinect)
    {
        std::vector<osg::Vec3> kinectRefPoints;

        if (selectPoints.size() > 3)
        {
            for (int i = 0; i < 4; i++)
            {
                osg::Vec3 center = selectPoints[i]->getDrawable(0)->computeBound().center();
                // cerr << "Center:" << center.x() << " " << center.z() << "\n";
                kinectRefPoints.push_back(center);
            }
        }

        CalibrateKinect* calibrateTool = new CalibrateKinect();
        calibrateTool->triangulateKinect(0, kinectRefPoints, "ref");
    }

    if (menuItem == _triangulateIrKinect)
    {
        std::vector<osg::Vec3> kinectRefPoints;

        if (selectPoints.size() > 3)
        {
            for (int i = 0; i < 4; i++)
            {
                osg::Vec3 center = selectPoints[i]->getDrawable(0)->computeBound().center();
                // cerr << "Center:" << center.x() << " " << center.z() << "\n";
                kinectRefPoints.push_back(center);
            }
        }

        CalibrateKinect* calibrateTool = new CalibrateKinect();
        calibrateTool->triangulateKinect(0, kinectRefPoints, "ir");
    }

    for (int i = 0; i < screen_list.size(); i++)
    {
        if (menuItem == screen_list[i])
        {
            if (screen_list[i]->getValue())
            {
                if (screenGroup[i]->getNumChildren() == 0)
                {
                    osg::Group* sGroup;
                    CalibrateKinect* calibrateTool = new CalibrateKinect();
                    sGroup = calibrateTool->generateScreen(screen_path[i]);

                    if (sGroup != NULL)
                    {
                        screenGroup[i] = sGroup;
                        _root->addChild(screenGroup[i].get());
                    }
                }
                else
                {
                    _root->addChild(screenGroup[i].get());
                }
            }
            else
            {
                if (screenGroup[i]->getNumChildren() > 0)
                {
                    _root->removeChild(screenGroup[i].get());
                }
            }
        }
    }

    if (menuItem == _showRefPoints)
    {
        if (_showRefPoints->getValue())
        {
            CalibrateKinect* calibrateTool = new CalibrateKinect();
            refPointsGroup = calibrateTool->getPoints("Ref");

            if (refPointsGroup != NULL)
            {
                _root->addChild(refPointsGroup);
                cerr << "added\n";
            }
        }
        else
        {
            if (refPointsGroup != NULL)
            {
                _root->removeChild(refPointsGroup);
            }
        }
    }

    if (menuItem == _showIRPoints)
    {
        if (_showIRPoints->getValue())
        {
            CalibrateKinect* calibrateTool = new CalibrateKinect();
            irPointsGroup = calibrateTool->getPoints("Ir");

            if (irPointsGroup != NULL)
            {
                _root->addChild(irPointsGroup);
            }
        }
        else
        {
            if (irPointsGroup != NULL)
            {
                _root->removeChild(irPointsGroup);
            }
        }
    }

    if (menuItem == _toggleButton0)
    {
        if (_toggleButton0->getValue())
        {
            inputManager->buttonDown(0);
        }
        else
        {
            inputManager->buttonUp(0);
        }
    }

    if (menuItem == _toggleNavigation)
    {
        if (_toggleNavigation->getValue())
        {
            for (int i = 0; i < kinects->size(); i++)
            {
                kinects->at(i)->toggleNavigation(true);
            }
        }
        else
        {
            for (int i = 0; i < kinects->size(); i++)
            {
                kinects->at(i)->toggleNavigation(true);
            }
        }
    }

    if (menuItem == _testInteract)
    {
        sendEvents();
    }

    if (menuItem == sliderX)
    {
        kinectX = sliderX->getValue();
    }

    if (menuItem == sliderY)
    {
        kinectY = sliderY->getValue();
    }

    if (menuItem == sliderZ)
    {
        kinectZ = sliderZ->getValue();
    }

    if (menuItem == slider2X)
    {
        kinect2X = slider2X->getValue();
    }

    if (menuItem == slider2Y)
    {
        kinect2Y = slider2Y->getValue();
    }

    if (menuItem == slider2Z)
    {
        kinect2Z = slider2Z->getValue();
    }
}

void KinectDemo::preFrame()
{
    if (!useKinect || !kinectInitialized) return;

    if (kinects != NULL && kinects->size() > 0)
    {
        for (int i = 0; i < kinects->size(); i++)
        {
            if (kinects->at(i)->_firstRun)
            {
                kinects->at(i)->cloudUpdate();
            }

            kinects->at(i)->skeletonUpdate();
            bcounter = ++bcounter % (int)colorfps;

            if (bcounter == (int)colorfps - 1)
            {
                kinects->at(i)->cameraUpdate();
                kinects->at(i)->depthUpdate();
            }
        }

        updateInfoPanel();
    }

    handleSkeleton();
    
}
void KinectDemo::handleSkeleton()
{
    if (!skeletonThreaded)
    {
        // for every skeleton in mapIdSkel - draw, navigation spheres, check intersection with objects
        //std::map< osg::ref_ptr<osg::Geode>, int >::iterator iter;
        CalibrateKinect* calibrateTool = new CalibrateKinect();

        for (int kinect_id = 0; kinect_id < kinects->size(); kinect_id++)
        {
            std::map<int, Skeleton>* skel_map = kinects->at(kinect_id)->skeletonGetMap();

            if (kinect_id == masterKinectServer)
            {
                int id = masterKinectServer;

                if (skel_map != NULL)
                {
                    if (inputManager != NULL)
                    {
                        Matrix kinectMat = kinects->at(id)->getTransform();
                        int count = 0;
                        checkSkelGesture(skel_map);
                        checkSkelMaster(skel_map);

                        if (kinects->at(id)->helmertSArray.size() > 0)
                        {
                            inputManager->updateSkeletonInteract(kinect_id, masterKinect, wandLockedToSkeleton, skel_map, kinectMat, kinects->at(id)->helmertTArray[count], kinects->at(id)->helmertMArray[count], kinects->at(id)->helmertSArray[count]);
                        }
                        else
                        {
                            Vec3 blankVec;
                            Matrix blankM;
                            float blankF = 0;
                            inputManager->updateSkeletonInteract(kinect_id, masterKinect, wandLockedToSkeleton, skel_map, kinectMat, blankVec, blankM, blankF);
                        }
                    }
                }
            }

            for (std::map<int, Skeleton>::iterator it = skel_map->begin(); it != skel_map->end(); ++it)
            {
                int sk_id = it->first;
                Skeleton* sk = &(it->second);

                /******One hand selection ********/
                if (sk->cylinder.attached == false)
                    checkHandsIntersections(sk_id, skel_map);

                //        cout << sk->joints[M_LHAND].position.x() << " lbusy " << sk->leftHandBusy << endl;
                Vec3 StartPoint = sk->joints[M_LHAND].position;
                Vec3 EndPoint = sk->joints[M_RHAND].position;
                // if cylinder would be >distanceMIN && <distanceMAX, draw it and check for collisions
                double distance = (StartPoint - EndPoint).length();
                float HAND_ELBOW_OFFSET = -0.15;

                //printf("Distance Cylinder: %g \n",distance);
                if (((sk->joints[9].position.z() - sk->joints[7].position.z() > HAND_ELBOW_OFFSET) && (sk->joints[15].position.z() - sk->joints[13].position.z() > HAND_ELBOW_OFFSET))) sk->cylinder.handsBeenAboveElbows = true;

                sk->cylinder.update(StartPoint, EndPoint);

                if (sk->cylinder.attached)
                {
                    bool detachC = false;

                    if (sk->cylinder.locked && ((sk->joints[9].position.z() - sk->joints[7].position.z() < HAND_ELBOW_OFFSET) && (sk->joints[15].position.z() - sk->joints[13].position.z() < HAND_ELBOW_OFFSET)))
                        detachC = true;

                    if (sk->cylinder.handsBeenAboveElbows && (distance > distanceMAX * 0.667  || distance < distanceMIN || detachC))
                    {
                        sk->cylinder.detach(kinects->at(kinect_id)->switchNode);

                        // unlock all the objects that were locked by this cylinder
                        for (int j = 0; j < selectableItems.size(); j++)
                        {
                            if (selectableItems[j].lock == 1000 * kinect_id + sk_id) selectableItems[j].unlock();
                        }

                        sk->cylinder.locked = false;
                    }
                    else
                    {
                        // if sk->leton's cylinder is locked to some object, do not lock any more
                        if (sk->cylinder.locked == false)
                        {
                            // for every selectable item, check if it intersects with the current cylinder
                            const osg::BoundingBox& bboxCyl = sk->cylinder.geode->getDrawable(0)->getBound();

                            for (int j = 0; j < selectableItems.size(); j++)
                            {
                                // fake sphere to easily calculate boundary
                                Vec3 center2 = Vec3(0, 0, 0) * (selectableItems[j].mt->getMatrix());
                                Box* fakeSphere = new Box(center2, selectableItems[j].scale);
                                ShapeDrawable* ggg2 = new ShapeDrawable(fakeSphere);
                                const osg::BoundingBox& fakeBbox = ggg2->getBound();

                                if (bboxCyl.intersects(fakeBbox) && selectableItems[j].lock == -1)
                                {
                                    sk->cylinder.locked = true;
                                    selectableItems[j].lockType = 0;
                                    selectableItems[j].lockTo(1000 * kinect_id + sk_id);
                                    sk->cylinder.prevVec = (sk->joints[M_LHAND].position - osg::Vec3d((StartPoint.x() + EndPoint.x()) / 2, (StartPoint.y() + EndPoint.y()) / 2, (StartPoint.z() + EndPoint.z()) / 2));
                                    sk->cylinder.handsBeenAboveElbows = false;
                                    break; // lock only one object
                                }
                            }
                        }
                    }
                }
                else
                {
                    // Cylinder is not attached
                    // Don't create a cylinder between hands if any of them is holding an object
                    if (sk->leftHandBusy == false && sk->rightHandBusy == false)
                    {
                        // CONDITIONS TO CREATE CYLINDER
                        if (distance < distanceMAX / 3 && distance > distanceMIN /*&& ((sk->joints[9].position.z() - sk->joints[7].position.z() > HAND_ELBOW_OFFSET) && (sk->joints[15].position.z() - sk->joints[13].position.z() > HAND_ELBOW_OFFSET))*/)
                        {
                            sk->cylinder.attach(kinects->at(kinect_id)->switchNode);
                        }
                    }
                }
            }

            // move all the objects that are locked to centers and rotate to centers rotation

            for (int j = 0; j < selectableItems.size(); j++)
            {
                SelectableItem sel = selectableItems[j];

                if (sel.lock == -1) continue;

                if (sel.lockType == 0)
                {
                    int cylinderId = 1000 * kinect_id + sel.lock;

                    if (skel_map->count(cylinderId) > 0)
                    {
                        Matrix rotMat0;
                        rotMat0.makeRotate(skel_map->at(cylinderId).cylinder.prevVec, skel_map->at(cylinderId).cylinder.currVec);
                        selectableItems[j].rt->postMult(rotMat0);
                        Matrix posMat;
                        posMat.setTrans(skel_map->at(cylinderId).cylinder.center);
                        selectableItems[j].mt->setMatrix(posMat);
                        double newscale = selectableItems[j].scale;
                        newscale *= (1 + ((skel_map->at(cylinderId).cylinder.length - skel_map->at(cylinderId).cylinder.prevLength) / (500 / 1.0)));

                        if (newscale < 1  / 1000.0) newscale = 1  / 1000.0;

                        selectableItems[j].setScale(newscale);
                    }
                }
                else if (sel.lockType != -1)
                {
                    if (skel_map->count(1000 * kinect_id + sel.lock) > 0)
                    {
                        // moving artifact by one hand
                        Matrix posMat;

                        if (sel.lockType == 1) posMat.setTrans(skel_map->at(1000 * kinect_id + sel.lock).joints[M_LHAND].position);
                        else if (sel.lockType == 2) posMat.setTrans(skel_map->at(1000 * kinect_id + sel.lock).joints[M_RHAND].position);
                        else cout << "Error - unknown type of a lock (" << sel.lockType << ") on an object" << endl;

                        selectableItems[j].mt->setMatrix(posMat);
                    }
                }
            }
        }
    }
}
void KinectDemo::kinectInit()
{
    // moving from points to spheres in kinect point cloud
    //bang
    skeletonThreaded = false;
    fixXY = false;
    ignoreZeros = false;
    filterBackground = false;
    assignPointsToSkeletons = false;
    classifyPoints = false;
    kinects = new std::vector<KinectObject*>();
    //Get KinectSkeleton Offset
    kinect2X =  kinectX = ConfigManager::getFloat("x", "Plugin.KinectDemo.KinectSkeleton", 0.0f);
    kinect2Y =  kinectY = ConfigManager::getFloat("y", "Plugin.KinectDemo.KinectSkeleton", 0.0f);
    kinect2Z =  kinectZ = ConfigManager::getFloat("z", "Plugin.KinectDemo.KinectSkeleton", 0.0f);
    kinect2RX =  kinectRX = ConfigManager::getFloat("rx", "Plugin.KinectDemo.KinectSkeleton", 0.0f);
    kinect2RY =  kinectRY = ConfigManager::getFloat("ry", "Plugin.KinectDemo.KinectSkeleton", 0.0f);
    kinect2RZ =  kinectRZ = ConfigManager::getFloat("rz", "Plugin.KinectDemo.KinectSkeleton", 0.0f);
    kinect2RW =  kinectRW = ConfigManager::getFloat("rw", "Plugin.KinectDemo.KinectSkeleton", 0.0f);
    //Show info Panel
    kShowInfoPanel = ConfigManager::getBool("Plugin.KinectDemo.KinectDefaultOn.ShowInfoPanel");

    if (kShowInfoPanel)
    {
        _kShowInfoPanel->setValue(true);
        _infoPanel->setVisible(true);
        sliderX = new MenuRangeValue("X", -1000.0, 1000.0, kinectX, 0.01);
        sliderX->setCallback(this);
        _infoPanel->addMenuItem(sliderX);
        sliderY = new MenuRangeValue("Y", -1000.0, 1000.0, kinectY, 0.01);
        sliderY->setCallback(this);
        _infoPanel->addMenuItem(sliderY);
        sliderZ = new MenuRangeValue("Z", -1000.0, 1000.0, kinectZ, 0.01);
        sliderZ->setCallback(this);
        _infoPanel->addMenuItem(sliderZ);
        slider2X = new MenuRangeValue("X-2", -1000.0, 1000.0, kinect2X, 0.01);
        slider2X->setCallback(this);
        _infoPanel->addMenuItem(slider2X);
        slider2Y = new MenuRangeValue("Y-2", -1000.0, 1000.0, kinect2Y, 0.01);
        slider2Y->setCallback(this);
        _infoPanel->addMenuItem(slider2Y);
        slider2Z = new MenuRangeValue("Z-2", -1000.0, 1000.0, kinect2Z, 0.01);
        slider2Z->setCallback(this);
        _infoPanel->addMenuItem(slider2Z);
        /*
              sliderRX = new MenuRangeValue("RX",-1000.0,1000.0,kinectRX,0.01);
              sliderRX->setCallback(this);
              _infoPanel->addMenuItem(sliderRX);

              sliderRY = new MenuRangeValue("RY",-1000.0,1000.0,kinectRY,0.01);
              sliderRY->setCallback(this);
              _infoPanel->addMenuItem(sliderRY);

              sliderRZ = new MenuRangeValue("RZ",-1000.0,1000.0,kinectRZ,0.01);
              sliderRZ->setCallback(this);
              _infoPanel->addMenuItem(sliderRZ);

              sliderRW = new MenuRangeValue("RW",-1000.0,1000.0,kinectRW,0.01);
              sliderRW->setCallback(this);
              _infoPanel->addMenuItem(sliderRW);
        */
    }

    float camX = ConfigManager::getFloat("x", "Plugin.KinectDemo.CamStart", 0.0f);
    float camY = ConfigManager::getFloat("y", "Plugin.KinectDemo.CamStart", 0.0f);
    float camZ = ConfigManager::getFloat("z", "Plugin.KinectDemo.CamStart", 0.0f);
    float camS = ConfigManager::getFloat("s", "Plugin.KinectDemo.CamStart", 0.0f);
    float camRX = ConfigManager::getFloat("rx", "Plugin.KinectDemo.CamStart", 0.0f);
    float camRY = ConfigManager::getFloat("ry", "Plugin.KinectDemo.CamStart", 0.0f);
    float camRZ = ConfigManager::getFloat("rz", "Plugin.KinectDemo.CamStart", 0.0f);
    float camRW = ConfigManager::getFloat("rw", "Plugin.KinectDemo.CamStart", 0.0f);

    //moveCam(273.923, 0.178878, -1.27967, 1.64388, -0.0247491, 0.294768, 0.952783, 0.0685912);
    //moveCam(camS, camX, camY, camZ, camRX, camRY, camRZ, camRW);

    if (!skeletonThreaded)
    {
        distanceMAX = ConfigManager::getFloat("Plugin.KinectDemo.Cylinder.Max");
        distanceMIN = ConfigManager::getFloat("Plugin.KinectDemo.Cylinder.Min");
        bcounter = 0;
        //    _modelFileNode1 = osgDB::readNodeFile(ConfigManager::getEntry("Plugin.KinectDemo.3DModelFolder").append("kinect_mm.obj"));
        _modelFileNode2 = osgDB::readNodeFile(ConfigManager::getEntry("Plugin.KinectDemo.3DModelFolder").append("dumptruck.osg"));////append("50563.ply"));//dumptruck.osg"));
        _modelFileNode5 = osgDB::readNodeFile(ConfigManager::getEntry("Plugin.KinectDemo.3DModelFolder").append("cessna.osg"));
        _modelFileNode4 = osgDB::readNodeFile(ConfigManager::getEntry("Plugin.KinectDemo.3DModelFolder").append("cow.osg"));
        _modelFileNode3 = osgDB::readNodeFile(ConfigManager::getEntry("Plugin.KinectDemo.3DModelFolder").append("robot.osg"));
        _sphereRadius = 0.07 * 500;
        //  Group* kinectgrp = new Group();
        //  kinectgrp->addChild(_modelFileNode3);
        // createSceneObject(kinectgrp);
        // createSelObj(Vec3(-0.70,  -2.0,  0.15),   string("DD"), 0.002, _modelFileNode1);
        //createSelObj(Vec3(0.00,  3000.0,  0.0),   string("FD"), 1, _modelFileNode2);
        createSelObj(Vec3(0.00,  3000.0,  0.0),   string("FD"), 10, _modelFileNode2);
        createSelObj(Vec3(-400,   3000.0,  0.15),   string("GD"), 20,  _modelFileNode3);
        createSelObj(Vec3(400,   3000.0,  0.15),   string("ED"), 20,  _modelFileNode4);
        createSelObj(Vec3(700,   3000.0,  0.15),   string("ZD"), 10, _modelFileNode5);
        //CreateSceneObject for all Kinect Data
        //createSceneObject();
        //createSceneObject2();
    }

    kinectInitialized = true;
    int num_kinects = ConfigManager::getInt("Plugin.KinectDemo.KinectServer.NumKinects");

    for (int i = 0; i < num_kinects; i++)
    {
        string cloud_server = ConfigManager::getEntry("Plugin.KinectDemo.KinectServer.PointCloud" + std::to_string((long long int)i + 1));
        string skeleton_server = ConfigManager::getEntry("Plugin.KinectDemo.KinectServer.Skeleton" + std::to_string((long long int)i + 1));
        string color_server = ConfigManager::getEntry("Plugin.KinectDemo.KinectServer.ColorMap" + std::to_string((long long int)i + 1));
        string depth_server = ConfigManager::getEntry("Plugin.KinectDemo.KinectServer.DepthMap" + std::to_string((long long int)i + 1));
        string name = std::to_string((long long int)(i));
        KinectObject* kinect = new KinectObject(name, cloud_server, skeleton_server, color_server, depth_server, osg::Vec3(0, 0, 0));
        PluginHelper::registerSceneObject(kinect, name);
        kinect->attachToScene();
        kinects->push_back(kinect);
    }

    inputManager = new InputManager();
    //inputManager->start();
    //   kinectTransform();
}

//Add as global class CalibrateKinect
//string name = "1";
//_calibraterTool = new CalibrateKinect(name);


void KinectDemo::createSelObj(osg::Vec3 pos, string color, float scalenumber, osg::Node* model)
{
    Matrixd scale;
    double snum = scalenumber;
    scale.makeScale(snum, snum, snum);
    MatrixTransform* modelScaleTrans = new MatrixTransform();
    modelScaleTrans->setMatrix(scale);
    modelScaleTrans->addChild(model);
    osg::Geode* boxGeode = new osg::Geode;
    MatrixTransform* rotate = new osg::MatrixTransform();
    Matrix rotMat;
    rotMat.makeRotate(0, 1, 0, 1);
    rotate->setMatrix(rotMat);
    MatrixTransform* translate = new osg::MatrixTransform();
    osg::Matrixd tmat;
    tmat.makeTranslate(pos);
    translate->setMatrix(tmat);
    translate->addChild(rotate);
    _root->addChild(translate);
    selectableItems.push_back(SelectableItem(boxGeode, modelScaleTrans, translate, rotate, snum));
}

void KinectDemo::cloudOff()
{
    for (int i = 0; i < kinects->size(); i++)
    {
        kinects->at(i)->cloudOff();
    }
}
void KinectDemo::depthOff()
{
    for (int i = 0; i < kinects->size(); i++)
    {
        kinects->at(i)->depthOff();
    }
}
void KinectDemo::colorOff()
{
    for (int i = 0; i < kinects->size(); i++)
    {
        kinects->at(i)->cameraOff();
    }
}
void KinectDemo::colorOn()
{
    for (int i = 0; i < kinects->size(); i++)
    {
        kinects->at(i)->cameraOn();
    }
}
void KinectDemo::depthOn()
{
    for (int i = 0; i < kinects->size(); i++)
    {
        kinects->at(i)->depthOn();
    }
}

void KinectDemo::cloudOn()
{
    for (int i = 0; i < kinects->size(); i++)
    {
        kinects->at(i)->cloudOn();
    }
}


void KinectDemo::moveWithCamOff()
{
    Skeleton::moveWithCam = false;
}
void KinectDemo::moveWithCamOn()
{
    Skeleton::moveWithCam = true;
}

void KinectDemo::moveCam(double bscale, double x, double y, double z, double o1, double o2, double o3, double o4)
{
    Vec3 trans = Vec3(x, y, z) * bscale;
    Matrix tmat;
    tmat.makeTranslate(trans);
    Matrix rot;
    rot.makeRotate(osg::Quat(o1, o2, o3, o4));
    Matrixd gotoMat = rot * tmat;
    Matrixd camMat = PluginHelper::getObjectMatrix();
    float cscale = PluginHelper::getObjectScale();
    Vec3 camTrans = camMat.getTrans();
    Quat camQuad = camMat.getRotate();
    PluginHelper::setObjectMatrix(gotoMat);
    PluginHelper::setObjectScale(bscale);
}

void KinectDemo::kinectOff()
{
    //    TrackingManager::instance()->setUpdateHeadTracking(true);
    printf("turning kinect off\n");

    for (int i = 0; i < selectableItems.size(); i++)
        _root->removeChild(selectableItems[i].mt);

    cloudOff();
    colorOff();
    depthOff();
    _kShowColor->setValue(false);
    _kShowDepth->setValue(false);
    _kShowPCloud->setValue(false);
    this->menuCallback(_kShowColor);
    this->menuCallback(_kShowDepth);
    this->menuCallback(_kShowPCloud);
    selectableItems.clear();

    for (int i = 0; i < kinects->size(); i++)
    {
        KinectObject* k = kinects->at(i);
        k->skeletonOff();
        delete k;
    }

    kinects->clear();
}

void KinectDemo::checkHandsIntersections(int skel_id, std::map<int, Skeleton>* skel_map)
{
    Skeleton* skel = &skel_map->at(skel_id);
    Sphere* handSphere = new Sphere(skel->joints[M_LHAND].position, 50);
    ShapeDrawable* ggg3 = new ShapeDrawable(handSphere);
    const osg::BoundingBox& bboxHandL = ggg3->getBound();
    handSphere = new Sphere(skel->joints[M_RHAND].position, 50);
    ggg3 = new ShapeDrawable(handSphere);
    const osg::BoundingBox& bboxHandR = ggg3->getBound();

    for (int j = 0; j < selectableItems.size(); j++)
    {
        Vec3 center2 = Vec3(0, 0, 0) * (selectableItems[j].mt->getMatrix());
        Box* fakeSphere = new Box(center2, 25 * sqrt(selectableItems[j].scale));
        ShapeDrawable* ggg2 = new ShapeDrawable(fakeSphere);
        const osg::BoundingBox& fakeBbox = ggg2->getBound();

        if (selectableItems[j].lock == skel_id && (selectableItems[j].lockType == 1))
        {
            if (bboxHandL.intersects(fakeBbox) == false)
            {
                selectableItems[j].unlock();
                skel->leftHandBusy = false;
            }
        }

        if (selectableItems[j].lock == skel_id && (selectableItems[j].lockType == 2))
        {
            if (bboxHandR.intersects(fakeBbox) == false)
            {
                selectableItems[j].unlock();
                skel->rightHandBusy = false;
            }
        }
    }

    for (int j = 0; j < selectableItems.size(); j++)
    {
        // fake sphere to easily calculate boundary
        Vec3 center2 = Vec3(0, 0, 0) * (selectableItems[j].mt->getMatrix());
        Box* fakeSphere = new Box(center2, 25 * sqrt(selectableItems[j].scale));
        ShapeDrawable* ggg2 = new ShapeDrawable(fakeSphere);
        const osg::BoundingBox& fakeBbox = ggg2->getBound();

        if (bboxHandL.intersects(fakeBbox) && selectableItems[j].lock == -1 && skel->leftHandBusy == false)
        {
            cout << "_locking left" << endl;
            selectableItems[j].lockTo(skel_id);
            selectableItems[j].lockType = 1;
            skel->leftHandBusy = true;
            break;
        }

        if (bboxHandR.intersects(fakeBbox) && selectableItems[j].lock == -1 && skel->rightHandBusy == false)
        {
            cout << "_locking right" << endl;
            selectableItems[j].lockTo(skel_id);
            selectableItems[j].lockType = 2;
            skel->rightHandBusy = true;
            break;
        }
    }
}

void KinectDemo::updateInfoPanel()
{
    std::stringstream ss;

    if (false)
    {
        for (int i = 0; i < kinects->size(); i++)
        {
            for (int n = 0; n < 3; n++)
            {
                float radius = kinects->at(i)->cm->userRadius[n];

                if (!kinects->at(i)->_firstRun)
                {
                    int count = kinects->at(i)->cm->userVerticesArray[n]->size();
                    int count2 = kinects->at(i)->cm->lHandVerticesArray[n]->size();
                    float radius = kinects->at(i)->cm->userRadius[n];
                    ss << "User" << n << "\n" << "Radius:" << count << " " << count2  << "\n";
                }
            }
        }
    }

    if (oldMasterKinect != masterKinect)
    {
        oldMasterKinect = masterKinect;
        ss << "User " << oldMasterKinect << " is Master\n";
        _infoPanel->updateTabWithText("Info", ss.str());
    }
}


bool KinectDemo::processEvent(InteractionEvent* event)
{
    TrackedButtonInteractionEvent* tie = event->asTrackedButtonEvent();

    if ((event->getInteraction() == BUTTON_DOWN) && tie->getHand() == 0)
    {
        //For Testing Simulated Hand Interaction
        // cerr << "Hand Button " << tie->getButton() << "\n";
        // inputManager->buttonUp(tie->getButton());
    }

    if ((event->getInteraction() == BUTTON_DOWN) && tie->getHand() == 1)
    {
        //For Testing Simulated Hand Interaction
        if (inputManager != NULL)
        {
            // inputManager->buttonDown(tie->getButton());
        }
    }

    if ((event->getInteraction() == BUTTON_DOWN) && _toggleCalibrate->getValue())
    {
        //Matrix handMat = PluginHelper::getHandMat(tie->getHand());
        Matrix handMat = PluginHelper::getHeadMat(0);
        cerr << "Hand " << tie->getHand() << " Button 0\n";
        calibCount++;

        if (kShowPCloud)
        {
            std::vector<Vec3Array*> kinectArrayVert;
            std::vector<Vec4Array*> kinectArrayColor;
            int count;

            if (false)
            {
                count = kinects->size();
            }
            else
            {
                count = 1;
            }

            if (kinects != NULL)
            {
                for (int i = 0; i < count; i++)
                {
                    cout << "KINECT FIRSTRUN " << kinects->at(i)->_firstRun << endl;

                    if (true || !kinects->at(i)->_firstRun)
                    {
                        cout << "SIZE " << kinects->at(i)->cm->userVerticesArray[1]->size() << endl;
                        kinectArrayVert.push_back(kinects->at(i)->cm->userVerticesArray[1]);
                        kinectArrayColor.push_back(kinects->at(i)->cm->userColoursArray[1]);
                    }
                }
            }

            CalibrateKinect* calibrateTool = new CalibrateKinect();
            calibrateTool->calibrateFromHand(calibCount, handMat, kinectArrayVert, kinectArrayColor);
        }

        if (calibCount == 4)
        {
            calibCount = 0;
            _toggleCalibrate->setValue(false);
        }
    }

    if ((event->getInteraction() == BUTTON_DOWN) && _toggleRefCalibrate->getValue())
    {
        Matrix handMat = PluginHelper::getHandMat(tie->getHand());
        cerr << "Hand " << tie->getHand() << " Button 0\n";
        calibCount++;

        if (kShowPCloud)
        {
            std::vector<Vec3Array*> kinectArrayVert;
            std::vector<Vec4Array*> kinectArrayColor;
            int count;

            if (false)
            {
                count = kinects->size();
            }
            else
            {
                count = 1;
            }

            if (kinects != NULL)
            {
                for (int i = 0; i < count; i++)
                {
                    if (!kinects->at(i)->_firstRun)
                    {
                        kinectArrayVert.push_back(kinects->at(i)->cm->userVerticesArray[0]);
                        kinectArrayColor.push_back(kinects->at(i)->cm->userColoursArray[0]);
                    }
                }
            }

            CalibrateKinect* calibrateTool = new CalibrateKinect();
            calibrateTool->calibrateFromRef(kinectArrayVert, kinectArrayColor);
        }

        calibCount = 0;
        _toggleRefCalibrate->setValue(false);
    }

    if ((event->getInteraction() == BUTTON_DOWN) && (_kinectCreateSelectPoint->getValue() || _kinectCreateIrPoint->getValue()))
    {
        osg::Matrix handMat = tie->getTransform();
        CalibrateKinect* calibrateTool = new CalibrateKinect();
        osg::Geode* newSelectPoint;
        bool found = false;

        if (_kinectCloud1.group == NULL && _kinectCloud2.group == NULL && _kinectCloud3.group == NULL && _kinectCloud4.group == NULL && _kinectCloudRef.group == NULL)
        {
            cerr << "Please turn on a Ref Point Cloud\n";
        }
        else
        {
            osg::Geode* points;
            bool foundP = false;

            if (_kinectCloud1.group != NULL && _kinectPC1->getValue())
            {
                points = _kinectCloud1.group->getChild(0)->asGeode();
                foundP = true;
            }
            else if (_kinectCloud2.group != NULL && _kinectPC2->getValue())
            {
                points = _kinectCloud2.group->getChild(0)->asGeode();
                foundP = true;
            }
            else if (_kinectCloud3.group != NULL && _kinectPC3->getValue())
            {
                points = _kinectCloud3.group->getChild(0)->asGeode();
                foundP = true;
            }
            else if (_kinectCloud4.group != NULL && _kinectPC4->getValue())
            {
                points = _kinectCloud4.group->getChild(0)->asGeode();
                foundP = true;
            }
            else if (_kinectCloudRef.group != NULL && _kinectRef->getValue())
            {
                points = _kinectCloudRef.group->getChild(0)->asGeode();
                foundP = true;
            }

            if (foundP)
            {
                osg::Geometry* nodeGeom = points->getDrawable(0)->asGeometry();
                osg::Vec3Array* vecPoints = dynamic_cast<Vec3Array*>(nodeGeom->getVertexArray());
                osg::Vec3 currentPos = calibrateTool->findBestSelectedPoint(handMat, vecPoints);
                cerr << "Found Point:" << currentPos.x() << " " << currentPos.y() << " " << currentPos.z() << "\n";

                if (currentPos.x() == 0)
                {
                    found = false;
                }
                else
                {
                    newSelectPoint = calibrateTool->createSelectSphere(currentPos);
                    found = true;
                }
            }
        }

        if (kinects->size() > 0 && found)
        {
            cerr << "Found!\n";
            kinects->at(0)->switchNode->addChild(newSelectPoint);
            selectPoints.push_back(newSelectPoint);
        }

        _kinectCreateSelectPoint->setValue(false);
        _kinectCreateIrPoint->setValue(false);
    }

    return false;
}
void KinectDemo::loadScreensMenu()
{
    SubMenu* screensMenu = new SubMenu("Screens Menu");
    _avMenu->addItem(screensMenu);
    string directory = cvr::ConfigManager::getEntry("Plugin.KinectDemo.ScreenConfigLocation");
    DIR* dir;
    class dirent* ent;
    class stat st;
    dir = opendir(directory.c_str());

    while ((ent = readdir(dir)) != NULL) {
        const string file_name = ent->d_name;
        const string full_file_name = directory + "/" + file_name;

        if (file_name[0] == '.')
            continue;

        if (stat(full_file_name.c_str(), &st) == -1)
            continue;

        const bool is_directory = (st.st_mode & S_IFDIR) != 0;

        if (is_directory)
            continue;

        string check = file_name;
        int found = check.find("creen");

        if (found >= 0)
        {
        }
        else
        {
            continue;
        }

        MenuCheckbox* b = new MenuCheckbox(file_name, false); //"test " + i);
        screensMenu->addItem(b);
        b->setCallback(this);
        screen_list.push_back(b);
        screen_path.push_back(full_file_name);
        screenGroup.push_back(new osg::Group);
    }

    closedir(dir);
}
void KinectDemo::checkSkelMaster(std::map<int, Skeleton>* skel_map)
{
    //Using tempKinect because don't want to update skeleton until sure it is attached
    bool skelFound = false;

    for (std::map<int, Skeleton>::iterator it = skel_map->begin(); it != skel_map->end(); ++it)
    {
        int sk_id = it->first;

        if (masterKinect == sk_id)
        {
            //cerr << "Found SkeletonInt\n";
            Skeleton* sk = &(it->second);
            bool attached = sk->attached;

            if (attached)
            {
                skelFound = true;
                break;
            }
        }
    }

    if (!skelFound)
    {
        int firstId = -1;
        bool mKhigher = false;

        for (std::map<int, Skeleton>::iterator it = skel_map->begin(); it != skel_map->end(); ++it)
        {
            int sk_id = it->first;

            if (firstId == -1)
            {
                firstId = sk_id;
            }

            if (masterKinect > sk_id)
            {
                mKhigher = true;
            }
            else
            {
                masterKinect = sk_id;
                mKhigher = false;
                break;
            }
        }

        if (mKhigher)
        {
            masterKinect = firstId;
        }
    }
}
void KinectDemo::checkSkelGesture(std::map<int, Skeleton>* skel_map)
{
    for (std::map<int, Skeleton>::iterator it = skel_map->begin(); it != skel_map->end(); ++it)
    {
        int sk_id = it->first;

        if (sk_id == masterKinect)
        {
            Skeleton* sk = &(it->second);
            Vec3 lHand = sk->joints[M_LHAND].position;
            Vec3 rHand = sk->joints[M_RHAND].position;
            Vec3 head = sk->joints[M_HEAD].position;
            gestureSurrender(lHand, rHand, head);
            gestureLeftClick(lHand, rHand, head);
            gestureRightClick(lHand, rHand, head);
            break;
        }
    }
}
void KinectDemo::gestureSurrender(osg::Vec3 lHand, osg::Vec3 rHand, osg::Vec3 head)
{
    float offset = 200.0;
    head += Vec3(0, 0, offset);

    if (rHand.z() > head.z() && lHand.z() > head.z())
    {
        masterKinect++;
        //cerr << "inc " << masterKinect << "\n";
    }
}
void KinectDemo::gestureLeftClick(osg::Vec3 lHand, osg::Vec3 rHand, osg::Vec3 head)
{
    float offset = 100.0;
    head -= Vec3(0, offset, 0);

    if (lHand.y() < head.y() && lHand.z() < head.z())
    {
        if (!buttonDown)
        {
            buttonDown = true;
            //cerr << "ButtonDown\n";
            inputManager->buttonDown(0);
        }
    }

    if (lHand.y() > head.y() || lHand.z() > head.z())
    {
        if (buttonDown)
        {
            buttonDown = false;
            //cerr << "ButtonUp\n";
            inputManager->buttonUp(0);
        }
    }
}
void KinectDemo::gestureRightClick(osg::Vec3 lHand, osg::Vec3 rHand, osg::Vec3 head)
{
    float offset = 400.0;
    float offset2 = 200.0;
    head += Vec3(offset, 0, 0);
    head -= Vec3(0, 0, offset2);

    if (lHand.x() > head.x() && lHand.y() > head.y())
    {
        if (!rightButtonDown)
        {
            rightButtonDown = true;
            //cerr << "RightButtonDown\n";
            inputManager->buttonDown(1);
        }
    }

    if (lHand.x() < head.x())
    {
        if (rightButtonDown)
        {
            rightButtonDown = false;
            //cerr << "RightButtonUp\n";
            inputManager->buttonUp(1);
        }
    }
}

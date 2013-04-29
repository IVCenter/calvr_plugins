#include "SkeletonManager.h"

#include <cvrKernel/ComController.h>
#include <sstream>
#include <algorithm>
#include <cstdio>

#include <sys/syscall.h>
#include <sys/stat.h>

//#include "Skeleton.h"
using namespace cvr;
using namespace std;
using namespace osg;

SkeletonManager::SkeletonManager()
{
    should_quit = false;
    sf2 = new RemoteKinect::SkeletonFrame();
    kNavSpheres = false;
    kMoveWithCam = false;
    _root = new osg::MatrixTransform();
    SceneManager::instance()->getObjectsRoot()->addChild(_root);
    //   _root->addChild(kinectgrp);
}

SkeletonManager::~SkeletonManager()
{
}

bool SkeletonManager::isCacheDone()
{
    return _cacheDone;
}

void SkeletonManager::update()
{
    static bool frameLoading = false;
    bool cDone;

    if (true)
    {
        if (cvr::ComController::instance()->isMaster())
        {
            cDone = _cacheDone;
            int numSlaves = cvr::ComController::instance()->getNumSlaves();
            bool sDone[numSlaves];
            cvr::ComController::instance()->readSlaves(sDone, sizeof(bool));

            for (int i = 0; i < numSlaves; i++)
            {
                cDone = cDone && sDone[i];
            }

            cvr::ComController::instance()->sendSlaves(&cDone, sizeof(bool));
        }
        else
        {
            cDone = _cacheDone;
            cvr::ComController::instance()->sendMaster(&cDone, sizeof(bool));
            cvr::ComController::instance()->readMaster(&cDone, sizeof(bool));
        }

        if (!cDone)
        {
            //std::cerr << "Waiting for load to finish." << std::endl;
            return;
        }

        cvr::ComController::instance()->sync();
        //Add not here?
        return;
    }

    if (cDone)
    {
        std::cerr << "Load Finished." << std::endl;
        //Add loaded node to root
    }
    else
    {
        //  std::cerr << "Waiting for GPU load finish." << std::endl;
    }
}

void SkeletonManager::run()
{
    //Do functions
    //cerr << ".";
    if (!should_quit)
    {
        zmq::context_t context3(1);
        skelT_socket = NULL;
        skelT_socket = new SubSocket<RemoteKinect::SkeletonFrame> (context3, ConfigManager::getEntry("Plugin.KinectDemo.KinectServer.Skeleton"));

        while (!should_quit)
        {
            if (Skeleton::moveWithCam)
            {
                Matrix camMat = PluginHelper::getWorldToObjectTransform(); //This will get us actual real world coordinates that the camera is at (not sure about how it does rotation)
                float cscale = 1; //Want to keep scale to actual Kinect which is is meters
                Vec3 camTrans = camMat.getTrans();
                Quat camQuad = camMat.getRotate();  //Rotation of cam will cause skeleton to be off center--need Fix!!
                double xOffset = (camTrans.x() / cscale);
                double yOffset = (camTrans.y() / cscale) + 3; //Added Offset of Skeleton so see a little ways from camera (i.e. 5 meters, works at this scale,only)
                double zOffset = (camTrans.z() / cscale);
                Skeleton::camPos = Vec3d(xOffset, yOffset, zOffset);
                Skeleton::camRot = camQuad;
            }

            while (skelT_socket->recv(*sf2))
            {
                //    cerr << ".";

                //return;
                // remove all the skeletons that are no longer reported by the server
                for (std::map<int, Skeleton>::iterator it2 = mapIdSkel.begin(); it2 != mapIdSkel.end(); ++it2)
                {
                    bool found = false;

                    for (int i = 0; i < sf2->skeletons_size(); i++)
                    {
                        if (sf2->skeletons(i).skeleton_id() == it2->first)
                        {
                            found = true;
                            break;
                        }
                    }

                    if (!found)
                    {
                        mapIdSkel[it2->first].detach(_root);
                    }
                }

                //cerr << sf->skeletons_size();

                // update all skeletons' joints' positions
                for (int i = 0; i < sf2->skeletons_size(); i++)
                {
                    // Skeleton reported but not in the map -> create a new one
                    if (mapIdSkel.count(sf2->skeletons(i).skeleton_id()) == 0)
                    {
                        mapIdSkel[sf2->skeletons(i).skeleton_id()] = Skeleton(); ///XXX remove Skeleton(); part
                        mapIdSkel[sf2->skeletons(i).skeleton_id()].attach(_root);
                    }

                    // Skeleton previously detached (stopped being reported), but is again reported -> reattach
                    if (mapIdSkel[sf2->skeletons(i).skeleton_id()].attached == false)
                        mapIdSkel[sf2->skeletons(i).skeleton_id()].attach(_root);

                    for (int j = 0; j < sf2->skeletons(i).joints_size(); j++)
                    {
                        float test;
                        test = sf2->skeletons(i).joints(j).type();
                        test = sf2->skeletons(i).joints(j).x() / 1000;
                        test = sf2->skeletons(i).joints(j).z() / -1000;
                        test = sf2->skeletons(i).joints(j).y() / 1000;
                        test = sf2->skeletons(i).joints(j).qx();
                        test = sf2->skeletons(i).joints(j).qz();
                        test = sf2->skeletons(i).joints(j).qy();
                        test = sf2->skeletons(i).joints(j).qw();
                        /*
                                        mapIdSkel[sf2->skeletons(i).skeleton_id()].update(
                                            sf2->skeletons(i).joints(j).type(),
                                            sf2->skeletons(i).joints(j).x() / 1000,
                                            sf2->skeletons(i).joints(j).z() / -1000,
                                            sf2->skeletons(i).joints(j).y() / 1000,
                                            sf2->skeletons(i).joints(j).qx(),
                                            sf2->skeletons(i).joints(j).qz(),
                                            sf2->skeletons(i).joints(j).qy(),
                                            sf2->skeletons(i).joints(j).qw());
                        */
                    }
                }
            }
        }

        if (should_quit)
        {
        }
    }

    //When Finished
    _cacheDone = true;
    std::cerr << "All frames loaded." << std::endl;
}

void  SkeletonManager::quit()
{
    should_quit = true;
}


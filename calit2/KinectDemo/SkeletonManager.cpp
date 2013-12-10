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

SkeletonManager::SkeletonManager(std::string server)
{
    should_quit = false;
    //sf2 = new RemoteKinect::SkeletonFrame();
   // kNavSpheres = false;
   // kMoveWithCam = false;
   // _root = new osg::MatrixTransform();
    //   _root->addChild(kinectgrp);
    switchNode = new osg::Switch;
    SceneManager::instance()->getObjectsRoot()->addChild(switchNode);
    skeletonServer = server;
    context = new zmq::context_t(1);
    skel_frame = new RemoteKinect::SkeletonFrame();
    skel_socket = new SubSocket<RemoteKinect::SkeletonFrame>(*context, skeletonServer);
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
}

void SkeletonManager::run()
{
  while (!should_quit)
  {
    while (skel_socket->recv(*skel_frame))
    {
           // cerr << "+";
        // remove all the skeletons that are no longer reported by the server
        for (std::map<int, Skeleton>::iterator it2 = mapIdSkel.begin(); it2 != mapIdSkel.end(); ++it2)
        {
            bool found = false;

            for (int i = 0; i < skel_frame->skeletons_size(); i++)
            {
                if (skel_frame->skeletons(i).skeleton_id() == it2->first)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                mapIdSkel[it2->first].detach(switchNode);
            }
        }

        //if (skel_frame->skeletons_size() > 0)

            //  cerr << "Skels:" << skel_frame->skeletons_size() << "\n";
            // update all skeletons' joints' positions
            for (int i = 0; i < skel_frame->skeletons_size(); i++)
            {
                // Skeleton reported but not in the map -> create a new one
                if (mapIdSkel.count(skel_frame->skeletons(i).skeleton_id()) == 0)
                {
                    mapIdSkel[skel_frame->skeletons(i).skeleton_id()] = Skeleton(); ///XXX remove Skeleton(); part
                    // mapIdSkel[sf->skeletons(i).skeleton_id()].attach(_root);
                    // cerr << "Found Skeleton\n";
                    mapIdSkel[skel_frame->skeletons(i).skeleton_id()].attach(switchNode);
                }

                // Skeleton previously detached (stopped being reported), but is again reported -> reattach
                if (mapIdSkel[skel_frame->skeletons(i).skeleton_id()].attached == false)
                    mapIdSkel[skel_frame->skeletons(i).skeleton_id()].attach(switchNode);

                for (int j = 0; j < skel_frame->skeletons(i).joints_size(); j++)
                {
                    mapIdSkel[skel_frame->skeletons(i).skeleton_id()].update(
                        skel_frame->skeletons(i).joints(j).type(),
                        skel_frame->skeletons(i).joints(j).x(),
                        skel_frame->skeletons(i).joints(j).z(),
                        skel_frame->skeletons(i).joints(j).y(),
                        skel_frame->skeletons(i).joints(j).qx(),
                        skel_frame->skeletons(i).joints(j).qz(),
                        skel_frame->skeletons(i).joints(j).qy(),
                        skel_frame->skeletons(i).joints(j).qw());
                }
            }
    }
   }
   //Closed
}

void  SkeletonManager::quit()
{
    should_quit = true;
}

std::map<int, Skeleton>* SkeletonManager::skeletonGetMap()
{
    return &mapIdSkel;
}

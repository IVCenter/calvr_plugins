//#ifndef _ANIMATION_MANAGER_MULT
//#define _ANIMATION_MANAGER_MULT

#include <osgDB/ReadFile>
#include <osgDB/FileUtils>
#include <cvrKernel/SceneManager.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuRangeValue.h>

#include <shared/PubSub.h>
#include <protocol/skeletonframe.pb.h>
#include <protocol/colormap.pb.h>
#include <protocol/depthmap.pb.h>
#include <protocol/pointcloud.pb.h>
#include <zmq.hpp>
#include "kUtils.h"
#include "Skeleton.h"
#include <unordered_map>


#include <iostream>
#include <string>
#include <map>

#include <OpenThreads/Thread>


//	zmq::context_t contextCloud(1);

class SkeletonManager : public OpenThreads::Thread {

    public:

	SkeletonManager();
        ~SkeletonManager();
	//contextCloud(1);
SubSocket<RemoteKinect::SkeletonFrame>* skelT_socket;
        void update();
        bool isCacheDone();

        RemoteKinect::SkeletonFrame* sf2;
        virtual void run();
        void quit();
    protected:

    std::map<int, Skeleton> mapIdSkel;
    bool kNavSpheres;
    bool kMoveWithCam;
        bool _cacheDone;
    bool should_quit;
    osg::MatrixTransform* _root;
};

//#endif

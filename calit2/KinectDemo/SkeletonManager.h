//#ifndef _ANIMATION_MANAGER_MULT
//#define _ANIMATION_MANAGER_MULT

#include <cvrKernel/SceneManager.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneObject.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuRangeValue.h>

#include <shared/PubSub.h>
#include <protocol/skeletonframe.pb.h>
#include <zmq.hpp>
#include "Skeleton.h"



#include <OpenThreads/Thread>

class SkeletonManager : public OpenThreads::Thread {

public:

    SkeletonManager(std::string server);
    ~SkeletonManager();
    RemoteKinect::SkeletonFrame* skel_frame;
    SubSocket<RemoteKinect::SkeletonFrame>* skel_socket;
    std::map<int, Skeleton> mapIdSkel;
    osg::Switch* switchNode;

    void update();
    bool isCacheDone();

    virtual void run();
    void quit();
    std::map<int, Skeleton>* skeletonGetMap();

    zmq::context_t* context;
protected:

    bool kNavSpheres;
    bool kMoveWithCam;
    bool _cacheDone;
    bool should_quit;
    std::string skeletonServer;
};

//#endif

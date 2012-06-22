#ifndef CVRPLUGIN_KINECT_HELPER_H
#define CVRPLUGIN_KINECT_HELPER_H

#include <cvrKernel/CVRPlugin.h>

#include <string>

class KinectHelper : public cvr::CVRPlugin
{
    public:
        KinectHelper();
        virtual ~KinectHelper();

        bool init();
        void preFrame();

    protected:
        void sendState();

        enum InteractionState
        {
            NO_INTERACTION = 0,
            MENU_INTERACTION,
            PICKING_INTERACTION,
            NAVIGATION_INTERACTION
        };

        std::string _ncCMD;
        InteractionState _state;
};

#endif

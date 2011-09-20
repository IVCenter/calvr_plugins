#ifndef ANDROID_NAVIGATOR_H
#define ANDROID_NAVIGATOR_H

#include <kernel/CVRPlugin.h>

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <X11/Xlib.h>
#include <menu/SubMenu.h>
#include <menu/MenuCheckbox.h>
#include <menu/MenuButton.h>
#include <osg/MatrixTransform>
#include <menu/MenuSystem.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <queue>
#include <OpenThreads/Mutex>
#include <OpenThreads/Thread>
#include <osg/Geode>
#include <kernel/InteractionManager.h>
#include <osg/Shape>
#include <osg/Node>
#include "AndroidTransform.h"
#include "AndroidVisitor.h"

#include <iostream>
#include <string>
#include <vector>

class AndroidNavigator : public cvr::CVRPlugin, public cvr::MenuCallback, public OpenThreads::Thread
{
    public:
        AndroidNavigator();
        ~AndroidNavigator();

        bool init();
        void preFrame();
        void menuCallback(cvr::MenuItem * item);
            // Makes a background thread to take in data
        void makeThread();
            // Selects a node to adjust
        void nodeSelect();
            // Adjusts selected node
        void adjustNode(double height, double magnitude, int position);
            // Allows Vec3 comparison
        class compare{
            public: 
                bool operator() (const osg::Vec3 vec1, const osg::Vec3 vec2){
                    return(vec1.length() < vec2.length());
                }
        };
        
    protected:
        osg::MatrixTransform * _root;
        cvr::MenuCheckbox *_isOn;
        cvr::SubMenu *_andMenu;
        float transMult, rotMult;
        float transcale, rotscale, scale;
        cvr::MenuSystem* _menu;
        int _tagCommand;
        int sock;
        socklen_t addr_len;
        struct sockaddr_in server_addr;
        struct sockaddr_in client_addr;
        double velocity;
        std::queue<std::string> queue;
        bool _mkill;
        OpenThreads::Mutex _mutex;
        bool newMode;
        bool flip;
        AndroidTransform * node;
        double old_ry;
        std::map<char*, AndroidTransform*> nodeMap;
        char* node_name;
        bool nodeChange;

        // Runs the thread to take in android data
        virtual void run();
        
};

#endif

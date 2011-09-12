#ifndef ANDROID_TRANSFORM_H
#define ANDROID_TRANSFORM_H

#include <kernel/CVRPlugin.h>

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <osg/MatrixTransform>
#include <sys/types.h>
#include <sys/socket.h>
#include <queue>
#include <OpenThreads/Mutex>
#include <OpenThreads/Thread>

#include <iostream>
#include <string>
#include <vector>

class AndroidTransform :public osg::MatrixTransform
{
    public:
        AndroidTransform(char* name);
        virtual ~AndroidTransform();

        void setName(char* name);        
        char* getName();
        virtual AndroidTransform* asAndroidTransform();

        void setTrans(float x, float y, float z);
        void setTrans(double x, double y, double z);
        void setTrans(osg::Vec3f vec);
        void setTrans(osg::Vec3d vec);
        osg::Vec3 getTrans();
        void setRotation(float rx, osg::Vec3 xa, float ry, osg::Vec3 ya, float rz, osg::Vec3 za);
        void setRotation(double rx, osg::Vec3 xa, double ry, osg::Vec3 ya, double rz, osg::Vec3 za);
        void setRotation(float angle, osg::Vec3f vec);
        void setRotation(double angle, osg::Vec3d vec);
        osg::Quat getRotation();
        void setScale(float sx, float sy, float sz);
        void setScale(double sx, double sy, double sz);
        void setScale(osg::Vec3f vec);
        void setScale(osg::Vec3d vec);
        osg::Vec3 getScale();

    protected:
        char* _name;
        long _len;
        osg::Matrix transmat;
        osg::Matrix rotmat;
};
#endif



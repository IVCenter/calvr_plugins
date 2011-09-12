#ifndef _AV_H
#define _AV_H

#include <osg/Geode>
#include <osg/Node>
#include <osg/Matrix>
#include <osg/Transform>
#include <osg/MatrixTransform>
#include "AndroidTransform.h"

class AndroidVisitor : public osg::NodeVisitor
{
    private:
        std::map<char*, AndroidTransform*> nodeMap;

    public:
        AndroidVisitor();
        virtual void apply(osg::Transform&);
        std::map<char*, AndroidTransform*> getMap();

};
#endif


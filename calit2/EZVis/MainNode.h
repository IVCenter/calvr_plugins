#ifndef _MAINNODE_H
#define _MAINNODE_H

#include <osg/MatrixTransform>
#include "ThreadQueue.h"

using namespace osg;
using namespace std;

class UpdateCallback : public osg::NodeCallback
{
    virtual void operator()(osg::Node* node, osg::NodeVisitor* nv);
};

class MainNode : public osg::MatrixTransform
{
    friend class UpdateCallback;

    private:
        ThreadQueue< std::pair<osg::Node* , bool> > _nodes;

    protected:
        MainNode(){};

    public:
        MainNode(float scale, bool rotaxis);
	    ~MainNode();

        void addElement(osg::Node* );
        void removeElement(osg::Node* );
};
#endif

#include "MainNode.h"

#include <iostream>

void UpdateCallback::operator()(osg::Node* node, osg::NodeVisitor* nv)
{
    MainNode* mainNode = dynamic_cast<MainNode*> (node);
    if( mainNode )
    {
        std::pair<osg::Node*, bool> element;
        while( mainNode->_nodes.get(element) )
        {
            if( element.second == true ) // add node
                mainNode->addChild(element.first);
            else
                mainNode->removeChild(element.first);
        }
    }

    traverse(node,nv);
}

MainNode::MainNode(float scale, bool rotaxis)
{
    // make scale multiplier matrix
    osg::Matrix m;
    m.makeScale(scale, scale, scale);


    if( rotaxis )
    {
        osg::Matrix r;
        r.makeRotate(M_PI_2, osg::Vec3(1, 0, 0));
        m = m * r; 
    }

    // set the global matrix
    setMatrix(m);

    // set default callback
    setUpdateCallback(new UpdateCallback());    
}

MainNode::~MainNode()
{
    setUpdateCallback(NULL);
}

void MainNode::addElement(osg::Node* node)
{
    _nodes.add(std::pair<osg::Node* , bool> (node, true));    
}

void MainNode::removeElement(osg::Node* node)
{
    _nodes.add(std::pair<osg::Node* , bool> (node, false));    
}

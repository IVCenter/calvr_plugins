#include "MainNode.h"

#include <iostream>

void UpdateCallback::operator()(osg::Node* node, osg::NodeVisitor* nv)
{
    MainNode* mainNode = dynamic_cast<MainNode*> (node);
    if( mainNode )
    {
        osg::Node* child = NULL;
        while( mainNode->_additionNodes.get(child) )
            mainNode->addChild(child);

        while( mainNode->_removalNodes.get(child) )
            mainNode->removeChild(child);
    }

    traverse(node,nv);
}

MainNode::MainNode()
{
    setUpdateCallback(new UpdateCallback());    
}

MainNode::~MainNode()
{
    setUpdateCallback(NULL);
}

void MainNode::addElement(osg::Node* node)
{
    _additionNodes.add(node);    
}

void MainNode::removeElement(osg::Node* node)
{
    _removalNodes.add(node);    
}

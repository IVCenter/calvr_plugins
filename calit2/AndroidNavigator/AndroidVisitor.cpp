#include <iostream>
#include <osg/ShapeDrawable>
#include "AndroidVisitor.h"

using namespace std;
using namespace osg;

  AndroidVisitor::AndroidVisitor():osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
  {
  }

  void AndroidVisitor::apply(osg::Transform& node)
  {
    AndroidTransform* trans = dynamic_cast<AndroidTransform* > (&node);
    if(trans)
    { 
        nodeMap.insert(std::pair<char*, AndroidTransform*>(trans->getName(), trans));
    }   
    traverse(node);
  }

  std::map<char*, AndroidTransform*> AndroidVisitor::getMap(){
      return nodeMap;
  }



#include <iostream>
#include <osg/ShapeDrawable>
#include "ComputeBBVisitor.h"

using namespace std;
using namespace osg;

	// run from mainNode
  ComputeBBVisitor::ComputeBBVisitor(const Matrix &mat):osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
  {
      //cerr << "\n\nNEW CBBV:\n";
    m_curMatrix = mat;
    m_bb.init();
  }

  void ComputeBBVisitor::apply(osg::Geode &geode)
  {
    for(unsigned int i = 0; i < geode.getNumDrawables(); i++)
    {
      osg::BoundingBox bb = geode.getDrawable(i)->computeBound();
      m_bb.expandBy(bb.corner(0)*m_curMatrix);
      m_bb.expandBy(bb.corner(1)*m_curMatrix);
      m_bb.expandBy(bb.corner(2)*m_curMatrix);
      m_bb.expandBy(bb.corner(3)*m_curMatrix);
      m_bb.expandBy(bb.corner(4)*m_curMatrix);
      m_bb.expandBy(bb.corner(5)*m_curMatrix);
      m_bb.expandBy(bb.corner(6)*m_curMatrix);
      m_bb.expandBy(bb.corner(7)*m_curMatrix);
    }
  }

  void ComputeBBVisitor::apply(osg::Transform& node)
  {

    if(node.asMatrixTransform() || node.asPositionAttitudeTransform())
    {
    	osg::Matrix prevMatrix = m_curMatrix;

    	m_curMatrix.preMult(node.asMatrixTransform()->getMatrix());

    	traverse(node);

    	m_curMatrix = prevMatrix;
    }   
  }

  BoundingBox ComputeBBVisitor::getBound()
  {
    return m_bb; 
  }

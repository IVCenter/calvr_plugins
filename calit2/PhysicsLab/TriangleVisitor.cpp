#include "TriangleVisitor.h"

std::vector< Triangle > * TriangleVisitor::_triangles;
osg::Matrixd TriangleVisitor::_matrix;
osg::Vec3 TriangleVisitor::max_v, TriangleVisitor::min_v;

TriangleVisitor::TriangleVisitor():
    osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
{
    _triangles = new std::vector< Triangle >();
}

TriangleVisitor::~TriangleVisitor()
{
    delete _triangles;
}

void TriangleVisitor::apply(osg::Geode& geode)
{
    _matrix.set(osg::computeLocalToWorld(this->getNodePath()));

    for(int i = 0; i < geode.getNumDrawables(); i++)
    {
            // add world space triangle to vector 
            geode.getDrawable(i)->accept(tf);
    }

}


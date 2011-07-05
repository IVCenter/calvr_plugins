#include "TransparencyVisitor.h"

#include <iostream>
#include <cstdio>
#include <cstring>
//osg
#include <osg/NodeVisitor>
#include <osg/Material>
#include <osg/StateSet>
#include <osg/ShapeDrawable>
#include <osg/Geode>
#include <osg/CullFace>
#include <osg/PolygonMode>

using namespace std;

TransparencyVisitor::TransparencyVisitor() : NodeVisitor( NodeVisitor::TRAVERSE_ALL_CHILDREN )
{
	_currentMode = ALL_OPAQUE;
}

TransparencyVisitor::~TransparencyVisitor()
{

}

void TransparencyVisitor::setMode(Mode mode)
{
    _currentMode = mode;
}

TransparencyVisitor::Mode TransparencyVisitor::getMode()
{
	return _currentMode;
}

void TransparencyVisitor::apply(osg::Geode& thisGeode)
{
	float level;

	if(_currentMode==ALL_OPAQUE)
            level=1.0f;
	else //if(_currentMode==ALL_TRANSPARENT)
            level=0.1f;

        osg::StateSet * stateset = thisGeode.getOrCreateStateSet();
        osg::Material * mm = dynamic_cast<osg::Material*>(stateset->getAttribute
            (osg::StateAttribute::MATERIAL));

        if (!mm)
            mm = new osg::Material;

        mm->setAlpha(osg::Material::FRONT_AND_BACK, level);

        stateset->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE |
            osg::StateAttribute::ON );
//        stateset->setMode(GL_LIGHTING,osg::StateAttribute::OVERRIDE |
//            osg::StateAttribute::ON );
        stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        stateset->setAttributeAndModes( mm, osg::StateAttribute::OVERRIDE |
            osg::StateAttribute::ON);

        thisGeode.setStateSet(stateset);
}

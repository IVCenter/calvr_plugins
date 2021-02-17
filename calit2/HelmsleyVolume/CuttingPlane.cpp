#include "CuttingPlane.h"

#include "cvrConfig/ConfigManager.h"

using namespace cvr;

void CuttingPlane::init()
{
	float size = ConfigManager::getFloat("Plugin.HelmsleyVolume.CuttingPlaneSize", 500.0f);

	osg::Drawable* cpd1 = new osg::ShapeDrawable(new osg::Box(osg::Vec3(size * 0.495, 0, 0), size * 0.01, size * 0.001, size));
	osg::Drawable* cpd2 = new osg::ShapeDrawable(new osg::Box(osg::Vec3(size * -0.495, 0, 0), size * 0.01, size * 0.001, size));
	osg::Drawable* cpd3 = new osg::ShapeDrawable(new osg::Box(osg::Vec3(0, 0, size * 0.495), size, size * 0.001, size * 0.01));
	osg::Drawable* cpd4 = new osg::ShapeDrawable(new osg::Box(osg::Vec3(0, 0, size * -0.495), size, size * 0.001, size * 0.01));

	osg::Geode* cuttingPlaneGeode = new osg::Geode();
	cuttingPlaneGeode->addDrawable(cpd1);
	cuttingPlaneGeode->addDrawable(cpd2);
	cuttingPlaneGeode->addDrawable(cpd3);
	cuttingPlaneGeode->addDrawable(cpd4);
	cuttingPlaneGeode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

	addChild(cuttingPlaneGeode);

	_lock = new MenuCheckbox("Lock", false);
	this->addMenuItem(_lock);
	_lock->setCallback(this);
}

bool CuttingPlane::processEvent(InteractionEvent* ie)
{

	return SceneObject::processEvent(ie);
}

void CuttingPlane::menuCallback(MenuItem* item)
{
	if (item == _lock)
	{
		this->setMovable(!_lock->getValue());
		this->setShowBounds(!_lock->getValue());
	}
	else {
		SceneObject::menuCallback(item);
	}
}

void CuttingPlane::updateCallback(int handID, const osg::Matrix& mat)
{

	changePlane();
}


void CenterlineUpdate::dirtyPlane() {
	((CuttingPlane*)_object)->changePlane();
}
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

	osg::Matrix obj2wrl = getObjectToWorldMatrix();

	osg::Matrix wrl2obj = _so->getWorldToObjectMatrix();
	osg::Matrix wrl2obj2 = _volume->getWorldToObjectMatrix();

	osg::Matrix posmat = obj2wrl * wrl2obj * wrl2obj2;

	osg::Quat q = osg::Quat();
	osg::Quat q2 = osg::Quat();
	osg::Vec3 v = osg::Vec3();
	osg::Vec3 v2 = osg::Vec3();
	osg::Matrix m = osg::Matrix();

	obj2wrl.decompose(v, q, v2, q2);
	m.makeRotate(q);

	wrl2obj.decompose(v, q, v2, q2);
	m.postMultRotate(q);

	wrl2obj2.decompose(v, q, v2, q2);
	m.postMultScale(osg::Vec3(1.0 / v2.x(), 1.0 / v2.y(), 1.0 / v2.z()));
	m.postMultRotate(q);

	osg::Vec4d normal = osg::Vec4(0, 1, 0, 0) * m;
	osg::Vec3 norm = osg::Vec3(normal.x(), normal.y(), normal.z());

	osg::Vec4f position = osg::Vec4(0, 0, 0, 1) * posmat;
	osg::Vec3f pos = osg::Vec3(position.x(), position.y(), position.z());


	//std::cerr << "Position: " << pos.x() << ", " << pos.y() << ", " << pos.z() << std::endl;
	//std::cerr << "Normal: " << norm.x() << ", " << norm.y() << ", " << norm.z() << std::endl << std::endl;


	_volume->_PlanePoint->set(pos);
	_volume->_PlaneNormal->set(norm);
}


void CenterlineUpdate::dirtyPlane() {
	((CuttingPlane*)_object)->changePlane();
}
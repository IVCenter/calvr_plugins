#include "MCylinder.h"
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osgText/Text>
#include <osg/Vec4>

#include <osgUtil/SceneView>
#include <osg/Camera>

#include <osg/CullFace>
#include <osg/Matrix>
#include <osg/ShapeDrawable>
#include <osgDB/ReadFile>
#include <osg/PositionAttitudeTransform>
#include <osg/PolygonMode>


MCylinder::MCylinder()
{
    osg::Vec3d currVec;
    osg::Vec3d prevVec;
    StartPoint = osg::Vec3(0, 0, 0);
    EndPoint = osg::Vec3(0, 0, 0);
    center = osg::Vec3(0, 0, 0);
    translate = new osg::MatrixTransform();
    osg::Quat rotation;
    geode     = new osg::Geode();
    attached = false;
    locked = false;
    translate->addChild(geode);
    radius = 0.01;
    CylinderColor = osg::Vec4(1.0, 0.4, 0.2, 1.0);
}

MCylinder::MCylinder(float _radius, osg::Vec4 _color)
{
    osg::Vec3d currVec;
    osg::Vec3d prevVec;
    StartPoint = osg::Vec3(0, 0, 0);
    EndPoint = osg::Vec3(0, 0, 0);
    center = osg::Vec3(0, 0, 0);
    translate = new osg::MatrixTransform();
    osg::Quat rotation;
    geode     = new osg::Geode();
    attached = false;
    locked = false;
    translate->addChild(geode);
    radius = _radius;
    CylinderColor = _color;
}

// TODO stop detaching&attaching all the time
void MCylinder::update(osg::Vec3 startP, osg::Vec3 endP)
{
    prevLength = length;
    prevVec = currVec;
    StartPoint = startP;
    EndPoint = endP;
    //attempt to remove 2 (why 2 is a good question) cylinders from the geode
    geode->removeDrawables(0, 2);
    float height;
    osg::ref_ptr<osg::Cylinder> cylinder;
    osg::ref_ptr<osg::ShapeDrawable> cylinderDrawable;
    osg::ref_ptr<osg::Material> pMaterial;
    height = (StartPoint - EndPoint).length();
    center = osg::Vec3d((StartPoint.x() + EndPoint.x()) / 2, (StartPoint.y() + EndPoint.y()) / 2, (StartPoint.z() + EndPoint.z()) / 2);
    currVec = osg::Vec3d(StartPoint.x() - center.x(), StartPoint.y() - center.y(), StartPoint.z() - center.z());
    // This is the default direction for the cylinders to face in OpenGL
    osg::Vec3   z = osg::Vec3(0, 0, 1);
    // Get diff between two points you want cylinder along
    osg::Vec3 p = (StartPoint - EndPoint);
    // Get CROSS product (the axis of rotation)
    osg::Vec3   t = z ^  p;
    // Get angle. length is magnitude of the vector
    double angle = acos((z * p) / p.length());
    //   Create a cylinder between the two points with the given radius
    cylinder = new osg::Cylinder(center, radius, height);
    rotation = osg::Quat(angle, osg::Vec3(t.x(), t.y(), t.z()));
    cylinder->setRotation(rotation);
    //   A geode to hold our cylinder
    cylinderDrawable = new osg::ShapeDrawable(cylinder);
    geode->addDrawable(cylinderDrawable);
    //   Set the color of the cylinder that extends between the two points.
    pMaterial = new osg::Material;
    pMaterial->setDiffuse(osg::Material::FRONT, CylinderColor);
    geode->getOrCreateStateSet()->setAttribute(pMaterial, osg::StateAttribute::OVERRIDE);
    length = height;//osg::Vec3(startP - endP).length();

    if (prevLength == 0) prevLength = length;
}

void MCylinder::attach(osg::MatrixTransform* parent)
{
    if (attached) return;

    parent->addChild(geode);
    attached = true;
    prevLength = length;
}

void MCylinder::detach(osg::MatrixTransform* parent)
{
    attached = false;
    parent->removeChild(geode);
}


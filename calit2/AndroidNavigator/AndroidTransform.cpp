#include "AndroidTransform.h"

#define PORT 8888

#include <stdio.h>
#include <kernel/PluginHelper.h>
#include <osg/Vec4>
#include <osg/Vec3>
#include <osg/Quat>
#include <osg/Matrix>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;
using namespace osg;
using namespace cvr;


AndroidTransform::AndroidTransform(char* name){
    _name = name;
    _len = strlen(name);
}

AndroidTransform::~AndroidTransform(){
}

void AndroidTransform::setName(char* name){
    _name = name;
}

char* AndroidTransform::getName(){
    return _name;
}

AndroidTransform* AndroidTransform::asAndroidTransform(){
    return this;
}

void AndroidTransform::setTrans(float x, float y, float z){
    transmat.makeTranslate(Vec3(x, y, z));
    setMatrix(rotmat * transmat);
}

void AndroidTransform::setTrans(double x, double y, double z){
    transmat.makeTranslate(Vec3(x, y, z)); 
    setMatrix(rotmat * transmat);
}

void AndroidTransform::setTrans(osg::Vec3f vec){
    transmat.makeTranslate(vec);
    setMatrix(rotmat * transmat);
}

void AndroidTransform::setTrans(osg::Vec3d vec){
    transmat.makeTranslate(vec);
    setMatrix(rotmat * transmat);
}

osg::Vec3 AndroidTransform::getTrans(){
    return transmat.getTrans();
}
void AndroidTransform::setRotation(float rx, Vec3 xa, float ry, Vec3 ya, float rz, Vec3 za){
    rotmat.makeRotate(rx, xa, ry, ya, rz, za);
    Matrix inverseT = transmat.inverse(transmat);
    setMatrix(inverseT * rotmat * transmat);
}

void AndroidTransform::setRotation(double rx, Vec3 xa, double ry, Vec3 ya, double rz, Vec3 za){
    rotmat.makeRotate(rx, xa, ry, ya, rz, za);
    setMatrix(rotmat * transmat);
}

void AndroidTransform::setRotation(float angle, Vec3f axis){
    rotmat.makeRotate(angle, axis);
    setMatrix(rotmat * transmat); 
}

void AndroidTransform::setRotation(double angle, Vec3d axis){
    rotmat.makeRotate(angle, axis);
    setMatrix(rotmat * transmat); 
}

osg::Quat AndroidTransform::getRotation(){
   return rotmat.getRotate();
}

void AndroidTransform::setScale(float sx, float sy, float sz){
    Matrix mat;
    mat.makeScale(sx, sy, sz);
    setMatrix(mat);
}

void AndroidTransform::setScale(double sx, double sy, double sz){
    Matrix mat;
    mat.makeScale(sx, sy, sz);
    setMatrix(mat);
}

void AndroidTransform::setScale(osg::Vec3f vec){
    Matrix mat;
    mat.makeScale(vec);
    setMatrix(mat);
}

void AndroidTransform::setScale(osg::Vec3d vec){
    Matrix mat;
    mat.makeScale(vec);
    setMatrix(mat);
}

osg::Vec3 AndroidTransform::getScale(){
    Matrix mat = getMatrix();
    return mat.getScale();

}

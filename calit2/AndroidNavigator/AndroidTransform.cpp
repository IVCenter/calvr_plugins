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

// All AndroidTransforms have names for identification with phone
AndroidTransform::AndroidTransform(char* name){
    _name = name;
}

AndroidTransform::~AndroidTransform(){
}

void AndroidTransform::setName(char* name){
    _name = name;
}

char* AndroidTransform::getName(){
    return _name;
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
    setRotation((double)rx, xa, (double) ry, ya, (double) rz, za);
}

void AndroidTransform::setRotation(double rx, Vec3 xa, double ry, Vec3 ya, double rz, Vec3 za){
    Quat x, y, z, main;
    x.makeRotate(rx, xa);
    y.makeRotate(ry, ya);
    z.makeRotate(rz, za);
    main = rotmat.getRotate();
    rotmat.makeRotate(main * x * y* z);
    setMatrix(rotmat * transmat);
}

osg::Quat AndroidTransform::getRotation(){
   return rotmat.getRotate();
}


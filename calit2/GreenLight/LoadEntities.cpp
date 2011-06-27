#include "GreenLight.h"

#include <iostream>
#include <kernel/NodeMask.h>
#include <osgDB/ReadFile>

bool GreenLight::loadBox()
{
    // files to load
    string boxFile = "/home/covise/data/GreenLight/Models/box.WRL";
    string doorFLfile = "/home/covise/data/GreenLight/Models/frontleft.WRL";
    string doorFRfile = "/home/covise/data/GreenLight/Models/frontright.WRL";
    string doorFIfile = "/home/covise/data/GreenLight/Models/frontinner.WRL";
    string doorBLfile = "/home/covise/data/GreenLight/Models/backleft.WRL";
    string doorBRfile = "/home/covise/data/GreenLight/Models/backright.WRL";
    string doorBIfile = "/home/covise/data/GreenLight/Models/backinner.WRL";
    string doorBIIfile = "/home/covise/data/GreenLight/Models/backinnerinner.WRL";

    // Load the models
    ref_ptr<osg::Node> box = osgDB::readNodeFile(boxFile);
    ref_ptr<osg::Node> doorFL = osgDB::readNodeFile(doorFLfile);
    ref_ptr<osg::Node> doorFR = osgDB::readNodeFile(doorFRfile);
    ref_ptr<osg::Node> doorFI = osgDB::readNodeFile(doorFIfile);
    ref_ptr<osg::Node> doorBL = osgDB::readNodeFile(doorBLfile);
    ref_ptr<osg::Node> doorBR = osgDB::readNodeFile(doorBRfile);
    ref_ptr<osg::Node> doorBI = osgDB::readNodeFile(doorBIfile);
    ref_ptr<osg::Node> doorBII = osgDB::readNodeFile(doorBIIfile);

    // if any files failed to load, report them and cancel loadBox()
    if (!box || !doorFL)
    {
        cerr << "Error (LoadEntities.cpp): Failed to load files(s):" << endl;

        if (!box)
            cerr << "\t" << boxFile << endl;
        if (!doorFL)
            cerr << "\t" << doorFLfile << endl;
        if (!doorFR)
            cerr << "\t" << doorFRfile << endl;
        if (!doorFI)
            cerr << "\t" << doorFIfile << endl;
        if (!doorBL)
            cerr << "\t" << doorBLfile << endl;
        if (!doorBR)
            cerr << "\t" << doorBRfile << endl;
        if (!doorBI)
            cerr << "\t" << doorBIfile << endl;
        if (!doorBII)
            cerr << "\t" << doorBIIfile << endl;

        return false;
    }

    // All loaded -- Create Entities & Animation Paths
    _box = new Entity(box);
    //box->setNodeMask(box->getNodeMask() & ~INTERSECT_MASK); // No interaction

    Vec3 doorOffset;
    AnimationPath::ControlPoint cp;

    // Door 0 - Front Left
    doorOffset = Vec3(-46.978,-118.808,0);
    _door.push_back(new Entity(doorFL, Matrix::translate(doorOffset)));
    _door[0]->path = new AnimationPath();
    _door[0]->path->setLoopMode(AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(Quat(0,Vec3(0,0,1)));
    _door[0]->path->insert(0,cp);
    cp.setRotation(Quat(-osg::PI*3/4,Vec3(0,0,1)));
    _door[0]->path->insert(1,cp);

    // Door 1 - Front Right
    doorOffset = Vec3(47.047,-118.851,0);
    _door.push_back(new Entity(doorFR, Matrix::translate(doorOffset)));
    _door[1]->path = new AnimationPath();
    _door[1]->path->setLoopMode(AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(Quat(0,Vec3(0,0,1)));
    _door[1]->path->insert(0,cp);
    cp.setRotation(Quat(osg::PI*3/4,Vec3(0,0,1)));
    _door[1]->path->insert(1,cp);

    // Door 2 - Front Inner
    doorOffset = Vec3(15.906,-104.9,0);
    _door.push_back(new Entity(doorFI, Matrix::translate(doorOffset)));
    _door[2]->path = new AnimationPath();
    _door[2]->path->setLoopMode(AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(Quat(0,Vec3(0,0,1)));
    _door[2]->path->insert(0,cp);
    cp.setRotation(Quat(osg::PI/2,Vec3(0,0,1)));
    _door[2]->path->insert(1,cp);

    // Door 3 - Back Left
    doorOffset = Vec3(46.993,118.757,0);
    _door.push_back(new Entity(doorBL, Matrix::translate(doorOffset)));
    _door[3]->path = new AnimationPath();
    _door[3]->path->setLoopMode(AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(Quat(0,Vec3(0,0,1)));
    _door[3]->path->insert(0,cp);
    cp.setRotation(Quat(-osg::PI*3/4,Vec3(0,0,1)));
    _door[3]->path->insert(1,cp);

    // Door 4 - Back Right
    doorOffset = Vec3(-47.117,113.765,0);
    _door.push_back(new Entity(doorBR, Matrix::translate(doorOffset)));
    _door[4]->path = new AnimationPath();
    _door[4]->path->setLoopMode(AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(Quat(0,Vec3(0,0,1)));
    _door[4]->path->insert(0,cp);
    cp.setRotation(Quat(osg::PI*3/4,Vec3(0,0,1)));
    _door[4]->path->insert(1,cp);

    // Door 5 - Back Inner
    doorOffset = Vec3(18.339,96.197,0);
    _door.push_back(new Entity(doorBI, Matrix::translate(doorOffset)));
    _door[5]->path = new AnimationPath();
    _door[5]->path->setLoopMode(AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(Quat(0,Vec3(0,0,1)));
    _door[5]->path->insert(0,cp);
    cp.setRotation(Quat(-osg::PI/2,Vec3(0,0,1)));
    _door[5]->path->insert(1,cp);

    // Door 6 - Back Inner Inner
    doorOffset = Vec3(15.505,81.835,0);
    _door.push_back(new Entity(doorBII, Matrix::translate(doorOffset)));
    _door[6]->path = new AnimationPath();
    _door[6]->path->setLoopMode(AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(Quat(0,Vec3(0,0,1)));
    _door[6]->path->insert(0,cp);
    cp.setRotation(Quat(-osg::PI/2,Vec3(0,0,1)));
    _door[6]->path->insert(1,cp);

    // Setup animation groups
    _door[0]->group.push_back(_door[1]);
    _door[1]->group.push_back(_door[0]);
    _door[3]->group.push_back(_door[4]);
    _door[4]->group.push_back(_door[3]);

    // Add it all to the box transform
    for (int d = 0; d < _door.size(); d++)
        _box->transform->addChild(_door[d]->transform);
    
    return true;
}

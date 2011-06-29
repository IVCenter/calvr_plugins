#include "GreenLight.h"

#include <iostream>
#include <osgDB/ReadFile>
#include <osgUtil/Optimizer>

#include <osg/ShapeDrawable>

bool GreenLight::loadScene()
{
    // files to load
    string modelsDir      = "/home/covise/data/GreenLight/Models/";
    string boxFile        = modelsDir + "box.WRL";
    string electricalFile = modelsDir + "electrical.WRL";
    string pipesFile      = modelsDir + "waterpipes.WRL";
    string doorFLfile     = modelsDir + "frontleft.WRL";
    string doorFRfile     = modelsDir + "frontright.WRL";
    string doorFIfile     = modelsDir + "frontinner.WRL";
    string doorBLfile     = modelsDir + "backleft.WRL";
    string doorBRfile     = modelsDir + "backright.WRL";
    string doorBIfile     = modelsDir + "backinner.WRL";
    string doorBIIfile    = modelsDir + "backinnerinner.WRL";
    string fansFile       = modelsDir + "fans_reduced.WRL";
    string rack1File      = modelsDir + "rack1_c.WRL";
    string rack2File      = modelsDir + "rack2_c.WRL";
    string rack3File      = modelsDir + "rack3_c.WRL";
    string rack4File      = modelsDir + "rack4_c.WRL";
    string rack5File      = modelsDir + "rack5_c.WRL";
    string rack6File      = modelsDir + "rack6_c.WRL";
    string rack7File      = modelsDir + "rack7_c.WRL";
    string rack8File      = modelsDir + "rack8_c.WRL";

    // Load the models
cerr<<"\nLoading "<<boxFile<<"\n";
    ref_ptr<Node> box = osgDB::readNodeFile(boxFile);
cerr<<"Loading "<<electricalFile<<"\n";
    ref_ptr<Node> electrical = osgDB::readNodeFile(electricalFile);
cerr<<"Loading "<<pipesFile<<"\n";
    ref_ptr<Node> pipes = osgDB::readNodeFile(pipesFile);
cerr<<"Loading "<<doorFLfile<<"\n";
    ref_ptr<Node> doorFL = osgDB::readNodeFile(doorFLfile);
cerr<<"Loading "<<doorFRfile<<"\n";
    ref_ptr<Node> doorFR = osgDB::readNodeFile(doorFRfile);
cerr<<"Loading "<<doorFIfile<<"\n";
    ref_ptr<Node> doorFI = osgDB::readNodeFile(doorFIfile);
cerr<<"Loading "<<doorBLfile<<"\n";
    ref_ptr<Node> doorBL = osgDB::readNodeFile(doorBLfile);
cerr<<"Loading "<<doorBRfile<<"\n";
    ref_ptr<Node> doorBR = osgDB::readNodeFile(doorBRfile);
cerr<<"Loading "<<doorBIfile<<"\n";
    ref_ptr<Node> doorBI = osgDB::readNodeFile(doorBIfile);
cerr<<"Loading "<<doorBIIfile<<"\n";
    ref_ptr<Node> doorBII = osgDB::readNodeFile(doorBIIfile);
cerr<<"Loading "<<fansFile<<"\n";
    ref_ptr<Node> fans = osgDB::readNodeFile(fansFile);
cerr<<"Loading "<<rack1File<<"\n";
    ref_ptr<Node> rack1 = osgDB::readNodeFile(rack1File);
cerr<<"Loading "<<rack2File<<"\n";
    ref_ptr<Node> rack2 = osgDB::readNodeFile(rack2File);
cerr<<"Loading "<<rack3File<<"\n";
    ref_ptr<Node> rack3 = osgDB::readNodeFile(rack3File);
cerr<<"Loading "<<rack4File<<"\n";
    ref_ptr<Node> rack4 = osgDB::readNodeFile(rack4File);
cerr<<"Loading "<<rack5File<<"\n";
    ref_ptr<Node> rack5 = osgDB::readNodeFile(rack5File);
cerr<<"Loading "<<rack6File<<"\n";
    ref_ptr<Node> rack6 = osgDB::readNodeFile(rack6File);
cerr<<"Loading "<<rack7File<<"\n";
    ref_ptr<Node> rack7 = osgDB::readNodeFile(rack7File);
cerr<<"Loading "<<rack8File<<"\n";
    ref_ptr<Node> rack8 = osgDB::readNodeFile(rack8File);
cerr<<"Done Loading\n";

    // if any files failed to load, report them and cancel loadScene()
    if (!box || !electrical || !pipes || !doorFL || !doorFR || !doorFI 
    || !doorBL || !doorBR || !doorBI || !doorBII || /*!fans ||*/ !rack1 || !rack2
    || !rack3 || !rack4 || !rack5 || !rack6 || !rack7 || !rack8)
    {
        cerr << "Error (LoadEntities.cpp): Failed to load files(s):" << endl;

        if (!box)
            cerr << "\t" << boxFile << endl;
        if (!electrical)
            cerr << "\t" << electricalFile << endl;
        if (!pipes)
            cerr << "\t" << pipesFile << endl;
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
        if (!fans)
            cerr << "\t" << fansFile << endl;
        if (!rack1)
            cerr << "\t" << rack1File << endl;
        if (!rack2)
            cerr << "\t" << rack2File << endl;
        if (!rack3)
            cerr << "\t" << rack3File << endl;
        if (!rack4)
            cerr << "\t" << rack4File << endl;
        if (!rack5)
            cerr << "\t" << rack5File << endl;
        if (!rack6)
            cerr << "\t" << rack6File << endl;
        if (!rack7)
            cerr << "\t" << rack7File << endl;
        if (!rack8)
            cerr << "\t" << rack8File << endl;

        return false;
    }

    // All loaded -- Create Entities & Animation Paths
    _box = new Entity(box, Matrix::scale(25.237011,25.237011,25.237011));

    _electrical = new Entity(electrical);
    _waterPipes = new Entity(pipes);
    _fans = new Entity(fans);

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
    cp.setRotation(Quat(-PI*3/4,Vec3(0,0,1)));
    _door[0]->path->insert(1,cp);

    // Door 1 - Front Right
    doorOffset = Vec3(47.047,-118.851,0);
    _door.push_back(new Entity(doorFR, Matrix::translate(doorOffset)));
    _door[1]->path = new AnimationPath();
    _door[1]->path->setLoopMode(AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(Quat(0,Vec3(0,0,1)));
    _door[1]->path->insert(0,cp);
    cp.setRotation(Quat(PI*3/4,Vec3(0,0,1)));
    _door[1]->path->insert(1,cp);

    // Door 2 - Front Inner
    doorOffset = Vec3(15.906,-104.9,0);
    _door.push_back(new Entity(doorFI, Matrix::translate(doorOffset)));
    _door[2]->path = new AnimationPath();
    _door[2]->path->setLoopMode(AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(Quat(0,Vec3(0,0,1)));
    _door[2]->path->insert(0,cp);
    cp.setRotation(Quat(PI/2,Vec3(0,0,1)));
    _door[2]->path->insert(1,cp);

    // Door 3 - Back Left
    doorOffset = Vec3(46.993,118.757,0);
    _door.push_back(new Entity(doorBL, Matrix::translate(doorOffset)));
    _door[3]->path = new AnimationPath();
    _door[3]->path->setLoopMode(AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(Quat(0,Vec3(0,0,1)));
    _door[3]->path->insert(0,cp);
    cp.setRotation(Quat(-PI*3/4,Vec3(0,0,1)));
    _door[3]->path->insert(1,cp);

    // Door 4 - Back Right
    doorOffset = Vec3(-47.117,113.765,0);
    _door.push_back(new Entity(doorBR, Matrix::translate(doorOffset)));
    _door[4]->path = new AnimationPath();
    _door[4]->path->setLoopMode(AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(Quat(0,Vec3(0,0,1)));
    _door[4]->path->insert(0,cp);
    cp.setRotation(Quat(PI*3/4,Vec3(0,0,1)));
    _door[4]->path->insert(1,cp);

    // Door 5 - Back Inner
    doorOffset = Vec3(18.339,96.197,0);
    _door.push_back(new Entity(doorBI, Matrix::translate(doorOffset)));
    _door[5]->path = new AnimationPath();
    _door[5]->path->setLoopMode(AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(Quat(0,Vec3(0,0,1)));
    _door[5]->path->insert(0,cp);
    cp.setRotation(Quat(-PI/2,Vec3(0,0,1)));
    _door[5]->path->insert(1,cp);

    // Door 6 - Back Inner Innera
    doorOffset = Vec3(15.505,81.835,0);
    _door.push_back(new Entity(doorBII, Matrix::translate(doorOffset)));
    _door[6]->path = new AnimationPath();
    _door[6]->path->setLoopMode(AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(Quat(0,Vec3(0,0,1)));
    _door[6]->path->insert(0,cp);
    cp.setRotation(Quat(-PI/2,Vec3(0,0,1)));
    _door[6]->path->insert(1,cp);

    // Racks
    Matrix rackMat;
    rackMat.setTrans(-26.962,-85.953,0);
    _rack.push_back(new Entity(rack1,rackMat));
    rackMat.setRotate(Quat(PI,Vec3(0,0,1)));
    rackMat.setTrans(-28.28,-33.44,0);
    _rack.push_back(new Entity(rack2,rackMat));
    rackMat.setTrans(-28.28,10.43,0);
    _rack.push_back(new Entity(rack3,rackMat));
    rackMat.setTrans(-28.28,54.31,0);
    _rack.push_back(new Entity(rack4,rackMat));
    rackMat.setRotate(Quat());
    rackMat.setTrans(28.16,54.31,0);
    _rack.push_back(new Entity(rack5,rackMat));
    rackMat.setTrans(28.16,10.44,0);
    _rack.push_back(new Entity(rack6,rackMat));
    rackMat.setTrans(28.16,-33.44,0);
    _rack.push_back(new Entity(rack7,rackMat));
    rackMat.setTrans(28.16,-77.31,0);
    _rack.push_back(new Entity(rack8,rackMat));

    for (int r = 0; r < _rack.size(); r++)
    {
        _rack[r]->path = new AnimationPath();
        _rack[r]->path->setLoopMode(AnimationPath::NO_LOOPING);
        cp.setRotation(_rack[r]->transform->getMatrix().getRotate());
        cp.setPosition(_rack[r]->transform->getMatrix().getTrans());
        _rack[r]->path->insert(0,cp);
        cp.setPosition(_rack[r]->transform->getMatrix().getTrans() + Vec3((r < 4)?25:-25,0,0));
        _rack[r]->path->insert(1,cp);
    }

    // Setup animation groups
    _door[0]->group.push_back(_door[1]);
    _door[1]->group.push_back(_door[0]);
    _door[3]->group.push_back(_door[4]);
    _door[4]->group.push_back(_door[3]);

    // Add it all to the box transform
    _box->addChild(_electrical);
    _box->addChild(_waterPipes);
    _box->addChild(_fans);

    for (int d = 0; d < _door.size(); d++)
        _box->addChild(_door[d]);

    for (int r = 0; r < _rack.size(); r++)
        _box->addChild(_rack[r]);

    loadHardwareFile();
    
cerr<<"Optimizing.\n";
    osgUtil::Optimizer o;
    o.optimize(_box->transform.get());
cerr<<"Done\n";

    // Menu Setup
    _displayComponentsMenu = new SubMenu("Display Components", "Display Components");
    _displayComponentsMenu->setCallback(this);
    _glMenu->addItem(_displayComponentsMenu);

    _displayFrameCheckbox = new MenuCheckbox("Box Frame",true);
    _displayFrameCheckbox->setCallback(this);
    _displayComponentsMenu->addItem(_displayFrameCheckbox);

    _displayDoorsCheckbox = new MenuCheckbox("Doors",true);
    _displayDoorsCheckbox->setCallback(this);
    _displayComponentsMenu->addItem(_displayDoorsCheckbox);

    _displayWaterPipesCheckbox = new MenuCheckbox("Water Pipes",true);
    _displayWaterPipesCheckbox->setCallback(this);
    _displayComponentsMenu->addItem(_displayWaterPipesCheckbox);

    _displayElectricalCheckbox = new MenuCheckbox("Electrical",true);
    _displayElectricalCheckbox->setCallback(this);
    _displayComponentsMenu->addItem(_displayElectricalCheckbox);

    _displayFansCheckbox = new MenuCheckbox("Fans",true);
    _displayFansCheckbox->setCallback(this);
    _displayComponentsMenu->addItem(_displayFansCheckbox);

    _displayRacksCheckbox = new MenuCheckbox("Racks",true);
    _displayRacksCheckbox->setCallback(this);
    _displayComponentsMenu->addItem(_displayRacksCheckbox);

    return true;
}

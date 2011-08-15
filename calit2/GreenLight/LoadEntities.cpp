#include "GreenLight.h"

#include <iostream>
#include <osgDB/ReadFile>
#include <osgUtil/Optimizer>

// local functions
osg::ref_ptr<osg::Node> loadModelFile(std::string file);

bool GreenLight::loadScene()
{
    // load model files
    std::string modelsDir = cvr::ConfigManager::getEntry("Plugin.GreenLight.ModelsDir");

    osg::ref_ptr<osg::Node> box = loadModelFile(modelsDir + "box.WRL");
    osg::ref_ptr<osg::Node> electrical = loadModelFile(modelsDir + "electrical.WRL");
    osg::ref_ptr<osg::Node> Pipes = loadModelFile(modelsDir + "waterpipes.WRL");
    osg::ref_ptr<osg::Node> doorFL = loadModelFile(modelsDir + "frontleft.WRL");
    osg::ref_ptr<osg::Node> doorFR = loadModelFile(modelsDir + "frontright.WRL");
    osg::ref_ptr<osg::Node> doorFI = loadModelFile(modelsDir + "frontinner.WRL");
    osg::ref_ptr<osg::Node> doorBL = loadModelFile(modelsDir + "backleft.WRL");
    osg::ref_ptr<osg::Node> doorBR = loadModelFile(modelsDir + "backright.WRL");
    osg::ref_ptr<osg::Node> doorBI = loadModelFile(modelsDir + "backinner.WRL");
    osg::ref_ptr<osg::Node> doorBII = loadModelFile(modelsDir + "backinnerinner.WRL");
    osg::ref_ptr<osg::Node> fans = loadModelFile(modelsDir + "fans_reduced.WRL");
    osg::ref_ptr<osg::Node> rack1 = loadModelFile(modelsDir + "rack1_c.WRL");
    osg::ref_ptr<osg::Node> rack2 = loadModelFile(modelsDir + "rack2_c.WRL");
    osg::ref_ptr<osg::Node> rack3 = loadModelFile(modelsDir + "rack3_c.WRL");
    osg::ref_ptr<osg::Node> rack4 = loadModelFile(modelsDir + "rack4_c.WRL");
    osg::ref_ptr<osg::Node> rack5 = loadModelFile(modelsDir + "rack5_c.WRL");
    osg::ref_ptr<osg::Node> rack6 = loadModelFile(modelsDir + "rack6_c.WRL");
    osg::ref_ptr<osg::Node> rack7 = loadModelFile(modelsDir + "rack7_c.WRL");
    osg::ref_ptr<osg::Node> rack8 = loadModelFile(modelsDir + "rack8_c.WRL");


    // all or nothing -- cancel loadScene if anythign failed
    if (!box || !electrical || !Pipes || !doorFL || !doorFR || !doorFI 
    || !doorBL || !doorBR || !doorBI || !doorBII || !fans || !rack1 || !rack2
    || !rack3 || !rack4 || !rack5 || !rack6 || !rack7 || !rack8)
    {
        return false;
    }

    // All loaded -- Create Entities & Animation Paths
    _box = new Entity(box, osg::Matrix::scale(25.237011,25.237011,25.237011));
    _electrical = new Entity(electrical);
    _waterPipes = new Entity(Pipes);
    _fans = new Entity(fans);

    osg::Vec3 doorOffset;
    osg::AnimationPath::ControlPoint cp;

    // Door 0 - Front Left
    doorOffset = osg::Vec3(-46.978,-118.808,0);
    _door.push_back(new Entity(doorFL, osg::Matrix::translate(doorOffset)));
    _door[0]->path = new osg::AnimationPath();
    _door[0]->path->setLoopMode(osg::AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(osg::Quat(0,osg::Vec3(0,0,1)));
    _door[0]->path->insert(0,cp);
    cp.setRotation(osg::Quat(-osg::PI*3/4,osg::Vec3(0,0,1)));
    _door[0]->path->insert(1,cp);

    // Door 1 - Front Right
    doorOffset = osg::Vec3(47.047,-118.851,0);
    _door.push_back(new Entity(doorFR, osg::Matrix::translate(doorOffset)));
    _door[1]->path = new osg::AnimationPath();
    _door[1]->path->setLoopMode(osg::AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(osg::Quat(0,osg::Vec3(0,0,1)));
    _door[1]->path->insert(0,cp);
    cp.setRotation(osg::Quat(osg::PI*3/4,osg::Vec3(0,0,1)));
    _door[1]->path->insert(1,cp);

    // Door 2 - Front Inner
    doorOffset = osg::Vec3(15.906,-104.9,0);
    _door.push_back(new Entity(doorFI, osg::Matrix::translate(doorOffset)));
    _door[2]->path = new osg::AnimationPath();
    _door[2]->path->setLoopMode(osg::AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(osg::Quat(0,osg::Vec3(0,0,1)));
    _door[2]->path->insert(0,cp);
    cp.setRotation(osg::Quat(osg::PI/2,osg::Vec3(0,0,1)));
    _door[2]->path->insert(1,cp);

    // Door 3 - Back Left
    doorOffset = osg::Vec3(46.993,118.757,0);
    _door.push_back(new Entity(doorBL, osg::Matrix::translate(doorOffset)));
    _door[3]->path = new osg::AnimationPath();
    _door[3]->path->setLoopMode(osg::AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(osg::Quat(0,osg::Vec3(0,0,1)));
    _door[3]->path->insert(0,cp);
    cp.setRotation(osg::Quat(-osg::PI*3/4,osg::Vec3(0,0,1)));
    _door[3]->path->insert(1,cp);

    // Door 4 - Back Right
    doorOffset = osg::Vec3(-47.117,113.765,0);
    _door.push_back(new Entity(doorBR, osg::Matrix::translate(doorOffset)));
    _door[4]->path = new osg::AnimationPath();
    _door[4]->path->setLoopMode(osg::AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(osg::Quat(0,osg::Vec3(0,0,1)));
    _door[4]->path->insert(0,cp);
    cp.setRotation(osg::Quat(osg::PI*3/4,osg::Vec3(0,0,1)));
    _door[4]->path->insert(1,cp);

    // Door 5 - Back Inner
    doorOffset = osg::Vec3(18.339,96.197,0);
    _door.push_back(new Entity(doorBI, osg::Matrix::translate(doorOffset)));
    _door[5]->path = new osg::AnimationPath();
    _door[5]->path->setLoopMode(osg::AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(osg::Quat(0,osg::Vec3(0,0,1)));
    _door[5]->path->insert(0,cp);
    cp.setRotation(osg::Quat(-osg::PI/2,osg::Vec3(0,0,1)));
    _door[5]->path->insert(1,cp);

    // Door 6 - Back Inner Innera
    doorOffset = osg::Vec3(15.505,81.835,0);
    _door.push_back(new Entity(doorBII, osg::Matrix::translate(doorOffset)));
    _door[6]->path = new osg::AnimationPath();
    _door[6]->path->setLoopMode(osg::AnimationPath::NO_LOOPING);
    cp.setPosition(doorOffset);
    cp.setRotation(osg::Quat(0,osg::Vec3(0,0,1)));
    _door[6]->path->insert(0,cp);
    cp.setRotation(osg::Quat(-osg::PI/2,osg::Vec3(0,0,1)));
    _door[6]->path->insert(1,cp);

    // Racks
    osg::Matrix rackMat;
    rackMat.setTrans(-26.962,-77.31,0);
    _rack.push_back(new Entity(rack1,rackMat));
    rackMat.setRotate(osg::Quat(osg::PI,osg::Vec3(0,0,1)));
    rackMat.setTrans(-28.28,-33.44,0);
    _rack.push_back(new Entity(rack2,rackMat));
    rackMat.setTrans(-28.28,10.43,0);
    _rack.push_back(new Entity(rack3,rackMat));
    rackMat.setTrans(-28.28,54.31,0);
    _rack.push_back(new Entity(rack4,rackMat));
    rackMat.setRotate(osg::Quat());
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
        _rack[r]->path = new osg::AnimationPath();
        _rack[r]->path->setLoopMode(osg::AnimationPath::NO_LOOPING);
        cp.setRotation(_rack[r]->transform->getMatrix().getRotate());
        cp.setPosition(_rack[r]->transform->getMatrix().getTrans());
        _rack[r]->path->insert(0,cp);
        cp.setPosition(_rack[r]->transform->getMatrix().getTrans() + osg::Vec3((r < 4)?25:-25,0,0));
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

    // populate racks
    parseHardwareFile();
    
    std::cerr<<"Optimizing.\n";
    osgUtil::Optimizer o;
    o.optimize(_box->mainNode.get());
    o.optimize(_electrical->mainNode.get());
    o.optimize(_waterPipes->mainNode.get());
    o.optimize(_fans->mainNode.get());
    for (int d = 0; d < _door.size(); d++)
        o.optimize(_door[d]->mainNode.get());
    for (int r = 0; r < _rack.size(); r++)
        o.optimize(_rack[r]->mainNode.get());

    // Menu Setup
    _displayComponentsMenu = new cvr::SubMenu("Display Components", "Display Components");
    _displayComponentsMenu->setCallback(this);
    _glMenu->addItem(_displayComponentsMenu);

    _xrayViewCheckbox = new cvr::MenuCheckbox("X-ray Vision",false);
    _xrayViewCheckbox->setCallback(this);
    _displayComponentsMenu->addItem(_xrayViewCheckbox);

    _displayFrameCheckbox = new cvr::MenuCheckbox("Box Frame",true);
    _displayFrameCheckbox->setCallback(this);
    _displayComponentsMenu->addItem(_displayFrameCheckbox);

    _displayDoorsCheckbox = new cvr::MenuCheckbox("Doors",true);
    _displayDoorsCheckbox->setCallback(this);
    _displayComponentsMenu->addItem(_displayDoorsCheckbox);

    _displayWaterPipesCheckbox = new cvr::MenuCheckbox("Water Pipes",true);
    _displayWaterPipesCheckbox->setCallback(this);
    _displayComponentsMenu->addItem(_displayWaterPipesCheckbox);

    _displayElectricalCheckbox = new cvr::MenuCheckbox("Electrical",true);
    _displayElectricalCheckbox->setCallback(this);
    _displayComponentsMenu->addItem(_displayElectricalCheckbox);

    _displayFansCheckbox = new cvr::MenuCheckbox("Fans",true);
    _displayFansCheckbox->setCallback(this);
    _displayComponentsMenu->addItem(_displayFansCheckbox);

    _displayRacksCheckbox = new cvr::MenuCheckbox("Racks",true);
    _displayRacksCheckbox->setCallback(this);
    _displayComponentsMenu->addItem(_displayRacksCheckbox);

    _displayComponentTexturesCheckbox = new cvr::MenuCheckbox("Component Textures",true);
    _displayComponentTexturesCheckbox->setCallback(this);
    _displayComponentsMenu->addItem(_displayComponentTexturesCheckbox);
    Component::_displayTexturesUni->setElement(0,_displayComponentTexturesCheckbox->getValue());
    Component::_displayTexturesUni->dirty();

    _powerMenu = new cvr::SubMenu("Power Consumption", "Power Consumption");
    _powerMenu->setCallback(this);
    _glMenu->addItem(_powerMenu);

    _loadPowerButton = new cvr::MenuButton("Load Recent Data");
    _loadPowerButton->setCallback(this);
    _powerMenu->addItem(_loadPowerButton);

    _pollHistoricalDataCheckbox = new cvr::MenuCheckbox("Poll Historical Data",false);
    _pollHistoricalDataCheckbox->setCallback(this);
    _powerMenu->addItem(_pollHistoricalDataCheckbox);

    _hardwareSelectionMenu = new cvr::SubMenu("Hardware Selection", "Hardware Selection");
    _hardwareSelectionMenu->setCallback(this);
    _glMenu->addItem(_hardwareSelectionMenu);

    _selectionModeCheckbox = new cvr::MenuCheckbox("Selection Enabled",false);
    _selectionModeCheckbox->setCallback(this);
    _hardwareSelectionMenu->addItem(_selectionModeCheckbox);

    _hoverDialog = new cvr::DialogPanel(400, "Intersected Component");
    _hoverDialog->setText("(nothing)");
    _hoverDialog->setVisible(_selectionModeCheckbox->getValue());

    if (_cluster.size() > 0)
    {
        _selectClusterMenu = new cvr::SubMenu("Cluster Selection", "Selected Clusters");
        _selectClusterMenu->setCallback(this);
        // Added to _hardwareSelectionMenu when selection mode is enabled

        std::map< std::string, std::set< Component * > * >::iterator cit;
        for (cit = _cluster.begin(); cit != _cluster.end(); cit++)
        {
            cvr::MenuCheckbox * checkbox = new cvr::MenuCheckbox(cit->first, true);
            checkbox->setCallback(this);
            _selectClusterMenu->addItem(checkbox);
            _clusterCheckbox.insert(checkbox);
        }
    }

    _selectAllButton = new cvr::MenuButton("Select All");
    _selectAllButton->setCallback(this);
    // Added to _hardwareSelectionMenu when selection mode is enabled

    _deselectAllButton = new cvr::MenuButton("Deselect All");
    _deselectAllButton->setCallback(this);
    // Added to _hardwareSelectionMenu when selection mode is enabled

    // Back face culling
    _box->transform->getOrCreateStateSet()->setMode(GL_CULL_FACE, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);


    return true;
}

osg::ref_ptr<osg::Node> loadModelFile(std::string file)
{
    std::cerr << "Loading " << file << "... ";
    osg::ref_ptr<osg::Node> model = osgDB::readNodeFile(file);

    if (!model)
        std::cerr << "FAILED." << std::endl;
    else
        std::cerr << "done." << std::endl;

    return model.get();
}

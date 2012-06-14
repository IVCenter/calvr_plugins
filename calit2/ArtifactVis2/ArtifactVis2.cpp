#include "ArtifactVis2.h"
#include "vvtokenizer.h"

#ifdef WITH_OSSIMPLANET
#include "../OssimPlanet/OssimPlanet.h"
#endif

#include <iostream>
#include <sstream>
#include <iostream>
#include <fstream>

#include <cmath>
#include <algorithm>
#include <vector>
#include <sys/stat.h>
#include <time.h>

#include <cstdlib>

#ifndef WIN32
#include <unistd.h>
#else
#include <direct.h>
#endif

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/PluginManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/InteractionManager.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrUtil/LocalToWorldVisitor.h>
#include <PluginMessageType.h>
#include <cvrKernel/ComController.h>
//#include <cvrInput/TrackerVRPN.h>


#include <osg/Vec4>

#include <osgUtil/SceneView>
#include <osg/Camera>


#include <osg/CullFace>
#include <osg/Matrix>
#include <osg/ShapeDrawable>
#include <osgDB/ReadFile>
#include <osg/PositionAttitudeTransform>
#include <osg/PolygonMode>
//#include </home/calvr/calvr_plugins/calit2/SpaceNavigator/SpaceNavigator.h>

#include <mxml.h>

#include <vrpn_Tracker.h>
#include <quat.h>

using namespace osg;
using namespace std;
using namespace cvr;

CVRPLUGIN(ArtifactVis2)
ArtifactVis2 * ArtifactVis2::_artifactvis2 = NULL;
ArtifactVis2::ArtifactVis2()
{

}
ArtifactVis2::~ArtifactVis2()
{

}
/*
* Returns an instance for use with other programs.
*/

ArtifactVis2 * ArtifactVis2::instance()
{
    if(!_artifactvis2)
    {
        _artifactvis2 = new ArtifactVis2();
    }
    return _artifactvis2;
}

void ArtifactVis2::message(int type, char * data)
{
    /*
    if(type == OE_TRANSFORM_POINTER)
    {
	OsgEarthRequest * request = (OsgEarthRequest*) data;

    }
    */
    _osgearth = true;

}
bool ArtifactVis2::init()
{






    std::cerr << "ArtifactVis2 init\n";
    _root = new osg::MatrixTransform();

    _tablesMenu = NULL;

    loadModels();

    //Algorithm for generating colors based on DC.
    for(int i = 0; i < 729; i++)
    {
        _colors[i] = Vec4(1-float((i%9)*0.125),1-float(((i/9)%9)*0.125),1-float(((i/81)%9)*0.125),1);
    }

    //Menu Setup:
    _avMenu = new SubMenu("ArtifactVis2", "ArtifactVis2");
    _avMenu->setCallback(this);

    _displayMenu = new SubMenu("Display");
    _avMenu->addItem(_displayMenu);

    //Generates the menu for selecting models to load
    setupSiteMenu();

    //Generates the menus to toggle each query on/off.
    setupQuerySelectMenu();

    //Generates the menus to query each table.
    setupTablesMenu();
    for(int i = 0; i < _tables.size(); i++)
    {
        setupQueryMenu(_tables[i]);
    }
    if(_tablesMenu)
    {
	_avMenu->addItem(_tablesMenu);
    }

    _selectArtifactCB = new MenuCheckbox("Select Artifact",false);
    _selectArtifactCB->setCallback(this);
    _avMenu->addItem(_selectArtifactCB);

    _scaleBar = new MenuCheckbox("Scale Bar",false); //new
    _scaleBar->setCallback(this);
    _avMenu->addItem(_scaleBar);

    _selectCB = new MenuCheckbox("Select box", false);
    _selectCB->setCallback(this);
    //_avMenu->addItem(_selectCB);  //Removed for now until fixed.

    _defaultMaterial =new Material();
    _defaultMaterial->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    _defaultMaterial->setDiffuse(Material::FRONT,osg::Vec4(1.0,1.0,1.0,1.0));

    //create wireframe selection box
    osg::Box * sbox = new osg::Box(osg::Vec3(0,0,0),1.0,1.0,1.0);
    osg::ShapeDrawable * sd = new osg::ShapeDrawable(sbox);
    osg::StateSet * stateset = sd->getOrCreateStateSet();
    osg::PolygonMode * polymode = new osg::PolygonMode;
    polymode->setMode(osg::PolygonMode::FRONT_AND_BACK,osg::PolygonMode::LINE);
    stateset->setAttributeAndModes(polymode,osg::StateAttribute::OVERRIDE|osg::StateAttribute::ON);

    osg::Geode * geo = new osg::Geode();
    geo->addDrawable(sd);

    _selectBox = new osg::MatrixTransform();
    _selectBox->addChild(geo);

    // create select mark for wand
    osg::Sphere * ssph = new osg::Sphere(osg::Vec3(0,0,0),10);
    sd = new osg::ShapeDrawable(ssph);
    sd->setColor(osg::Vec4(1.0,0,0,1.0));
    stateset = sd->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    stateset->setAttributeAndModes(_defaultMaterial,osg::StateAttribute::ON);

    geo = new osg::Geode();
    geo->addDrawable(sd);

    _selectMark = new osg::MatrixTransform();
    _selectMark->addChild(geo);

    MenuSystem::instance()->addMenuItem(_avMenu);
    SceneManager::instance()->getObjectsRoot()->addChild(_root);

    _sphereRadius = 0.03;
    _activeArtifact  = -1;

    _picFolder = ConfigManager::getEntry("value","Plugin.ArtifactVis2.PicFolder","");

    //Tabbed dialog for selecting artifacts
    _artifactPanel = new TabbedDialogPanel(400,30,4,"Selected Artifact","Plugin.ArtifactVis2.ArtifactPanel");
    _artifactPanel->addTextTab("Info","");
    _artifactPanel->addTextureTab("Side","");
    _artifactPanel->addTextureTab("Top","");
    _artifactPanel->addTextureTab("Bottom","");
    _artifactPanel->setVisible(false);
    _artifactPanel->setActiveTab("Info");

    _selectionStatsPanel = new DialogPanel(450,"Selection Stats","Plugin.ArtifactVis2.SelectionStatsPanel");
    _selectionStatsPanel->setVisible(false);
    //_testA = 0;
    std::cerr << "ArtifactVis2 init done.\n";
    //std::cerr << selectArtifactSelected() << "\n";

    //SpaceNavigator Included
    statusSpnav = false; //made global
    if(ComController::instance()->isMaster())
    {
	if(spnav_open()==-1)
	{
	    cerr << "SpaceNavigator: Failed to connect to the space navigator daemon" << endl;
	}
	else
	{
	    statusSpnav = true;
	}
	ComController::instance()->sendSlaves((char *)&statusSpnav, sizeof(bool));
    }
    else
    {
	ComController::instance()->readMaster((char *)&statusSpnav, sizeof(bool));
    }

    transMult = ConfigManager::getFloat("Plugin.SpaceNavigator.TransMult", 1.0);
    rotMult = ConfigManager::getFloat("Plugin.SpaceNavigator.RotMult", 1.0);
    cout << "Sp:" << transMult;
    transcale = -0.05 * transMult;
    rotscale = -0.000009 * rotMult;

    //Kinect Test Init
    //remote = new vrpn_Tracker_Remote("Tracker0@192.168.0.100");
    //remote->register_change_handler(NULL, tracker_handler);
    //remote->shutup = true;
    if(remote)
    {
        cout << "Connection Established\n";
    }
    else
    {

        cout << "Error with Connection\n";
    }


    //........................................................................
    return true;
}


/*
 Loads in all existing models of the form 3dModelFolder/DC/DC.obj, where DC is the two letter DCode.
 Has space for ALL possible DC codes (26^2).
*/
void ArtifactVis2::loadModels()
{
    for(int i = 0; i < 26; i++)
    {
        for(int j = 0 ; j < 26; j++)
        {
            char c1 = i+65;
            char c2 = j+65;
            stringstream ss;
            ss << c1 << c2;
            string dc = ss.str();
            string modelPath = ConfigManager::getEntry("Plugin.ArtifactVis2.3DModelFolder").append("dcode_models/Finished/obj/"+dc+"/"+dc+".obj");
            if(modelExists(modelPath.c_str()))
            {
                cout << "Model " << modelPath << " Exists \n";
                _models[i*26+j] = osgDB::readNodeFile(modelPath);
                _modelLoaded[i*26+j] = true;
            }
            else
            {
               // cout << "Model " << modelPath << " Not Exists \n";
                _models[i*26+j] = NULL;
                _modelLoaded[i*26+j] = false;
            }
        }
    }
}
bool ArtifactVis2::processEvent(InteractionEvent * event)
{
    TrackedButtonInteractionEvent * tie = event->asTrackedButtonEvent();
    if(!tie)
    {
	return false;
    }

    if((event->getInteraction() == BUTTON_DOWN) && tie->getHand() == 0 && tie->getButton() == 0)
    {

        //bang
    }

    if((event->getInteraction() == BUTTON_DOWN || event->getInteraction() == BUTTON_DOUBLE_CLICK) && tie->getHand() == 0 && tie->getButton() == 0)
    {
    //Artifact Selection
	if(_selectArtifactCB->getValue() )
	{
	    if(!_selectCB->getValue())
	    {
                osg::Matrix w2l = PluginHelper::getWorldToObjectTransform();
		osg::Vec3 start(0,0,0);
		osg::Vec3 end(0,1000000,0);

		start = start * tie->getTransform() * w2l;
		end = end * tie->getTransform() * w2l;

		int index = -1;
                int queryIndex = -1;
		double distance;
                for(int q = 0; q < _query.size(); q++)
                    {
                    vector<Artifact*> artifacts = _query[q]->artifacts;
                    if(_query[q]->active)
                    {
                        for(int i = 0; i < artifacts.size(); i++)
                        {
                            osg::Vec3 num = (artifacts[i]->modelPos - start) ^ (artifacts[i]->modelPos - end);
                            osg::Vec3 denom = end - start;
                            double point2line = num.length() / denom.length();
                            if(point2line <= _sphereRadius)
                            {
                                double point2start = (artifacts[i]->modelPos - start).length2();
                                if(index == -1 || point2start < distance)
                                {
                                    distance = point2start;
                                    index = i;
                                    queryIndex = q;
			        }
		            }
		        }
                    }
                }
		if(index != -1)
		{
		    //std::cerr << "Got sphere intersection with index " << index << std::endl;
		    setActiveArtifact(index,queryIndex);
		    return true;
		}
	    }
	}
        //Box selection
	else if(_selectCB->getValue() && tie->getInteraction() == BUTTON_DOUBLE_CLICK)
	{
            osg::Matrix w2l = PluginHelper::getWorldToObjectTransform();
	    if(!_selectActive)
	    {
		_selectStart = osg::Vec3(0,1000,0);
		_selectStart = _selectStart * tie->getTransform() * w2l;
		_selectActive = true;
	    }
	    else
	    {
		_selectCurrent = osg::Vec3(0,1000,0);
		_selectCurrent = _selectCurrent * tie->getTransform() * w2l;
		_selectActive = false;
	    }
	    return true;
	}
    }

    return false;
}

//Gets the string of the query that would be sent to PGSQL via ArchInterface.
//Includes only the current 'OR' statement, not previous ones. Those are stored in the current_query variable for Tables.
std::string ArtifactVis2::getCurrentQuery(Table * t)
{
        std::stringstream ss;
        std::vector<cvr::SubMenu *>::iterator menu = t->querySubMenu.begin();
        int index = 0;
        bool conditionSelected = false;
        ss << "(";
        for(; menu < t->querySubMenu.end(); menu++)
        {
           if(!t->queryOptions[index]->firstOn().empty())
           {
             if(conditionSelected) ss << " AND ";
             ss << (*menu)->getName() << "=\'" << t->queryOptions[index]->firstOn() << "\'";
             conditionSelected = true;
           }
           index++;
        }
        for(int i = 0; i < t->querySlider.size(); i++ )
        {
            if(t->querySlider[i]->getValue())
            {
               if(conditionSelected) ss << " AND ";
               ss <<  t->querySubMenuSlider[i]->getName() << "=\'" << t->queryOptionsSlider[i]->getValue() << "\'";
               conditionSelected = true;
            }
        }
        ss << ")";
       if(!conditionSelected)
           {
           return "";
       }
       return ss.str();
}

void ArtifactVis2::menuCallback(MenuItem* menuItem)
{
#ifdef WITH_OSSIMPLANET
    if(!_ossim&&OssimPlanet::instance())
    {
        _ossim = true;
        _sphereRadius /= 1000;
        cout << "Loaded into OssimPlanet." << endl;
    }
#endif
    for(int i = 0; i < _showModelCB.size(); i++)
    {
        if(menuItem == _showModelCB[i])
        {
            if(_showModelCB[i]->getValue())
	    {
                if(_siteRoot[i]->getNumChildren()==0)
		    readSiteFile(i);
                if(!_ossim)
	        _root->addChild(_siteRoot[i]);
	    }
	    else
	    {
	        _root->removeChild(_siteRoot[i]);
	    }
        }
        if(menuItem == _reloadModel[i])
        {
            _root->removeChild(_siteRoot[i].get());
            reloadSite(i);
            readSiteFile(i);
            _root->addChild(_siteRoot[i].get());




        }
    }
    for(int i = 0; i < _showPCCB.size(); i++)
    {
        if(menuItem == _showPCCB[i])
        {
            if(_showPCCB[i]->getValue())
	    {
                if(_pcRoot[i]->getNumChildren()==0)
		    readPointCloud(i);

	        _root->addChild(_pcRoot[i].get());
	    }
	    else
	    {
	        _root->removeChild(_pcRoot[i].get());
	    }
        }
        if(menuItem == _reloadPC[i])
        {
            _root->removeChild(_pcRoot[i].get());
            reloadSite(i);
            readPointCloud(i);
            _root->addChild(_pcRoot[i].get());




        }
    }
    std::vector<Table *>::iterator t = _tables.begin();
    for(; t < _tables.end(); t++)
    {
        for(int i = 0; i < (*t)->querySlider.size(); i++)
        {
            if(menuItem == (*t)->querySlider[i])
                (*t)->query_view->setText((*t)->current_query+getCurrentQuery((*t)));
        }
        for(int i = 0; i < (*t)->queryOptionsSlider.size(); i++)
        {
            if(menuItem == (*t)->queryOptionsSlider[i])
                (*t)->query_view->setText((*t)->current_query+getCurrentQuery((*t)));
        }
        for(int i = 0; i < (*t)->queryOptions.size(); i++)
        {
            if(menuItem == (*t)->queryOptions[i])
                (*t)->query_view->setText((*t)->current_query+getCurrentQuery((*t)));
        }
        if(menuItem == (*t)->clearConditions)
        {
            clearConditions((*t));
            (*t)->current_query = "";
        }
        if(menuItem == (*t)->addOR)
        {
            (*t)->current_query.append(getCurrentQuery((*t)));
            (*t)->current_query.append(" OR ");
            clearConditions((*t));
        }
        if(menuItem == (*t)->removeOR)
        {
            (*t)->current_query = (*t)->current_query.substr(0,(*t)->current_query.rfind("("));
        }
        if(menuItem == (*t)->genQuery)
        {
            //_query[0]->sphereRoot->setNodeMask(0);
            //_query[1]->sphereRoot->setNodeMask(0);
            bool status;
            if(ComController::instance()->isMaster())
            {
                std::stringstream ss;
                ss <<  "./ArchInterface -b ";
                ss << "\"";
                ss << (*t)->name;
                ss << "\" ";
                ss << "\"";
                ss << (*t)->current_query;
                ss << getCurrentQuery((*t));
                ss << "\"";
		const char* current_path = getcwd(NULL, 0);
                chdir(ConfigManager::getEntry("Plugin.ArtifactVis2.ArchInterfaceFolder").c_str());
                cout <<ss.str().c_str() << endl;
                system(ss.str().c_str());
		chdir(current_path);
    	        ComController::instance()->sendSlaves(&status,sizeof(bool));
            }
            else
            {
    	        ComController::instance()->readMaster(&status,sizeof(bool));
            }
            cout << (*t)->name << "\n";
            if((*t)->name.find("_a",0)!=string::npos)
            {
                cout << "query0 \n";
                cout << _query[0]->kmlPath << "\n";
                readQuery(_query[0]);
            }
            else
            {
                cout << "query1 \n";
                cout << _query[1]->kmlPath << "\n";
                readQuery(_query[1]);
                _root->addChild(_query[1]->sphereRoot);
                _query[1]->sphereRoot->setNodeMask(0xffffffff);

            }
            if(_queryOption[0]->getValue())
            {
                 if((*t)->name.find("_a",0)!=string::npos)
                 {
                     displayArtifacts(_query[0]);
                     _query[0]->updated = false;
                 }
            }
            if(_queryOption[1]->getValue())
            {
                 if((*t)->name.find("_a",0)==string::npos)
                 {
                     _query[1]->updated = false;
                 }
            }
            setupQuerySelectMenu();
        }
        if(menuItem == (*t)->saveQuery)
        {
	    const char* current_path = getcwd(NULL, 0);
            chdir(ConfigManager::getEntry("Plugin.ArtifactVis2.ArchInterfaceFolder").c_str());
            if((*t)->name.find("_a",0)!=string::npos)
            {
                bool status;
                if(ComController::instance()->isMaster())
                {
                    system("./ArchInterface -r \"query\"");
    	            ComController::instance()->sendSlaves(&status,sizeof(bool));
                }
                else
                {
    	            ComController::instance()->readMaster(&status,sizeof(bool));
                }

                setupQuerySelectMenu();
            }
            else
            {
                bool status;
                if(ComController::instance()->isMaster())
                {
                    system("./ArchInterface -r \"querp\"");
    	            ComController::instance()->sendSlaves(&status,sizeof(bool));
                }
                else
                {
    	            ComController::instance()->readMaster(&status,sizeof(bool));
                }
                setupQuerySelectMenu();
            }
	    chdir(current_path);
        }
    }
    for(int i = 0; i < _queryOption.size(); i++)
    {
        if(menuItem==_queryOption.at(i))
        {
            _query[i]->active = _queryOption[i]->getValue();
            if(_queryOption[i]->getValue())
            {
                if(_query[i]->updated)
                {
                    if(_query[i]->sf)
                        displayArtifacts(_query[i]);
                    _query[i]->updated = false;
                }
		_root->addChild(_query[i]->sphereRoot);
                _query[i]->sphereRoot->setNodeMask(0xffffffff);
            }
            else
            {
                _query[i]->sphereRoot->setNodeMask(0);
            }
        }
        if(menuItem==_eraseQuery[i])
        {
            bool status;
            if(ComController::instance()->isMaster())
            {
		const char* current_path = getcwd(NULL, 0);
                chdir(ConfigManager::getEntry("Plugin.ArtifactVis2.ArchInterfaceFolder").c_str());
                stringstream ss;
                ss << "./ArchInterface -n \"" << _query[i]->name << "\"";
                cout << ss.str() << endl;
                system(ss.str().c_str());
		chdir(current_path);
    	        ComController::instance()->sendSlaves(&status,sizeof(bool));
            }
            else
            {
    	        ComController::instance()->readMaster(&status,sizeof(bool));
            }
            _root->removeChild(_query[i]->sphereRoot);
            _query.erase(_query.begin()+i);
            setupQuerySelectMenu();
        }
        if(menuItem==_centerQuery[i])
        {
	    /*
            Matrixd mat;
            mat.makeTranslate(_query[i]->center*-1);
            cout << _query[i]->center.x() << ", " << _query[i]->center.y() << ", " << _query[i]->center.z() << endl;
            SceneManager::instance()->setObjectMatrix(mat);
	    */
        }
 	if(menuItem==_toggleLabel[i])
        {
            if(_toggleLabel[i]->getValue())
            {
                //cout << "on\n";

             //cout << _query[i]->artifacts.size() << "\n";
             for(int j = 0; j < _query[i]->artifacts.size(); j++)
              _query[i]->artifacts[j]->showLabel = true;

            }
            else
            {
                //cout << "off\n";
                //cout << _query[i]->artifacts.size() << "\n";
                for(int j = 0; j < _query[i]->artifacts.size(); j++)
              _query[i]->artifacts[j]->showLabel = false;

            }
        }
    }
    if(menuItem == _locusDisplayMode)
    {
        cout <<"LocusDisplayMode=" << _locusDisplayMode->firstOn() << "\n";
        if(_locusDisplayMode->firstOn()=="Wireframe")
        {
            for(int i = 0; i < _query.size(); i++)
            {
                if(!_query[i]->sf)
                {
                    for(int j = 0; j < _query[i]->loci.size(); j++)
                    {
                        _query[i]->sphereRoot->removeChild(_query[i]->loci[j]->top_geode.get());
                        _query[i]->sphereRoot->removeChild(_query[i]->loci[j]->fill_geode.get());
                        _query[i]->sphereRoot->removeChild(_query[i]->loci[j]->line_geode.get());
                        _query[i]->sphereRoot->addChild(_query[i]->loci[j]->fill_geode.get());
                        _query[i]->sphereRoot->addChild(_query[i]->loci[j]->line_geode.get());
                        StateSet * state = _query[i]->loci[j]->fill_geode->getOrCreateStateSet();
                        Material * mat = dynamic_cast<Material*>(state->getAttribute(StateAttribute::MATERIAL,0));
                        mat->setAlpha(Material::FRONT_AND_BACK,0.01);
                    }
                }
            }
        }
        else if(_locusDisplayMode->firstOn()=="Solid")
        {
            for(int i = 0; i < _query.size(); i++)
            {
                if(!_query[i]->sf)
                {
                    for(int j = 0; j < _query[i]->loci.size(); j++)
                    {
                        _query[i]->sphereRoot->removeChild(_query[i]->loci[j]->top_geode.get());
                        _query[i]->sphereRoot->removeChild(_query[i]->loci[j]->fill_geode.get());
                        _query[i]->sphereRoot->removeChild(_query[i]->loci[j]->line_geode.get());
                        _query[i]->sphereRoot->addChild(_query[i]->loci[j]->fill_geode.get());
                        _query[i]->sphereRoot->addChild(_query[i]->loci[j]->line_geode.get());
                        StateSet * state = _query[i]->loci[j]->fill_geode->getOrCreateStateSet();
                        Material * mat = dynamic_cast<Material*>(state->getAttribute(StateAttribute::MATERIAL,0));
                        mat->setAlpha(Material::FRONT_AND_BACK,0.99);
                    }
                }
            }
        }
        else if(_locusDisplayMode->firstOn()=="Top")
        {
            for(int i = 0; i < _query.size(); i++)
            {
                if(!_query[i]->sf)
                {
                    cout << _query[i]->loci.size() << " size\n";
                    for(int j = 0; j < _query[i]->loci.size(); j++)
                    {
                        //cout << j << "\n";
                        //_query[i]->sphereRoot->removeChild(_query[i]->loci[j]->top_geode);
                        _query[i]->sphereRoot->removeChild(_query[i]->loci[j]->fill_geode.get());
                        _query[i]->sphereRoot->removeChild(_query[i]->loci[j]->line_geode.get());
                        //osg::Geode * test=_query[i]->loci[j]->top_geode;

                        _query[i]->sphereRoot->addChild(_query[i]->loci[j]->top_geode.get());

                        //StateSet * state = _query[i]->loci[j]->top_geode->getOrCreateStateSet();
                        //Material * mat = dynamic_cast<Material*>(state->getAttribute(StateAttribute::MATERIAL,0));
                        //mat->setAlpha(Material::FRONT_AND_BACK,0.4);

                    }
                }
            }
        }
        else
        {
            for(int i = 0; i < _query.size(); i++)
            {
                if(!_query[i]->sf)
                {
                    for(int j = 0; j < _query[i]->loci.size(); j++)
                    {
                        _query[i]->sphereRoot->removeChild(_query[i]->loci[j]->top_geode.get());
                        _query[i]->sphereRoot->removeChild(_query[i]->loci[j]->fill_geode.get());
                        _query[i]->sphereRoot->removeChild(_query[i]->loci[j]->line_geode.get());
                        _query[i]->sphereRoot->addChild(_query[i]->loci[j]->fill_geode.get());
                        _query[i]->sphereRoot->addChild(_query[i]->loci[j]->line_geode.get());
                        StateSet * state = _query[i]->loci[j]->fill_geode->getOrCreateStateSet();
                        Material * mat = dynamic_cast<Material*>(state->getAttribute(StateAttribute::MATERIAL,0));
                        mat->setAlpha(Material::FRONT_AND_BACK,0.4);
                    }
                }
            }
        }
    }
    if(menuItem == _scaleBar)
    {
        if(_scaleBar->getValue())
        {
            osg::Matrix w2l = PluginHelper::getWorldToObjectTransform();
            osg::Vec3d start(0,0,0);
            start = start * w2l;
            cout << start.x() << " " << start.y() << " " << start.z() << "\n";
            std::cerr << selectArtifactSelected() << "\n";
            loadScaleBar(start);



        }
        else
        {
            _root->removeChild(_scaleBarModel);
        }
    }
    if(menuItem == _selectArtifactCB)
    {
	if(_selectArtifactCB->getValue())
	{
	    if(_selectCB->getValue())
	    {
		_selectCB->setValue(false);
		menuCallback(_selectCB);
	    }
	}
	_artifactPanel->setVisible(_selectArtifactCB->getValue());
	if (!_selectArtifactCB->getValue()) //New Add
	{
        _root->removeChild(_selectModelLoad.get());
	}
    }
    if(menuItem == _selectCB)
    {
	if(_selectCB->getValue())
	{
	    if(_selectArtifactCB->getValue())
	    {
		_selectArtifactCB->setValue(false);
		menuCallback(_selectArtifactCB);
	    }
            for(int q = 0; q < _query.size(); q++)
            {
                vector<Artifact*> artifacts = _query[q]->artifacts;
 	        for(int i = 0; i < artifacts.size(); i++)
	        {
		    artifacts[i]->selected = false;
		    osg::ShapeDrawable * sd = dynamic_cast<osg::ShapeDrawable*>(artifacts[i]->drawable);
		    if(sd)
		    {
		        osg::Vec4 color = sd->getColor();
		        color.x() = color.x() * 0.5;
		        color.y() = color.y() * 0.5;
		        color.z() = color.z() * 0.5;
		        sd->setColor(color);
		    }
	        }
            }
	    _selectStart = osg::Vec3(0,0,0);
	    _selectCurrent = osg::Vec3(0,0,0);
	    _root->addChild(_selectBox);
	    if(PluginHelper::getNumHands())
	    {
		PluginHelper::getScene()->addChild(_selectMark);
	    }
	    _selectionStatsPanel->setVisible(true);
	}
	else
	{
            for(int q = 0; q < _query.size(); q++)
            {
                vector<Artifact*> artifacts = _query[q]->artifacts;
	        for(int i = 0; i < artifacts.size(); i++)
	        {
		    if(!artifacts[i]->selected)
		    {
		        osg::ShapeDrawable * sd = dynamic_cast<osg::ShapeDrawable*>(artifacts[i]->drawable);
		        if(sd)
		        {
			    osg::Vec4 color = sd->getColor();
			    color.x() = color.x() * 2.0;
			    color.y() = color.y() * 2.0;
			    color.z() = color.z() * 2.0;
			    sd->setColor(color);
		        }
		    }
	       }
            }
	    _root->removeChild(_selectBox);
	    if(PluginHelper::getNumHands())
	    {
		PluginHelper::getScene()->removeChild(_selectMark);
	    }
	    _selectionStatsPanel->setVisible(false);
	}
	_selectActive = false;
    }
}
//Removes all conditions set in the query for the selected table.
void ArtifactVis2::clearConditions(Table * t)
{
        std::vector<cvr::MenuCheckbox *>::iterator button;
        for(button = t->querySlider.begin(); button < t->querySlider.end(); button++)
            (*button)->setValue(false);
        for(int i = 0; i < t->queryOptions.size(); i++)
            t->queryOptions[i]->setValue(t->queryOptions[i]->firstOn(),false);

}
//Converts the DC into a unique integer, 0 through (26^2 - 1).
int ArtifactVis2::dc2Int(string dc)
{
    char letter1 = dc.c_str()[0];
    char letter2 = dc.c_str()[1];
    int char1 = letter1-65;
    int char2 = letter2-65;
    int tot = char1*26 + char2;
    return tot;
}
std::string ArtifactVis2::getTimeModified(std::string file)
{
    struct tm* clock;
    struct stat attrib;
    stat(file.c_str(),&attrib);
    clock = gmtime(&(attrib.st_mtime));
    stringstream ss;
    ss << clock->tm_year+1900;
    if(clock->tm_yday + 1 < 100) ss << "0";
    if(clock->tm_yday + 1 < 10) ss << "0";
    ss << clock->tm_yday + 1;
    if(clock->tm_hour < 10) ss << "0";
    ss << clock->tm_hour;
    if(clock->tm_min < 10) ss << "0";
    ss << clock->tm_min;
    if(clock->tm_sec < 10) ss << "0";
    ss << clock->tm_sec;
    string output = ss.str();
    return output;
}
void ArtifactVis2::preFrame()
{
    std::vector<Artifact*> allArtifacts;
    for(int i = 0; i < _query.size(); i++)
    {
       if(_queryDynamicUpdate[i]->getValue())
       {
          string path = _query[i]->kmlPath;
          string newTime = getTimeModified(path);
          if(newTime!=_query[i]->timestamp)
          {
              cout << "New query found for " << _query[i]->name << "." << endl;
              readQuery(_query[i]);
              if(_queryOption[i]->getValue())
              {
                 if(_query[i]->sf)
                     displayArtifacts(_query[i]);
                 _query[i]->updated = false;
                 _root->addChild(_query[i]->sphereRoot);

              }
          }
       }
       if(_query[i]->active&&_query[i]->sf)
          for(int j = 0; j < _query[i]->artifacts.size(); j++)
             allArtifacts.push_back(_query[i]->artifacts[j]);

    }
    std::sort(allArtifacts.begin(),allArtifacts.end(),compare());
    Matrixd viewOffsetM;
    Vec3f viewOffset = Vec3f(ConfigManager::getFloat("x","ViewerPosition",0.0f),
                             ConfigManager::getFloat("y","ViewerPosition",0.0f),
                             ConfigManager::getFloat("z","ViewerPosition",0.0f))*-1;
    viewOffsetM.makeTranslate(viewOffset);
    Matrixd objMat =  PluginHelper::getObjectMatrix()*viewOffsetM;
    Vec3f vec = objMat*Vec3f(0,0,1);
    double norm = vec.length();
    //for(int i = 0; i < allArtifacts.size(); i++) allArtifacts[i]->showLabel = true;
    for(int i = 0; i < allArtifacts.size(); i++)
    {
        string dc = allArtifacts[i]->dc;
        Vec4 color = _colors[dc2Int(dc)];
        color.x() = std::min((double)1.0,color.x() + .25);
        color.y() = std::min((double)1.0,color.y() + .25);
        color.z() = std::min((double)1.0,color.z() + .25);
        Vec3d vec1 = allArtifacts[i]->modelPos*objMat;
        allArtifacts[i]->distToCam = (vec1).length();
	/*
        if(vec1.y() > 0)
          for(int j = i+1; j < allArtifacts.size(); j++)
          {
            if(allArtifacts[i]->showLabel)
            {
                Vec3f vec2 = allArtifacts[j]->modelPos*objMat;
                double angle = acos(vec1*vec2/(vec1.length()*vec2.length()));
                if(angle  < atan2(1.0f,24.0f))
                {
                    allArtifacts[j]->showLabel = false;
                }
            }
          }
	*/
        allArtifacts[i]->label->setColor(color*int(allArtifacts[i]->showLabel));

        allArtifacts[i]->label->setPosition(allArtifacts[i]->modelPos+vec/norm*1.2*_sphereRadius);

    }
    if(_selectCB->getValue())
    {
	updateSelect();
    }

    //......................................................................
    //SpaceNavigator Add
    //......................................................................
    if (statusSpnav)
    {
        Matrixd finalmat;

        if(ComController::instance()->isMaster())
        {

        //static double transcale = -0.03;
        //static double rotscale = -0.00006;

        spnav_event sev;

        double x, y, z;
        x = y = z = 0.0;
        double rx, ry, rz;
        rx = ry = rz = 0.0;
        double rx2, ry2, rz2;
        rx2 = ry2 = rz2 = 0.0;

        while(spnav_poll_event(&sev))
        {
            if(sev.type == SPNAV_EVENT_MOTION)
            {
            x += sev.motion.x;
            y += sev.motion.z;
            z += sev.motion.y;
            rx += sev.motion.rx;
            ry += sev.motion.rz;
            rz += sev.motion.ry;

            rx2 += sev.motion.rx;
            ry2 += sev.motion.rz;
            rz2 += sev.motion.ry;
            // printf("got motion event: t(%d, %d, %d) ", sev.motion.x, sev.motion.y, sev.motion.z);
            // printf("r(%d, %d, %d)\n", sev.motion.rx, sev.motion.ry, sev.motion.rz);



            }
            else
            {	/* SPNAV_EVENT_BUTTON */
            //printf("got button %s event b(%d)\n", sev.button.press ? "press" : "release", sev.button.bnum);
            if(sev.button.press)
            {
                /*switch(sev.button.bnum)
                {
                case 0:
                    transcale *= 1.1;
                    break;
                case 1:
                    transcale *= 0.9;
                    break;
                case 2:
                    rotscale *= 1.1;
                    break;
                case 3:
                    rotscale *= 0.9;
                    break;
                default:
                    break;

                }
                cerr << "Translate Scale: " << transcale << " Rotate Scale: " << rotscale << endl;
                */

            }
            }
        }




        x *= transcale;
        y *= transcale;
        z *= transcale;
        rx *= rotscale;
        ry *= rotscale;
        rz *= rotscale;
        //cout << "Trans: " << x << "," << y << "," << z << " Rotation: " << rx << "," << ry << "," << rz << "\n";

        Matrix view = PluginHelper::getHeadMat();

        Vec3 campos = view.getTrans();
        Vec3 trans = Vec3(x, y, z);

        trans = (trans * view) - campos;

        Matrix tmat;
        tmat.makeTranslate(trans);
        Vec3 xa = Vec3(1.0, 0.0, 0.0);
        Vec3 ya = Vec3(0.0, 1.0, 0.0);
        Vec3 za = Vec3(0.0, 0.0, 1.0);

        xa = (xa * view) - campos;
        ya = (ya * view) - campos;
        za = (za * view) - campos;

        Matrix rot;
        rot.makeRotate(rx, xa, ry, ya, rz, za);

        Matrix ctrans, nctrans;
        ctrans.makeTranslate(campos);
        nctrans.makeTranslate(-campos);
        if(_selectModelLoad && _selectArtifactCB->getValue())
        {
            double rotscale2 = 0.01;

                rx2 *= rotscale2;
                ry2 *= rotscale2;
                rz2 *= rotscale2;

            rotateModel(rx2,ry2,rz2);
        }
        else
        {
        finalmat = PluginHelper::getObjectMatrix() * nctrans * rot * tmat * ctrans;

        ComController::instance()->sendSlaves((char *)finalmat.ptr(), sizeof(double[16]));
        }
        }
        else
        {
        ComController::instance()->readMaster((char *)finalmat.ptr(), sizeof(double[16]));
        }

        if(_selectModelLoad && _selectArtifactCB->getValue())
        {
            //Disables movement
        }
        else
        {
        PluginHelper::setObjectMatrix(finalmat);
        }
    }
    else
    {
        //Manipulate Artifact Using Mouse bang

        if(_selectModelLoad && _selectArtifactCB->getValue())
        {
            double rx2, ry2, rz2;
            rx2 = ry2 = rz2 = 0.0;

            int tempX = PluginHelper::getMouseX();
            int tempY = PluginHelper::getMouseY();
            int tempZ = 0;
            //cout << tempX << "," << tempY << "," << tempZ << "\n";

            if (tempX == 0)
            {

            }
            else if (_xRotMouse == 0)
            {
                rx2 = 1;
                _xRotMouse = tempX;
            }
            else
            {
                rx2 = _xRotMouse - tempX;
                _xRotMouse = tempX;
            }

            if (tempY == 0)
            {

            }
            else if (_yRotMouse == 0)
            {
                ry2 = 1;
                _yRotMouse = tempY;
            }
            else
            {
                ry2 = _yRotMouse - tempY;
                _yRotMouse = tempY;
            }


            double rotscale2 = 0.5;

                rx2 *= rotscale2;
                ry2 *= rotscale2;
                rz2 *= rotscale2;

            if(rx2 != 0 && ry2 != 0)
            {
            //cout << rx2 << "," << ry2 << "," << rz2 << "\n";
            rotateModel(rx2,ry2,rz2);
            }
        }

    }
        //........................................................................


    //bool kinectOn = true;
    if (remote)
    {
        float tessellation = ConfigManager::getFloat("Plugin.ArtifactVis2.Tessellation",.2);
        //while (true)
        //{
                // Update the joint positions
                remote->mainloop();
/*
                printf("handle_tracker\tSensor %d is now at (%g,%g,%g)\n",
	 0,
	 position[0][0], position[0][1], position[0][2]);

*/

   //bango
    bool testS = false;
   if(position[0][0] != 0 && testS)
   {
    //Group * root_node;
    _root->removeChild(skeletonGeode);
    skeletonGeode = new osg::Geode();
    Vec3d center(0,0,0);
    for (int i=0; i<24; i++)
    {
        Vec3d pos(position[i][0], position[i][1], position[i][2]);

        osg::Drawable * g = createObject("BL-BLADE",tessellation, pos);
            //g->setUseDisplayList(true);
            //(*item)->drawable = g;
            skeletonGeode->addDrawable(g);

    }
    //root_node->addChild(sphereGeode);
    _root->addChild(skeletonGeode);
   }

    bool testK = true;
   if(position[0][0] != 0 && testK)
   {
       double x, y, z;
        x = y = z = 0.0;
        double rx, ry, rz;
        rx = ry = rz = 0.0;

        Matrixd finalmat;
       double transcale = 10;
       double rotscale = 0;



        double yscale;
       y = (orientation[6][0]); //Left Elbow Rotatation on X-axis
       if(y < 0)
       {
         yscale = 0.5;
       }
       else
       {
         yscale = -0.5;
       }
       y += yscale;
       x = (orientation[6][1] +0.5) * -1; //Left Elbow Rotation on Y-axis
       z = orientation[5][1]; //Left Shoulder Rotation on Y-axis
       rx = orientation[8][3] -0.5; //Left Hand Rotation on Z-axis
       ry = orientation[8][0] -0.5; //Left Hand Rotation on X-axis
       rz = orientation[8][1] +0.5; //Left Hand Rotation on Y-axis



       x *= 5;
        y *= 10;
        if(x <= -1.5 || x >= 1.5 )
        {

        }
        else
        {
            x = 0;
        }
        if(y <= -1.5 || y >= 1.5 )
        {

        }
        else
        {
            y = 0;
        }
        z *= transcale * 0;
        rx *= rotscale;
        ry *= rotscale;
        rz *= rotscale;

       cout << "Trans: " << x << "," << y << "," << z << " Rotation: " << rx << "," << ry << "," << rz << "\n";




        Matrix view = PluginHelper::getHeadMat();

        Vec3 campos = view.getTrans();
        Vec3 trans = Vec3(x, y, z);

        trans = (trans * view) - campos;

        Matrix tmat;
        tmat.makeTranslate(trans);
        Vec3 xa = Vec3(1.0, 0.0, 0.0);
        Vec3 ya = Vec3(0.0, 1.0, 0.0);
        Vec3 za = Vec3(0.0, 0.0, 1.0);

        xa = (xa * view) - campos;
        ya = (ya * view) - campos;
        za = (za * view) - campos;

        Matrix rot;
        rot.makeRotate(rx, xa, ry, ya, rz, za);

        Matrix ctrans, nctrans;
        ctrans.makeTranslate(campos);
        nctrans.makeTranslate(-campos);

        finalmat = PluginHelper::getObjectMatrix() * nctrans * rot * tmat * ctrans;

        ComController::instance()->sendSlaves((char *)finalmat.ptr(), sizeof(double[16]));

        PluginHelper::setObjectMatrix(finalmat);

   }
                /*
                if(position[0][0]!=0)
                {
                // Clear console
                //system("cls");

                for (int i=0; i<24; i++)
                {
                        // Convert orientation quaternion to euler angles
                        //q_to_euler(eulerOrientation, orientation[i]);

                        // Output updated data
                        printf("\nJoint: %i", i);
                        printf("Position:\nX: %d\nY: %d\nZ: %d", position[i][0], position[i][1], position[i][2]);
                        //printf("Orientation:\nYaw: %d\nPitch: %d\nRoll: %d", eulerOrientation[0], eulerOrientation[1], eulerOrientation[2]);
                }
                }

   }
   */
    //}
    }
}
void ArtifactVis2::loadScaleBar(osg::Vec3d start)
{
    string modelPath = ConfigManager::getEntry("Plugin.ArtifactVis2.3DModelFolder").append("scale_5_m/scale_5_m.obj");
    Node* modelFileNode = osgDB::readNodeFile(modelPath);
                //_modelLoaded[i*26+j] = true;

            PositionAttitudeTransform* modelTrans = new PositionAttitudeTransform();
            Matrixd scale;
            double snum = 0.58166;
            scale.makeScale(snum,snum,snum);
            MatrixTransform* scaleTrans = new MatrixTransform();
            scaleTrans->setMatrix(scale);
            scaleTrans->addChild(modelFileNode);

             MatrixTransform * siteRote = new MatrixTransform();
                         //Matrixd rot2;
                        //rot2.makeRotate(osg::DegreesToRadians(1.0), 1, 0, 0);
                        Matrix pitchMat;
                        Matrix yawMat;
                        Matrix rollMat;
                        pitchMat.makeRotate(DegreesToRadians(0.0),1,0,0);
                        yawMat.makeRotate(DegreesToRadians(0.0),0,1,0);
                        rollMat.makeRotate(DegreesToRadians(0.0),0,0,1);

                        siteRote->setMatrix(pitchMat*yawMat*rollMat);
                        siteRote->addChild(scaleTrans);

            //_modelartPos = artifacts[art]->modelPos + position;
            Vec3d position(39942.500,73212.802,1451.860);
            modelTrans->setPosition(start);
            //modelTrans->addChild(scaleTrans);
            modelTrans->addChild(siteRote);
            //root_node->addChild(modelTrans);

            _scaleBarModel = new osg::MatrixTransform();
            _scaleBarModel->addChild(modelTrans);
            //_selectModelLoad->addChild(siteRote);

            StateSet * ss=_scaleBarModel->getOrCreateStateSet();
	    ss->setMode(GL_LIGHTING, StateAttribute::ON | osg::StateAttribute::OVERRIDE);

	    Material* mat =new Material();
	    mat->setColorMode(Material::AMBIENT_AND_DIFFUSE);
	    Vec4 color_dif(1,1,1,1);
	    mat->setDiffuse(Material::FRONT_AND_BACK,color_dif);
	    ss->setAttribute(mat);
	    ss->setAttributeAndModes( mat, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);

            _root->addChild(_scaleBarModel);
}
void ArtifactVis2::setActiveArtifact(int art, int q)
{
            vector<Artifact*> artifacts = _query[q]->artifacts;
            if(art < 0 || art >= artifacts.size())
            {
	        return;
            }
	    cout << "Active Artfiact: ";
            cout << artifacts[art]->modelPos[0] << " " << artifacts[art]->modelPos[1] << " " << artifacts[art]->modelPos[2] << endl;
            if(art == _activeArtifact)
            {
	         return;
            }

            std::stringstream ss;
            for (int i = 0; i < artifacts[art]->fields.size(); i++)
            {
                 ss << artifacts[art]->fields.at(i) << " " << artifacts[art]->values.at(i) << endl;
            }
            ss << "Position: " << endl;
            ss << "-Longitude: " << artifacts[art]->pos[0] << endl;
            ss << "-Latitude: " << artifacts[art]->pos[1] << endl;
            ss << "-Altitude: " << artifacts[art]->pos[2] << endl;

            _artifactPanel->updateTabWithText("Info",ss.str());
 	    string picPath = ConfigManager::getEntry("Plugin.ArtifactVis2.PicFolder").append("photos/");

            string side =  (picPath+artifacts[art]->values[1]+"/"+"SF.jpg");
            string top = (picPath+artifacts[art]->values[1]+"/"+ "T.jpg");
            string bottom = (picPath+artifacts[art]->values[1]+"/" + "B.jpg");
            string check = top;
            if(!modelExists(check.c_str()))
            {
            //side =  (picPath+"50563_s.jpg");
            //top = (picPath+"50563_t.jpg");
            //bottom = (picPath+"50563_b.jpg");

            }
            cout << check << "\n";
            cout << top << "\n";
            /*
            side <<  _picFolder << "/" << artifacts[art]->values.at(0) << "_s.jpg";
            top << _picFolder << "/" << artifacts[art]->values.at(0) << "_t.jpg";
            bottom << _picFolder << "/" << artifacts[art]->values.at(0) << "_b.jpg";
            */
            _artifactPanel->updateTabWithTexture("Side",side);
            _artifactPanel->updateTabWithTexture("Top",top);
            _artifactPanel->updateTabWithTexture("Bottom",bottom);

    //std::cerr << "Side texture: " << side.str() << std::endl;
    //std::cerr << "Top texture: " << top.str() << std::endl;
    //std::cerr << "Bottom texture: " << bottom.str() << std::endl;
            _root->removeChild(_selectModelLoad.get());
            if(true)
            { //bang
            //Once an artifact is selected we search to see if a 3D model of that artifact is available, otherwise a default model is loaded
            //The Basket number is used to load the appropriate arftifact.
            //double snum;
            cout << "Basket: " << artifacts[art]->values[1] << "\n";
            string basket = artifacts[art]->values[1];
            string modelPath = ConfigManager::getEntry("Plugin.ArtifactVis2.ScanFolder").append(""+basket+"/"+basket+".ply");
            string dc;
            dc = artifacts[art]->dc;
            cout << dc << "\n";
            _snum = 0.01;
            double xrot = 0;
            double yrot = 0;
            double zrot = 0;
            Vec3d position;


            if(!modelExists(modelPath.c_str()))
            {
            //dc = "ZZ";
            modelPath = ConfigManager::getEntry("Plugin.ArtifactVis2.3DModelFolder").append("photos/"+basket+"/test.obj");
            _snum = 0.05;
            }
            if(!modelExists(modelPath.c_str()))
            {

                modelPath = ConfigManager::getEntry("Plugin.ArtifactVis2.3DModelFolder").append("photos/"+basket+"/frame.obj");
                _snum = 0.1;
                xrot = 90;
                if(modelExists(modelPath.c_str()))
                {

                    string file = ConfigManager::getEntry("Plugin.ArtifactVis2.3DModelFolder").append("photos/"+basket+"/frame.xml");

                    FILE * fp = fopen(file.c_str(),"r");

                    mxml_node_t * tree;
                    tree = mxmlLoadFile(NULL, fp, MXML_TEXT_CALLBACK);
                    fclose(fp);
                    mxml_node_t * child;
                    double trans[3];
                    double scale[3];
                    //double rot[3];
                    child = mxmlFindElement(tree, tree, "easting", NULL, NULL, MXML_DESCEND);
                    trans[0] = atof(child->child->value.text.string);
                    child = mxmlFindElement(tree, tree, "northing", NULL, NULL, MXML_DESCEND);
                    trans[1] = atof(child->child->value.text.string);
                    child = mxmlFindElement(tree, tree, "elevation", NULL, NULL, MXML_DESCEND);
                    trans[2] = atof(child->child->value.text.string);
                    child = mxmlFindElement(tree, tree, "x", NULL, NULL, MXML_DESCEND);
                    scale[0] = atof(child->child->value.text.string);
                    child = mxmlFindElement(tree, tree, "y", NULL, NULL, MXML_DESCEND);
                    scale[1] = atof(child->child->value.text.string);
                    child = mxmlFindElement(tree, tree, "z", NULL, NULL, MXML_DESCEND);
                    scale[2] = atof(child->child->value.text.string);
                    child = mxmlFindElement(tree, tree, "heading", NULL, NULL, MXML_DESCEND);
                    xrot = atof(child->child->value.text.string);
                    child = mxmlFindElement(tree, tree, "tilt", NULL, NULL, MXML_DESCEND);
                    yrot = atof(child->child->value.text.string);
                    child = mxmlFindElement(tree, tree, "roll", NULL, NULL, MXML_DESCEND);
                    zrot = atof(child->child->value.text.string);
                    position = Vec3d(trans[0], trans[1], trans[2]);
                    cout << "Read ZB XML: " << trans[2] << " " << scale[0] << " " << xrot << "\n";
                    _snum = scale[0];
                }
            }
            if(!modelExists(modelPath.c_str()))
            {
            //dc = "ZZ";
            //modelPath = ConfigManager::getEntry("Plugin.ArtifactVis2.3DModelFolder").append("photos/default/test.obj");
            _snum = 0.1;
            }

            if(modelExists(modelPath.c_str()))
            {
                cout << "Select Model " << modelPath << " Exists \n";
                _modelFileNode = osgDB::readNodeFile(modelPath);


            PositionAttitudeTransform* modelTrans = new PositionAttitudeTransform();
            Matrixd scale;

            scale.makeScale(_snum,_snum,_snum);
            MatrixTransform* scaleTrans = new MatrixTransform();
            scaleTrans->setMatrix(scale);
            scaleTrans->addChild(_modelFileNode);

             MatrixTransform * siteRote = new MatrixTransform();

                        Matrix pitchMat;
                        Matrix yawMat;
                        Matrix rollMat;
                        pitchMat.makeRotate(DegreesToRadians(xrot),1,0,0);
                        yawMat.makeRotate(DegreesToRadians(yrot),0,1,0);
                        rollMat.makeRotate(DegreesToRadians(zrot),0,0,1);

                        siteRote->setMatrix(pitchMat*yawMat*rollMat);
                        siteRote->addChild(scaleTrans);

            _modelartPos = artifacts[art]->modelPos + position;
            modelTrans->setPosition(_modelartPos);

            modelTrans->addChild(siteRote);
            _selectModelLoad = new osg::MatrixTransform();
            _selectModelLoad->addChild(modelTrans);

            _root->addChild(_selectModelLoad.get());
            }
            //sphereGeode->addDrawable(artifacts[art]->drawable);
            //root_node->addChild(sphereGeode);


            }
            _activeArtifact = art;
}
void ArtifactVis2::readQuery(QueryGroup * query)
{

    _root->removeChild(query->sphereRoot);
    query->updated = true;
    std::vector<Artifact*> generatedArtifacts;
    string filename = query->kmlPath;
    cerr << "Reading query: " << filename << endl;
    FILE * fp;
    mxml_node_t * tree;
    fp = fopen(filename.c_str(),"r");
    if(fp == NULL)
    {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }
    tree = mxmlLoadFile(NULL, fp, MXML_TEXT_CALLBACK);
    fclose(fp);
    if(tree == NULL)
    {
        std::cerr << "Unable to parse XML file: " << filename << std::endl;
        return;
    }
    mxml_node_t *node;
    node = mxmlFindElement(tree, tree, "name", NULL, NULL, MXML_DESCEND);
    query->name = node->child->value.text.string;
    node = mxmlFindElement(tree, tree, "query", NULL, NULL, MXML_DESCEND);
    query->query = node->child->value.text.string;
    node = mxmlFindElement(tree, tree, "timestamp", NULL, NULL, MXML_DESCEND);
    query->timestamp = getTimeModified(filename);
    if(!query->sf)
    {
        readLocusFile(query);
        cout << "Query read!" << endl;
        return;
    }
    for (node = mxmlFindElement(tree, tree, "Placemark", NULL, NULL, MXML_DESCEND); node != NULL; node = mxmlFindElement(node, tree, "Placemark", NULL, NULL, MXML_DESCEND))
    {
        Artifact* newArtifact = new Artifact();
        newArtifact->label = new osgText::Text();
        mxml_node_t * desc_node = mxmlFindElement(node, tree, "description", NULL, NULL, MXML_DESCEND);
        mxml_node_t * desc_child;
        string dc;
        string basket;
        for(desc_child = desc_node->child; desc_child != NULL; desc_child = desc_child->next)
        {
            char * desc_text = desc_child->value.text.string;

            string desc = desc_text;
            if(desc.find(":",0)!=string::npos)
            {
                if(desc.find("dccode:",0)!=string::npos) dc = desc_child->next->value.text.string;
                if(desc.find("d_code:",0)!=string::npos) dc = desc_child->next->value.text.string;
                if(desc.find("the_geom:",0)!=string::npos)
                {
		    string coord;
                    for(int i = 0; i < 3; i++)
                    {
                        desc_child = desc_child->next;
                        std::istringstream ss;
                        //std::cout.precision(11);
                        coord = desc_child->value.text.string;
                        //coord = coord.erase(coord.find(".")+4);
                        //pos[i] = atof(coord.c_str());

                        ss.str(coord);
                        ss >> newArtifact->pos[i];
                    }
                }
                else
                {
                    string value_text = desc_child->next->value.text.string;
                    if(desc.find("basket",0)!=string::npos)
                        basket = value_text;
                    newArtifact->fields.push_back(desc);
                    if(value_text.find("NULL",0)==string::npos)
                        newArtifact->values.push_back(value_text);
                    else
                        newArtifact->values.push_back("-");

               }
            }
        }
        newArtifact->dc = dc;
        newArtifact->label->setText(basket+dc.substr(0,2));

        newArtifact->visible = true;
        generatedArtifacts.push_back(newArtifact);
    }
    query->artifacts = generatedArtifacts;
    cout <<"Query read!" <<endl;
}
void ArtifactVis2::listArtifacts()
{
    for(int q = 0; q < _query.size(); q++)
    {
        vector<Artifact*> artifacts = _query[q]->artifacts;
        cerr << "Listing " << artifacts.size() << " elements:" << endl;
        vector<Artifact*>::iterator item = artifacts.begin();
        for (; item < artifacts.end(); item++)
        {
            for(int i = 0; i < (*item)->fields.size(); i++)
            {
                 cout << (*item)->fields.at(i) << " " << (*item)->values.at(i) << " ";
            }
            cout << "Position: " << (*item)->pos[0] << ", " << (*item)->pos[1] << ", " << (*item)->pos[2];
            cout << endl;
        }
    }
}
bool ArtifactVis2::modelExists(const char * filename)
{
    ifstream ifile(filename);
    return !ifile.fail();
}
void ArtifactVis2::displayArtifacts(QueryGroup * query)
{
    Group * root_node = query->sphereRoot;
    while(root_node->getNumChildren()!=0)
    {
        root_node->removeChild(root_node->getChild(0));
    }
    const double M_TO_MM = 1.0f;
    //const double LATLONG_FACTOR = 100000.0f;
    std::vector<Artifact*> artifacts = query->artifacts;
    cerr << "Creating " << artifacts.size() << " artifacts..." << endl;
    vector<Artifact*>::iterator item = artifacts.begin();
    float tessellation = ConfigManager::getFloat("Plugin.ArtifactVis2.Tessellation",.2);
    Vec3d offset = Vec3d(
                   ConfigManager::getFloat("Plugin.ArtifactVis2.Offset.X",0),
                   ConfigManager::getFloat("Plugin.ArtifactVis2.Offset.Y",0),
                   ConfigManager::getFloat("Plugin.ArtifactVis2.Offset.Z",0));
    osg::Geode * sphereGeode = new osg::Geode();
    Vec3d center(0,0,0);
    for (; item < artifacts.end();item++)
    {
        Vec3d position((*item)->pos[0], (*item)->pos[1], (*item)->pos[2]);
        osg::Vec3d pos;
        if(!_ossim)
        {

            Matrixd trans;
            trans.makeTranslate(position);
            Matrixd scale;
            scale.makeScale(M_TO_MM, M_TO_MM, M_TO_MM);


            //scale.makeScale(M_TO_MM*LATLONG_FACTOR, M_TO_MM*LATLONG_FACTOR, M_TO_MM);
            //Matrixd rot1;
            //rot1.makeRotate(osg::DegreesToRadians(-90.0), 0, 1, 0);
            //Matrixd rot2;
            //rot2.makeRotate(osg::DegreesToRadians(90.0), 1, 0, 0);
            //Matrixd mirror;
            //mirror.makeScale(1, -1, 1);
            Matrixd offsetMat;
            offsetMat.makeTranslate(offset);
            //pos = osg::Vec3d(0,0,0) * mirror * trans * scale * mirror * rot2 * rot1 * offsetMat;
            pos = osg::Vec3d(0,0,0) * trans * offsetMat;

            //pos = position;
            //printf("artifact %f %f %f\n", pos[0], pos[1], pos[2]);
            (*item)->modelPos = pos;
            //center+=pos;
        }
        else
        {
            pos = position;
            //center+=pos;
            (*item)->modelPos = pos;
        }
    }
    //center/=artifacts.size();
    //bango
    for(item = artifacts.begin(); item < artifacts.end();item++)
    {
        //cerr<<"Creating object "<<(item-artifacts.begin())<<" out of "<<artifacts.size()<<endl;
        if(_ossim)
        {
           // (*item)->modelPos-=center;
        }
        Vec3d pos = (*item)->modelPos;
        int dcInt = dc2Int((*item)->dc);
        if(!_modelLoaded[dcInt])
       // if(true)
        {
            osg::Drawable * g = createObject((*item)->dc,tessellation, pos);
            g->setUseDisplayList(false);
            (*item)->drawable = g;
            sphereGeode->addDrawable((*item)->drawable);
        }
        else
        {
            PositionAttitudeTransform* modelTrans = new PositionAttitudeTransform();
            Matrixd scale;
            double snum = 0.05;
            scale.makeScale(snum,snum,snum);
            MatrixTransform* scaleTrans = new MatrixTransform();
            scaleTrans->setMatrix(scale);
            scaleTrans->addChild(_models[dcInt]);
            modelTrans->setPosition(pos);
            modelTrans->addChild(scaleTrans);
            root_node->addChild(modelTrans);
            (*item)->drawable = (*item)->label;
        }
        sphereGeode->addDrawable((*item)->label);
        (*item)->label->setUseDisplayList(false);
        (*item)->label->setAxisAlignment(osgText::Text::SCREEN);
        (*item)->label->setPosition((*item)->modelPos+Vec3f(0,0,_sphereRadius*1.1));
        (*item)->label->setAlignment(osgText::Text::CENTER_CENTER);
        (*item)->label->setCharacterSize(15);
        (*item)->label->setCharacterSizeMode(osgText::Text::SCREEN_COORDS);
    }
    //query->center = center;
    cerr << "done" << endl;

    StateSet * ss=sphereGeode->getOrCreateStateSet();

    ss->setMode(GL_BLEND, StateAttribute::ON);
    ss->setMode(GL_LIGHTING, StateAttribute::ON);
    ss->setRenderingHint( osg::StateSet::OPAQUE_BIN );

    ss->setAttribute(_defaultMaterial);

    osg::CullFace * cf=new osg::CullFace();
    cf->setMode(osg::CullFace::BACK);

    ss->setAttributeAndModes( cf, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);
   // cout << center.x() << ", " << center.y() << endl;
    if(_ossim)
    {
#ifdef WITH_OSSIMPLANET
        OssimPlanet::instance()->addModel(sphereGeode,center.y(),center.x(),Vec3(1.0,1.0,1.0),10,0,0,0);
#endif
    }
    else
        root_node->addChild(sphereGeode);
}

osg::Drawable * ArtifactVis2::createObject(std::string dc, float tessellation, Vec3d & pos)
{
    TessellationHints * hints = new TessellationHints();
    hints->setDetailRatio(tessellation);
    Sphere* sphereShape = new Sphere(pos, _sphereRadius);
    ShapeDrawable * shapeDrawable = new ShapeDrawable(sphereShape);
    shapeDrawable->setTessellationHints(hints);
    shapeDrawable->setColor(_colors[dc2Int(dc)]);
    return shapeDrawable;
}
void ArtifactVis2::readPointCloud(int index)
{
     //string filename = ConfigManager::getEntry("Plugin.ArtifactVis2.ArchInterfaceFolder").append("Model/").append(_showPCCB[index]->getText());
    Vec3Array * coords;
    Vec4Array * colors;

    //_coordsPC[index] = new Vec3Array();
    //_colorsPC[index] = new Vec4Array();
    string filename = ConfigManager::getEntry("Plugin.ArtifactVis2.3DModelFolder").append(_showPCCB[index]->getText()); //Replaced
    string type = filename;
    //Vec3d avgOffset;
    if(!_coordsPC[index])
    {


    coords = new Vec3Array();
    colors =new Vec4Array();
    cout << "Reading point cloud data from : " << filename <<  "..." << endl;
    ifstream file(filename.c_str());

    type.erase(0,(type.length()-3));
    cout << type << "\n";
    string line;
    bool read = false;


    cout << _pcFactor[index] << "\n";
    int factor = _pcFactor[index];
    if(file.is_open())
    {
        int pcount = 0;
        int bten = factor-1;
        int cptx = 0;
        double transx = 0;
        double transy = 0;
        double transz = 0;
        while(file.good())
        {
            getline(file,line);
            string lineout = line;
            if(type == "ptx")
            {

                vector<string> entries;
                //vector<string>array;
                string token;
    //string tmp = "this@is@a@line";

                istringstream iss(line);
                char lim = ' ';
                while ( getline(iss, token, lim) )
                {
                entries.push_back(token);
                }
                if (entries.size() < 7)
                {
                    cout << lineout << "\n";

                    if (cptx == 9)
                    {
                    cptx = 0;
                    //cout << lineout << "\n";
                    std::stringstream ss;
                    ss.precision(19);
                    transx = atof(entries[0].c_str());
                    transy = atof(entries[0].c_str());
                    transz = atof(entries[0].c_str());
                    }
                    else
                    {
                    cptx++;

                    }
                }
                else
                {
                    std::stringstream ss;
                    ss.precision(19);
                double x = atof(entries[0].c_str()) + transx;
                double y = atof(entries[1].c_str()) + transy;
                double z = atof(entries[2].c_str()) + transz;
                double r = atof(entries[4].c_str())/255;
                double g = atof(entries[5].c_str())/255;
                double b = atof(entries[6].c_str())/255;
                coords->push_back(Vec3d(x,y,z));
                _avgOffset[index]+=Vec3d(x,y,z);
                colors->push_back(Vec4d(r,g,b,1.0));
                bten = 0;
                pcount++;

                }
            }

            if(line.empty()) break;

            if(read)
            {
                bten++;
                vector<string> entries;
                int ck = 0;
                /*
                while(true)
                {
                    if(line.find(" ")==line.rfind(" "))
                        break;
                    else
                    {
                        entries.push_back(line.substr(0,line.find(" ")));
                        line = line.substr(line.find(" ")+1);
                        cout << line << "\n";
                    }
                }
                */
                if (bten == factor)
                {
                while(ck < 7)
                {

                        entries.push_back(line.substr(0,line.find(" ")));
                        line = line.substr(line.find(" ")+1);
                        //cout << line << "\n";
                        ck++;

                }
                }
                //cout << entries.size();
                if ((type == "ply") && (bten == factor))
                {
                float x = atof(entries[0].c_str());
                float y = atof(entries[1].c_str());
                float z = atof(entries[2].c_str());
                float r = atof(entries[3].c_str())/255;
                float g = atof(entries[4].c_str())/255;
                float b = atof(entries[5].c_str())/255;
                coords->push_back(Vec3d(x,y,z));
                _avgOffset[index]+=Vec3d(x,y,z);
                colors->push_back(Vec4f(r,g,b,1.0));
                pcount++;
                bten = 0;
                }
                if ((type == "txt") && (bten == factor))
                {
                float x = atof(entries[0].c_str());
                float y = atof(entries[1].c_str());
                float z = atof(entries[2].c_str());
                float r = atof(entries[3].c_str())/255;
                float g = atof(entries[4].c_str())/255;
                float b = atof(entries[5].c_str())/255;
                coords->push_back(Vec3d(x,y,z));
                _avgOffset[index]+=Vec3d(x,y,z);
                colors->push_back(Vec4f(r,g,b,1.0));
                pcount++;
                bten = 0;
                }
                else if ((type == "pts") && (bten == factor))
                {

                    std::stringstream ss;
                    ss.precision(19);
                double x = atof(entries[0].c_str());
                double y = atof(entries[1].c_str());
                double z = atof(entries[2].c_str());
                double r = atof(entries[4].c_str())/255;
                double g = atof(entries[5].c_str())/255;
                double b = atof(entries[6].c_str())/255;
                coords->push_back(Vec3d(x,y,z));
                _avgOffset[index]+=Vec3d(x,y,z);
                colors->push_back(Vec4d(r,g,b,1.0));
                bten = 0;
                pcount++;

                if (pcount == 7216)
                {
                    //std::stringstream ss;
                    //ss.precision(19);
                    printf("TOP %f %f %f\n", x, y, z);
                    cout << "Coords: " << entries[0].c_str() << " " << entries[1].c_str() << " " << entries[2].c_str() << "\n";
                    //cout << "Coords: " << coords[7216][0] << " " << coords[7216][1] << " " << coords[7216][2] << "\n";
                }
                }
            }
            if(type == "pts")
            {
                read = true;
            }
            if(line=="end_header")
                read = true;
        }
        _coordsPC[index] = coords;
        _colorsPC[index] = colors;
        cout << "Finished Loading, Total Vertices: " << pcount << "\n";
        //cout << "Coords: " << coords<< " " << y << " " << z << "\n";
    }
    else
    {
        cout << "Unable to open file: " << filename << endl;
        return;
    }
    }
    coords = _coordsPC[index];
    colors = _colorsPC[index];
    _avgOffset[index]/=coords->size();
    for(int i = 0; i < coords->size(); i++)
    {
        /*
        if (type == "ply")
        {
        coords->at(i)-=_avgOffset[index];
        }
        */
    }
    Geometry * pointCloud = new Geometry();
    Geode * pointGeode = new Geode();
    pointCloud->setVertexArray(coords);
    pointCloud->setColorArray(colors);
    pointCloud->setColorBinding(Geometry::BIND_PER_VERTEX);
    DrawElementsUInt* points = new DrawElementsUInt(PrimitiveSet::POINTS,0);
    for(int i = 0; i < coords->size(); i++)
    {
        points->push_back(i);
    }
    pointCloud->addPrimitiveSet(points);
    pointGeode->addDrawable(pointCloud);
    StateSet * ss=pointGeode->getOrCreateStateSet();
    ss->setMode(GL_LIGHTING, StateAttribute::OFF | osg::StateAttribute::OVERRIDE);
    MatrixTransform * rotTransform = new MatrixTransform();
    Matrix pitchMat;
    Matrix yawMat;
    Matrix rollMat;
    pitchMat.makeRotate(DegreesToRadians(_pcRot[index].x()),1,0,0);
    yawMat.makeRotate(DegreesToRadians(_pcRot[index].y()),0,1,0);
    rollMat.makeRotate(DegreesToRadians(_pcRot[index].z()),0,0,1);
    rotTransform->setMatrix(pitchMat*yawMat*rollMat);
    MatrixTransform * scaleTransform = new MatrixTransform();
    Matrix scaleMat;
    scaleMat.makeScale(_pcScale[index]);
    scaleTransform->setMatrix(scaleMat);
    MatrixTransform * posTransform = new MatrixTransform();
    Matrix posMat;
    Matrix invertMat;
    //invertMat.invert()
    posMat.makeTranslate(_pcPos[index]);
    posTransform->setMatrix(posMat);
    MatrixTransform * invTransform = new MatrixTransform();
    rotTransform->addChild(pointGeode);
    scaleTransform->addChild(rotTransform);
    posTransform->addChild(scaleTransform);
    //invertMat = posTransform->getInverseMatrix();
    //posTransform.invert(posTransform);
    //invTransform->setMatrix()
    _pcRoot[index] = new MatrixTransform();
    _pcRoot[index]->addChild(posTransform);
    //_pcRoot[index]->addChild(inversMat);
    //_pcRoot[index]->addChild(pointGeode);
    //cout << _pcRoot[index][5].x() << "\n";
}
void ArtifactVis2::readSiteFile(int index)
{
    const double INCH_IN_MM = 25.4f;
    const double M_TO_MM = 1.0f;


    //std::string modelFileName = ConfigManager::getEntry("Plugin.ArtifactVis2.ArchInterfaceFolder").append("Model/").append(_showModelCB[index]->getText());
    std::string modelFileName = ConfigManager::getEntry("Plugin.ArtifactVis2.3DModelFolder").append(_showModelCB[index]->getText()); //Replaced

    cerr << "Reading site file: " << modelFileName << " " << index << " ..." << endl;
    if (!modelFileName.empty())
    {
        //cout << _modelSFileNode.size() << "\n";
        if(!_modelSFileNode[index])
        {
            _modelSFileNode[index] = osgDB::readNodeFile(modelFileName);
        }

        Node* modelFileNode = _modelSFileNode[index];

	if (modelFileNode==NULL) cerr << "Error reading file" << endl;
	else
	{
            MatrixTransform * siteScale = new MatrixTransform();
            Matrix scaleMat;
            scaleMat.makeScale(_siteScale[index]*INCH_IN_MM);
            siteScale->setMatrix(scaleMat);

            siteScale->addChild(modelFileNode);

            if(_ossim)
            {
                _siteRoot[index]->addChild(siteScale);
#ifdef WITH_OSSIMPLANET
                OssimPlanet::instance()->addModel(_siteRoot[index],_sitePos[index].y(),_sitePos[index].x(),Vec3f(1,1,1),0,0,0,0);
#endif
                cout << _sitePos[index].y() << ", " << _sitePos[index].x() << endl;
            }
/*
            else if(_osgearth)
            {
		_siteRoot[index]->addChild(siteScale);
                OsgEarthRequest request;
	        request.lat = _sitePos[index].y();
	        request.lon = _sitePos[index].x();
	        cout << "Lat, Lon: " << _sitePos[index].y() << ", " << _sitePos[index].x() << endl;
	        request.height = 30000.0f;
	        request.trans = _siteRoot[index];
	        PluginManager::instance()->sendMessageByName("OsgEarth",OE_ADD_MODEL,(char *) &request);

            }
*/
	    else
	    {
                MatrixTransform * siteRot = new MatrixTransform();
                //Matrixd rot1;
                //rot1.makeRotate(osg::DegreesToRadians(-90.0), 0, 1, 0);
                //Matrixd rot2;
                //rot2.makeRotate(osg::DegreesToRadians(90.0), 1, 0, 0);
                Matrix pitchMat;
                Matrix yawMat;
                Matrix rollMat;
                pitchMat.makeRotate(DegreesToRadians(_siteRot[index].x()),1,0,0);
                yawMat.makeRotate(DegreesToRadians(_siteRot[index].y()),0,1,0);
                rollMat.makeRotate(DegreesToRadians(_siteRot[index].z()),0,0,1);
                siteRot->setMatrix(pitchMat*yawMat*rollMat);
                //siteRot->setMatrix(rot2);
                siteRot->addChild(siteScale);
                MatrixTransform * siteTrans = new MatrixTransform();
                Matrix transMat;
                transMat.makeTranslate(_sitePos[index]);
                siteTrans->setMatrix(transMat);
                siteTrans->addChild(siteRot);

                _siteRoot[index] = new MatrixTransform();
                _siteRoot[index]->addChild(siteTrans);


	    }
	    StateSet * ss=_siteRoot[index]->getOrCreateStateSet();
	    ss->setMode(GL_LIGHTING, StateAttribute::ON | osg::StateAttribute::OVERRIDE);

	    Material* mat =new Material();
	    mat->setColorMode(Material::AMBIENT_AND_DIFFUSE);
	    Vec4 color_dif(1,1,1,1);
	    mat->setDiffuse(Material::FRONT_AND_BACK,color_dif);
	    ss->setAttribute(mat);
	    ss->setAttributeAndModes( mat, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);

	    cerr << "File read." << endl;
	}
    }
    else
    {
	cerr << "Error: Plugin.ArtifactVis2.Topo needs to point to a .wrl 3D topography file" << endl;
    }
}

void ArtifactVis2::readLocusFile(QueryGroup * query)
{
    cout << "Reading Locus File..." << endl;
    const double M_TO_MM = 1.0f;
    //const double LATLONG_FACTOR = 100000.0f;
    Vec3d center(0,0,0);
    for(int i = 0; i < query->loci.size(); i++)
    {
        query->sphereRoot->removeChildren(0,query->sphereRoot->getNumChildren());
    }
    query->loci.clear();
    Vec3f offset = Vec3f(
        ConfigManager::getFloat("Plugin.ArtifactVis2.Offset.X",0),
        ConfigManager::getFloat("Plugin.ArtifactVis2.Offset.Y",0),
        ConfigManager::getFloat("Plugin.ArtifactVis2.Offset.Z",0));
    std::string locusFile = query->kmlPath;
    if(locusFile.empty())
    {
	std::cerr << "ArtifactVis2: Warning: No Plugin.ArtifactVis2.LociFile entry." << std::endl;
	return;
    }

    FILE * fp;
    mxml_node_t * tree;
    fp = fopen(locusFile.c_str(), "r");
    if(fp == NULL)
    {
	std::cerr << "Unable to open file: " << locusFile << std::endl;
	return;
    }
    tree = mxmlLoadFile(NULL, fp,MXML_TEXT_CALLBACK);
    fclose(fp);
    if(tree == NULL)
    {
	std::cerr << "Unable to parse XML file: " << locusFile  << std::endl;
	return;
    }

    mxml_node_t *node = mxmlFindElement(tree, tree, "Placemark", NULL, NULL, MXML_DESCEND);
    for(; node != NULL; node = mxmlFindElement(node, tree, "Placemark", NULL, NULL, MXML_DESCEND))
    {
        Locus * loc = new Locus;
        mxml_node_t * desc_node = mxmlFindElement(node, tree, "description", NULL, NULL, MXML_DESCEND);
        mxml_node_t * desc_child;
        for(desc_child = desc_node->child; desc_child != NULL; desc_child = desc_child->next)
        {
            char * desc_text = desc_child->value.text.string;

            string desc = desc_text;
            if(desc.find(":",0)!=string::npos&&desc.find("the_geom",0)==string::npos)
            {
                if(desc.find("locus:",0)!=string::npos) loc->id = desc_child->next->value.text.string;
                string value_text = desc_child->next->value.text.string;
                loc->fields.push_back(desc);
                if(value_text.find("NULL",0)==string::npos)
                    loc->values.push_back(value_text);
                else
                    loc->values.push_back("-");
            }
        }
        desc_node = mxmlFindElement(node,tree,"name",NULL,NULL,MXML_DESCEND);
        desc_child = desc_node->child;
        stringstream ss;
        ss << desc_child->value.text.string << " ";
        for(desc_child = desc_child->next; desc_child != NULL; desc_child = desc_child->next)
            ss << desc_child->value.text.string << " ";
        loc->name = ss.str();
        mxml_node_t * coord_node;
        //mxml_node_t * polyhedron_node;
        Vec3Array * coords = new Vec3Array();
        coord_node = mxmlFindElement(node, tree, "coordTop", NULL, NULL, MXML_DESCEND);
        mxml_node_t * child;
        Vec3d locCenter(0,0,0);
        for(child = coord_node->child; child != NULL; child = child->next)
        {
           // std::istringstream ss;
            //std::cout.precision(15);
            double pos[3];
	    string coord;
            for(int i = 0; i < 3; i++)
            {
              //   ss.str(child->value.text.string);
                coord = child->value.text.string;
                //coord = coord.erase(coord.find(".")+4);
                pos[i] = atof(coord.c_str());
                if(i!=2)
                child = child->next;
            }
            Vec3d position = Vec3d(pos[0],pos[1],pos[2]);
            if(_ossim)
            {
                loc->coordsTop.push_back(position);
                coords->push_back(position);
            }
            else
            {
                Matrixd scale;
                //scale.makeScale(M_TO_MM, M_TO_MM, M_TO_MM);
                //scale.makeScale(M_TO_MM*1000, M_TO_MM*1000, M_TO_MM*1000);

                Matrixd trans;
                trans.makeTranslate(position);
                //scale.makeScale(M_TO_MM*LATLONG_FACTOR, M_TO_MM*LATLONG_FACTOR, M_TO_MM);
                //Matrixd rot1;
                //rot1.makeRotate(osg::DegreesToRadians(-90.0), 0, 1, 0);
                //Matrixd rot2;
                //rot2.makeRotate(osg::DegreesToRadians(90.0), 1, 0, 0);
                //Matrixd mirror;
                //mirror.makeScale(1, -1, 1);
                Matrixd offsetMat;
                //offset = Vec3f(-700000,-3300000,0);
                offsetMat.makeTranslate(offset);
                osg::Vec3d posVec = osg::Vec3d(0,0,0) * trans * offsetMat ;


		//posVec[2] = -posVec[2];

                //osg::Vec3d posVec = position;
		//posVec[1] = -posVec[1];
	    	//printf("TOP %f %f %f\n", posVec[0], posVec[1], posVec[2]);
                loc->coordsTop.push_back(posVec);

                coords->push_back(posVec);
            }
        }
        coords->pop_back();
        coord_node = mxmlFindElement(node, tree, "coordBottom", NULL, NULL, MXML_DESCEND);
        for(child = coord_node->child; child != NULL; child = child->next)
        {

           // std::istringstream ss;
          //  std::cout.precision(15);
            double pos[3];
           string coord;
            for(int i = 0; i < 3; i++)
            {
              //  ss.str(child->value.text.string);
                coord = child->value.text.string;
                //coord = coord.erase(coord.find(".")+4);
                pos[i] = atof(coord.c_str());
                if(i!=2)
                child = child->next;
            }
            Vec3d position = Vec3d(pos[0],pos[1],pos[2]);
            if(_ossim)
            {
                loc->coordsBot.push_back(position);
                coords->push_back(position);
            }
            else
            {
 		 Matrixd scale;
                scale.makeScale(M_TO_MM, M_TO_MM, M_TO_MM);
                //scale.makeScale(M_TO_MM*1000, M_TO_MM*1000, M_TO_MM*1000);
                Matrixd trans;
                trans.makeTranslate(position);
                //scale.makeScale(M_TO_MM*LATLONG_FACTOR, M_TO_MM*LATLONG_FACTOR, M_TO_MM);
                //Matrixd rot1;
                //rot1.makeRotate(osg::DegreesToRadians(-90.0), 0, 1, 0);
                //Matrixd rot2;
                //rot2.makeRotate(osg::DegreesToRadians(90.0), 1, 0, 0);
                //Matrixd mirror;
                //mirror.makeScale(1, -1, 1);
                Matrixd offsetMat;
              //  offset = Vec3f(-700000,-3300000,0);
                offsetMat.makeTranslate(offset);
                osg::Vec3d posVec = osg::Vec3d(0,0,0) * trans * scale * offsetMat ;


//                osg::Vec3d posVec = position;
		//posVec[2] = -posVec[2];
		//posVec[1] = -posVec[1];
	    	//printf("BOTTOM %f %f %f\n", posVec[0], posVec[1], posVec[2]);
                loc->coordsBot.push_back(posVec);
                coords->push_back(posVec);
            }
        }
        coords->pop_back();
        int size = coords->size()/2;
        if(size>0)
        {
                      Geometry * geom = new Geometry();
            Geometry * tgeom = new Geometry();
            Geode * fgeode = new Geode();
            Geode * lgeode = new Geode();
            Geode * tgeode = new Geode();
            geom->setVertexArray(coords);
            tgeom->setVertexArray(coords);
            for(int i = 0; i < size; i++)
            {

                DrawElementsUInt* face = new DrawElementsUInt(PrimitiveSet::QUADS,0);
                face->push_back(i);
                face->push_back(i+size);
                face->push_back(((i+1)%size)+size);
                face->push_back((i+1)%size);
                geom->addPrimitiveSet(face);

              if(i < size-1)  //Commented out for now, adds caps to the polyhedra.
                {
                    face = new DrawElementsUInt(PrimitiveSet::TRIANGLES,0);
                    face->push_back(0);
                    face->push_back(i);
                    face->push_back(i+1);
                    geom->addPrimitiveSet(face);
                    tgeom->addPrimitiveSet(face);
                    face = new DrawElementsUInt(PrimitiveSet::TRIANGLES,0);
                    face->push_back(size);
                    face->push_back(size+i);
                    face->push_back(size+i+1);
                    geom->addPrimitiveSet(face);
                    //tgeom->addPrimitiveSet(face);
                }
            }
            StateSet * state(fgeode->getOrCreateStateSet());
            Material * mat(new Material);
            mxml_node_t* color_node = mxmlFindElement(node, tree, "color", NULL, NULL, MXML_DESCEND);
            double colors[3];
            double colorsl[3];
            mxml_node_t* color_child = color_node->child;
            for(int i = 0; i < 4; i++) //New
            {
                colors[i] = atof(color_child->value.text.string);
                colorsl[i] = atof(color_child->value.text.string);
                if ((colorsl[i] != 0) && (i != 3))
                {
                    colorsl[i] = colorsl[i] - 20;
                }
                if (i != 3)
                {
                    colors[i] = colors[i]/255;
                    colorsl[i] = colorsl[i]/255;
                }
                color_child = color_child->next;
            }
            //Vec4f color = Vec4f(colors[0],colors[1],colors[2],0.4);
            Vec4f color = Vec4f(colors[0],colors[1],colors[2],colors[3]); //Replaced
            Vec4f colorl = Vec4f(colorsl[0],colorsl[1],colorsl[2],colorsl[3]); //New
            Vec4f colort = Vec4f(colors[0],colors[1],colors[2],colors[3]); //New
            mat->setColorMode(Material::DIFFUSE);
            mat->setDiffuse(Material::FRONT_AND_BACK,color);
            state->setAttribute(mat);
            state->setRenderingHint(StateSet::TRANSPARENT_BIN);
            state->setMode(GL_BLEND, StateAttribute::ON);
            state->setMode(GL_LIGHTING, StateAttribute::OFF);
            osg::PolygonMode * polymode = new osg::PolygonMode;
            polymode->setMode(osg::PolygonMode::FRONT_AND_BACK,osg::PolygonMode::FILL);
            state->setAttributeAndModes(polymode,osg::StateAttribute::OVERRIDE|osg::StateAttribute::ON);
            fgeode->setStateSet(state);
            fgeode->addDrawable(geom);

            StateSet * state2(lgeode->getOrCreateStateSet());
            Material * mat2(new Material);
            state2->setRenderingHint(StateSet::OPAQUE_BIN);
            mat2->setColorMode(Material::DIFFUSE);
            mat2->setDiffuse(Material::FRONT_AND_BACK,colorl);
            state2->setAttribute(mat2);
            state->setMode(GL_BLEND, StateAttribute::ON);
            state2->setMode(GL_LIGHTING, StateAttribute::OFF);
            osg::PolygonMode * polymode2 = new osg::PolygonMode;
            polymode2->setMode(osg::PolygonMode::FRONT_AND_BACK,osg::PolygonMode::LINE);
            state2->setAttributeAndModes(polymode2,osg::StateAttribute::OVERRIDE|osg::StateAttribute::ON);
            lgeode->setStateSet(state2);
            lgeode->addDrawable(geom);

            StateSet * state3(tgeode->getOrCreateStateSet());
            Material * mat3(new Material);
            state3->setRenderingHint(StateSet::OPAQUE_BIN);
            //state3->setRenderingHint(StateSet::TRANSPARENT_BIN);
            mat3->setColorMode(Material::DIFFUSE);
            mat3->setDiffuse(Material::FRONT_AND_BACK,colort);
            state3->setAttribute(mat3);
            state3->setMode(GL_BLEND, StateAttribute::ON);
            state3->setMode(GL_LIGHTING, StateAttribute::OFF);
            osg::PolygonMode * polymode3 = new osg::PolygonMode;
            polymode3->setMode(osg::PolygonMode::FRONT_AND_BACK,osg::PolygonMode::FILL);
            state3->setAttributeAndModes(polymode3,osg::StateAttribute::OVERRIDE|osg::StateAttribute::ON);
            tgeode->setStateSet(state3);
            tgeode->addDrawable(tgeom);

            loc->geom = geom;
            loc->fill_geode = fgeode;
            loc->line_geode = lgeode;
            loc->top_geode = tgeode;
            query->sphereRoot->addChild(loc->fill_geode);
            query->sphereRoot->addChild(loc->line_geode);
           // query->sphereRoot->addChild(loc->top_geode);
            //query->sphereRoot->removeChild(loc->top_geode);
            //setNodeMask(0xffffffff)

	    Geode * textGeode = new Geode();
            loc->text_geode = textGeode;
            StateSet * textSS = textGeode->getOrCreateStateSet();
            textSS->setRenderingHint(StateSet::TRANSPARENT_BIN);
            textSS->setMode(GL_BLEND, StateAttribute::ON);
            textSS->setMode(GL_LIGHTING, StateAttribute::OFF);
            loc->label = new osgText::Text();
            loc->label->setText(loc->name);
            loc->label->setAlignment(osgText::Text::CENTER_CENTER);
            textGeode->addDrawable(loc->label);
            for(int i = 0; i < size; i++)
            {

                float width = abs(loc->coordsTop[i].z()-loc->coordsBot[i].z());
                //printf("Width %f\n", width);
                Vec3d edge = (loc->coordsBot[(i+1)%size]-loc->coordsBot[i]);
                Matrix scale;
              // double scaleFactor = (min(min((float)edge.length()/600,(float)width/60),0.7f))/10;
                double scaleFactor = 0.000200;
                //printf("Scale %f\n", scaleFactor);//0.000167
                scale.makeScale(scaleFactor,scaleFactor,scaleFactor);

                MatrixTransform * scaleTrans = new MatrixTransform();
                scaleTrans->setMatrix(scale);

                Matrix rot1;

                // rot1.makeRotate(acos(edge.x()/edge.length()),Vec3f(0,-edge.z(),edge.y()));
                 rot1.makeRotate(acos(edge.x()/edge.length()),Vec3f(0,0,edge.y()));
                 //rot1.makeRotate(acos(edge.x()/edge.length()),Vec3f(edge.y(),0,0));
                 //rot1.makeRotate(acos(edge.x()/edge.length()),Vec3f(0.5,edge.z(),edge.y()));
                 //rot1.makeRotate(acos(edge.x()/edge.length()),Vec3d(edge.x(),edge.y(),edge.z()));
                 //printf("Angle %f %f %f %f\n",acos(edge.x()/edge.length()), edge.x(), edge.y(), edge.z());
                 //test
                Matrix rot2;
                rot2.makeRotate(osg::DegreesToRadians(90.0),edge.x(),edge.y(),0);
                Matrix rot3;
                rot3.makeRotate(osg::DegreesToRadians(180.0),0,0,1);
                MatrixTransform * rotTrans = new MatrixTransform();
                rotTrans->setMatrix(rot1 * rot2 * rot3);
                //rotTrans->setMatrix(rot1);


                Matrix pos;

                Vec3d norm = (loc->coordsBot[i]-loc->coordsTop[i])^(loc->coordsTop[(i+1)%size]-loc->coordsTop[i]);
                norm/=norm.length();
                norm*=5;
                //printf("Norm %f %f %f\n",norm[0], norm[1], norm[2]);

                Vec3d posF = ((loc->coordsBot[i]+loc->coordsBot[(i+1)%size]+loc->coordsTop[i]+loc->coordsTop[(i+1)%size])/4);
                 //printf("Face %f %f %f\n",posF[0], posF[1], posF[2]);

                //pos.makeTranslate((loc->coordsBot[i]+loc->coordsBot[(i+1)%size]+loc->coordsTop[i]+loc->coordsTop[(i+1)%size])/4+norm);
                pos.makeTranslate(posF);
                 //printf("Position %f %f %f\n",pos[0], pos[1], pos[2]);
                MatrixTransform * posTrans = new MatrixTransform();
                posTrans->setMatrix(pos);
                scaleTrans->addChild(loc->text_geode);
                rotTrans->addChild(scaleTrans);
                posTrans->addChild(rotTrans);
                //posTrans->addChild(scaleTrans);
                query->sphereRoot->addChild(posTrans);

                locCenter+=loc->coordsTop[i];

            }
            bool topLabel = true;
            if (topLabel)
            {
                //Vec3d posF = loc->coordsTop[0];
                double tl;
                double old;
                int vertice = 0;
                for(int i = 0; i < loc->coordsTop.size(); i++)
                {
                    tl = ((loc->coordsTop[i].x())*-1) + loc->coordsTop[i].y();
                    if(i == 0)
                    {
                     old = tl;
                    }
                    else
                    {
                        if(tl > old)
                        {
                            old = tl;
                            vertice = i;
                        }

                    }


                }
                //double edge = loc->coordsTop[0].y() - loc->coordsTop[1].y();
                double y;
                Vec3d posF = loc->coordsTop[vertice];
                int loclength = loc->name.length() /2;
                double locOffset = loclength * 0.1;
                posF = posF + Vec3d (locOffset,-0.1,0.03);
                Matrix pos;
                pos.makeTranslate(posF);
                MatrixTransform * posTrans = new MatrixTransform();
                posTrans->setMatrix(pos);

                double scaleFactor = 0.000300;

                Matrix scale;
                scale.makeScale(scaleFactor,scaleFactor,scaleFactor);

                MatrixTransform * scaleTrans = new MatrixTransform();
                scaleTrans->setMatrix(scale);
                scaleTrans->addChild(loc->text_geode);
                posTrans->addChild(scaleTrans);
                query->sphereRoot->addChild(posTrans);
                //cout << pos.x() << " " << pos.x() << " " << pos.x() << " "




            }

            locCenter/=size;
            loc->label->setCharacterSize(300);
        }
        //center+=locCenter;
        query->loci.push_back(loc);
    }
    //center/=query->loci.size();
   // query->center = center;
#ifdef WITH_OSSIMPLANET
    if(_ossim)
        OssimPlanet::instance()->addModel(query->sphereRoot,center.y(),center.x(),Vec3(1,1,1),10,0,0,0);
#endif
    std::cerr << "Loci Loaded." << std::endl;
}
void ArtifactVis2::setupSiteMenu()
{
    _modelDisplayMenu = new SubMenu("Models");
    _displayMenu->addItem(_modelDisplayMenu);

    _pcDisplayMenu = new SubMenu("Point Clouds");
    _displayMenu->addItem(_pcDisplayMenu);
    cout << "Generating Model menu..."<<endl;
    string file = ConfigManager::getEntry("Plugin.ArtifactVis2.3DModelFolder").append("models.kml"); //Replaced
    FILE * fp = fopen(file.c_str(),"r");
    if(fp == NULL)
    {
        std::cerr << "Unable to open file: " << file << std::endl;
        return;
    }
    const double M_TO_MM = 1.0f;
   // const double LATLONG_FACTOR = 100000.0f;
    Vec3d offset = Vec3d(
                   ConfigManager::getFloat("Plugin.ArtifactVis2.Offset.X",0),
                   ConfigManager::getFloat("Plugin.ArtifactVis2.Offset.Y",0),
                   ConfigManager::getFloat("Plugin.ArtifactVis2.Offset.Z",0));

    mxml_node_t * tree;
    tree = mxmlLoadFile(NULL, fp, MXML_TEXT_CALLBACK);
    fclose(fp);
    if(tree == NULL)
    {
      std::cerr << "Unable to parse XML file: " << file << std::endl;
      return;
    }
    mxml_node_t * node = mxmlFindElement(tree, tree, "Placemark", NULL, NULL, MXML_DESCEND);
    for(; node != NULL; node = mxmlFindElement(node, tree, "Placemark", NULL, NULL, MXML_DESCEND))
    {
         mxml_node_t * child = mxmlFindElement(node, tree, "name", NULL, NULL, MXML_DESCEND);
        string child_text = child->child->value.text.string;
        bool isPC = child_text.find("PointCloud")!=string::npos;
        child = mxmlFindElement(node, tree, "href", NULL, NULL, MXML_DESCEND);
        MenuCheckbox * site = new MenuCheckbox(child->child->value.text.string,false);
        site->setCallback(this);
        MenuButton * reload = new MenuButton("-Reload");
        reload->setCallback(this);
        int factor[1];
        double trans[3];
        double scale[3];
        double rot[3];
        child = mxmlFindElement(node, tree, "altitudeMode", NULL, NULL, MXML_DESCEND);
        factor[0] = atoi(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "longitude", NULL, NULL, MXML_DESCEND);
        trans[0] = atof(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "latitude", NULL, NULL, MXML_DESCEND);
        trans[1] = atof(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "altitude", NULL, NULL, MXML_DESCEND);
        trans[2] = atof(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "x", NULL, NULL, MXML_DESCEND);
        scale[0] = atof(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "y", NULL, NULL, MXML_DESCEND);
        scale[1] = atof(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "z", NULL, NULL, MXML_DESCEND);
        scale[2] = atof(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "tilt", NULL, NULL, MXML_DESCEND);
        rot[0] = atof(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "heading", NULL, NULL, MXML_DESCEND);
        rot[1] = atof(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "roll", NULL, NULL, MXML_DESCEND);
        rot[2] = atof(child->child->value.text.string);
        Vec3d position(trans[0], trans[1], trans[2]);
        Matrixd transMat;
        transMat.makeTranslate(position);
        Matrixd scaleMat;
        //scaleMat.makeScale(M_TO_MM*LATLONG_FACTOR, M_TO_MM*LATLONG_FACTOR, M_TO_MM);
        scaleMat.makeScale(M_TO_MM, M_TO_MM, M_TO_MM);
        Matrixd rot1;
        rot1.makeRotate(osg::DegreesToRadians(-90.0), 0, 1, 0);
        Matrixd rot2;
        rot2.makeRotate(osg::DegreesToRadians(90.0), 1, 0, 0);
        Matrixd mirror;
        mirror.makeScale(1, -1, 1);
        Matrixd offsetMat;
        offsetMat.makeTranslate(offset);
        if(isPC)
        {
            _pcRoot.push_back(new MatrixTransform());
            Vec3d pos = Vec3d(0,0,0) * transMat;
            _pcDisplayMenu->addItem(site);
            _pcDisplayMenu->addItem(reload);
            _showPCCB.push_back(site);
            _reloadPC.push_back(reload);
            _pcPos.push_back(pos);
            _pcScale.push_back(Vec3d(scale[0],scale[1],scale[2]));
            _pcRot.push_back(Vec3d(rot[0],rot[1],rot[2]));
            _pcFactor.push_back(factor[0]);
        }
        else
        {
            _siteRoot.push_back(new MatrixTransform());

            Vec3d pos;
            //if(_ossim)
            if(true)
            {
                pos = position;
                cout << "Ossim position: ";
            }
            else
            {
                //pos = Vec3d(0,0,0) * mirror * transMat * scaleMat * mirror * rot2 * rot1 * offsetMat;
            }
            cout << pos[0] << ", " << pos[1] << ", " << pos[2] << endl;
            _modelDisplayMenu->addItem(site);
            _modelDisplayMenu->addItem(reload);
            _showModelCB.push_back(site);
            _reloadModel.push_back(reload);
            _sitePos.push_back(pos);
            _siteScale.push_back(Vec3d(scale[0],scale[1],scale[2]));
            _siteRot.push_back(Vec3d(rot[0],rot[1],rot[2]));

        }

    }
    int countMP = _pcRoot.size() + _siteRoot.size();

    cout << "Total Models and PC loaded: " << countMP << "\n";
    _coordsPC.resize(countMP);
    _colorsPC.resize(countMP);
    _avgOffset.resize(countMP);
    _modelSFileNode.resize(countMP);
    cout << "done." << endl;
}
void ArtifactVis2::reloadSite(int index)
{

    //string file = ConfigManager::getEntry("Plugin.ArtifactVis2.ArchInterfaceFolder").append("Model/KISterrain2.kml");
    string file = ConfigManager::getEntry("Plugin.ArtifactVis2.3DModelFolder").append("models.kml"); //Replaced
    FILE * fp = fopen(file.c_str(),"r");
    if(fp == NULL)
    {
        std::cerr << "Unable to open file: " << file << std::endl;
        return;
    }
    const double M_TO_MM = 1.0f;
    //const double LATLONG_FACTOR = 100000.0f;
    //const double LATLONG_FACTOR = 1000.0f;
    Vec3d offset = Vec3d(
                   ConfigManager::getFloat("Plugin.ArtifactVis2.Offset.X",0),
                   ConfigManager::getFloat("Plugin.ArtifactVis2.Offset.Y",0),
                   ConfigManager::getFloat("Plugin.ArtifactVis2.Offset.Z",0));
//bang1
    mxml_node_t * tree;
    tree = mxmlLoadFile(NULL, fp, MXML_TEXT_CALLBACK);
    fclose(fp);
    if(tree == NULL)
    {
      std::cerr << "Unable to parse XML file: " << file << std::endl;
      return;
    }
    mxml_node_t * node = mxmlFindElement(tree, tree, "Placemark", NULL, NULL, MXML_DESCEND);
    int incPC = 0;
    int incModel = 0;
    for(; node != NULL; node = mxmlFindElement(node, tree, "Placemark", NULL, NULL, MXML_DESCEND))
    {
        mxml_node_t * child = mxmlFindElement(node, tree, "name", NULL, NULL, MXML_DESCEND);
        string child_text = child->child->value.text.string;
        bool isPC = child_text.find("PointCloud")!=string::npos;
        child = mxmlFindElement(node, tree, "href", NULL, NULL, MXML_DESCEND);
        //MenuCheckbox * site = new MenuCheckbox(child->child->value.text.string,false);
        //site->setCallback(this);
        //MenuButton * reload = new MenuButton("-Reload");
        //reload->setCallback(this);
        int factor[1];
        double trans[3];
        double scale[3];
        double rot[3];
        child = mxmlFindElement(node, tree, "altitudeMode", NULL, NULL, MXML_DESCEND);
        factor[0] = atoi(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "longitude", NULL, NULL, MXML_DESCEND);
        trans[0] = atof(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "latitude", NULL, NULL, MXML_DESCEND);
        trans[1] = atof(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "altitude", NULL, NULL, MXML_DESCEND);
        trans[2] = atof(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "x", NULL, NULL, MXML_DESCEND);
        scale[0] = atof(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "y", NULL, NULL, MXML_DESCEND);
        scale[1] = atof(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "z", NULL, NULL, MXML_DESCEND);
        scale[2] = atof(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "tilt", NULL, NULL, MXML_DESCEND);
        rot[0] = atof(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "heading", NULL, NULL, MXML_DESCEND);
        rot[1] = atof(child->child->value.text.string);
        child = mxmlFindElement(node, tree, "roll", NULL, NULL, MXML_DESCEND);
        rot[2] = atof(child->child->value.text.string);
        Vec3d position(trans[0], trans[1], trans[2]);
        Matrixd transMat;
        transMat.makeTranslate(position);
        Matrixd scaleMat;
        //scaleMat.makeScale(M_TO_MM*LATLONG_FACTOR, M_TO_MM*LATLONG_FACTOR, M_TO_MM);
        scaleMat.makeScale(M_TO_MM, M_TO_MM, M_TO_MM);
        Matrixd rot1;
        rot1.makeRotate(osg::DegreesToRadians(-90.0), 0, 1, 0);
        Matrixd rot2;
        rot2.makeRotate(osg::DegreesToRadians(90.0), 1, 0, 0);
        Matrixd mirror;
        mirror.makeScale(1, -1, 1);
        Matrixd offsetMat;
        offsetMat.makeTranslate(offset);
        if(isPC)
        {
            if(index == incPC)
            {
            //_pcRoot.push_back(new MatrixTransform());
            Vec3d pos = Vec3d(0,0,0) * transMat;
            //_pcDisplayMenu->addItem(site);
            //_pcDisplayMenu->addItem(reload);
            //_showPCCB.push_back(site);
            //_reloadPC.push_back(reload);
            _pcPos[incPC] = pos;
            _pcScale[incPC] = Vec3d(scale[0],scale[1],scale[2]);
            _pcRot[incPC] = Vec3d(rot[0],rot[1],rot[2]);
            _pcFactor[incPC] = factor[0];

            }
            incPC++;
        }
        else
        {
            //_siteRoot.push_back(new MatrixTransform());

            Vec3d pos;
            //if(_ossim)
            if(true)
            {
                pos = position;
                //cout << "Ossim position: ";
            }
            else
            {
                //pos = Vec3d(0,0,0) * mirror * transMat * scaleMat * mirror * rot2 * rot1 * offsetMat;
            }
            if(index == incModel)
            {
            //cout << pos[0] << ", " << pos[1] << ", " << pos[2] << endl;
            //_modelDisplayMenu->addItem(site);
            //_modelDisplayMenu->addItem(reload);
            //_showModelCB.push_back(site);
            //_reloadModel.push_back(reload);
            //cout << "Reset _SiteRoot\n";
            //_siteRoot[incModel] = new MatrixTransform();
            _sitePos[incModel] = pos;
            _siteScale[incModel] = Vec3d(scale[0],scale[1],scale[2]);
            _siteRot[incModel] = Vec3d(rot[0],rot[1],rot[2]);
            }
            incModel++;

        }


    }
    //int countMP = _pcRoot.size() + _siteRoot.size();

    //cout << "Total Models and PC loaded: " << countMP << "\n";
    //_coordsPC.resize(countMP);
   //_colorsPC.resize(countMP);
   // _avgOffset.resize(countMP);
    //_modelSFileNode.resize(countMP);
    cout << "done." << endl;
}
void ArtifactVis2::setupQuerySelectMenu()
{
    vector<std::string> queryNames;
    vector<bool> queryActive;

    _displayMenu->removeItem(_artifactDisplayMenu);
    _displayMenu->removeItem(_locusDisplayMenu);

    _artifactDisplayMenu = new SubMenu("Artifacts");
    _displayMenu->addItem(_artifactDisplayMenu);

    _locusDisplayMenu = new SubMenu("Loci");
    _displayMenu->addItem(_locusDisplayMenu);
    for(int i = 0; i < _query.size(); i++)
    {
        _root->removeChild(_query[i]->sphereRoot);
        queryNames.push_back(_query[i]->name);
        queryActive.push_back(_query[i]->active);
    }
    _query.clear();
    _locusDisplayMode = new MenuTextButtonSet(true,300,30,1);
    _locusDisplayMode->addButton("Fill");
    _locusDisplayMode->setValue("Fill",true);
    _locusDisplayMode->addButton("Wireframe");
    _locusDisplayMode->addButton("Solid");
    _locusDisplayMode->addButton("Top");
    _locusDisplayMode->setCallback(this);
    _locusDisplayMenu->addItem(_locusDisplayMode);
    _queryOptionMenu.clear();
    _queryOption.clear();
    _showQueryInfo.clear();
    _queryInfo.clear();
    _queryDynamicUpdate.clear();
    _eraseQuery.clear();
    _centerQuery.clear();
    _toggleLabel.clear();
    //std::string file = ConfigManager::getEntry("Plugin.ArtifactVis2.ArchInterfaceFolder").append("kmlfiles.xml");
    std::string file = ConfigManager::getEntry("Plugin.ArtifactVis2.Database").append("kmlfiles.xml"); //Replaced

    FILE * fp = fopen(file.c_str(),"r");
    if(fp == NULL)
    {
        std::cerr << "Unable to open file: " << file << std::endl;
        return;
    }

    mxml_node_t * tree;
    tree = mxmlLoadFile(NULL, fp, MXML_TEXT_CALLBACK);
    fclose(fp);
    if(tree == NULL)
    {
      std::cerr << "Unable to parse XML file: " << file << std::endl;
      return;
    }
    mxml_node_t * table = mxmlFindElement(tree, tree, "kmlfiles", NULL, NULL, MXML_DESCEND);
    for(mxml_node_t * child = table-> child; child != NULL; child = child->next)
    {
        std::string kmlName = child->value.text.string;
        //std::string kmlfile = ConfigManager::getEntry("Plugin.ArtifactVis2.ArchInterfaceFolder").append("queries/").append(kmlName).append(".kml");
        std::string kmlfile = ConfigManager::getEntry("Plugin.ArtifactVis2.Database").append(kmlName).append(".kml"); //Replaced

        FILE * fpkml = fopen(kmlfile.c_str(),"r");
        if(fpkml == NULL)
        {
            std::cerr << "Unable to open file: " << kmlfile << std::endl;
            return;
        }
        mxml_node_t * querytree;
        querytree = mxmlLoadFile(NULL, fpkml, MXML_TEXT_CALLBACK);
        fclose(fpkml);
        if(querytree == NULL)
        {
            std::cerr << "Unable to parse XML file: " << kmlfile << std::endl;
            return;
        }
        bool sf = mxmlFindElement(querytree,querytree,"Polyhedron",NULL,NULL,MXML_DESCEND) == NULL;
        QueryGroup * query = new QueryGroup;
        query->kmlPath = kmlfile;
        query->sf = sf;
        query->sphereRoot = new MatrixTransform();
        readQuery(query);
        SubMenu * queryOptionMenu = new SubMenu(query->name);
        bool isActive = false;
        for(int i = 0; i < queryNames.size(); i++)
        {
            if(kmlName==queryNames[i])
            {
                isActive = queryActive[i];
            }
        }
        query->active = isActive;
        _query.push_back(query);
        MenuCheckbox * queryOption = new MenuCheckbox("Use this Query",isActive);
        queryOption->setCallback(this);
        SubMenu * showInfo = new SubMenu("Show info");
        stringstream ss;
        ss << "Query: " << query->query << "\n";
        ss << "Size: " << query->artifacts.size() << " Artifacts\n";
        MenuText * info = new MenuText(ss.str(),1,false,400);
        showInfo->addItem(info);
        MenuCheckbox * dynamic = new MenuCheckbox("Dynamically Update", false);
        MenuButton * erase = new MenuButton("Delete this Query");
        erase->setCallback(this);
        MenuButton * center = new MenuButton("Center Query");
        center->setCallback(this);
        MenuCheckbox * toglabel = new MenuCheckbox("Labels OnOff", true);
        toglabel->setCallback(this);
        _queryOptionMenu.push_back(queryOptionMenu);
        _queryOption.push_back(queryOption);
        _showQueryInfo.push_back(showInfo);
        _queryInfo.push_back(info);
        _queryDynamicUpdate.push_back(dynamic);
        _eraseQuery.push_back(erase);
        _centerQuery.push_back(center);
        _toggleLabel.push_back(toglabel);
        queryOptionMenu->addItem(queryOption);
        queryOptionMenu->addItem(showInfo);
        queryOptionMenu->addItem(dynamic);
        if(kmlName!="query"&&kmlName!="querp")
            queryOptionMenu->addItem(erase);
        queryOptionMenu->addItem(center);
        queryOptionMenu->addItem(toglabel); //new
        _root->addChild(query->sphereRoot);
        if(!isActive) query->sphereRoot->setNodeMask(0);
        else
            if(query->sf) displayArtifacts(query);
        if(sf)
            _artifactDisplayMenu->addItem(queryOptionMenu);
        else
            _locusDisplayMenu->addItem(queryOptionMenu);


    }
    cout << "Menu loaded." << endl;
}
void ArtifactVis2::setupTablesMenu()
{
    std::string file = ConfigManager::getEntry("Plugin.ArtifactVis2.Database").append("tables.xml"); //Replaced Problemo
    FILE * fp = fopen(file.c_str(),"r");
    if(fp == NULL)
    {
        std::cerr << "Unable to open file: " << file << std::endl;
        return;
    }

    mxml_node_t * tree;
    tree = mxmlLoadFile(NULL, fp, MXML_TEXT_CALLBACK);
    fclose(fp);
    if(tree == NULL)
    {
      std::cerr << "Unable to parse XML file: " << file << std::endl;
      return;
    }
    _tablesMenu = new SubMenu("Query Database");
    mxml_node_t * table = mxmlFindElement(tree, tree, "tables", NULL, NULL, MXML_DESCEND);
    for(mxml_node_t * child = table-> child; child != NULL; child = child->next)
    {
        std::string tableName = child->value.text.string;
        Table* table = new Table;
        table->name = tableName;
        SubMenu * tableMenu = new SubMenu(tableName);
        table->queryMenu = tableMenu;
        _tablesMenu->addItem(tableMenu);
        _tables.push_back(table);
    }
}
void ArtifactVis2::setupQueryMenu(Table * table)
{
     bool status;
    std::string file = ConfigManager::getEntry("Plugin.ArtifactVis2.Database").append(table->name).append(".xml"); //New
    FILE * fp = fopen(file.c_str(),"r");
    if(ComController::instance()->isMaster())
    {
        if (fp == NULL)
        {
        chdir(ConfigManager::getEntry("Plugin.ArtifactVis2.ArchInterfaceFolder").c_str());
        stringstream ss;
        ss << "./ArchInterface -m \"" << table->name << "\"";
        system(ss.str().c_str());
    	ComController::instance()->sendSlaves(&status,sizeof(bool));
        }
    }
    else
    {
	ComController::instance()->readMaster(&status,sizeof(bool));
    }
    table->query_view = new MenuText("",1,false,400);
    //std::string file = ConfigManager::getEntry("Plugin.ArtifactVis2.ArchInterfaceFolder").append("menu.xml");
    //std::string file = ConfigManager::getEntry("Plugin.ArtifactVis2.Database").append("menu.xml"); //Replaced
    if (fp == NULL)
    {
    fp = fopen(file.c_str(),"r");
    }
    if(fp == NULL)
    {
        std::cerr << "Unable to open file: " << file << std::endl;
        return;
    }
    mxml_node_t * tree;
    tree = mxmlLoadFile(NULL, fp, MXML_TEXT_CALLBACK);
    fclose(fp);
    if(tree == NULL)
    {
        std::cerr << "Unable to parse XML file: " << file << std::endl;
        return;
    }
    table->conditions = new SubMenu("Edit Conditions");
    table->queryMenu->addItem(table->conditions);
    mxml_node_t * node = mxmlFindElement(tree,tree,NULL,NULL,NULL,MXML_DESCEND);
    for (; node != NULL; node = mxmlFindElement(node, tree, NULL, NULL, NULL, MXML_DESCEND))
    {
        string name(node->value.element.name);
        SubMenu *menu = new SubMenu(name);
        mxml_node_t *child;
        int childcount = 0;
        for(child = node->child; child != NULL; child = child->next)
        {
            childcount++;
        }
        if(childcount <= 30)
        {
             MenuTextButtonSet * optionSet = new MenuTextButtonSet(true,500,30,1);
             std::vector<string> children;
             for(child = node->child; child != NULL; child = child->next)
             {
                  std::string optionName = child->value.text.string;
                  if(optionName.empty())break;
                  children.push_back(optionName);
             }
             sort(children.begin(),children.end());
             for(int i = 0; i < children.size(); i++)
             {
                 optionSet->addButton(children[i]);
             }
             optionSet->setCallback(this);
             table->queryOptions.push_back(optionSet);
             menu->addItem(optionSet);
             table->querySubMenu.push_back(menu);
        }
        else
        {
            std::vector<string> children;
            for(child = node->child; child != NULL; child = child->next)
            {
                 string childText = child->value.text.string;
                 if(childText.empty())break;
                 children.push_back(childText);
            }
            sort(children.begin(),children.end());
            table->sliderEntry.push_back(children);
            MenuList *slider = new MenuList();
            slider->setValues(children);
            slider->setCallback(this);
            table->queryOptionsSlider.push_back(slider);
            MenuCheckbox *useSlider = new MenuCheckbox("Use Value",false);
            useSlider->setCallback(this);
            table->querySlider.push_back(useSlider);
            menu->addItem(useSlider);
            menu->addItem(slider);
            table->querySubMenuSlider.push_back(menu);
        }
        table->conditions->addItem(menu);
    }
    table->addOR = new MenuButton("Add OR Condition");
    table->addOR->setCallback(this);
    table->queryMenu->addItem(table->addOR);
    table->removeOR = new MenuButton("Remove Last OR Condition");
    table->removeOR->setCallback(this);
    table->queryMenu->addItem(table->removeOR);
    table->viewQuery = new SubMenu("View Current Query");
    table->viewQuery->addItem(table->query_view);
    table->queryMenu->addItem(table->viewQuery);
    table->genQuery = new MenuButton("Generate Query");
    table->genQuery->setCallback(this);
    table->queryMenu->addItem(table->genQuery);
    table->clearConditions = new MenuButton("Clear All Conditions");
    table->clearConditions->setCallback(this);
    table->queryMenu->addItem(table->clearConditions);
    table->saveQuery = new MenuButton("Save Current Query");
    table->saveQuery->setCallback(this);
    table->queryMenu->addItem(table->saveQuery);
    _tablesMenu->addItem(table->queryMenu);
}
void ArtifactVis2::updateSelect()
{
    osg::Vec3 markPos(0,1000,0);
    markPos = markPos * PluginHelper::getHandMat();
    osg::Matrix markTrans;
    markTrans.makeTranslate(markPos);
    _selectMark->setMatrix(markTrans);

    if(_selectActive)
    {
        osg::Matrix w2l = PluginHelper::getWorldToObjectTransform();
	_selectCurrent = osg::Vec3(0,1000,0);
	_selectCurrent = _selectCurrent * PluginHelper::getHandMat() * w2l;
    }

    if(_selectStart.length2() > 0)
    {

	osg::BoundingBox bb;
	osg::Vec3 minvec, maxvec;
	minvec.x() = std::min(_selectStart.x(),_selectCurrent.x());
	minvec.y() = std::min(_selectStart.y(),_selectCurrent.y());
	minvec.z() = std::min(_selectStart.z(),_selectCurrent.z());

	maxvec.x() = std::max(_selectStart.x(),_selectCurrent.x());
	maxvec.y() = std::max(_selectStart.y(),_selectCurrent.y());
	maxvec.z() = std::max(_selectStart.z(),_selectCurrent.z());

	bb.set(minvec, maxvec);

	osg::Matrix scale, trans;
	trans.makeTranslate(bb.center());
	scale.makeScale(maxvec.x() - minvec.x(), maxvec.y() - minvec.y(), maxvec.z() - minvec.z());

	_selectBox->setMatrix(scale * trans);

	std::map<string,int> dcCount;
	int totalSelected = 0;
        for(int q = 0; q < _query.size(); q++)
        {
            vector<Artifact*> artifacts = _query[q]->artifacts;
            if(_query[q]->active)
            for(int i = 0; i < artifacts.size(); i++)
            {
	        if(bb.contains(artifacts[i]->modelPos) && !artifacts[i]->selected)
	        {
		    osg::ShapeDrawable * sd = dynamic_cast<osg::ShapeDrawable*>(artifacts[i]->drawable);
		    if(sd)
		    {
		        osg::Vec4 color = sd->getColor();
		        color.x() = color.x() * 2.0;
		        color.y() = color.y() * 2.0;
		        color.z() = color.z() * 2.0;
		        sd->setColor(color);
	            }
                    artifacts[i]->selected = true;
	        }
	        else if((!artifacts[i]->visible || !bb.contains(artifacts[i]->modelPos)) && artifacts[i]->selected)
	        {
		    osg::ShapeDrawable * sd = dynamic_cast<osg::ShapeDrawable*>(artifacts[i]->drawable);
		    if(sd)
		    {
		        osg::Vec4 color = sd->getColor();
		        color.x() = color.x() * 0.5;
		        color.y() = color.y() * 0.5;
		        color.z() = color.z() * 0.5;
		        sd->setColor(color);
		    }
                artifacts[i]->selected = false;
                }

                if(artifacts[i]->selected)
                {
                    dcCount[artifacts[i]->dc]++;
                    totalSelected++;
                }
            }
        }

	std::stringstream ss;
	ss << "Region Size: " << fabs(_selectStart.x() - _selectCurrent.x()) << " x " << fabs(_selectStart.y() - _selectCurrent.y()) << " x " << fabs(_selectStart.z() - _selectCurrent.z()) << std::endl;
	ss << "Artifacts Selected: " << totalSelected;
	for(std::map<std::string,int>::iterator it = dcCount.begin(); it != dcCount.end(); it++)
	{
	    ss << std::endl << it->first << ": " << it->second;
	}

	_selectionStatsPanel->setText(ss.str());
    }
}

std::vector<Vec3> ArtifactVis2::getArtifactsPos(){
    vector<Vec3f> positions;
    for(int q = 0; q < _query.size(); q++)
    {
        vector<Artifact*>::iterator item = _query[q]->artifacts.begin();

        for (int i = 0; item != _query[q]->artifacts.end();item++)
        {
            positions.push_back((*item)->modelPos);
            i++;
        }
    }

    return positions;
}
float ArtifactVis2::selectArtifactSelected()
{

    //float transMult = SpaceNavigator::instance()->transMultF();
    return transMult;

}
void ArtifactVis2::testSelected()
{

    cout << _activeArtifact;

}
osg::Matrix ArtifactVis2::getSelectMatrix()
{

    return _selectModelLoad.get()->getMatrix();

}
void ArtifactVis2::setSelectMatrix(osg::Matrix & mat)
{
    _selectModelLoad.get()->setMatrix(mat);
}
void ArtifactVis2::rotateModel(double rx, double ry, double rz)
{


               //cout << rx << "," << ry << "," << rz << "\n";
                //cout << _selectRotx << "," << _selectRoty << "," << _selectRotz << "\n";

                _selectRotx += rx;
                _selectRoty += ry;
                _selectRotz += rz;
                //cout << _selectRotx << "," << _selectRoty << "," << _selectRotz << "\n";
                        _root->removeChild(_selectModelLoad.get());

                        Matrixd scale;

                        scale.makeScale(_snum,_snum,_snum);
                        MatrixTransform* scaleTrans = new MatrixTransform();
                        scaleTrans->setMatrix(scale);
                        scaleTrans->addChild(_modelFileNode);


                        MatrixTransform * siteRote = new MatrixTransform();  //Andrew here it works, with no problem
                         Matrixd rotx;
                        rotx.makeRotate(osg::DegreesToRadians(_selectRotx), 1, 0, 0); //Rotating it 10 degrees
                        Matrixd roty;
                        roty.makeRotate(osg::DegreesToRadians(_selectRoty), 0, 1, 0); //Rotating it 10 degrees
                        Matrixd rotz;
                        rotz.makeRotate(osg::DegreesToRadians(_selectRotz), 0, 1, 0); //Rotating it 10 degrees

                        siteRote->setMatrix(rotx*roty*rotz);
                        siteRote->addChild(scaleTrans);

                        PositionAttitudeTransform* modelTrans = new PositionAttitudeTransform();
                        modelTrans->setPosition(_modelartPos);

                        modelTrans->addChild(siteRote);

                        _selectModelLoad = new osg::MatrixTransform();

                        _selectModelLoad->addChild(modelTrans);

                        _root->addChild(_selectModelLoad.get()); //reattach the transform matrix.



}

static void VRPN_CALLBACK tracker_handler(void *userdata, const vrpn_TRACKERCB t)
{
        // Store the updated position and orientation of the targeted joint
        //
        q_vec_copy(position[t.sensor], t.pos);
        q_copy(orientation[t.sensor], t.quat);
        //t_user_callback	*t_data = (t_user_callback *)userdata;

        // Make sure we have a count value for this sensor
/*
    printf("handle_tracker\tSensor %d is now at (%g,%g,%g)\n",
	 t.sensor,
	 t.pos[0], t.pos[1], t.pos[2]);
	 */

	// See if we have gotten enough reports from this sensor that we should
	// print this one.  If so, print and reset the count.
	/*
        t.sensor
		printf("Tracker %s, sensor %d:\n        pos (%5.2f, %5.2f, %5.2f); quat (%5.2f, %5.2f, %5.2f, %5.2f)\n",
			t_data->t_name,
			t.sensor,
			t.pos[0], t.pos[1], t.pos[2],
			t.quat[0], t.quat[1], t.quat[2], t.quat[3]);
			*/

}

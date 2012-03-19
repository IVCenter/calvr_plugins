#include "ArtifactVis.h"
#include "vvtokenizer.h"

#ifdef WITH_OSSIMPLANET
#include "../OssimPlanet/OssimPlanet.h"
#endif

#include <iostream>
#include <sstream>

#include <config/ConfigManager.h>
#include <kernel/PluginHelper.h>
#include <kernel/SceneManager.h>
#include <kernel/InteractionManager.h>
#include <menu/MenuSystem.h>
#include <input/TrackingManager.h>
#include <util/LocalToWorldVisitor.h>

#include <osg/CullFace>
#include <osg/Matrix>
#include <osg/ShapeDrawable>
#include <osg/PolygonMode>
#include <osgDB/ReadFile>

#include <mxml.h>

using namespace osg;
using namespace std;
using namespace cvr;

CVRPLUGIN(ArtifactVis)

ArtifactVis::ArtifactVis()
{

}

bool ArtifactVis::init()
{
    std::cerr << "ArtifactVis init\n";

    _root = new osg::MatrixTransform();

    _avMenu = new SubMenu("ArtifactVis", "ArtifactVis");
    _avMenu->setCallback(this);

    _showSiteCB = new MenuCheckbox("Show Site", false);
    _showSiteCB->setCallback(this);
    _avMenu->addItem(_showSiteCB);

    _showSpheresCB = new MenuCheckbox("Show Artifacts", false);
    _showSpheresCB->setCallback(this);
    _avMenu->addItem(_showSpheresCB);

    setupDCFilter();

    _selectArtifactCB = new MenuCheckbox("Select Artifact",false);
    _selectArtifactCB->setCallback(this);
    _avMenu->addItem(_selectArtifactCB);

    _selectCB = new MenuCheckbox("Select box", false);
    _selectCB->setCallback(this);
    _avMenu->addItem(_selectCB);

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

    //_selectMark = new osg::MatrixTransform();
    //_selectMark->addChild(geo);

    for(int i = 0; i < PluginHelper::getNumHands(); i++)
    {
	_selectMarks.push_back(new osg::MatrixTransform());
	// no need to mark the mouse, kind of a hack, should probably use scene objects
	if(TrackingManager::instance()->getHandTrackerType(i) != TrackerBase::MOUSE)
	{
	    _selectMarks[i]->addChild(geo);
	}
    }

    MenuSystem::instance()->addMenuItem(_avMenu);
    SceneManager::instance()->getObjectsRoot()->addChild(_root);

    //_my_own_root = new LOD();
    _sphereRoot = new MatrixTransform();

    _sphereRadius = 50.0;
    _activeArtifact  = -1;
    //_LODmaxRange = ConfigManager::getFloat("Plugins.ArtifactVis.MaxVisibleRange", 30.0);

    _picFolder = ConfigManager::getEntry("value","Plugin.ArtifactVis.PicFolder","");

    readLocusFile();

    _artifactPanel = new TabbedDialogPanel(400,30,4,"Selected Artifact","Plugin.ArtifactVis.ArtifactPanel");
    _artifactPanel->addTextTab("Info","");
    _artifactPanel->addTextureTab("Side","");
    _artifactPanel->addTextureTab("Top","");
    _artifactPanel->addTextureTab("Bottom","");
    _artifactPanel->setVisible(false);
    _artifactPanel->setActiveTab("Info");

    _selectionStatsPanel = new DialogPanel(450,"Selection Stats","Plugin.ArtifactVis.SelectionStatsPanel");
    _selectionStatsPanel->setVisible(false);

    std::cerr << "ArtifactVis init done.\n";
    return true;
}


ArtifactVis::~ArtifactVis()
{
}

bool ArtifactVis::processEvent(InteractionEvent * event)
{
    TrackedButtonInteractionEvent * tie = event->asTrackedButtonEvent();

    if(tie)
    {
	if((tie->getInteraction() == BUTTON_DOWN || tie->getInteraction() == BUTTON_DOUBLE_CLICK) && tie->getButton() == 0)
	{
	    if(_selectArtifactCB->getValue() && _showSpheresCB->getValue())
	    {
		if(!_selectCB->getValue())
		{
		    osg::Matrix l2w = getLocalToWorldMatrix(_sphereRoot.get());
		    osg::Matrix w2l = osg::Matrix::inverse(l2w);

		    osg::Vec3 start(0,0,0);
		    osg::Vec3 end(0,1000000,0);

		    start = start * tie->getTransform() * w2l;
		    end = end * tie->getTransform() * w2l;

		    int index = -1;
		    double distance;
		    for(int i = 0; i < _artifacts.size(); i++)
		    {
			if(!_artifacts[i]->visible)
			{
			    continue;
			}
			osg::Vec3 num = (_artifacts[i]->modelPos - start) ^ (_artifacts[i]->modelPos - end);
			osg::Vec3 denom = end - start;
			double point2line = num.length() / denom.length();
			if(point2line <= _sphereRadius)
			{
			    double point2start = (_artifacts[i]->modelPos - start).length2();
			    if(index == -1 || point2start < distance)
			    {
				distance = point2start;
				index = i;
			    }
			}
		    }

		    if(index != -1)
		    {
			//std::cerr << "Got sphere intersection with index " << index << std::endl;
			setActiveArtifact(index);
			return true;
		    }
		}
	    }
	    else if(_showSpheresCB->getValue() && _selectCB->getValue() && tie->getInteraction() == BUTTON_DOUBLE_CLICK)
	    {
		if(_selectActive && tie->getHand() != _selectHand)
		{
		    return false;
		}
		osg::Matrix l2w = getLocalToWorldMatrix(_sphereRoot.get());
		osg::Matrix w2l = osg::Matrix::inverse(l2w);
		if(!_selectActive)
		{
		    _selectStart = osg::Vec3(0,1000,0);
		    _selectStart = _selectStart * tie->getTransform() * w2l;
		    _selectActive = true;
		    _selectHand = tie->getHand();
		    for(int i = 0; i < PluginHelper::getNumHands(); i++)
		    {
			if(i == _selectHand)
			{
			    continue;
			}
			PluginHelper::getScene()->removeChild(_selectMarks[i]);
		    }
		}
		else
		{
		    _selectCurrent = osg::Vec3(0,1000,0);
		    _selectCurrent = _selectCurrent * tie->getTransform() * w2l;
		    _selectActive = false;
		    for(int i = 0; i < PluginHelper::getNumHands(); i++)
		    {
			if(i == _selectHand)
			{
			    continue;
			}
			PluginHelper::getScene()->addChild(_selectMarks[i]);
		    }
		}
		return true;
	    }
	}
    }

    return false;
}

/*bool ArtifactVis::buttonEvent(int type, int button, int hand, const osg::Matrix & mat)
{
    if((type == BUTTON_DOWN || type == BUTTON_DOUBLE_CLICK) && hand == 0 && button == 0)
    {
	if(_selectArtifactCB->getValue() && _showSpheresCB->getValue())
	{
	    if(!_selectCB->getValue())
	    {
		osg::Matrix l2w = getLocalToWorldMatrix(_sphereRoot.get());
		osg::Matrix w2l = osg::Matrix::inverse(l2w);

		osg::Vec3 start(0,0,0);
		osg::Vec3 end(0,1000000,0);

		start = start * mat * w2l;
		end = end * mat * w2l;

		int index = -1;
		double distance;
		for(int i = 0; i < _artifacts.size(); i++)
		{
		    if(!_artifacts[i]->visible)
		    {
			continue;
		    }
		    osg::Vec3 num = (_artifacts[i]->modelPos - start) ^ (_artifacts[i]->modelPos - end);
		    osg::Vec3 denom = end - start;
		    double point2line = num.length() / denom.length();
		    if(point2line <= _sphereRadius)
		    {
			double point2start = (_artifacts[i]->modelPos - start).length2();
			if(index == -1 || point2start < distance)
			{
			    distance = point2start;
			    index = i;
			}
		    }
		}

		if(index != -1)
		{
		    //std::cerr << "Got sphere intersection with index " << index << std::endl;
		    setActiveArtifact(index);
		    return true;
		}
	    }
	}
	else if(_showSpheresCB->getValue() && _selectCB->getValue() && type == BUTTON_DOUBLE_CLICK)
	{
	    osg::Matrix l2w = getLocalToWorldMatrix(_sphereRoot.get());
	    osg::Matrix w2l = osg::Matrix::inverse(l2w);
	    if(!_selectActive)
	    {
		_selectStart = osg::Vec3(0,1000,0);
		_selectStart = _selectStart * mat * w2l;
		_selectActive = true;
	    }
	    else
	    {
		_selectCurrent = osg::Vec3(0,1000,0);
		_selectCurrent = _selectCurrent * mat * w2l;
		_selectActive = false;
	    }
	    return true;
	}
    }

    return false;
}

bool ArtifactVis::mouseButtonEvent(int type, int button, int x, int y, const osg::Matrix & mat)
{
    if(type == MOUSE_BUTTON_DOWN)
    {
	if(!_selectCB->getValue())
	{
	    return buttonEvent(BUTTON_DOWN, button, 0, mat);
	}
    }
    if(type == MOUSE_DOUBLE_CLICK)
    {
	if(!_selectCB->getValue())
	{
	    return buttonEvent(BUTTON_DOUBLE_CLICK, button, 0, mat);
	}
    }

    return false;
}*/

void ArtifactVis::menuCallback(MenuItem* menuItem)
{
    if(menuItem == _showSpheresCB)
    {
        if (_showSpheresCB->getValue())
        {
            // load artifacts and send them to OssimPlanet (once)
            static bool load = true;
            if (load)
            {
                readArtifactsFile(ConfigManager::getEntry("Plugin.ArtifactVis.Database"));
                //listArtifacts(); // uncomment this line to Debug
                displayArtifacts(_sphereRoot);

#ifdef WITH_OSSIMPLANET
                if(OssimPlanet::instance() && OssimPlanet::instance()->addModel(_sphereRoot,
                    ConfigManager::getFloat("Plugins.ArtifactVis.Site.Latitude",0),
                    ConfigManager::getFloat("Plugins.ArtifactVis.Site.Longitude",0),
		    osg::Vec3(0.6, 0.6, 0.6), 0.0, 0.0, 0.0, 135.0))
                {
		    std::cerr << "Spheres added to planet." << std::endl;
                }
		else
#endif
		{
		    //PluginHelper::getObjectsRoot()->addChild( _my_sphere_root );
		    //_root->addChild(_sphereRoot);
		}
                load = false;
            }
	    
	    _root->addChild(_sphereRoot);
            // enable spheres
            //_my_own_root->setRange(0,0,_LODmaxRange);
        }
        else
        {
	    //TODO: add OssimPlanet remove
            // disable spheres
            //_my_own_root->setRange(0,0,0);
	    _root->removeChild(_sphereRoot);
        }
    }

    if(menuItem == _showSiteCB)
    {
	if(_showSiteCB->getValue())
	{
	    if(!_siteRoot.valid())
	    {
		readSiteFile();
	    }

	    _root->addChild(_siteRoot);
	}
	else
	{
	    _root->removeChild(_siteRoot);
	}
    }

    if(menuItem == _dcFilterShowAll)
    {
	_dcFilterAuto->setValue(false);
	for(int i = 0; i < _dcFilterItems.size(); i++)
	{
	    _dcFilterItems[i]->setValue(true);
	}

	for(std::map<std::string,bool>::iterator it = _dcVisibleMap.begin(); it != _dcVisibleMap.end(); it++)
	{
	    it->second = true;
	}

	updateVisibleStatus();
    }

    if(menuItem == _dcFilterShowNone)
    {
	_dcFilterAuto->setValue(false);
	for(int i = 0; i < _dcFilterItems.size(); i++)
	{
	    _dcFilterItems[i]->setValue(false);
	}

	for(std::map<std::string,bool>::iterator it = _dcVisibleMap.begin(); it != _dcVisibleMap.end(); it++)
	{
	    it->second = false;
	}

	updateVisibleStatus();
    }

    if(menuItem == _dcFilterAuto)
    {
	if(_dcFilterAuto->getValue())
	{
	    menuCallback(_dcFilterShowNone);
	    _dcFilterAuto->setValue(true);
	}
	else
	{
	    menuCallback(_dcFilterShowNone);
	}

	_filterTime = 0.0;
    }

    for(int i = 0; i < _dcFilterItems.size(); i++)
    {
	if(menuItem == _dcFilterItems[i])
	{
	    if(_dcFilterAuto->getValue())
	    {
		_dcFilterItems[i]->setValue(!_dcFilterItems[i]->getValue());
		return;
	    }

	    setDCVisibleStatus(_dcList[i], _dcFilterItems[i]->getValue());
	    break;
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
	    for(int i = 0; i < _artifacts.size(); i++)
	    {
		_artifacts[i]->selected = false;
		osg::ShapeDrawable * sd = dynamic_cast<osg::ShapeDrawable*>(_artifacts[i]->drawable);
		if(sd)
		{
		    osg::Vec4 color = sd->getColor();
		    color.x() = color.x() * 0.5;
		    color.y() = color.y() * 0.5;
		    color.z() = color.z() * 0.5;
		    sd->setColor(color);
		}
	    }
	    _selectStart = osg::Vec3(0,0,0);
	    _selectCurrent = osg::Vec3(0,0,0);
	    _sphereRoot->addChild(_selectBox);
	    for(int i = 0; i < PluginHelper::getNumHands(); i++)
	    {
		PluginHelper::getScene()->addChild(_selectMarks[i]);
	    }
	    /*if(PluginHelper::getNumHands())
	    {
		PluginHelper::getScene()->addChild(_selectMark);
	    }*/
	    _selectionStatsPanel->setVisible(true);
	}
	else
	{
	   for(int i = 0; i < _artifacts.size(); i++)
	    {
		if(!_artifacts[i]->selected)
		{
		    osg::ShapeDrawable * sd = dynamic_cast<osg::ShapeDrawable*>(_artifacts[i]->drawable);
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
	    _sphereRoot->removeChild(_selectBox);
	    for(int i = 0; i < PluginHelper::getNumHands(); i++)
	    {
		PluginHelper::getScene()->removeChild(_selectMarks[i]);
	    }
	    /*if(PluginHelper::getNumHands())
	    {
		PluginHelper::getScene()->removeChild(_selectMark);
	    }*/
	    _selectionStatsPanel->setVisible(false);
	}
	_selectActive = false;
    }
}

void ArtifactVis::preFrame()
{
    if(!_showSpheresCB->getValue())
    {
	return;
    }

    if(_dcFilterAuto->getValue())
    {
	_filterTime += PluginHelper::getLastFrameDuration();

	int index = ((int)(_filterTime / 5.0)) % _dcList.size();
	for(int i = 0; i < _dcList.size(); i++)
	{
	    if(i == index)
	    {
		if(!_dcVisibleMap[_dcList[i]])
		{
		    setDCVisibleStatus(_dcList[i],true);
		}
	    }
	    else
	    {
		if(_dcVisibleMap[_dcList[i]])
		{
		    setDCVisibleStatus(_dcList[i],false);
		}
	    }
	}
    }

    for(int i = 0; i < _artifacts.size(); i++)
    {
	osg::ShapeDrawable * sd = dynamic_cast<osg::ShapeDrawable*>(_artifacts[i]->drawable);
	if(sd)
	{
	    if(!_artifacts[i]->visible && sd->getColor().a() > 0.0)
	    {
		osg::Vec4 color = sd->getColor();
		color.a() = std::max(color.a() - PluginHelper::getLastFrameDuration(), (double) 0.0);
		sd->setColor(color);
	    }
	    else if(_artifacts[i]->visible && sd->getColor().a() < 1.0)
	    {
		osg::Vec4 color = sd->getColor();
		color.a() = std::min(color.a() + PluginHelper::getLastFrameDuration(), (double) 1.0);
		sd->setColor(color);
	    }
	}
    }

    if(_selectCB->getValue())
    {
	updateSelect();
    }
}

void ArtifactVis::setDCVisibleStatus(std::string dc, bool status)
{
    for(int i = 0; i < _artifacts.size(); i++)
    {
	if(_artifacts[i]->dc == dc)
	{
	    _artifacts[i]->visible = status;
	}
    }

    for(int i = 0; i < _dcFilterItems.size(); i++)
    {
	if(_dcList[i] == dc)
	{
	    _dcFilterItems[i]->setValue(status);
	    break;
	}
    }

    _dcVisibleMap[dc] = status;
}

void ArtifactVis::updateVisibleStatus()
{
    for(int i = 0; i < _artifacts.size(); i++)
    {
	_artifacts[i]->visible = _dcVisibleMap[_artifacts[i]->dc];
    }
}

void ArtifactVis::setActiveArtifact(int art)
{
    if(art < 0 || art >= _artifacts.size())
    {
	return;
    }

    if(art == _activeArtifact)
    {
	return;
    }

    std::stringstream ss;
    ss << "EDM: " << _artifacts[art]->edm << std::endl;
    ss << "DC: " << _artifacts[art]->dc << std::endl;
    ss << "Basket: " << _artifacts[art]->basket;

    _artifactPanel->updateTabWithText("Info",ss.str());

    std::stringstream side, top, bottom;
    side <<  _picFolder << "/" << _artifacts[art]->edm << "_s.jpg";
    top << _picFolder << "/" << _artifacts[art]->edm << "_t.jpg";
    bottom << _picFolder << "/" << _artifacts[art]->edm << "_b.jpg";

    _artifactPanel->updateTabWithTexture("Side",side.str());
    _artifactPanel->updateTabWithTexture("Top",top.str());
    _artifactPanel->updateTabWithTexture("Bottom",bottom.str());

    //std::cerr << "Side texture: " << side.str() << std::endl;
    //std::cerr << "Top texture: " << top.str() << std::endl;
    //std::cerr << "Bottom texture: " << bottom.str() << std::endl;

    _activeArtifact = art;
}

void ArtifactVis::readArtifactsFile(std::string filename)
{
  cerr << "Reading artifacts file: " << filename << endl;

  vvTokenizer::TokenType ttype;
  FILE* fp = fopen(filename.c_str(), "rb");
  if (fp==NULL)
  {
    cerr << "Cannot read file: " << filename << endl;
    return;
  }
  vvTokenizer* tokenizer = new vvTokenizer(fp);
  tokenizer->setEOLisSignificant(true);
  tokenizer->setCaseConversion(vvTokenizer::VV_UPPER);
  tokenizer->setParseNumbers(true);
  tokenizer->setAlphaCharacter(' ');
  tokenizer->setWhitespaceCharacter('\t');
  tokenizer->setCommentCharacter('#');
  while ((ttype = tokenizer->nextToken()) != vvTokenizer::VV_EOF)
  {
    Artifact* newArtifact = new Artifact();
    
    // EDM:
    if (tokenizer->ttype != vvTokenizer::VV_NUMBER) 
    {
      cerr << "Error: expected EDM in line " << tokenizer->getLineNumber() << endl;
      break;
    }
    newArtifact->edm = int(tokenizer->nval);
    
    // DC:
    tokenizer->nextToken();
    if (tokenizer->ttype != vvTokenizer::VV_WORD) 
    {
      cerr << "Error: expected DC Code in line " << tokenizer->getLineNumber() << endl;
      break;
    }    
    string dcString(tokenizer->sval);
    newArtifact->dc = dcString;

    //Add this descriptor to the list, if it does not exist yet

    vector<std::string>::iterator dc_item = _descriptor_list.begin();
    int foundit=0;
    for (; dc_item < _descriptor_list.end(); dc_item++)
    {
      if ((*dc_item).compare(dcString)==0)
        {
         foundit=1;
         break;
        }
    }

    if (foundit==0){
      _descriptor_list.push_back(dcString);
      _descriptor_list_colors.push_back(Vec4((float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX,1.00));
      //cerr << dcString << endl;
    }


    // Locus:
    tokenizer->nextToken();
    if (tokenizer->ttype != vvTokenizer::VV_NUMBER)
    {
      cerr << "Error: expected Locus in line " << tokenizer->getLineNumber() << endl;
      cerr << "Read: " << tokenizer->sval << endl;
      break;
    } 

    newArtifact->locus = int(tokenizer->nval);

    // Basket:
    tokenizer->nextToken();
    if (tokenizer->ttype != vvTokenizer::VV_NUMBER)
    {
      cerr << "Error: expected Basket in line " << tokenizer->getLineNumber() << endl;
      break;
    }
    newArtifact->basket = int(tokenizer->nval);

    // Square:
    tokenizer->nextToken();
    string square(tokenizer->sval);
    newArtifact->square = square;

    // Date:
    tokenizer->nextToken();
    string date(tokenizer->sval);
    newArtifact->date = date;

    // Area:
    tokenizer->nextToken();
    if (tokenizer->ttype != vvTokenizer::VV_WORD)
    {
      cerr << "Error: expected Area in line " << tokenizer->getLineNumber() << ", " << tokenizer->sval << endl;
    }
    newArtifact->area = (tokenizer->sval[0]);
    
    // Site:
    tokenizer->nextToken();
    if (tokenizer->ttype != vvTokenizer::VV_WORD) 
    {
      cerr << "Error: expected Site in line " << tokenizer->getLineNumber() << endl;
      break;
    }    
    string siteString(tokenizer->sval);
    newArtifact->site = siteString;

    // Pos[0]
    tokenizer->nextToken();
    if (tokenizer->ttype != vvTokenizer::VV_NUMBER)
    {
      cerr<< "Error: expected Northing in line " << tokenizer->getLineNumber() << endl;
      break;
    }
    newArtifact->pos[0] = tokenizer->nval;

    // Pos[1]
    tokenizer->nextToken();
    if (tokenizer->ttype != vvTokenizer::VV_NUMBER)
    {
      cerr<< "Error: expected Easting in line " << tokenizer->getLineNumber() << endl;
      break;
    }
    newArtifact->pos[1] = tokenizer->nval;

    // Pos[2]
    tokenizer->nextToken();
    if (tokenizer->ttype != vvTokenizer::VV_NUMBER)
    {
      cerr<< "Error: expected Elevation in line " << tokenizer->getLineNumber() << endl;
      break;
    }
    newArtifact->pos[2] = tokenizer->nval;

    newArtifact->visible = true;

    // done with line:
    _artifacts.push_back(newArtifact);
    tokenizer->nextLine();      // won't need this once all elements in line are read
  }
  delete tokenizer;
  fclose(fp); 
}

void ArtifactVis::listArtifacts()
{
  cerr << "Listing " << _artifacts.size() << " elements:" << endl;
  vector<Artifact*>::iterator item = _artifacts.begin();
  for (; item < _artifacts.end(); item++)
  {
    cerr << "edm: " << (*item)->edm << ", dc: " << (*item)->dc << ", basket: " << (*item)->basket << ", northing: " << (*item)->pos[0] << ", easting: " << (*item)->pos[1] << ", elevation: " << (*item)->pos[2] << endl;
  }

    vector<std::string>::iterator dc_item = _descriptor_list.begin();
 
    int ind=-1;
 for (; dc_item < _descriptor_list.end(); dc_item++)
  {
    ind++;
    cerr <<ind<< "  dc: " << (*dc_item)<<endl;
  }

   cerr<<"Num descriptors = "<<_descriptor_list.size()<<endl;
}

void ArtifactVis::displayArtifacts(Group * root_node)
{
    const double M_TO_MM = 1000.0f;

    cerr << "Creating " << _artifacts.size() << " artifacts...";
    vector<Artifact*>::iterator item = _artifacts.begin();

    Vec3d offset = Vec3d(
        ConfigManager::getDouble("Plugin.ArtifactVis.Offset.X",0),
        ConfigManager::getDouble("Plugin.ArtifactVis.Offset.Y",0),
        ConfigManager::getDouble("Plugin.ArtifactVis.Offset.Z",0));

    float tessellation = ConfigManager::getFloat("Plugin.ArtifactVis.Tessellation",.2);

    int artCount = _artifacts.size();

    osg::Geode * sphereGeode = new osg::Geode();

    for (int objCount = 0; item < _artifacts.end();item++)
    {
        //cerr<<"Creating object "<<++objCount<<" out of"<<artCount<<endl;
        Vec3d position((*item)->pos[0], (*item)->pos[1], (*item)->pos[2]);

        Matrixd trans;
        trans.makeTranslate(position + offset);
        Matrixd scale;
        scale.makeScale(M_TO_MM, M_TO_MM, M_TO_MM);
        Matrixd rot1;
        rot1.makeRotate(osg::DegreesToRadians(-90.0), 0, 1, 0);
        Matrixd rot2;
        rot2.makeRotate(osg::DegreesToRadians(90.0), 1, 0, 0);
        Matrixd mirror;
        mirror.makeScale(1, -1, 1);

        vector<std::string>::iterator dc_item = _descriptor_list.begin();
        int index=0;
        for (; dc_item < _descriptor_list.end(); dc_item++)
        {
            if ((*dc_item).compare((*item)->dc)==0)
                break;
            index++;
        }
	
	osg::Vec3d pos = osg::Vec3d(0,0,0) * mirror * trans * scale * mirror * rot2 * rot1;

	(*item)->modelPos = pos;

	Drawable* g = createObject(index,tessellation, pos);
	g->setUseDisplayList(false);
	_artifacts[objCount]->drawable = g;
	sphereGeode->addDrawable(g);
	objCount++;
    }

    cerr << "done" << endl;

    StateSet * ss=sphereGeode->getOrCreateStateSet();
  
    ss->setMode(GL_BLEND, StateAttribute::ON);
    ss->setMode(GL_LIGHTING, StateAttribute::ON);
    ss->setRenderingHint( osg::StateSet::TRANSPARENT_BIN );

    /*Material * mat =new Material();	
    mat->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    mat->setDiffuse(Material::FRONT,osg::Vec4(1.0,1.0,1.0,1.0));*/
    ss->setAttribute(_defaultMaterial);
    
    osg::CullFace * cf=new osg::CullFace();
    cf->setMode(osg::CullFace::BACK);
  
    ss->setAttributeAndModes( cf, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);

    root_node->addChild(sphereGeode);
}

Drawable * ArtifactVis::createObject(int index, float tessellation, Vec3d & pos)
{
    //const double M_TO_MM = 1000.0f;
    //const double radius = 0.05f * M_TO_MM;
    vector<Vec4>::iterator dc_color = _descriptor_list_colors.begin();
    dc_color+=index;
    
    TessellationHints * hints = new TessellationHints();
    hints->setDetailRatio(tessellation);

    Sphere* sphereShape = new Sphere(pos, _sphereRadius); 
    ShapeDrawable * shapeDrawable = new ShapeDrawable(sphereShape);
    shapeDrawable->setTessellationHints(hints);
    shapeDrawable->setColor((*dc_color));//(*dc_color));
    return shapeDrawable;
}

void ArtifactVis::readSiteFile()
{
    const double INCH_IN_MM = 25.4f;

    std::string modelFileName = ConfigManager::getEntry("Plugin.ArtifactVis.TopoFile");
    cerr << "Reading site file: " << modelFileName << " ..." << endl;
    if (!modelFileName.empty()) 
    {
	_siteRoot = new osg::MatrixTransform();
	Node* modelFileNode = osgDB::readNodeFile(modelFileName);
	if (modelFileNode==NULL) cerr << "Error reading file" << endl;
	else
	{
	    Matrix scale;
	    scale.makeScale(INCH_IN_MM, INCH_IN_MM, INCH_IN_MM);
	    _siteRoot->setMatrix(scale);
	    _siteRoot->addChild(modelFileNode);

	    StateSet * ss=_siteRoot->getOrCreateStateSet();

	    osg::CullFace * cf=new osg::CullFace();
	    cf->setMode(osg::CullFace::BACK);

	    ss->setAttributeAndModes( cf, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);

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
	cerr << "Error: Plugin.ArtifactVis.Topo needs to point to a .wrl 3D topography file" << endl;
    }
}

void ArtifactVis::readLocusFile()
{
    std::string locusFile = ConfigManager::getEntry("value","Plugin.ArtifactVis.LociFile","");
    if(locusFile.empty())
    {
	std::cerr << "ArtifactVis: Warning: No Plugin.ArtifactVis.LociFile entry." << std::endl;
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

    mxml_node_t *node;
    int nonLocusCount = 0;
    for (node = mxmlFindElement(tree, tree, "name", NULL, NULL, MXML_DESCEND); node != NULL; node = mxmlFindElement(node, tree, "name", NULL, NULL, MXML_DESCEND))
    {
	char * text =  node->child->value.text.string;
	int num = atoi(text);
	if(num!=0)
	{
	    Locus * locus = new Locus;
	    locus->id = num;
	    _locusList.push_back(locus);
	}
	else nonLocusCount++;
	if(nonLocusCount==3)break;
    }

    mxml_node_t *nodeChild;
    int index = 0;
    for (node = mxmlFindElement(tree, tree, "coordinates", NULL, NULL, MXML_DESCEND); node != NULL; node = mxmlFindElement(node, tree, "coordinates", NULL, NULL, MXML_DESCEND))
    {
	if(index==_locusList.size())break;
	for(nodeChild = node->child; nodeChild != NULL; nodeChild=nodeChild->next)
	{
	    char * text = nodeChild->value.text.string;

	    std::string coordsText = text;
	    double x,y,z;
	    size_t pos;
	    pos = coordsText.find(",");

	    if(pos != std::string::npos)
	    {
		x = atof((coordsText.substr(0,pos)).c_str());
		coordsText = coordsText.substr(pos+1);
	    }

	    pos = coordsText.find(",");

	    if(pos != std::string::npos)
	    {
		y = atof((coordsText.substr(0,pos)).c_str());
		coordsText = coordsText.substr(pos+1);
	    }

	    z = atof((coordsText.substr(0,pos)).c_str());

	    _locusList[index]->coords.push_back(osg::Vec3d(x,y,z));
	}
	index++;
    }

    /*for(int i = 0; i <_locusList.size(); i++)
    {
	std::cerr << _locusList[i]->id << ": ";
	for(int j = 0; j < _locusList[i]->coords.size(); j++)
	{
	    std::cerr << _locusList[i]->coords[j].x() << "," << _locusList[i]->coords[j].y() << "," << _locusList[i]->coords[j].z() << endl;
	}
	std::cerr << endl;
    }*/

    std::cerr << "Loci Loaded." << std::endl;
}

void ArtifactVis::setupDCFilter()
{
    std::string dcFile = ConfigManager::getEntry("value","Plugin.ArtifactVis.DCInfoFile","");
    if(dcFile.empty())
    {
	std::cerr << "ArtifactVis: Warning: no entry for Plugin.ArtifactVis.DCInfoFile" << std::endl;
    }

    vvTokenizer::TokenType ttype;
    FILE* fp = fopen(dcFile.c_str(), "rb");
    if (fp==NULL)
    {
	cerr << "Cannot read file: " << dcFile << endl;
	return;
    }

    _dcFilterMenu = new SubMenu("Filter");
    _avMenu->addItem(_dcFilterMenu);

    _dcFilterAuto = new MenuCheckbox("Auto", false);
    _dcFilterAuto->setCallback(this);
    _dcFilterMenu->addItem(_dcFilterAuto);

    _dcFilterShowAll = new MenuButton("Show All");
    _dcFilterShowAll->setCallback(this);
    _dcFilterMenu->addItem(_dcFilterShowAll);

    _dcFilterShowNone = new MenuButton("Show None");
    _dcFilterShowNone->setCallback(this);
    _dcFilterMenu->addItem(_dcFilterShowNone);

    vvTokenizer* tokenizer = new vvTokenizer(fp);
    tokenizer->setEOLisSignificant(true);
    tokenizer->setCaseConversion(vvTokenizer::VV_UPPER);
    tokenizer->setParseNumbers(true);
    tokenizer->setAlphaCharacter(' ');
    tokenizer->setWhitespaceCharacter('\t');
    tokenizer->setCommentCharacter('#');
    while ((ttype = tokenizer->nextToken()) != vvTokenizer::VV_EOF)
    {
	string s(tokenizer->sval);
	if(s.find("-") ==0)
	{
	    MenuCheckbox * cb = new MenuCheckbox(s.substr(1),true);
	    cb->setCallback(this);
	    _dcFilterSubMenus.back()->addItem(cb);
	    _dcFilterItems.push_back(cb);
	    _dcList.push_back(s.substr(1));
	    _dcVisibleMap[s.substr(1)] = true;
	}
	else if (s.size() !=0)
	{
	    SubMenu* subMenu = new SubMenu(s);
	    _dcFilterMenu->addItem(subMenu);
	    _dcFilterSubMenus.push_back(subMenu);
	}
    }
}

void ArtifactVis::updateSelect()
{
    if(_selectActive)
    {
	osg::Vec3 markPos(0,1000,0);
	markPos = markPos * PluginHelper::getHandMat(_selectHand);
	osg::Matrix markTrans;
	markTrans.makeTranslate(markPos);
	_selectMarks[_selectHand]->setMatrix(markTrans);
    }
    else
    {
	for(int i = 0; i < PluginHelper::getNumHands(); i++)
	{
	    osg::Vec3 markPos(0,1000,0);
	    markPos = markPos * PluginHelper::getHandMat(i);
	    osg::Matrix markTrans;
	    markTrans.makeTranslate(markPos);
	    _selectMarks[i]->setMatrix(markTrans);
	}
    }

    if(_selectActive)
    {
	osg::Matrix l2w = getLocalToWorldMatrix(_sphereRoot.get());
	osg::Matrix w2l = osg::Matrix::inverse(l2w);
	_selectCurrent = osg::Vec3(0,1000,0);
	_selectCurrent = _selectCurrent * PluginHelper::getHandMat(_selectHand) * w2l;
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

	for(int i = 0; i < _artifacts.size(); i++)
	{
	    if(_artifacts[i]->visible && bb.contains(_artifacts[i]->modelPos) && !_artifacts[i]->selected)
	    {
		osg::ShapeDrawable * sd = dynamic_cast<osg::ShapeDrawable*>(_artifacts[i]->drawable);
		if(sd)
		{
		    osg::Vec4 color = sd->getColor();
		    color.x() = color.x() * 2.0;
		    color.y() = color.y() * 2.0;
		    color.z() = color.z() * 2.0;
		    sd->setColor(color);
		}
		_artifacts[i]->selected = true;
	    }
	    else if((!_artifacts[i]->visible || !bb.contains(_artifacts[i]->modelPos)) && _artifacts[i]->selected)
	    {
		osg::ShapeDrawable * sd = dynamic_cast<osg::ShapeDrawable*>(_artifacts[i]->drawable);
		if(sd)
		{
		    osg::Vec4 color = sd->getColor();
		    color.x() = color.x() * 0.5;
		    color.y() = color.y() * 0.5;
		    color.z() = color.z() * 0.5;
		    sd->setColor(color);
		}
		_artifacts[i]->selected = false;
	    }

	    if(_artifacts[i]->selected)
	    {
		dcCount[_artifacts[i]->dc]++;
		totalSelected++;
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

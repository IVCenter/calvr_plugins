#include "Network.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/ComController.h>
#include <cvrUtil/CVRSocket.h>
//#include <cvrKernel/SceneObject.h>
#include <cvrKernel/NodeMask.h>
#include <cvrUtil/ComputeBoundingBoxVisitor.h>

#include <fcntl.h>
#include <iostream>
#include <cstring>
#include <sstream>

#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/CullFace>
#include <osg/Material>
#include <osg/Group>
#include <osg/LineWidth>
#include <osg/BoundingBox>
#include <osgDB/ReadFile>
#include <osgDB/FileNameUtils>
#include <osgUtil/Optimizer>

#include "NetworkWallSceneObject.h"

#include <json/json.h>

using namespace osg;
using namespace std;
using namespace cvr;

CVRPLUGIN( Network )

TextSizeVisitor::TextSizeVisitor(float fontsize, bool read) : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
{
    _fontsize = fontsize;
    _read = read;
}

void TextSizeVisitor::apply(osg::Geode& geode)
{
    for(int i = 0; i < geode.getNumDrawables(); i++)
    {
	osgText::Text* text = dynamic_cast<osgText::Text*> (geode.getDrawable(i));
        if( text )
        {
	   if( _read )
	   {
	      _fontsize = text->getCharacterHeight();
	   }
	   else
	   { 
	      // set font size
	      text->setCharacterSize(_fontsize);
	   }
	}
    }
}

// load in vertex mapping file to allow for graph adjustments (when removing graph delete current mapping look up)
// update to load OTU mapping as well TODO
float Network::loadJsonVertexMappingFile(std::string fileName, std::map<std::string, VertexMap> & vertexSampleLookup, std::map<std::string, VertexMap> & vertexObservLookup)
{
    // read in vert mapping file       
    ifstream ifs(fileName.c_str());
    
    Json::Reader reader;
    Json::Value obj;
    reader.parse(ifs, obj);
    ifs.close();

    Json::Value sampleList = obj["Samples"];

    // get list of mapping elements           
    for(int i = 0; i < sampleList.size(); i++)
    {
    
	int sampleIndex = obj["Samples"][i]["index"].asInt();
	std::string sampleName = obj["Samples"][i]["sampleName"].asString();
		
	// add mapping
	vertexSampleLookup[sampleName].pointIndex = sampleIndex;

	Json::Value links = obj["Samples"][i]["links"];
	for( int j = 0; j < links.size(); j++)
	{
	     LineVertices lv;
	     lv.sInd = obj["Samples"][i]["links"][j]["sampIndex"].asInt();
	     lv.oInd = obj["Samples"][i]["links"][j]["otuIndex"].asInt();
	     vertexSampleLookup[sampleName].lineIndexs.push_back(lv);
	}
    }
    
    Json::Value observList = obj["Otus"];

    // get list of mapping elements           
    for(int i = 0; i < observList.size(); i++)
    {
    
	int otuIndex = obj["Otus"][i]["index"].asInt();
	std::string otuName = obj["Otus"][i]["otuName"].asString();
		
	// add mapping
	vertexObservLookup[otuName].pointIndex = otuIndex;

	Json::Value links = obj["Otus"][i]["links"];
	for( int j = 0; j < links.size(); j++)
	{
	     LineVertices lv;
	     lv.sInd = obj["Otus"][i]["links"][j]["sampIndex"].asInt();
	     lv.oInd = obj["Otus"][i]["links"][j]["otuIndex"].asInt();
	     vertexObservLookup[otuName].lineIndexs.push_back(lv);
	}
    }


    // need to get list of OTU mapping data
    return obj["HighestWeight"].asFloat();
}

// returns a visual key
void Network::applyMapping(osg::Geode* sampleNode, osg::Geode* edgeNode,
                        ColorMap & colorMapping, // property to color
			std::map<std::string, string> & sampleMapping, // sample to property
			std::map<std::string, VertexMap > & vertexMapping) // vertex mapping happens when graph is loaded
{
    // temp vector so key can be created
    std::vector< std::pair< std::string, osg::Vec3 > > keydata;

    // update uniforms    
    std::map<std::string, int> quickLookup;
   
    // currently just support 12 colors for mapping (can change at a later date)
    for( int i = 0; i < colorMapping.mapping.size(); i++)
    {
	sampleNode->getStateSet()->getUniform("colorTable")->setElement(i, osg::Vec3f(colorMapping.mapping[i].color));
	edgeNode->getStateSet()->getUniform("colorTable")->setElement(i, osg::Vec3f(colorMapping.mapping[i].color));

        // to make setting actual indexes fast
        quickLookup[colorMapping.mapping.at(i).property] = i;

	// add color to keydata
	keydata.push_back(std::pair<std::string, osg::Vec3 > (colorMapping.mapping.at(i).property, colorMapping.mapping.at(i).color));
    }

    // if no default key set use set it as -1 in quick lookup
    if (quickLookup.find("Default") == quickLookup.end() )
    {
	quickLookup["Default"] = -1;
    }

    std::cerr << "Default is: " << quickLookup["Default"] << std::endl;
    int defaultColorRef = quickLookup["Default"];


    // set the rest of the colors (this assumes there is a default value)
    for( int i = colorMapping.mapping.size(); i < 12; i++ )
    {
	// set default color to rest of look up table
	//sampleNode->getStateSet()->getUniform("colorTable")->setElement(i, keydata.at(quickLookup["Default"]).second);
	//edgeNode->getStateSet()->getUniform("colorTable")->setElement(i, keydata.at(quickLookup["Default"]).second);
	sampleNode->getStateSet()->getUniform("colorTable")->setElement(i, osg::Vec3(1.0, 1.0, 1.0));
	edgeNode->getStateSet()->getUniform("colorTable")->setElement(i, osg::Vec3(1.0,1.0,1.0));
    }

    // vector that colds vertex mapping
    std::vector< IndexValue >  * sampleUpdates = new std::vector< IndexValue >();
    std::vector< IndexValue > * lineUpdates = new std::vector< IndexValue >();

    // add mappings
    IndexValue iValue;
    //int colorInt = 11;
    int colorInt = defaultColorRef;

    //TODO check if there is a Default color in map, if not set colorInt to -1 so it wont render

    // TODO faster to do in reverse and iterate though vertex mapping and then check sample makking for value
    // less values to change
    
    // loop through all samples and look up mapping value
    std::map<std::string, VertexMap >::iterator it =  vertexMapping.begin();
    for(; it != vertexMapping.end(); ++it)
    {
	std::string sampName = it->first;
	    
	VertexMap vert = vertexMapping[sampName];

        if( sampleMapping.find(sampName) != sampleMapping.end() )	
	{
	    std::string optionName = sampleMapping[sampName];

	    colorInt = defaultColorRef;
	    if( quickLookup.find(optionName) != quickLookup.end() )
		colorInt = quickLookup[optionName];

	    //std::cerr << "Looking for option: " << optionName << " color returned is: " << colorInt << std::endl;
	
	    iValue.index = vert.pointIndex;
	    iValue.value = colorInt; // default to last value in lookup
	    sampleUpdates->push_back(iValue); 
	
	    for(int j = 0; j < vert.lineIndexs.size(); j++)
	    {
		    iValue.index = vert.lineIndexs.at(j).oInd;
		    iValue.value = colorInt;
		    lineUpdates->push_back(iValue);
	    }
	
	}
	//else
	//{
	//    std::cerr << "Error: no mapping meta data found for sample: " << sampName << std::endl;   
	//}	
    } 

/*
    // loop through all samples and look up mapping value
    std::map<std::string, std::string>::iterator it = sampleMapping.begin();
    for(; it != sampleMapping.end(); ++it)
    {
	std::string sampName = it->first;
	std::string optionName = it->second;

        if( vertexMapping.find(sampName) != vertexMapping.end() )	
	{

	    VertexMap vert = vertexMapping[sampName];

	    colorInt = 11;
	    if( quickLookup.find(optionName) != quickLookup.end() )
		colorInt = quickLookup[optionName];
	    //else
	    //	std::cerr << "No option found for " << optionName << std::endl;
	
	    iValue.index = vert.pointIndex;
	    iValue.value = colorInt; // default to last value in lookup
	    sampleUpdates->push_back(iValue); 
	
	    // update line updates
	    //for(int j = 0; j < vert.lineIndexs->size(); j++)
	    //{
		//iValue.index = vert.lineIndexs->at(j).oInd;
	    for(int j = 0; j < vert.lineIndexs.size(); j++)
	    {
		iValue.index = vert.lineIndexs.at(j).oInd;
		iValue.value = colorInt;
		lineUpdates->push_back(iValue);
	    }
	
	}	
    } 
*/
    //std::cerr << "update sizes " << lineUpdates->size() << " " << sampleUpdates->size() << std::endl;
   
    // TODO create outside and attache to drawables and pass into function
    //create DrawableUpdateCallback and apply
    DrawableUpdateCallback * sampCall = new DrawableUpdateCallback();
    DrawableUpdateCallback * edgeCall = new DrawableUpdateCallback();
    sampleNode->getDrawable(0)->setUpdateCallback(sampCall);
    edgeNode->getDrawable(0)->setUpdateCallback(edgeCall);
    sampCall->applyUpdate(sampleUpdates);
    edgeCall->applyUpdate(lineUpdates);
    
    // update key 
    if ( _root != NULL )
	_root->addChild(createVisualKey(keydata)); 
}

osg::Geode* Network::createVisualKey(std::vector<std::pair<std::string, osg::Vec3> > & elements)
{
    // TODO put title on key
    osg::ref_ptr<osgText::Font> font = osgText::readRefFontFile("/home/calvr/CalVR/resources/arial.ttf");
    osg::Vec4 layoutColor(0.5f,0.5f,0.5f,1.0f);
    float fontSize = 40.0;
    float xOffset = 0.0; 
    float gap = fontSize * 0.1;
    float distVert = fontSize + gap;
    
    osg::Geode* geode = new osg::Geode();
    osg::Geometry* geom = new osg::Geometry();
    osg::Vec3Array* vertices = new osg::Vec3Array();
    osg::Vec4Array* colors = new osg::Vec4Array();

    for(int i = 0; i < elements.size(); i++)
    {
	// create text
        osgText::Text* text = new osgText::Text;
        text->setUseVertexBufferObjects(true);
        text->setFont(font);
        text->setColor(layoutColor);
        text->setCharacterSize(fontSize);
        text->setPosition(osg::Vec3(0.0, -1.0, i * -distVert));
        text->setAxisAlignment(osgText::Text::XZ_PLANE);
        text->setAlignment(osgText::Text::LEFT_CENTER); // was BOTTOM
        text->setFontResolution(40,40);
        text->setText(elements.at(i).first.c_str());    	
        geode->addDrawable(text);
    }

    osg::BoundingBox bounds = geode->getBoundingBox();
    xOffset = bounds.xMax();

    float zMin = bounds.zMin() - gap;
    float zMax = bounds.zMax();
   
    // add an initial black quad to act as a background for the key
    vertices->push_back(osg::Vec3(0.0, 0.0, zMin));
    vertices->push_back(osg::Vec3(0.0, 0.0, zMax));
    vertices->push_back(osg::Vec3(bounds.xMax(), 0.0, zMax));
    vertices->push_back(osg::Vec3(bounds.xMax(), 0.0, zMin));
    colors->push_back(osg::Vec4(0, 0, 0, 1));
    colors->push_back(osg::Vec4(0, 0, 0, 1));
    colors->push_back(osg::Vec4(0, 0, 0, 1));
    colors->push_back(osg::Vec4(0, 0, 0, 1));
    
    // update vertices
    for(int i = 0; i < elements.size(); i++)
    {
	// create colored quad
	vertices->push_back(osg::Vec3(xOffset, 0.0,  zMax - fontSize - (i * (gap + fontSize))));
	vertices->push_back(osg::Vec3(xOffset, 0.0, zMax - fontSize - (i * (gap + fontSize) - fontSize)));
	vertices->push_back(osg::Vec3(xOffset + fontSize, 0.0, zMax - fontSize - (i * (gap + fontSize) - fontSize)));
	vertices->push_back(osg::Vec3(xOffset + fontSize , 0.0, zMax - fontSize - (i * (gap + fontSize))));
	
	colors->push_back(osg::Vec4(elements.at(i).second[0], elements.at(i).second[1], elements.at(i).second[2], 1.0));
	colors->push_back(osg::Vec4(elements.at(i).second[0], elements.at(i).second[1], elements.at(i).second[2], 1.0));
	colors->push_back(osg::Vec4(elements.at(i).second[0], elements.at(i).second[1], elements.at(i).second[2], 1.0));
	colors->push_back(osg::Vec4(elements.at(i).second[0], elements.at(i).second[1], elements.at(i).second[2], 1.0));
    }    
    
    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,vertices->size()));
    geom->setVertexArray(vertices);
    geom->setColorArray(colors, osg::Array::BIND_PER_VERTEX);

    geode->addDrawable(geom);

    return geode;
}


Network::Network() 
{
}

// load network also needs to adjust and rebuild the menu options correctly
void Network::loadNetwork(std::string file, float highestWeight)
{
    // check if file exists and can be loaded
    osg::Node* node = osgDB::readNodeFile(file.c_str());    
    if( !node )
        return;

    // remove existing scene object
    if( _so )
    {
        delete _so;
    }
    _so = NULL;

    if( _root )
    {
	while( _root->getNumChildren() )
	    _root->removeChild(0,1);
    }
    
    // load in the new network
    _so = new NetworkWallSceneObject(std::string("Network"),false,true,false,true,false);

    // need to check how registration works
    PluginHelper::registerSceneObject(_so,"Network");
    _so->attachToScene();
    _so->addMoveMenuItem();
    _so->addNavigationMenuItem();
    _so->setBoundsCalcMode(cvr::SceneObject::MANUAL);
    _so->setTiledWallMovement(true);
    
    // compute bound ( use info to set a default size and also set the center correctly for scaling )
    ComputeBoundingBoxVisitor cbbv;
    node->accept(cbbv);
    osg::BoundingBox bb = cbbv.getBound();

    // added fix for default scaling if the graph if large in the y axis more so that the x axis
    float scale = ConfigManager::getFloat("value", "Plugin.NetworkNew.MaxInitialSize",12000.0) / (bb.xMax() - bb.xMin());
    float yscale = ConfigManager::getFloat("value", "Plugin.NetworkNew.MaxInitialSize",12000.0) / (bb.zMax() - bb.zMin());
    if( yscale < scale )
	scale = yscale;

    // set matrix
    osg::Matrix mat;
    mat.makeScale(osg::Vec3(scale, 1.0,scale));
    mat.setTrans(osg::Vec3( (-bb.xMin() - ((bb.xMax() - bb.xMin()) * 0.5)) * scale, 0.0, (-bb.zMin() - ((bb.zMax() - bb.zMin()) * 0.5)) * scale)); 
    osg::MatrixTransform* matt = new osg::MatrixTransform();
    matt->setMatrix(mat);

    // add node and then attach to sceneobject
    matt->addChild(node);
    
    ComputeBoundingBoxVisitor cbbvv;
    matt->accept(cbbvv);
    osg::BoundingBox newBound = cbbvv.getBound();
    newBound.yMin() = 0.0;
    newBound.yMax() = 0.0;
   
    _so->addChild(matt);
    _so->setBoundingBox(newBound);

    // dangerous but want to read values (clean up later)
    osg::Group* data = _so->getChildNode(0)->asGroup()->getChild(0)->asGroup();

    // read in text height
    TextSizeVisitor textHeight(1.0, true);
    
    data->getChild(3)->accept(textHeight);
    float sampleTextHeight = textHeight.getFontSize();
    
    data->getChild(4)->accept(textHeight);
    float elementTextHeight = textHeight.getFontSize();

    // reset menu items (scale) remove old elements and replace with new initialized ones
    _sampleText->setValue(true);
    _elementText->setValue(true);
    

    int position = 0;
    if( _edgeWeight )
    {
	position = _subMenu->getItemPosition(_edgeWeight);
	_subMenu->removeItem(_edgeWeight);
    }
    _edgeWeight = new MenuRangeValue("EdgeWeight", 0.0, highestWeight, 0.0);
    _edgeWeight->setCallback(this);
    _subMenu->addItem(_edgeWeight, position);
  
    // set back to default 
    _edgeWidth->setValue(2.0);
    
    _colorEdges->setValue(false);
    _sampleText->setValue(true); 
    _elementText->setValue(true); 

/*
    if( _otuEdges )
    {
	position = _subMenu->getItemPosition(_otuEdges);
	_subMenu->removeItem(_otuEdges);
    }
    _otuEdges = new MenuRangeValue("NumOTUEdges", 0.0, numSamples, 0.0);
    _otuEdges->setCallback(this);
    _subMenu->addItem(_otuEdges, position);
    
    if( _sampleEdges )
    {
	position = _subMenu->getItemPosition(_sampleEdges);
	_subMenu->removeItem(_sampleEdges);
    }
    _sampleEdges = new MenuRangeValue("NumSampleEdges", 0.0, numOtus, 0.0);
    _sampleEdges->setCallback(this);
    _subMenu->addItem(_sampleEdges, position);
*/    
    if( _sampleTextSize )
    {
	position = _subMenu->getItemPosition(_sampleTextSize);
	_subMenu->removeItem(_sampleTextSize);
    }
    _sampleTextSize = new MenuRangeValue("SampleTextSize", sampleTextHeight, sampleTextHeight * 10.0, sampleTextHeight);
    _sampleTextSize->setCallback(this);
    _subMenu->addItem(_sampleTextSize, position);
    
    if( _elementTextSize )
    {
	position = _subMenu->getItemPosition(_elementTextSize);
	_subMenu->removeItem(_elementTextSize);
    }
    _elementTextSize = new MenuRangeValue("ElementTextSize", elementTextHeight, elementTextHeight * 10.0, elementTextHeight);
    _elementTextSize->setCallback(this);
    _subMenu->addItem(_elementTextSize, position);
   
    // read uniform for default size value 
    if( _samplePointSize )
    {
	position = _subMenu->getItemPosition(_samplePointSize);
	_subMenu->removeItem(_samplePointSize);
    }

    float pointsizes = 0.0;
    data = _so->getChildNode(0)->asGroup()->getChild(0)->asGroup();
    data->getChild(0)->getStateSet()->getUniform("point_size")->get(pointsizes);
    _samplePointSize = new MenuRangeValue("SamplePointSize", pointsizes * 0.1, pointsizes * 10.0, pointsizes);
    _samplePointSize->setCallback(this);
    _subMenu->addItem(_samplePointSize, position);
    
    if( _elementPointSize )
    {
	position = _subMenu->getItemPosition(_elementPointSize);
	_subMenu->removeItem(_elementPointSize);
    }

    data = _so->getChildNode(0)->asGroup()->getChild(0)->asGroup();
    data->getChild(2)->getStateSet()->getUniform("point_size")->get(pointsizes);
    _elementPointSize = new MenuRangeValue("ElementPointSize", pointsizes * 0.1, pointsizes * 10.0, pointsizes);
    _elementPointSize->setCallback(this);
    _subMenu->addItem(_elementPointSize, position);

}

// load color mapping file and build menu
void Network::loadColorMappingFile(std::string vertMapFileName, std::map< std::string, Graph > & vertColorMapping, 
				   cvr::SubMenu* menu, std::map< cvr::MenuItem* , std::pair< std::string, std::string> > & buttonMapping)
{
   // read in vert mapping file       
   ifstream ifs(vertMapFileName.c_str());
   
   Json::Reader reader;
   Json::Value obj;
   reader.parse(ifs, obj);
   ifs.close();
   
   //osg::Vec3 defaultColor(1.0, 1.0, 1.0); 
   
   // get list of mapping elements           
   for(int i = 0; i < obj.size(); i++)
   {
	// get types of graphs and parameters
	std::string name = obj[i]["Graph"].asString();
	std::string graphFileName = obj[i]["File"].asString();
	std::string vertexFileName = obj[i]["Vertexmap"].asString();
	std::string metaFileName = obj[i]["Metadata"].asString();
    
	// set information
	vertColorMapping[name].graphFileName = graphFileName;
	vertColorMapping[name].vertexFileName = vertexFileName;
	vertColorMapping[name].metaFileName = metaFileName;

	// create submenu
	cvr::SubMenu* graphMenu = new cvr::SubMenu(name, name);
	menu->addItem(graphMenu);

	// loop through mappings for this graph
	Json::Value mappings = obj[i]["Mappings"];
	for(int j = 0; j < mappings.size(); j++ )
	{

	    // TODO need to support sample coloring and observation coloring

	    std::string mapName = obj[i]["Mappings"][j]["MapName"].asString();
	    //vertColorMapping[mapName].metaName = obj[i]["Mappings"][j]["MetaProperty"].asString(); 
	    vertColorMapping[name].mappings[mapName].metaName = obj[i]["Mappings"][j]["MetaProperty"].asString(); 
	    vertColorMapping[name].mappings[mapName].type = obj[i]["Mappings"][j]["Type"].asString();

	    // add submenu buttons and attach callbacks
	    cvr::MenuButton* button = new cvr::MenuButton(mapName);
	    graphMenu->addItem(button);
	    button->setCallback(this);

	    // update button mapping lookup
	    buttonMapping[button] = std::pair<std::string, std::string> (name, mapName);

	    //std::cerr << "Name: " << name << std::endl;
	    //std::cerr << "Map Name: " << mapName << std::endl;

	    bool defaultColor = false;

	    Json::Value cm = obj[i]["Mappings"][j]["Properties"];
	    for( int k = 0; k < cm.size(); k++)
	    {
		//std::cerr << "Property: " << obj[i]["Mappings"][j]["Properties"][k]["Property"].asString();

		PropertyMap prop;
		prop.property = obj[i]["Mappings"][j]["Properties"][k]["Property"].asString();
		prop.color.set( obj[i]["Mappings"][j]["Properties"][k]["Color"][0].asFloat() / 255.0,
		                obj[i]["Mappings"][j]["Properties"][k]["Color"][1].asFloat() / 255.0,
				obj[i]["Mappings"][j]["Properties"][k]["Color"][2].asFloat() / 255.0);

		vertColorMapping[name].mappings[mapName].mapping.push_back(prop);
   
		// check for default color 
		if( vertColorMapping[name].mappings[mapName].mapping[k].property.compare("Default") == 0)
		{
		    defaultColor = true;
		}
	    }

	    // if no default color add one
	    /*
	    if ( ! defaultColor )
	    {
	    	PropertyMap prop;
		prop.property = "Default";
		prop.color.set(1.0, 1.0, 1.0);
		vertColorMapping[name].mappings[mapName].mapping.push_back(prop);
	    }
	    */
	    // set the rest of the colors using the default color
	    //for(int k = cm.size(); k < 12; k++)
	    //{
	    //	PropertyMap prop;
	    //
	    //	vertColorMapping[name].mappings[mapName].mapping[k].color.set(defaultColor);
	    //}
	}
    }
}

bool Network::init() 
{
    std::cerr << "Network init\n";
   
    _root = NULL;
    _so = NULL;

    // create sub menu to hold items
    _subMenu = new SubMenu("Network", "Network");
    MenuSystem::instance()->addMenuItem(_subMenu);

    _loadMenu = new SubMenu("Load","Load");
    _loadMenu->setCallback(this);
    _subMenu->addItem(_loadMenu);
  
    /// find position for the key
    _root = new osg::PositionAttitudeTransform();
  
    // TODO should adjust the defaults 
    float x = ConfigManager::getFloat("x", "Plugin.Network.KeyPosition",2400.0); 
    float y = ConfigManager::getFloat("y", "Plugin.Network.KeyPosition",-10.0); 
    float z = ConfigManager::getFloat("z", "Plugin.Network.KeyPosition",-800.0);
    _root->setPosition(osg::Vec3(x, y, z)); 
     
    // attach _root to the scene TODO 
    PluginHelper::getScene()->addChild(_root);
     
    std::vector<std::string> list;
 
    // set data directory TODO this needs to be moved to configuration file 
    //_dataDirectory = "/home/calvr/Philip/NetworkNew/data/";
    _dataDirectory = ConfigManager::getEntry("Plugin.Network.DataDir");
   
    // menu configuration
    loadColorMappingFile( _dataDirectory + "network_config.json", _vertColorMapping, _loadMenu, _buttonMapping); 
   
    _colorEdges = new MenuCheckbox("ColorEdgesToMeta", false);
    _colorEdges->setCallback(this);
    _subMenu->addItem(_colorEdges);
    
    _sampleText = new MenuCheckbox("SampleText", true);
    _sampleText->setCallback(this);
    _subMenu->addItem(_sampleText);
    
    _elementText = new MenuCheckbox("OTUText", true);
    _elementText->setCallback(this);
    _subMenu->addItem(_elementText);

    // add other menu options
    //_edgeWeight = new MenuRangeValue("EdgeWeight", 0.0, 2788.0, 0.0);
    
    //_edgeWeight = new MenuRangeValue("EdgeWeight", 0.0, 1.0, 0.0);
    _edgeWeight = new MenuRangeValue("EdgeWeight", 0.0, 0.0, 0.0);
    _edgeWeight->setCallback(this);
    //_so->addMenuItem(_edgeWeight);
    _subMenu->addItem(_edgeWeight);
    
    _edgeWidth = new MenuRangeValue("EdgeWidth", 2.0, 10.0, 2.0);
    _edgeWidth->setCallback(this);
    //_so->addMenuItem(_edgeWeight);
    _subMenu->addItem(_edgeWidth);
   
   
    // max this can be is the total number of samples 
    //_otuEdges = new MenuRangeValue("NumOTUEdges", 0.0, 500.0, 0.0);
   
/* 
    //_otuEdges = new MenuRangeValue("NumOTUEdges", 0.0, 60.0, 0.0);
    _otuEdges = new MenuRangeValue("NumOTUEdges", 0.0, 0.0, 0.0);
    _otuEdges->setCallback(this);
    //_so->addMenuItem(_otuEdges);
    _subMenu->addItem(_otuEdges);

    //_otuTotalWeight = new MenuRangeValue("TotalOTUWeight", 0.0, 1000.0, 0.0);
    //_otuTotalWeight->setCallback(this);
    //_so->addMenuItem(_otuTotalWeight);
    
    //_sampleEdges = new MenuRangeValue("NumSampleEdges", 0.0, 2788.0, 0.0);
    
    //_sampleEdges = new MenuRangeValue("NumSampleEdges", 0.0, 7387.0, 0.0);
    _sampleEdges = new MenuRangeValue("NumSampleEdges", 0.0, 0.0, 0.0);
    _sampleEdges->setCallback(this);
    //_so->addMenuItem(_sampleEdges);
    _subMenu->addItem(_sampleEdges);
*/    
    _sampleTextSize = new MenuRangeValue("SampleTextSize", 0.0, 0.0, 0.0);
    _sampleTextSize->setCallback(this);
    _subMenu->addItem(_sampleTextSize);
    
    _elementTextSize = new MenuRangeValue("ElementTextSize", 0.0, 0.0, 0.0);
    _elementTextSize->setCallback(this);
    _subMenu->addItem(_elementTextSize);
    
    _samplePointSize = new MenuRangeValue("SamplePointSize", 0.0, 0.0, 0.0);
    _samplePointSize->setCallback(this);
    _subMenu->addItem(_samplePointSize);
    
    _elementPointSize = new MenuRangeValue("ElementPointSize", 0.0, 0.0, 0.0);
    _elementPointSize->setCallback(this);
    _subMenu->addItem(_elementPointSize);
        
        
    _remove = new MenuButton("Delete");
    _remove->setCallback(this);
    _subMenu->addItem(_remove);
    
    //_sampleTotalWeight = new MenuRangeValue("TotalSampleWeight", 0.0, 1000.0, 0.0); 
    //_sampleTotalWeight->setCallback(this);
    //_so->addMenuItem(_sampleTotalWeight);

    return true;
}

void Network::preFrame()
{
}

void Network::menuCallback(cvr::MenuItem* menuItem)
{
    if( menuItem == _sampleText )
    {
	if( _so )
	{
	    osg::Group* data = _so->getChildNode(0)->asGroup()->getChild(0)->asGroup();
	    data->getChild(3)->setNodeMask((_sampleText->getValue())?~0:0);
	}
    }
    else if( menuItem == _elementText )
    {
	if( _so )
	{
	    osg::Group* data = _so->getChildNode(0)->asGroup()->getChild(0)->asGroup();
	    data->getChild(4)->setNodeMask((_elementText->getValue())?~0:0);
	}
    }
    else if( menuItem == _colorEdges )
    {
	if( _so )
	{
	    osg::Group* data = _so->getChildNode(0)->asGroup()->getChild(0)->asGroup();
	    osg::Uniform* currentUniform = NULL;
	    currentUniform = data->getChild(1)->getStateSet()->getUniform("colorEdgesToSample");
	    if( currentUniform)
		currentUniform->set((bool)_colorEdges->getValue());
	}
    }
    else if( menuItem == _edgeWidth )
    {
        if( _so )
        {
            osg::Group* data = _so->getChildNode(0)->asGroup()->getChild(0)->asGroup();
            osg::StateSet* state = data->getChild(1)->getStateSet();
	    osg::LineWidth* lw = dynamic_cast<osg::LineWidth*>(state->getAttribute(osg::StateAttribute::LINEWIDTH));
	    if ( lw )
	    {
		lw->setWidth(int(_edgeWidth->getValue()));	
	    }
        }
    }
    // first geode samples, second geode connections, third geode otus
    else if( menuItem == _edgeWeight )
    {
        if( _so )
        {
            osg::Group* data = _so->getChildNode(0)->asGroup()->getChild(0)->asGroup();
	    osg::Uniform* currentUniform = NULL;
            currentUniform = data->getChild(1)->getStateSet()->getUniform("minWeight");
	    if( currentUniform)
		currentUniform->set((float)_edgeWeight->getValue());
            currentUniform = data->getChild(2)->getStateSet()->getUniform("minWeight");
	    if( currentUniform)
		currentUniform->set((float)_edgeWeight->getValue());
	    currentUniform = data->getChild(4)->getStateSet()->getUniform("minWeight");
	    if( currentUniform)
		currentUniform->set((float)_edgeWeight->getValue());
	    currentUniform = data->getChild(3)->getStateSet()->getUniform("minWeight");
	    if( currentUniform)
		currentUniform->set((float)_edgeWeight->getValue());
        }
    }
    /*
    else if( menuItem == _otuEdges )
    {   if( _so )
        {
            osg::Group* data = _so->getChildNode(0)->asGroup()->getChild(0)->asGroup();
            data->getChild(2)->getStateSet()->getUniform("minEdges")->set((int)_otuEdges->getValue());
        }
    }
    else if( menuItem == _sampleEdges )
    {   if( _so )
        {
            osg::Group* data = _so->getChildNode(0)->asGroup()->getChild(0)->asGroup();
            data->getChild(0)->getStateSet()->getUniform("minEdges")->set((int)_sampleEdges->getValue());
        }
    }
    */
    else if( menuItem == _sampleTextSize )
    {   if( _so )
        {
            osg::Group* data = _so->getChildNode(0)->asGroup()->getChild(0)->asGroup();
	    TextSizeVisitor updateSize(_sampleTextSize->getValue(), false);
            data->getChild(3)->accept(updateSize);
        }
    }
    else if( menuItem == _elementTextSize )
    {   if( _so )
        {
            osg::Group* data = _so->getChildNode(0)->asGroup()->getChild(0)->asGroup();
	    TextSizeVisitor updateSize(_elementTextSize->getValue(), false);
            data->getChild(4)->accept(updateSize);
        }
    }
    else if( menuItem == _elementPointSize )
    {   
	if( _so )
        {
            osg::Group* data = _so->getChildNode(0)->asGroup()->getChild(0)->asGroup();
            data->getChild(2)->getStateSet()->getUniform("point_size")->set(_elementPointSize->getValue());
        }
    }
    else if( menuItem == _samplePointSize )
    {   if( _so )
        {
            osg::Group* data = _so->getChildNode(0)->asGroup()->getChild(0)->asGroup();
            data->getChild(0)->getStateSet()->getUniform("point_size")->set(_samplePointSize->getValue());
        }
    }
    else if( menuItem == _remove )
    {
        if(_so)
            delete _so;
        _so = NULL;

	if( _root != NULL)
	{
	    while(_root->getNumChildren() )
		_root->removeChild(0,1);

	}

	// reset graph name
	_currentLoadedGraph = "";
    }
    else
    {
	// found button load data
	if( _buttonMapping.find(menuItem) != _buttonMapping.end() )
	{
	    std::string graphName = _buttonMapping[menuItem].first;
	    std::string mappingName = _buttonMapping[menuItem].second;

	    // different graph need to load file in
	    if( _currentLoadedGraph.compare(graphName) != 0 )
	    {
		std::cerr << "Loading vertex map\n";

		std::string vertexFileName = _vertColorMapping[graphName].vertexFileName;
		std::string graphFileName = _vertColorMapping[graphName].graphFileName;

		// clear graph sample mapping
		_vertexSampleLookup.clear();
		_vertexObserLookup.clear();

		// load in vertex mapping data
		float highestWeight = loadJsonVertexMappingFile( _dataDirectory + vertexFileName, _vertexSampleLookup, _vertexObserLookup);
	
		std::cerr << "Finished Loading vertex map\n";

		// use graph name to look up file name to load TODO ignore highestWeight (because data is always normalized)
		loadNetwork( _dataDirectory + graphFileName, highestWeight);

		std::cerr << "Finished loading network\n";
		
		// set graph name	
		_currentLoadedGraph = graphName;
	    }

	    // apply color mapping and generate key
	    if ( _root != NULL )
	    {
		while(_root->getNumChildren() )
		    _root->removeChild(0,1);
	    }


	    //std::cerr << "Loading specific metadata\n";

	    // add more mapping data for the key to have colors for different sections


	    // Need to check add info is mapping is observation or sample
	
	    // TODO adjust so it supports edge data
	    // read in mapping specific to parameter 
	    std::map<std::string, std::string> generalMapping;
	    std::map<std::string, int> frequency; // can be used to order the layout (highest frequency at top)

	    // check type of graph
	    if( _vertColorMapping[graphName].mappings[mappingName].type.compare("sample") == 0 )
	    {
		std::cerr << "Using sample mapping\n";

		loadSpecificMetaData("#SampleID", _vertColorMapping[graphName].mappings[mappingName].metaName, 
				 _dataDirectory + _vertColorMapping[graphName].metaFileName, frequency, generalMapping);
	    
		//std::cerr << "Loading mapping name: " << mappingName << " for: " << graphName << std::endl;
	   
		osg::Group* data = _so->getChildNode(0)->asGroup()->getChild(0)->asGroup();
	     
		// load 
		applyMapping(data->getChild(0)->asGeode(), data->getChild(1)->asGeode(),
	                           _vertColorMapping[graphName].mappings[mappingName], // property to color
			           generalMapping, // sample to property (compiled per load from meta file)
			           _vertexSampleLookup); // name to vertex look up _rootcolorMapping.mapping.at(i) 
	    }
	    else if( _vertColorMapping[graphName].mappings[mappingName].type.compare("observation") == 0 )
		 // observation
	    {
		std::cerr << "Using obervation mapping\n";
		
		loadSpecificMetaData("#Observation", _vertColorMapping[graphName].mappings[mappingName].metaName, 
				 _dataDirectory + _vertColorMapping[graphName].metaFileName, frequency, generalMapping);
	    
		//std::cerr << "Loading mapping name: " << mappingName << " for: " << graphName << std::endl;
	  
		osg::Group* data = _so->getChildNode(0)->asGroup()->getChild(0)->asGroup();
	     
		// load 
		applyMapping(data->getChild(2)->asGeode(), data->getChild(1)->asGeode(),
	                           _vertColorMapping[graphName].mappings[mappingName], // property to color
			           generalMapping, // sample to property (compiled per load from meta file)
			           _vertexObserLookup); // name to vertex look up _rootcolorMapping.mapping.at(i) 
	    }
	
	    // TODO maybe create menu here, also attach menu as a scene object so it can be moved
	
	    
	    // hack print fequency
	    /*
	    std::map<std::string, int>::iterator it = frequency.begin();
	    for(; it != frequency.end(); ++it)
	    {
	        std::cerr << it->first << ": " << it->second <<std::endl;
	    }
	    */
	}
    }
}

// currently this just loads sample specific metadata
// need to make it generic (pass in key e.g. #SampleID, otuID etc)
void Network::loadSpecificMetaData(std::string key, std::string metaHeader, std::string metaDataFile, std::map<std::string, int> & types, 
				   std::map<std::string, string > & sampleMapping)
{
    // TODO check timing here ( if reading the file takes long, make it so it says in memory until different meta data is needed)
    std::cerr << "Reading meta file\n";

    // open json meta
    ifstream ifs(metaDataFile.c_str());
    
    Json::Reader reader;
    Json::Value obj;
    reader.parse(ifs, obj);
    ifs.close();

    std::cerr << "Finished reading file\n";
 
    // this is used to determine frequency 
    for(int i = 0; i < obj.size(); i++)
    {
	//std::string name = obj[i]["#SampleID"].asString();
	std::string name = obj[i][key].asString();
	std::string data = obj[i][metaHeader].asString();
	
	if( types.find(data) != types.end() )
	    types[data] += 1;
        else
            types[data] = 1;
	    
	// use map to remove repeats
	sampleMapping[name] = data;		
    }

    std::cerr << "Finished creating map\n";
}

Network::~Network() 
{
    std::cerr << "Network Destructor called\n";
}

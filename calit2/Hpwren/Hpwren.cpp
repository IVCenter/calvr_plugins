#include "Hpwren.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginManager.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/NodeMask.h>
#include <cvrMenu/MenuSystem.h>
#include <PluginMessageType.h>
#include <iostream>

#include <osg/Matrix>
#include <osg/Math>
#include <osg/LightModel>
#include <osg/Texture2D>
#include <osgDB/ReadFile>
#include <osg/Geode>
#include <osg/Material>
#include <osg/ShapeDrawable>
#include <osgEarth/ElevationQuery>
#include <osgEarthDrivers/feature_ogr/OGRFeatureOptions>
#include <osgEarthDrivers/model_feature_geom/FeatureGeomModelOptions>

#include <math.h>
#include <string.h>
//#include "SdgeReader.h"
#include "MesoReader.h"

using namespace osg;
using namespace std;
using namespace cvr;
using namespace osgEarth;
using namespace osgEarth::Drivers;
using namespace osgEarth::Features;
using namespace osgEarth::Symbology;

// Return this code to the OS in case of failure.
static const int NC_ERR = 2;

CVRPLUGIN(Hpwren)

// simple array of color to use for shape files
Color shapefileColors[] = {Color::White, Color::Silver, Color::Maroon, Color::Yellow, Color::Purple, Color::Orange, Color::Navy, Color::Red, Color::Teal};

// simple clamp function
inline float clamp(float x, float a, float b)
{
    return x < a ? a : (x > b ? b : x);
}

Hpwren::Hpwren()
{

}

bool Hpwren::init()
{
	//osg::setNotifyLevel( osg::INFO );

    std::cerr << "Hpwren init\n";
    _sensorThread = NULL;
    _map = NULL;
    _mapNode = NULL;

	// init netcdf params
	_direction = NULL;
	_speed = NULL;
	_nlat = 0;
	_nlon = 0;
	_time = 0;
	//_file = NULL;

	// try finding an exiusting planet attached to the scenegraph
    osgEarth::MapNode* _mapNode = MapNode::findMapNode( SceneManager::instance()->getObjectsRoot() );
    
    // will return true if OsgEarth is enabled in config file
    if( !_mapNode )
	{
		std::cerr << "ERROR: OsgEarth plugin needs to be active enable in configuration file" << std::endl;
		return false;
	}

	// try readin the netcdf file
	//if( NC_ERR == parseNetCDF(std::string("/home/pweber/Downloads/out.nc")) )
	//	std::cerr << "Error reading NetCDF File\n";

	_map = _mapNode->getMap();

	// init font
    _font = osgText::readFontFile(ConfigManager::getEntry("Plugin.Hpwren.Font"));
    _style = new osgText::Style;
    _style->setThicknessRatio(0.01);

    // create basic menus
    _hpwrenMenu = new SubMenu("Hpwren");
    _shapeMenu = new SubMenu("ShapeFile");
    _hpwrenMenu->addItem(_shapeMenu);

	// send message to get osgEarth menu back
	OsgEarthMenuRequest request;
	request.plugin = "Hpwren";
	request.oe_menu = NULL;
	PluginManager::instance()->sendMessageByName("OsgEarth",OE_MENU,(char *) &request);

	// note: replaced with getting main menu from the osgEarth
    //PluginHelper::addRootMenuItem(_hpwrenMenu);


    // NOTE hard coding min and max 0 - 40 celcius

    _minTemp = FLT_MAX;
    _maxTemp = FLT_MIN;

    // read in hpwren sensors first
    XmlReader* configs = new XmlReader(ConfigManager::getEntry("Plugin.Hpwren.Sensors"));

	// read in hpwren sensors
    osg::Group* flagGroup = new osg::Group();
    PluginHelper::getObjectsRoot()->addChild(flagGroup);
    
    // init sensors
    initSensors(flagGroup, configs);

	// read in shape file and create menu items
	std::vector<std::string> shapefileNames;
    ConfigManager::getChildren("Plugin.Hpwren.ShapeFiles",shapefileNames);

    for(int i = 0; i < shapefileNames.size(); i++)
    {
		// get shape location
		string location = ConfigManager::getEntry("Plugin.Hpwren.ShapeFiles." + shapefileNames[i]);

		OGRFeatureOptions feature_opt;
    		feature_opt.name() = shapefileNames[i];
    		feature_opt.url() = location;

		// a style for the building data:
		Style style;
		style.setName("default");
		style.getOrCreate<AltitudeSymbol>()->clamping() = AltitudeSymbol::CLAMP_TO_TERRAIN;
     		style.getOrCreate<ExtrusionSymbol>()->height() = 1000.0; // meters MSL
        	style.getOrCreate<PolygonSymbol>()->fill()->color() = Color(shapefileColors[0], 0.5);
	
		// assemble a stylesheet and add our styles to it:
		StyleSheet* styleSheet = new StyleSheet();
		styleSheet->addStyle( style );

		// create a model layer that will render the buildings according to our style sheet.
		FeatureGeomModelOptions fgm_opt;
		fgm_opt.featureOptions() = feature_opt;
		fgm_opt.styles() = styleSheet;

		// create a model layer	
		ModelLayer* layer = new ModelLayer( shapefileNames[i], fgm_opt);

		// add menu option (default is disabled)
		SubMenu* layerMenu = new SubMenu(shapefileNames[i]);	
	
        	MenuCheckbox* currentcheck = new MenuCheckbox(shapefileNames[i], false);
		currentcheck->setCallback(this);
        	layerMenu->addItem(currentcheck);

        	MenuRangeValue* currentheight = new MenuRangeValue("Height: ", 0.0, 4000.0, 1000.0);
		currentheight->setCallback(this);
        	layerMenu->addItem(currentheight);

        	MenuRangeValue* currentcolor = new MenuRangeValue("Color: ", 0.0, (float) (sizeof(shapefileColors) / sizeof(Color)), 0.0);
		currentcolor->setCallback(this);
        	layerMenu->addItem(currentcolor);

        	_shapeMenu->addItem(layerMenu);

	        map->addModelLayer(layer);
		layer->setVisible(currentcheck->getValue());
        	_shapeLayers.push_back(std::pair<MenuCheckbox*, ModelLayer* > (currentcheck, layer));
        	_heightLayers.push_back(std::pair<MenuRangeValue*, StyleSheet* > (currentheight, styleSheet));
        	_colorLayers.push_back(std::pair<MenuRangeValue*, StyleSheet* > (currentcolor, styleSheet));
    }
    std::cerr << "Hpwren init complete\n";
    return true;
}

void Hpwren::message(int type, char *&data, bool collaborative)
{
    if(type == OE_MENU)
    {
        // fly mode
        OsgEarthMenuRequest* oerequest = (OsgEarthMenuRequest*)data;
		_baseMenu = oerequest->oe_menu;

        // add menus
        _baseMenu->addItem(_hpwrenMenu);
    }
}

/*
int Hpwren::parseNetCDF(std::string fileName)
{
	//_file = new NcFile(fileName.c_str(), NcFile::ReadOnly);
	_file = new NcFile(fileName.c_str());

	if( _file->is_valid() )
	{
		_time = _file->get_dim(0)->size();
        _nlat = _file->get_dim(1)->size();
        _nlon = _file->get_dim(2)->size();

		std::cerr << "Time " << _time << " lat " << _nlat << " lon " << _nlon << std::endl;

		// read in the dimensions
		NcVar *latVar, *lonVar;

		if (!(latVar = _file->get_var("latitude")))
                return NC_ERR;

        if (!(lonVar = _file->get_var("longitude")))
                return NC_ERR;

        // temporary hold the dimensions to create arrows
		float lats[_nlat * _nlon], lons[_nlat * _nlon]; // TODO need to change not a rectangular grid

		if (!latVar->get(lats, _nlat, _nlon))
                return NC_ERR;
        
        if (!lonVar->get(lons, _nlat, _nlon))
                return NC_ERR;

		// TODO need to create arrows

		

		// need to create data set once (heap) TODO
		_direction = new float[_nlat * _nlon];
        _speed = new float[_nlat * _nlon];

		// read in first time step data
		if( NC_ERR == initNetCDFStep(0) )
			std::cerr << "Error trying to set NetCDF timestep\n";
	}
	else
	{
		_file = NULL;
		return 0;
	}

	return 1;
}

// TODO might need a lock on here
int Hpwren::initNetCDFStep(int index)
{
	NcVar *dirVar, *speedVar;

	if (!(dirVar = _file->get_var("WDIR")))
          return NC_ERR;

    if (!(speedVar  = _file->get_var("WGUST_10m")))
          return NC_ERR;

    if (!dirVar->set_cur(index, 0, 0))
    	return NC_ERR;
    if (!speedVar->set_cur(index, 0, 0))
    	return NC_ERR;

	if (!dirVar->get(_direction, 1, _nlat, _nlon))  
    	return NC_ERR;
    if (!speedVar->get(_speed, 1, _nlat, _nlon))  
        return NC_ERR;

	// should test to see if data was correctly written in
	std::cerr << "Direction " << _direction[0] << " speed " << _speed[0] << std::endl;
	
	return 1;
}
*/

void Hpwren::menuCallback(MenuItem * item)
{
    // toggle on and off
    for(int i = 0; i < (int) _shapeLayers.size(); i++)
    {
        MenuCheckbox * current = _shapeLayers.at(i).first;
        if( current == item)
        {
                // make sure layer is enabled
                _shapeLayers.at(i).second->setVisible(current->getValue());
                return;
        }
    }

    // set height of shape
    for(int i = 0; i < (int) _heightLayers.size(); i++)
    {
        MenuRangeValue * current = _heightLayers.at(i).first;
        if( current == item)
        {
		//change style sheet
		StyleSheet* sheet = _heightLayers.at(i).second;
		Style* style = sheet->getStyle("default");
		if( style )
			style->getOrCreate<ExtrusionSymbol>()->height() = current->getValue();	
                return;
        }
    }

    // set color of shape
    for(int i = 0; i < (int) _colorLayers.size(); i++)
    {
        MenuRangeValue * current = _colorLayers.at(i).first;
        if( current == item)
        {
		//change style sheet
		StyleSheet* sheet = _colorLayers.at(i).second;
		Style* style = sheet->getStyle("default");
		if( style && style->getOrCreate<PolygonSymbol>()->fill()->color() != shapefileColors[(int)current->getValue()])
			style->getOrCreate<PolygonSymbol>()->fill()->color() = shapefileColors[(int)current->getValue()];	
                return;
        }
    }
}

void Hpwren::initSensors(osg::Group* parent, XmlReader* configs)
{
    // set up hpwren sensors
    std::string baseName("Hpwren");
    std::vector<std::string> tagList;
    configs->getChildren(baseName, tagList);
    for(int i = 0; i < tagList.size(); i++)
    {
	std::string sensorName(baseName + "." + tagList[i]);
	Sensor sensor(true, _font, _style, ConfigManager::getBool("Plugin.Hpwren.Portrait", false));
	sensor.setCoord(configs->getFloat("lon", sensorName, 0.0), configs->getFloat("lat", sensorName, 0.0));
	_hpwrensensors.insert(std::pair<std::string, Sensor> (configs->getEntry("value", sensorName, ""), sensor));
    }

    // set up initial sdge sites
    
    int numSDGESensors = 0;

    // ONLY READ ON MASTER and then sync size
    if(ComController::instance()->isMaster())
    {
	//std::string name("http://anr.ucsd.edu/Sensors/SDGE"); OLD
	//std::string name("http://anr.ucsd.edu/cgi-bin/sm_sdge2.pl");
	//SdgeReader test(name, _sdgesensors, _font, _style, ConfigManager::getEntry("Plugin.Hpwren.SdgeBak"), ConfigManager::getBool("Plugin.Hpwren.Portrait", false));
	std::string name("https://firemap.sdsc.edu:5443/stations/data/latest?selection=withinRadius&lat=32.7157&lon=-117.1611&radius=50&observable=temperature&observable=wind_speed&observable=wind_direction");
	MesoReader test(name, _sdgesensors, _font, _style, ConfigManager::getEntry("Plugin.Hpwren.MesoBak"), ConfigManager::getBool("Plugin.Hpwren.Portrait", false));
	numSDGESensors = (int)_sdgesensors.size();
        ComController::instance()->sendSlaves(&numSDGESensors,sizeof(int));
    }
    else
    {
	ComController::instance()->readMaster(&numSDGESensors,sizeof(int));
    }

    std::cerr << "Num sensors: " << numSDGESensors << std::endl;

    // sync sensor data from head node
    std:vector< SensorData > updates(numSDGESensors);
    
    if(ComController::instance()->isMaster())
    {
	// add data to updates
	int index = 0;
	std::map< std::string, Sensor >::iterator it = _sdgesensors.begin();
	for(; it != _sdgesensors.end(); ++it )
	{
	    updates[index].velocity = it->second.getVelocity();
	    updates[index].direction = it->second.getDirection();
	    updates[index].temperature = it->second.getTemperature();
	    updates[index].pressure = it->second.getPressure();
	    updates[index].humidity = it->second.getHumidity();
	    it->second.getCoord(updates[index].lon, updates[index].lat);
	    strcpy(updates[index].name, it->first.c_str());
	    index++;
	}

        ComController::instance()->sendSlaves(&updates[0],sizeof(SensorData) * numSDGESensors);
    }
    else
    {
	ComController::instance()->readMaster(&updates[0],sizeof(SensorData) * numSDGESensors);

	// add sdge data to map
	// read data back into _sdgesensors (different between systems for some reason)
	for(int i = 0; i < (int) updates.size(); i++ )
	{
	    Sensor sens(true, _font, _style, ConfigManager::getBool("Plugin.Hpwren.Portrait", false));
	    sens.setVelocity(updates.at(i).velocity);
	    sens.setDirection(updates.at(i).direction);
	    sens.setTemperature(updates.at(i).temperature);
	    sens.setPressure(updates.at(i).pressure);
	    sens.setHumidity(updates.at(i).humidity);
	    sens.setCoord(updates.at(i).lon, updates.at(i).lat);
	    _sdgesensors.insert(std::pair< std::string, Sensor> (std::string(updates.at(i).name),sens));
/*
	    std::map<std::string, Sensor>::iterator it = _sdgesensors.find(std::string(updates.at(i).name));
	    if( it != _sdgesensors.end() )
	    {
		it->second.setVelocity(updates.at(i).velocity);
		it->second.setDirection(updates.at(i).direction);
		it->second.setTemperature(updates.at(i).temperature);
		it->second.setPressure(updates.at(i).pressure);
		it->second.setHumidity(updates.at(i).humidity);
	    }
*/
	}
    }

    // compute min and max temperatures
    std::map<std::string, Sensor>::iterator it = _sdgesensors.begin();
    for(; it != _sdgesensors.end(); ++it)
    {
	float temp = it->second.getTemperature();
        if(temp < _minTemp)
                _minTemp = temp;
        if(temp > _maxTemp)
                _maxTemp = temp;
        
        //std::cerr << "Name: " << it->first << " direction: " << it->second.getDirection() << " velocity " << it->second.getVelocity() 
        //                <<  " temp " << it->second.getTemperature() << std::endl;
    }


    // NOTE hard coding min and max 0 - 40 celcius
    _minTemp = 0.0;
    _maxTemp = 40.0;

	// initialize sdge tower data
    for(it = _sdgesensors.begin() ; it != _sdgesensors.end(); ++it)
    {
        // make tower active
        //_sdgesensors.at(i).setType(true);

        // set direction of the flag
        osg::Matrix mat;
        mat.makeRotate((it->second.getDirection()*osg::PI/180.0) + osg::PI_2,osg::Vec3(0,0,1));

        // set length of the flag
        osg::Matrix matScale;
        matScale.makeScale(osg::Vec3(it->second.getVelocity() / 5.0, 1.0, 1.0));
        mat = matScale * mat;
     
        mat.setTrans(it->second.getRotation().getTrans());
        it->second.setRotation(mat);
        //it->second.getScale()->setMatrix(matScale);

        // set the color of the flag
        it->second.getColor()->set(clamp((it->second.getTemperature() - _minTemp) / (_maxTemp - _minTemp), 0.0, 1.0));
        
        //std::cerr << "Updated tower info " << key << std::endl;
	std::stringstream ss;
	ss << it->first << ", ";
        ss << setprecision(1) << std::fixed;
	ss << "Velocity: ";
        ss << it->second.getVelocity();
        ss << "m/s, Temp: ";
        ss << it->second.getTemperature();
        ss << "C, Direction: ";
        ss << it->second.getDirection();
        ss << " degrees";
	it->second.getFlagText()->setText(ss.str());			
    }

	std::cerr << "min: " << _minTemp << " max: " << _maxTemp << std::endl;

    // if master create a sensor thread to listen for updates
    if(ComController::instance()->isMaster())
        _sensorThread = new SensorThread(_hpwrensensors);

	// locate flag shaders
    std::string shaderpath = ConfigManager::getEntry("Plugin.Hpwren.ShaderPath");
    osg::Program* program = new osg::Program;
    program->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile(shaderpath + "/flag.vert")));
    program->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile(shaderpath + "/flag.frag")));
    
    // locate color look up texture
    std::string colortable = ConfigManager::getEntry("Plugin.Hpwren.ColorTable");
    osg::Image* image = osgDB::readImageFile(colortable);
    osg::Texture2D* texture = new osg::Texture2D(image);
    texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
    texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
    parent->getOrCreateStateSet()->setTextureAttribute(0, texture);
   
    // create Towers
    createTowers(_hpwrensensors, osg::Vec4(0.0, 0.0, 1.0, 1.0), parent, 1000.0, program);
    createTowers(_sdgesensors, osg::Vec4(102.0 / 255.0, 0.0, 204.0 / 255.0, 1.0) , parent, 1000.0, program);
    //createTowers(_hpwrensensors, osg::Vec4(0.0, 0.0, 1.0, 1.0), parent, 1000.0, program);
    //createTowers(_sdgesensors, osg::Vec4(102.0 / 255.0, 0.0, 204.0 / 255.0, 1.0) , parent, 1000.0, program);
}

osg::Geode * Hpwren::createFlag(float heightAboveGround, osg::Vec4& color, int numWaves)
{
    // p1 ----- p3
    // |
    // |
    // p2

    // create geode flag
    osg::Geode* flag = new osg::Geode();
    osg::Vec3Array* vertices = new osg::Vec3Array();
    osg::Vec3Array* normals = new osg::Vec3Array();
    osg::Vec4Array* colors = new osg::Vec4Array();

    float flagLength = heightAboveGround * 0.3;
    float flagHeight = heightAboveGround * 0.2;
    float waveLength = flagLength / numWaves;
    float waveHeight = waveLength * 0.25;

    for(int i = 0; i < numWaves; i++)
    {
        float position = waveHeight * M_SQRT1_2;
        float center = (waveLength * i) + waveHeight;

        vertices->push_back(osg::Vec3(center - waveHeight, 0.0, heightAboveGround));
        vertices->push_back(osg::Vec3(center - waveHeight, 0.0, heightAboveGround - flagHeight));
        normals->push_back(osg::Vec3(-1, 0, 0));
        normals->push_back(osg::Vec3(-1, 0, 0));

        vertices->push_back(osg::Vec3(center - position, position, heightAboveGround));
        vertices->push_back(osg::Vec3(center - position, position, heightAboveGround - flagHeight));
        normals->push_back(osg::Vec3(-M_SQRT1_2, M_SQRT1_2, 0));
        normals->push_back(osg::Vec3(-M_SQRT1_2, M_SQRT1_2, 0));

        vertices->push_back(osg::Vec3(center, waveHeight, heightAboveGround));
        vertices->push_back(osg::Vec3(center, waveHeight, heightAboveGround - flagHeight));
        normals->push_back(osg::Vec3(0, 1, 0));
        normals->push_back(osg::Vec3(0, 1, 0));

		vertices->push_back(osg::Vec3(center + position, position, heightAboveGround));
        vertices->push_back(osg::Vec3(center + position, position, heightAboveGround - flagHeight));
        normals->push_back(osg::Vec3(M_SQRT1_2, M_SQRT1_2, 0));
        normals->push_back(osg::Vec3(M_SQRT1_2, M_SQRT1_2, 0));

        center = (waveLength * i) + (waveHeight * 3);

        vertices->push_back(osg::Vec3(center - waveHeight, 0.0, heightAboveGround));
        vertices->push_back(osg::Vec3(center - waveHeight, 0.0, heightAboveGround - flagHeight));
        normals->push_back(osg::Vec3(1, 0, 0));
        normals->push_back(osg::Vec3(1, 0, 0));

        vertices->push_back(osg::Vec3(center - position, -position, heightAboveGround));
        vertices->push_back(osg::Vec3(center - position, -position, heightAboveGround - flagHeight));
        normals->push_back(osg::Vec3(-M_SQRT1_2, -M_SQRT1_2, 0));
        normals->push_back(osg::Vec3(-M_SQRT1_2, -M_SQRT1_2, 0));

        vertices->push_back(osg::Vec3(center, -waveHeight, heightAboveGround));
        vertices->push_back(osg::Vec3(center, -waveHeight, heightAboveGround - flagHeight));
        normals->push_back(osg::Vec3(0, -1, 0));
        normals->push_back(osg::Vec3(0, -1, 0));

        vertices->push_back(osg::Vec3(center + position, -position, heightAboveGround));
        vertices->push_back(osg::Vec3(center + position, -position, heightAboveGround - flagHeight));
        normals->push_back(osg::Vec3(M_SQRT1_2, -M_SQRT1_2, 0));
        normals->push_back(osg::Vec3(M_SQRT1_2, -M_SQRT1_2, 0));
	}

    vertices->push_back(osg::Vec3(flagLength, 0.0, heightAboveGround));
    vertices->push_back(osg::Vec3(flagLength, 0.0, heightAboveGround - flagHeight));
    normals->push_back(osg::Vec3(1, 0, 0));
    normals->push_back(osg::Vec3(1, 0, 0));

    colors->push_back(color);

	osg::Geometry* geometry = new osg::Geometry();
    geometry->setVertexArray(vertices);
    geometry->setNormalArray(normals);
    geometry->setNormalBinding( osg::Geometry::BIND_PER_VERTEX );
    geometry->setColorArray(colors);
    geometry->setColorBinding( osg::Geometry::BIND_OVERALL );
    geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP,0, (int) vertices->size()));
    flag->addDrawable(geometry);

    float boundParam = heightAboveGround * 0.3;
    geometry->setInitialBound(osg::BoundingBox(-boundParam, -boundParam, -boundParam, boundParam, boundParam, boundParam));
    flag->addDrawable(geometry);

    osg::LightModel* ltModel = new osg::LightModel;
    ltModel->setTwoSided(true);
    flag->getOrCreateStateSet()->setAttribute(ltModel);

    return flag;
}

osg::Matrix Hpwren::computePosition(double lat, double lon, double height)
{
    double groundHeight = 0.0;
    osgEarth::ElevationQuery query( _map );
    osgEarth::GeoPoint point(_map->getProfile()->getSRS(), lon, lat);
    query.getElevation(point, groundHeight);

    osg::Matrixd output;
    _map->getProfile()->getSRS()->getEllipsoid()->computeLocalToWorldTransformFromLatLongHeight(
                                                             osg::DegreesToRadians(lat),
                                                             osg::DegreesToRadians(lon),
                                                             height + groundHeight,
                                                             output );

    return output;
}

void Hpwren::createTowers(std::map<std::string, Sensor> & sensors, osg::Vec4 baseColor, osg::Group* parent, float towerHeight, osg::Program * program)
{
    // loop through and create towers
	std::map<std::string, Sensor>::iterator it = sensors.begin();
    for(; it != sensors.end(); ++it)
    {
            // sample: compute a location on the planet
            double height = 0.0;   // on the surface (in meters)
	    double test = 0.0;

            double latitude;
            double longitude;
            it->second.getCoord(longitude , latitude);

            osgEarth::ElevationQuery query( _map );
            osgEarth::GeoPoint point(_map->getProfile()->getSRS(), longitude, latitude);
	    query.getElevation(point, height);

            osg::Matrixd output;
            _map->getProfile()->getSRS()->getEllipsoid()->computeLocalToWorldTransformFromLatLongHeight(
                    osg::DegreesToRadians(latitude),
                    osg::DegreesToRadians(longitude),
                    height,
                    output );

			// attach a tower
            osg::Geode* tower = new osg::Geode();
            osg::ShapeDrawable* shape = new osg::ShapeDrawable(new osg::Cylinder(osg::Vec3(0.0, 0.0, (towerHeight * 0.5)), (towerHeight * 0.01), towerHeight));
            tower->addDrawable(shape);

            // set tower color
            osg::ref_ptr<osg::StateSet> stateset = tower->getOrCreateStateSet();
            osg::ref_ptr<osg::Material> mm = dynamic_cast<osg::Material*>(stateset->getAttribute(osg::StateAttribute::MATERIAL));
            if (!mm)
                mm = new osg::Material;

            // flag pole base color
            mm->setDiffuse(osg::Material::FRONT, baseColor);
            stateset->setAttributeAndModes( mm, osg::StateAttribute::PROTECTED | osg::StateAttribute::ON );
            tower->setStateSet(stateset);

            osg::MatrixTransform * mat = new osg::MatrixTransform();
            mat->setMatrix(output);
            mat->addChild(tower);

	    if( it->second.getType() )   //  add flag if site has data
            {
                    // create geode flag
                    osg::Vec4 color(1.0, 0.0, 0.0, 1.0);
                    osg::Geode* flag = createFlag(towerHeight, color, 4);

                    // set flag color
                    stateset = flag->getOrCreateStateSet();
                    stateset->setAttribute(program);

                    // attach color uniform to flag
                    stateset->addUniform(it->second.getColor());

                    // attach flag to tower
                    osg::Matrix positionMat;
                    positionMat.setTrans(osg::Vec3(0.0, 0.0, towerHeight));
                    mat->addChild(it->second.getFlagTransform());
                    it->second.getFlagTransform()->addChild(flag);

					// add text and background to geode
					osg::Geode* textGeode = new osg::Geode();
					textGeode->addDrawable(it->second.getFlagText());

					// create auto transform to aligntext with screen
					osg::AutoTransform* at = new osg::AutoTransform;
					at->setPosition(osg::Vec3(0.0, 0.0, towerHeight));
					at->setAutoRotateMode(osg::AutoTransform::ROTATE_TO_CAMERA);

					// add a LOD Node to disable text from far away
					osg::LOD* lod = new osg::LOD;
					lod->addChild(textGeode, 0.0, 10000.0);
					at->addChild(lod);
					mat->addChild(at);
            }

            // create flag to attach   // adjust scale in x to stretch flag for speed
            parent->addChild(mat);
    }
}

Hpwren::~Hpwren()
{
   if( _sensorThread )
		delete _sensorThread;
   _sensorThread = NULL;
}

void Hpwren::preFrame()
{
	// synchronize map data
	std:vector< SensorData > updates((int)_hpwrensensors.size());
	
	if(ComController::instance()->isMaster())
	{
	    // get data from listening thread
	    _sensorThread->getData(updates);
	    ComController::instance()->sendSlaves(&updates[0],sizeof(SensorData) * (int) _hpwrensensors.size());
	}
	else
	{
	    ComController::instance()->readMaster(&updates[0],sizeof(SensorData) * (int) _hpwrensensors.size());
	}

   // loop through the updates.... then update the tower info
   for(int i = 0; i < (int)updates.size(); i++)
   {
	 std::string key(updates[i].name);
	 std::map<std::string, Sensor>::iterator it = _hpwrensensors.find(key);
	 if( it != _hpwrensensors.end() )
	 {
	     // set direction of the flag
	     osg::Matrix mat;
	     mat.makeRotate((updates[i].direction*osg::PI/180.0) + osg::PI_2,osg::Vec3(0,0,1));
	    
	     // set length of the flag
	     osg::Matrix matScale;
	     matScale.makeScale(osg::Vec3(updates[i].velocity / 5.0, 1.0, 1.0));
	     mat = matScale * mat;

	     mat.setTrans(it->second.getRotation().getTrans());
	     it->second.setRotation(mat);

	     // set the color of the flag
	     it->second.getColor()->set(clamp((updates[i].temperature - _minTemp) / (_maxTemp - _minTemp), 0.0, 1.0));

		 // set string
         std::stringstream ss;
         ss << setprecision(1) << std::fixed;
         ss << "Velocity: ";
         ss << updates[i].velocity;
         ss << "m/s, Temp: ";
         ss << updates[i].temperature;
         ss << "C, Direction: ";
         ss << updates[i].direction;
         ss << " degrees";
         it->second.getFlagText()->setText(ss.str());
      }
   }
}

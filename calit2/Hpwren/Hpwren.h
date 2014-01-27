#ifndef _HPWREN_
#define _HPWREN_

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <curl/curl.h>

#include <osgEarth/Map>
#include <osgEarth/MapNode>
#include <osgEarth/Utils>
#include <osgEarthDrivers/feature_ogr/OGRFeatureOptions>
#include <osgEarthDrivers/model_feature_geom/FeatureGeomModelOptions>

#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>

#include <osg/MatrixTransform>
#include <osg/Program>

#include <string>
#include <vector>

#include "netcdf.h"
#include "netcdfcpp.h"

#include "SensorThread.h"
#include "XmlReader.h"

class Hpwren : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:        
        Hpwren();
        virtual ~Hpwren();
        void preFrame(); 
		bool init();
		void menuCallback(cvr::MenuItem * item);
        virtual void message(int type, char *&data, bool collaborative = false);

    protected:
        osgEarth::Map * map;
		std::map < std::string, Sensor > _hpwrensensors;
		std::map < std::string, Sensor > _sdgesensors;
		SensorThread* _sensorThread;
		osg::Group* _root; // flag root node

		// model layers
		std::vector< std::pair<cvr::MenuCheckbox*, osgEarth::ModelLayer* > > _shapeLayers;
		std::vector< std::pair<cvr::MenuRangeValue*, osgEarth::Features::StyleSheet* > > _colorLayers;
		std::vector< std::pair<cvr::MenuRangeValue*, osgEarth::Features::StyleSheet* > > _heightLayers;
	
		// flag values
		float _waveTime;
		float _waveWidth;
		float _waveHeight;
		float _waveFreq;

		// tower height
		float _towerHeight;
		    
		// uniforms for adjustign the flag
		osg::Uniform* _waveTimeUniform;
		osg::Uniform* _waveWidthUniform;
		osg::Uniform* _waveHeightUniform;

		// create tower function
		void createTowers(std::map<std::string, Sensor> & sensors, osg::Vec4 baseColor, osg::Group* parent, float towerHeight, osg::Program * program);
        osg::Geode * createFlag(float heightaboveGround, osg::Vec4& color, int numWaves = 4);
		void initSensors(osg::Group*, XmlReader*);
		osg::Matrix computePosition(double lat, double lon, double height);

		// parse NetCDFFile
		int parseNetCDF(std::string fileName);
		int initNetCDFStep(int index);
		float* _direction;
		float* _speed;
		int _nlat;
		int _nlon;
		int _time;
		NcFile* _file;

		// font and style for all text
        osg::ref_ptr<osgText::Font> _font;
        osg::ref_ptr<osgText::Style> _style;

        // map vars
        osgEarth::Map * _map;
        osgEarth::MapNode * _mapNode;

		// temp range
		float _minTemp;
		float _maxTemp;

		// menu options
		cvr::SubMenu * _baseMenu;
		cvr::SubMenu * _hpwrenMenu;
		cvr::SubMenu * _shapeMenu;
        
};

#endif

#ifndef _NETWORK_
#define _NETWORK_

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/SceneObject.h>
#include <cvrKernel/TiledWallSceneObject.h>

#include <osg/PositionAttitudeTransform>

#include <sstream>
#include <fstream>
#include <iostream>
#include <osgText/Text>

using namespace std;
using namespace osg;

struct IndexValue
{
    int index;
    float value;
};

struct PropertyMap
{
    std::string property;
    osg::Vec3f color;
};

// object to hold full color mapping data
struct ColorMap
{
    std::string type;
    std::string metaName;
    std::vector<PropertyMap> mapping;
};

// can have a list of graphs
struct Graph
{
    std::string graphFileName;
    std::string vertexFileName;
    std::string metaFileName;
    std::map< std::string, ColorMap > mappings;
};

struct LineVertices
{
    unsigned int sInd;
    unsigned int oInd;
};

// structues to hold vertex mapping info
struct VertexMap
{
    unsigned int pointIndex;
    //std::vector< LineVertices > * lineIndexs; // first point always sample
    std::vector< LineVertices > lineIndexs; // first point always sample
};


class TextSizeVisitor : public osg::NodeVisitor
{
  public:
    TextSizeVisitor(float fontsize, bool read);
    virtual void apply(osg::Geode& geode);
    void setFontSize(float size) { _fontsize = size;};
    float getFontSize() { return _fontsize;};

  protected:
    TextSizeVisitor(){};
    float _fontsize;
    bool _read;
};

// TODO Need to do something about edge coloring from observations to samples
// call back to attach and use to change color mapping for vertexs
struct DrawableUpdateCallback : public osg::Drawable::UpdateCallback
{
    OpenThreads::Mutex _mutex;
    std::vector<IndexValue> * _changes;

    DrawableUpdateCallback() : _changes(NULL) {}

    void applyUpdate( std::vector<IndexValue> * changes)
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
        _changes = changes;
    }

    // need to protect to update list is not changed which callback is in use
    virtual void update(osg::NodeVisitor*, osg::Drawable* drawable)
    {
        // update the vertex color mapping 
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
        if( _changes )
        {
            osg::Geometry* geom = dynamic_cast<osg::Geometry* > (drawable);
            if( geom )
            {
                osg::Vec4Array* colors = dynamic_cast<osg::Vec4Array*> (geom->getColorArray());
                if ( colors )
                {
                    for(int i = 0; i < _changes->size(); i++)
                    {
                        osg::Vec4 *color = &colors->operator [](_changes->at(i).index);
                        color->set( (*color)[0], (*color)[1], (*color)[2], _changes->at(i).value );
                    }
                    geom->setColorArray(colors);
                }
            }

            // once update is applied clean up memory
            delete _changes;
            _changes = NULL;
        }
    }
};

class Network: public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:        
	Network();
	virtual ~Network();
	bool init();
        virtual void menuCallback(cvr::MenuItem * menuItem);
        virtual void preFrame();

    protected:
        cvr::TiledWallSceneObject* _so; // can hold standard objects or tiledWallSceneObjects (TODO screen multiple group)
	cvr::MenuCheckbox* _colorEdges;
	cvr::MenuCheckbox* _sampleText;
	cvr::MenuCheckbox* _elementText;
	cvr::MenuRangeValue* _edgeWeight;
	cvr::MenuRangeValue* _edgeWidth;
	cvr::MenuRangeValue* _otuEdges;
	cvr::MenuRangeValue* _sampleTextSize;
	cvr::MenuRangeValue* _elementTextSize;
	cvr::MenuRangeValue* _samplePointSize;
	cvr::MenuRangeValue* _elementPointSize;
	    //cvr::MenuRangeValue* _otuTotalWeight;
	cvr::MenuRangeValue* _sampleEdges;
	    //cvr::MenuRangeValue* _sampleTotalWeight;
        cvr::SubMenu * _subMenu;
        cvr::SubMenu * _loadMenu;
        cvr::MenuButton * _remove;

        // menu of networks (only allow one to be loaded at a time)
        //std::vector< std::pair<cvr::MenuButton*, std::string> > menuFileList;
        std::vector< std::pair<cvr::SubMenu*, std::string> > menuFileList;

        // holds the key menu
        osg::PositionAttitudeTransform* _root;
        
        // color mapping lookup
        std::map< std::string, Graph > _vertColorMapping;

        // button mapping
        std::map< cvr::MenuItem* , std::pair< std::string, std::string> > _buttonMapping;
        std::map<std::string, VertexMap>  _vertexSampleLookup;
        std::map<std::string, VertexMap>  _vertexObserLookup;

        // current loaded graph
        std::string _currentLoadedGraph;
    
        // data directory
        std::string _dataDirectory;

        // apply sample color mapping to current graph
        void applyMapping(osg::Geode* pointNode, osg::Geode* edgeNode,
                                ColorMap & colorMapping, // property to color
                                std::map<std::string, string> & mapping, // sample to property
                                std::map<std::string, VertexMap > & vertexMapping); // name to vertex look up
        
        // apply observation color mapping to current graph
        //void applyObservationMapping(osg::Geode* otuNode, osg::Geode* edgeNode,
        //                        ColorMap & colorMapping, // property to color
        //                        std::map<std::string, string> & sampleMapping, // sample to property
        //                        std::map<std::string, VertexMap > & vertexMapping); // name to vertex look up


        void loadSpecificMetaData(std::string key, std::string metaHeader, std::string metaDataFile, std::map<std::string, int> & types, 
                                  std::map<std::string, string > & sampleMapping);
        osg::Geode* createVisualKey(std::vector<std::pair<std::string, osg::Vec3> > & elements);
        
        // load in network 
        void loadNetwork(std::string fileName, float highestWeight);

        // load graph file and configure menu
        void loadColorMappingFile(std::string vertMapFileName, std::map< std::string, Graph > & vertColorMapping, 
                                  cvr::SubMenu* menu, std::map< cvr::MenuItem* , std::pair< std::string, std::string> > & buttonMapping);

        // vertex json mapping file load function, returns edge weight (for normalization)
        float loadJsonVertexMappingFile(std::string fileName, std::map<std::string, VertexMap> & vertexSampleLookup, std::map<std::string, VertexMap> & vertexObervLookup);
};

#endif

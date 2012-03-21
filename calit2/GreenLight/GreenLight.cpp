// John Mangan (Summer 2011)
//  Alfred Tarng (Spring 2012)
// Plugin for CalVR -- GreenLight Project
// Many models altered from prior Covise Plugin (BlackBoxInfo)

#include "GreenLight.h"

#include <fstream>
#include <iostream>
#include <stdlib.h>

#include <kernel/ComController.h>
#include <kernel/InteractionManager.h>
#include <kernel/PluginHelper.h>

#include <osgDB/ReadFile>

CVRPLUGIN(GreenLight)

using namespace osg;
using namespace std;
using namespace cvr;

#ifdef WITH_OSGEARTH
using namespace osgEarth;
#endif

// Static Variables
osg::ref_ptr<osg::Uniform> GreenLight::Component::_displayTexturesUni =
                                         new osg::Uniform("showTexture",false);
osg::ref_ptr<osg::Uniform> GreenLight::Component::_neverTextureUni =
                                         new osg::Uniform("showTexture",false);

Matrixd previousViewMatrix; // TODO: move this to .h file;
double  previousViewScale;
bool savedMatrix = false;
bool SOCM = false; // TOGGLE for Custom Menu.

float range = 64;

// TODO move this to green light and allow it to disable functionality.
int lodLevel = -1;//false;
osg::MatrixTransform * secondDegreeMT;// = new GreenLight::MTA();
/***
 * Accessor of a Scenemanager to get specific popup menu members.
 *
 */
class SMA : public SceneManager {

    public:
        int grabProtectedMember1(){ return _menuDefaultOpenButton; }
        float grabProtectedMember2(){ return _menuScale; }

} sma;

class LODA : public LOD {
    public:
        typedef LOD super;
    // Wrapper to toggle activity based on active child?
    //
        void traverse(NodeVisitor& nv){
//          super::traverse(nv);
        }
} loda;


void GreenLight::NodeA::accept(NodeVisitor& nv){
//    printf("check or call accept of lowestLOD\n");
    if (nv.validNodeMask(*this)) 
    {
        nv.pushOntoNodePath(this);
        nv.apply(*this);
        nv.popFromNodePath();
    }
}

void GreenLight::MTA::accept(NodeVisitor& nv){
    if (nv.validNodeMask(*this)) 
    {
        if ( nv.getTraversalMode() == nv.TRAVERSE_ACTIVE_CHILDREN )
        {
            lodLevel = LLOD;
        }
        nv.pushOntoNodePath(this);
        nv.apply(*this);
        nv.popFromNodePath();
    }
}

/***
 * Called when Navigate To Plugin button is clicked.
 * Sets Matrix and Scale to be in front of the Green Light Plugin using
 * hardcoded values.
 */
void zoom(){
    if (savedMatrix == false){
      previousViewMatrix = SceneManager::instance()->getObjectTransform()->getMatrix(); 
      previousViewScale = SceneManager::instance()-> getObjectScale();
      savedMatrix = true;
    }

            double xScale = 342.677490;
            Matrixd xMatrix = Matrixd(
            // Values gained from logging (keyboard event 'l')
            //
/*         
      0.890503,	 0.247582,	 -0.381720,	 0.000000,
	 -0.454463,	 0.523984,	 -0.720350,	 0.000000,
	 0.021670,	 0.814951,	 0.579125,	 0.000000,
	 -19185177.449566,	 102954049.534328,	 -2181004421.189106,	 1.000000
*/

              0.875374,   0.271526,   -0.399995,  0.000000,
             -0.483186,  0.464147,   -0.742360,  0.000000,
             -0.015913,  0.843115,   0.537499,   0.000000,
             -34501329.462374,   -7890269.618077,    -2183230472.145897,     1.000000
/*
        // Camera Position Based off original object position.
             0.894261,    0.247156,    -0.373112,   0.000000,
             -0.446530,   0.548915,    -0.706614,   0.000000,
             0.030163,    0.798503,    0.601235,    0.000000,
             -13083652.887404,    162726828.747752,    -2177405384.833411,  1.000000
*/
            );

            SceneManager::instance()->setObjectMatrix(xMatrix);
            SceneManager::instance()->setObjectScale( xScale ) ;
}
void restoreView(){
    if (savedMatrix){
        SceneManager::instance()->setObjectMatrix( previousViewMatrix );
        SceneManager::instance()->setObjectScale( previousViewScale ) ;

        savedMatrix = false;
    }
}

GreenLight::GreenLight()
{
    std::cerr << "GreenLight created." << std::endl;
    osgEarthInit = false;
    sma = *(SMA *) SceneManager::instance();
    _menuButton = sma.grabProtectedMember1();
                //(SceneManager::instance()->_menuDefaultOpenButton);
}

GreenLight::~GreenLight()
{
    if (_glMenu) delete _glMenu;
    if (_showSceneCheckbox) delete _showSceneCheckbox;

    if (_hardwareSelectionMenu) delete _hardwareSelectionMenu;
    if (_selectionModeCheckbox) delete _selectionModeCheckbox;
    if (_selectClusterMenu) delete _selectClusterMenu;

    if (_navigateToPluginButton) delete _navigateToPluginButton;
    if (_restorePreviousViewButton) delete _restorePreviousViewButton;

//    if (scaleMT) delete scaleMT; // can't delete... protected?
    if (scaleMatrix) delete scaleMatrix;
    if (scaleVector) delete scaleVector;

    std::set< cvr::MenuCheckbox * >::iterator chit;
    for (chit = _clusterCheckbox.begin(); chit != _clusterCheckbox.end(); chit++)
    {
        if (*chit) delete *chit;
    }
    _clusterCheckbox.clear();

    if (_selectAllButton) delete _selectAllButton;
    if (_deselectAllButton) delete _deselectAllButton;

    if (_displayComponentsMenu) delete _displayComponentsMenu;
    if (_xrayViewCheckbox) delete _xrayViewCheckbox;
    if (_displayFrameCheckbox) delete _displayFrameCheckbox;
    if (_displayDoorsCheckbox) delete _displayDoorsCheckbox;
    if (_displayWaterPipesCheckbox) delete _displayWaterPipesCheckbox;
    if (_displayElectricalCheckbox) delete _displayElectricalCheckbox;
    if (_displayFansCheckbox) delete _displayFansCheckbox;
    if (_displayRacksCheckbox) delete _displayRacksCheckbox;
    if (_displayComponentTexturesCheckbox) delete _displayComponentTexturesCheckbox;
    if (_powerMenu) delete _powerMenu;
    if (_displayPowerCheckbox) delete _displayPowerCheckbox;
    if (_loadPowerButton) delete _loadPowerButton;
    if (_legendText) delete _legendText;
    if (_legendGradient) delete _legendGradient;
    if (_legendTextOutOfRange) delete _legendTextOutOfRange;
    if (_legendGradientOutOfRange) delete _legendGradientOutOfRange;

    if (_timeFrom) delete _timeFrom;
    if (_timeTo) delete _timeTo;
    if (_yearText) delete _yearText;
    if (_monthText) delete _monthText;
    if (_dayText) delete _dayText;
    if (_hourText) delete _hourText;
    if (_minuteText) delete _minuteText;
    if (_yearFrom) delete _yearFrom;
    if (_monthFrom) delete _monthFrom;
    if (_dayFrom) delete _dayFrom;
    if (_hourFrom) delete _hourFrom;
    if (_minuteFrom) delete _minuteFrom;
    if (_yearTo) delete _yearTo;
    if (_monthTo) delete _monthTo;
    if (_dayTo) delete _dayTo;
    if (_hourTo) delete _hourTo;
    if (_minuteTo) delete _minuteTo;

    if (_hoverDialog) delete _hoverDialog;

    if (_box) delete _box;
    if (_waterPipes) delete _waterPipes;
    if (_electrical) delete _electrical;
    if (_fans) delete _fans;

    std::map< std::string, std::set< Component * > * >::iterator cit;
    for (cit = _cluster.begin(); cit != _cluster.end(); cit++)
    {
        if (cit->second) delete cit->second;
    }
    _cluster.clear();

    std::vector<Entity *>::iterator vit;
    for (vit = _door.begin(); vit != _door.end(); vit++)
    {
        if (*vit) delete *vit;
    }
    _door.clear();

    for (vit = _rack.begin(); vit != _rack.end(); vit++)
    {
        if (*vit) delete *vit;
    }
    _rack.clear();

    std::cerr << "GreenLight destroyed." << std::endl;
}

bool GreenLight::init()
{
    std::cerr << "GreenLight init()." << std::endl;

    OsgE_MT = new MatrixTransform();
//    MinMaxPair mmp = pair<500, 5000>;
//    scaleMT = new osg::MatrixTransform(); 
    scaleMT = new GreenLight::MTA();
    pluginMT = new osg::MatrixTransform();

    scaleMT -> addChild (pluginMT);

//  OsgE_MT->addChild( scaleMT );

#ifdef WITH_OSGEARTH 
    mapVariable = NULL; // doesn't seem neccessary.
    osgEarth::MapNode* mapNode = MapNode::findMapNode( SceneManager::instance()->getObjectsRoot() );
    _glLOD  = new LOD();

    if( mapNode )
    {
        _glLOD -> addChild( scaleMT );
        OsgE_MT -> addChild(_glLOD);
        printf("Attached First LOD \"Box\" \n");

        mapNode->setNodeMask(mapNode->getNodeMask() & ~2);

        // Execute OSGEarth Specific initialization.
        printf("Initializing GreenLight with OSGEarth configuration...\n");
        osgEarthInit = true;

        mapVariable = mapNode -> getMap();

        // POSITION:  Texture Based:      Original:
        double lat    =   32.874175,  //  32.874264,
               lon    = -117.236122,  //-117.236074,
               height =  0.0;

        osgEarth::ElevationQuery query( mapVariable );
        double query_resolution = 0.0; // 1/10th of a degree
        double out_resolution = 0.0;
        bool ret = query.getElevation(osg::Vec3d( lon, lat, 0),
        mapVariable->getProfile()->getSRS(), height, query_resolution, &out_resolution);

        mapVariable->getProfile()->getSRS()->getEllipsoid()->computeLocalToWorldTransformFromLatLongHeight(
            DegreesToRadians(lat),
            DegreesToRadians(lon),
            height,
            output
        );

        OsgE_MT->setMatrix( output );

    	// attach a silly shape
        osg::MatrixTransform * gMT = new GreenLight::MTA();
        osg::Geode* geode = new osg::Geode();
        double _swidth = 10, _slength = 30, _sheight = 12;
        osg::ShapeDrawable* shape = new osg::ShapeDrawable(
             new osg::Box(osg::Vec3(0.0, 0.0, 0.0), _swidth, _slength, _sheight ));

        osg::MatrixTransform * lodShapeHeightOffset = new MatrixTransform();
        osg::Matrixd * shapeHeightOffset = new Matrixd();
        shapeHeightOffset->makeTranslate( Vec3d(0,0, _sheight/2) );
        lodShapeHeightOffset->setMatrix( *shapeHeightOffset );

        geode->addDrawable(shape);
        lodShapeHeightOffset->addChild(geode);
        gMT->addChild( lodShapeHeightOffset );
//        lodShapeHeightOffset->addChild( geode );

        secondDegreeMT = new GreenLight::MTA();
        secondDegreeMT -> addChild(pluginMT);

        ((MTA *)scaleMT) ->LLOD = 0; // true;
        _glLOD -> addChild( secondDegreeMT );
        ((MTA *)secondDegreeMT) -> LLOD = 1;
        _glLOD -> addChild( gMT );
        ((MTA *)gMT)->LLOD = 2;

        _glLOD->setRange(0, 0, range);
        _glLOD->setRange(1, range, range * 4);
        _glLOD->setRange(2, range * 4, range * 1024);
        printf("Attached Second LOD \"Box\" \n");

        scaleMatrix = new osg::Matrixd();
        double scaleVal =  // 1.0/500.0;
                              1.0/691.0;
        scaleVector = new osg::Vec3d( scaleVal, scaleVal, scaleVal );

        scaleMatrix->makeScale( *scaleVector );
//        scaleMT->setMatrix( *scaleMatrix );
        pluginMT->setMatrix( *scaleMatrix );

//        mapNode->addChild( OsgE_MT );
        cvr::PluginHelper::getObjectsRoot()->addChild( OsgE_MT );
    }
    else
#endif
    {
        // Execute Default Initialization.
        printf("Initializing GreenLight with default configuration...\n");
        osgEarthInit = false;

        OsgE_MT -> addChild(scaleMT);
        cvr::PluginHelper::getObjectsRoot()->addChild( OsgE_MT );

        lodLevel = 0;
    }


    /*** Things for Context Menus.. ***/
    /******/
                         // name, tag(?), closable
    _myMenu = new PopupMenu("GreenLight", "", false);

    _customButton = new MenuButton("Custom Button");
    _customButton->setCallback(this);
    _myMenu -> addMenuItem(_customButton);
    
    /***************************************/


    /*** Menu Setup ***/
    _glMenu = new cvr::SubMenu("GreenLight","GreenLight");
    _glMenu->setCallback(this);
    cvr::PluginHelper::addRootMenuItem(_glMenu);

    _showSceneCheckbox = new cvr::MenuCheckbox("Load Scene",false);
    _showSceneCheckbox->setCallback(this);
    _glMenu->addItem(_showSceneCheckbox);

    _hardwareSelectionMenu = NULL;
    _selectClusterMenu = NULL;
    _selectionModeCheckbox = NULL;
    _selectAllButton = NULL;
    _deselectAllButton = NULL;

    _displayComponentsMenu = NULL;
    _xrayViewCheckbox = NULL;
    _displayFrameCheckbox = NULL;
    _displayDoorsCheckbox = NULL;
    _displayWaterPipesCheckbox = NULL;
    _displayElectricalCheckbox = NULL;
    _displayFansCheckbox = NULL;
    _displayRacksCheckbox = NULL;
    _displayComponentTexturesCheckbox = NULL;

    _powerMenu = NULL;
    _loadPowerButton = NULL;
    _pollHistoricalDataCheckbox = NULL;
    _displayPowerCheckbox = NULL;
    _magnifyRangeCheckbox = NULL;
    _legendText = NULL;
    _legendGradient = NULL;
    _legendTextOutOfRange = NULL;
    _legendGradientOutOfRange = NULL;

    _timeFrom = NULL;
    _timeTo = NULL;
    _yearText = NULL;
    _monthText = NULL;
    _dayText = NULL;
    _hourText = NULL;
    _minuteText = NULL;
    _yearFrom = NULL;
    _monthFrom = NULL;
    _dayFrom = NULL;
    _hourFrom = NULL;
    _minuteFrom = NULL;
    _yearTo = NULL;
    _monthTo = NULL;
    _dayTo = NULL;
    _hourTo = NULL;
    _minuteTo = NULL;

    _hoverDialog = NULL;

    _navigateToPluginButton = NULL;
    _restorePreviousViewButton = NULL;
    /*** End Menu Setup ***/

    /*** Defaults ***/
    _box = NULL;
    _waterPipes = NULL;
    _electrical = NULL;
    _fans = NULL;

    _shaderProgram = NULL;

    _mouseOver = NULL;
    _wandOver = NULL;
    /*** End Defaults ***/

    return true;
}

void GreenLight::menuCallback(cvr::MenuItem * item)
{
    std::set< cvr::MenuCheckbox * >::iterator chit;

/*
    if(locInit.find(models[i]->name) != locInit.end())
    {
        osg::Matrix scale;
        scale.makeScale(osg::Vec3(locInit[models[i]->name].first,locInit[models[i]->name].first,
        locInit[models[i]->name].first));
        so->setTransform(scale * locInit[models[i]->name].second);
    };
*/

    /*** END: Things for Context Menus.. ***/

    if (item == _showSceneCheckbox)
    {
        // Load as neccessary
        if (!_box)
        {
            if (!_shaderProgram)
            {
                // First compile shaders
                std::cerr<<"Loading shaders... ";
                _shaderProgram = new osg::Program;

                osg::ref_ptr<osg::Shader> vertShader = new osg::Shader( osg::Shader::VERTEX );
                osg::ref_ptr<osg::Shader> fragShader = new osg::Shader( osg::Shader::FRAGMENT );

                if (utl::loadShaderSource(vertShader, cvr::ConfigManager::getEntry("vertex",
                         "Plugin.GreenLight.Shaders", ""))
                && utl::loadShaderSource(fragShader, cvr::ConfigManager::getEntry("fragment",
                         "Plugin.GreenLight.Shaders", "")))
                {
                    _shaderProgram->addShader( vertShader );
                    _shaderProgram->addShader( fragShader );
                    std::cerr<<"done."<<std::endl;
                }
                else
                    std::cerr<<"failed!"<<std::endl;
                // Done with shaders
            }

            utl::downloadFile(cvr::ConfigManager::getEntry("download", "Plugin.GreenLight.Hardware", ""),
                              cvr::ConfigManager::getEntry("local", "Plugin.GreenLight.Hardware", ""),
                              _hardwareContents);

            if (loadScene())
                _showSceneCheckbox->setText("Show Scene");
            else
            {
                std::cerr << "Error: loadScene() failed." << std::endl;
                _showSceneCheckbox->setValue(false);
                return;
            }

        }

        if (_showSceneCheckbox->getValue())
        {
//            scaleMT -> addChild( _box -> transform );
            pluginMT -> addChild( _box -> transform );
        }
        else
        {
//            scaleMT -> removeChild( _box -> transform );
            pluginMT -> removeChild( _box -> transform );
        }
    }
    else if (item == _xrayViewCheckbox)
    {
        bool transparent = _xrayViewCheckbox->getValue();
        _box->setTransparency(transparent);
        _waterPipes->setTransparency(transparent);
        _electrical->setTransparency(transparent);
        _fans->setTransparency(transparent);
        for (int d = 0; d < _door.size(); d++)
            _door[d]->setTransparency(transparent);
        for (int r = 0; r < _rack.size(); r++)
            _rack[r]->setTransparency(transparent);
    }
    else if (item == _displayFrameCheckbox)
    {
        _box->showVisual(_displayFrameCheckbox->getValue());
    }
    else if (item == _displayDoorsCheckbox)
    {
        for (int d = 0; d < _door.size(); d++)
            _door[d]->showVisual(_displayDoorsCheckbox->getValue());
    }
    else if (item == _displayWaterPipesCheckbox)
    {
        _waterPipes->showVisual(_displayWaterPipesCheckbox->getValue());
    }
    else if (item == _displayElectricalCheckbox)
    {
        _electrical->showVisual(_displayFrameCheckbox->getValue());
    }
    else if (item == _displayFansCheckbox)
    {
        _fans->showVisual(_displayFansCheckbox->getValue());
    }
    else if (item == _displayRacksCheckbox)
    {
        for (int r = 0; r < _rack.size(); r++)
            _rack[r]->showVisual(_displayRacksCheckbox->getValue());
    }
    else if (item == _displayComponentTexturesCheckbox)
    {
        Component::_displayTexturesUni->setElement(0,_displayComponentTexturesCheckbox->getValue());
        Component::_displayTexturesUni->dirty();
    }
    else if (item == _loadPowerButton)
    {
        std::string selectedNames = "";

        /***
         * If the checkbox for <pollHistoricalData?> is checked?
         */
        if (_selectionModeCheckbox->getValue())
        {
            int selections = 0;
            std::set<Component *>::iterator sit;
            for (sit = _components.begin(); sit != _components.end(); sit++)
            {
                if ((*sit)->selected)
                {
                    if (selectedNames == "")
                        selectedNames = "&name=";
                    else
                        selectedNames += ",";
                    selectedNames += (*sit)->name;
                    selections++;
                }
            }
            if (_components.size() == selections) // we grabbed all of them
                selectedNames = "";
            else if (selections == 0) // shouldn't poll anything
                selectedNames = "&name=null";

        }

        std::string downloadUrl = cvr::ConfigManager::getEntry("download", "Plugin.GreenLight.Power", "");

        if (_timeFrom != NULL && _timeTo != NULL && _pollHistoricalDataCheckbox->getValue())
        {
            int monF = _monthFrom->getIndex() + 1;
            std::string monthF = (monF < 10 ? "0" : "") + utl::stringFromInt(monF);
            int monT = _monthTo->getIndex() + 1;
            std::string monthT = (monT < 10 ? "0" : "") + utl::stringFromInt(monT);

            downloadUrl += "&from=" + _yearFrom->getValue() + "-" + monthF + "-" + _dayFrom->getValue() + " " +
                                      _hourFrom->getValue() + ":" + _minuteFrom->getValue() + ":00";
            downloadUrl += "&to=" + _yearTo->getValue() + "-" + monthT + "-" + _dayTo->getValue() + " " +
                                     _hourTo->getValue() + ":" + _minuteTo->getValue() + ":00";
        }

        downloadUrl += selectedNames;

        size_t pos;
        while ((pos = downloadUrl.find(' ')) != std::string::npos)
        {
            downloadUrl.replace(pos,1,"%20");
        }

        utl::downloadFile(downloadUrl,
                          cvr::ConfigManager::getEntry("local", "Plugin.GreenLight.Power", ""),
                          _powerContents);

        if (!_displayPowerCheckbox)
        {
            std::ifstream file;
            file.open(cvr::ConfigManager::getEntry("local", "Plugin.GreenLight.Power", "").c_str());
            if (file)
            {
                _displayPowerCheckbox = new cvr::MenuCheckbox("Display Power Consumption",false);
                _displayPowerCheckbox->setCallback(this);
                _powerMenu->addItem(_displayPowerCheckbox);
            }
            file.close();

            if (!_magnifyRangeCheckbox)
            {
                _magnifyRangeCheckbox = new cvr::MenuCheckbox("Magnify Range", false);
                _magnifyRangeCheckbox->setCallback(this);
                _powerMenu->addItem(_magnifyRangeCheckbox);
            }
        }

        if (!_legendText)
        {
            _legendText = new cvr::MenuText("Low    <--Legend-->    High");
            _powerMenu->addItem(_legendText);
        }

        /***
         * Creates the Legend gradient....
         */ 
        if (!_legendGradient)
        {
            osg::ref_ptr<osg::Texture2D> tex = new osg::Texture2D;
            tex->setInternalFormat(GL_RGBA32F_ARB);
            tex->setFilter(osg::Texture::MIN_FILTER,osg::Texture::NEAREST);
            tex->setFilter(osg::Texture::MAG_FILTER,osg::Texture::NEAREST);
            tex->setResizeNonPowerOfTwoHint(false);  

            osg::ref_ptr<osg::Image> data = new osg::Image;
            data->allocateImage(100, 1, 1, GL_RGBA, GL_FLOAT);  

            for (int i = 0; i < 100; i++)
            {
                osg::Vec3 color = wattColor(i+1,1,101);
                for (int j = 0; j < 3; j++)
                {
                    ((float *)data->data(i))[j] = color[j];
                }
                ((float *)data->data(i))[3] = 1;
            }

            data->dirty();
            tex->setImage(data.get());

            _legendGradient = new cvr::MenuImage(tex,450,50);
            _powerMenu->addItem(_legendGradient);
        }

        if (!_legendTextOutOfRange)
        {
            _legendTextOutOfRange = new cvr::MenuText("   Off            |        Standby  ");
            //     | Too Low  | Too High");
            _powerMenu->addItem(_legendTextOutOfRange);
        }

        if (!_legendGradientOutOfRange)
        {
            osg::ref_ptr<osg::Texture2D> tex = new osg::Texture2D;
            tex->setInternalFormat(GL_RGBA32F_ARB);
            tex->setFilter(osg::Texture::MIN_FILTER,osg::Texture::NEAREST);
            tex->setFilter(osg::Texture::MAG_FILTER,osg::Texture::NEAREST);
            tex->setResizeNonPowerOfTwoHint(false);  

            osg::ref_ptr<osg::Image> data = new osg::Image;
            int sections = 2; //3;
            data->allocateImage( sections , 1, 1, GL_RGBA, GL_FLOAT );

            for (int i = 0; i < 2; i++)//Removing "Too High"  // 3; i++)
            {
                osg::Vec3 color = wattColor(i*2,3,3);
                for (int j = 0; j < 3; j++)
                {
                    ((float *)data->data(i))[j] = color[j];
                }
                ((float *)data->data(i))[3] = 1;
            }

            data->dirty();
            tex->setImage(data.get());

            _legendGradientOutOfRange = new cvr::MenuImage(tex,450,50);
            _powerMenu->addItem(_legendGradientOutOfRange);
        }

        if (_displayPowerCheckbox->getValue())
        {
            setPowerColors(true);
        }
    }
    else if (item == _navigateToPluginButton) // Do Navigate to Plugin location on OsgEarth.
    {
        zoom();
    }else if (item == _restorePreviousViewButton)
    {
        restoreView();
    }
    else if (item == _pollHistoricalDataCheckbox)
    {
        if (_timeFrom == NULL && _timeTo == NULL)
            createTimestampMenus();
    }
    else if (item == _displayPowerCheckbox)
    {
        setPowerColors(_displayPowerCheckbox->getValue());
    }
    else if (item == _magnifyRangeCheckbox)
    {
        if (_displayPowerCheckbox->getValue())
            setPowerColors(true);
    }
    else if (item == _selectionModeCheckbox)
    {
        // Toggle the non-selected hardware transparencies
        //Entity * ent;
        std::set< Component * >::iterator sit;
        for (sit = _components.begin(); sit != _components.end(); sit++)
        {
            if (!(*sit)->selected)
                (*sit)->setTransparency(_selectionModeCheckbox->getValue());
        }

        if (_selectionModeCheckbox->getValue())
        {
            if (_selectClusterMenu)
                _hardwareSelectionMenu->addItem(_selectClusterMenu);
            _hardwareSelectionMenu->addItem(_selectAllButton);
            _hardwareSelectionMenu->addItem(_deselectAllButton);
        }
        else
        {
            if (_selectClusterMenu)
                _hardwareSelectionMenu->removeItem(_selectClusterMenu);
            _hardwareSelectionMenu->removeItem(_selectAllButton);
            _hardwareSelectionMenu->removeItem(_deselectAllButton);
        }

        _hoverDialog->setVisible(_selectionModeCheckbox->getValue());
    }
    else if (item == _selectAllButton || item == _deselectAllButton)
    {
        std::set< Component * >::iterator sit;
        for (sit = _components.begin(); sit != _components.end(); sit++)
            selectComponent(*sit, item == _selectAllButton);
    }
    else if ((chit = _clusterCheckbox.find(dynamic_cast<cvr::MenuCheckbox *>(item))) != _clusterCheckbox.end())
    {
        cvr::MenuCheckbox * checkbox = *chit;

        std::map< std::string, std::set< Component * > * >::iterator cit = _cluster.find(checkbox->getText());
        if (cit == _cluster.end())
        {
            std::cerr << "Error: Cluster checkbox selected without a matching cluster (" <<
                 checkbox->getText() << ")" << std::endl;
            checkbox->setValue(checkbox->getValue());
            return;
        }

        std::set< Component * > * cluster = cit->second;
        selectCluster(cluster, checkbox->getValue());
    }
    else if (item == _yearFrom || item == _monthFrom || item == _dayFrom ||
             item == _hourFrom || item == _minuteFrom)
    {
        if (item == _monthFrom || item == _dayFrom)
        {
            int day = _dayFrom->getIndex() + 1; // +1 offsets indexing from 0
            if (day > 28)
            {
                int month = _monthFrom->getIndex();
                if (month == 1)
                {
                    if (month % 4 == 0 && (month % 100 != 0 || month % 400 == 0))
                        _dayFrom->setIndex(28); // 29th
                    else
                        _dayFrom->setIndex(27); // 28th
                }
                else if ((month % 2 == 0) != (month < 7) && day == 31)
                    _dayFrom->setIndex(29); // 30th
            }
        }

        if (_yearFrom->getIndex() > _yearTo->getIndex() || (_yearFrom->getIndex() == _yearTo->getIndex() &&
            (_monthFrom->getIndex() > _monthTo->getIndex() || (_monthFrom->getIndex() == _monthTo->getIndex() &&
            (_dayFrom->getIndex() > _dayTo->getIndex() || (_dayFrom->getIndex() == _dayTo->getIndex() &&
            (_hourFrom->getIndex() > _hourTo->getIndex() || (_hourFrom->getIndex() == _hourTo->getIndex() &&
            (_minuteFrom->getIndex() > _minuteTo->getIndex() || (_minuteFrom->getIndex() == _minuteTo->getIndex()
           ))))))))))
        {
            _yearTo->setIndex(_yearFrom->getIndex());
            _monthTo->setIndex(_monthFrom->getIndex());
            _dayTo->setIndex(_dayFrom->getIndex());
            _hourTo->setIndex(_hourFrom->getIndex());
            _minuteTo->setIndex(_minuteFrom->getIndex());
        }
    }
    else if (item == _yearTo || item == _monthTo || item == _dayTo ||
             item == _hourTo || item == _minuteTo)
    {
        if (item == _monthTo || item == _dayTo)
        {
            int day = _dayTo->getIndex() + 1; // +1 offsets indexing from 0
            if (day > 28)
            {
                int month = _monthTo->getIndex();
                if (month == 1)
                {
                    if (month % 4 == 0 && (month % 100 != 0 || month % 400 == 0))
                        _dayTo->setIndex(28); // 29th
                    else
                        _dayTo->setIndex(27); // 28th
                }
                else if ((month % 2 == 0) != (month < 7) && day == 31)
                    _dayTo->setIndex(29); // 30th
            }
        }

        if (_yearTo->getIndex() < _yearFrom->getIndex() || (_yearTo->getIndex() == _yearFrom->getIndex() &&
            (_monthTo->getIndex() < _monthFrom->getIndex() || (_monthTo->getIndex() == _monthFrom->getIndex() &&
            (_dayTo->getIndex() < _dayFrom->getIndex() || (_dayTo->getIndex() == _dayFrom->getIndex() &&
            (_hourTo->getIndex() < _hourFrom->getIndex() || (_hourTo->getIndex() == _hourFrom->getIndex() &&
            (_minuteTo->getIndex() < _minuteFrom->getIndex() || (_minuteTo->getIndex() == _minuteFrom->getIndex()
           ))))))))))
        {
            _yearFrom->setIndex(_yearTo->getIndex());
            _monthFrom->setIndex(_monthTo->getIndex());
            _dayFrom->setIndex(_dayTo->getIndex());
            _hourFrom->setIndex(_hourTo->getIndex());
            _minuteFrom->setIndex(_minuteTo->getIndex());
        }
    }
}

void GreenLight::preFrame()
{
    /***
     * TODO: Add LOD Call-back support.
     * within reach? Add two more call backs for out of sight and in sight.
     */

    if( true ){
    }else if(true){
    }else{
    }

    // update mouse and wand intersection with components
    if (_box)
    {
        // continue animations
        for (int d = 0; d < _door.size(); d++)
            _door[d]->handleAnimation();
        for (int r = 0; r < _rack.size(); r++)
            _rack[r]->handleAnimation();


//        if (cvr::ComController::instance()->isMaster())
        if( lodLevel == 0 ){
            handleHoverOver(cvr::PluginHelper::getMouseMat(), _mouseOver, cvr::ComController::instance()->isMaster());
//        else
            handleHoverOver(cvr::PluginHelper::getHandMat(), _wandOver, !cvr::ComController::instance()->isMaster());
        }
    }


}

void GreenLight::postFrame()
{
}


bool GreenLight::keyboardEvent(int key , int type )
{

    char c = key;

    if ( (osgEarthInit && _showSceneCheckbox->getValue()) || c == 'l' ){
      if( type == KEY_DOWN )
      {   
          osg::Vec3d v3d = scaleMT->getMatrix().getScale();
          if ( c == 'r' ) // revert to location on earth.
          {
              OsgE_MT->setMatrix( output );
          }else if( c == 'i' )  // set to Identity
          {
              osg::Matrixd identity = Matrixd();
              OsgE_MT->setMatrix( identity ); // reset position to middle.
              BoundingSphere bs = _box->transform->getBound();
              printf("Radius of the Bounding Sphere of the Box is: %f\n", bs.radius());

          }else if( c == 'g' )  // scaleUp
          {
//            v3d *= 1.5; //2;
              range *= 8;
              _glLOD->setRange(0, 0, range);
              _glLOD->setRange(1, range, range * 4);
              _glLOD->setRange(2, range * 4, range * 1024);
              printf("range is: %f \n", range);
          }else if( c == 's' )  // scaleDown
          {
//            v3d *= .75; //.5;
              range /= 8;
              _glLOD->setRange(0, 0, range);
              _glLOD->setRange(1, range, range * 4);
              _glLOD->setRange(2, range * 4, range * 1024);
              printf("range is: %f \n", range);
          }else if( c == 'l' ) // log.
          {
              printf("Current Matrix Config: \n");
              const MatrixTransform * mmt = SceneManager::instance()->getObjectTransform();
              Matrixd mmd = mmt->getMatrix();
              for ( int i = 0; i < 4; i++ )
              {
                for ( int j = 0; j < 4; j++ )
                {
                  printf("\t %f", mmd(i,j) );

                  if( !( i == 3 && j == 3 ) )
                      printf(",");
                }
                cout<<endl;
              }
              printf("Scale is: %f\n",SceneManager::instance()->getObjectScale());

              printf( "Scale of BlackBox is currently : x%f\n ", v3d.x() );
                
              printf( "LOD level: %d\n", lodLevel ) ;
 
          }else if( c == 'x' ){
            zoom();
          }else // do nothing.
          {
          }
          if( osgEarthInit ){
            scaleMatrix->makeScale( v3d );
            scaleMT->setMatrix( *scaleMatrix );
          }
        }
        return true;
      }
    return false;
}

/***
 * Will this get used at all?
 */
void GreenLight::closeMenu()
{
    if(_myMenu)
    {
        _myMenu->setVisible(false);
    }
}

bool GreenLight::processEvent(cvr::InteractionEvent * event)
{
    
    cvr::InteractionEvent * ie = event->asTrackedButtonEvent();
    if(!ie)
    { // If it is not a TrackedButton Event Check if it is a keyboardEvent.
        cvr::InteractionEvent * ie = event->asKeyboardEvent();
        if(!ie)
        	return false;
        else
        {
            KeyboardInteractionEvent * kie = (KeyboardInteractionEvent *) ie;
            return ( keyboardEvent( kie->getKey() , kie->getInteraction() ) );
        }
    }

    if ( lodLevel == 0 )
    {
    // cast to TrackedButtonEvent.
    cvr::TrackedButtonInteractionEvent * tie = (TrackedButtonInteractionEvent *) ie;

    
      if (SOCM)
      {
        if ( tie->getButton() == _menuButton )
        {
        if(tie->getInteraction() == BUTTON_DOWN)
        {
            if( !_myMenu->isVisible() )
            {
              /***
               * TODO: Look into incorporating the check, to make it so menu
               *       only shows up when hover over a component box.
               */
                if (_wandOver){
                    Component * comp = _wandOver->asComponent();
                    if (comp)
                    {
//                      selectComponent( comp, !comp->selected );
                        cout << "on a component." << endl;
                    } else
                    {
                        cout << "on a rack/door/other." << endl;
                    }
                }
              
                _myMenu->setVisible(true);
            
                osg::Vec3 start(0,0,0), end(0,1000,0);
                start = start * tie->getTransform();
                end = end * tie->getTransform();
                
                // Insert PositionInformation stuff here?
                Vec3 menuPoint(0, 1000.0, 0);
                menuPoint = menuPoint * tie->getTransform();

                Matrix m;
                m.makeTranslate(menuPoint);
                _myMenu->setTransform( m );
    
                _myMenu->setScale( sma.grabProtectedMember2() );
//                                SceneManager::instance()->_menuScale);
//              SceneManager::instance()->setMenuOpenObject(this);  //#
            }else
            {
                _myMenu->setVisible(false); // #
//              SceneManager::instance()->closeOpenObjectMenu();  //#
            }
          }
        }
      }


      // ???
      if (tie->getInteraction() != cvr::BUTTON_DOWN || tie->getButton() != 0 || tie->getHand() != 0 )
        return false;

      // if box is still not loaded?
      if (!_box)
        return false;

      // Should be hovering over it
      if (_wandOver)
      {
        Component * comp = _wandOver->asComponent();
        if (comp)
        {
            cout << "on a component." << endl;
            selectComponent( comp, !comp->selected );
        }
        else // _wandOver is a rack/door/etc.
        {
            _wandOver->beginAnimation();

            // Handle group animations
            std::list<Entity *>::iterator eit;
            for (eit = _wandOver->group.begin(); eit != _wandOver->group.end(); eit++)
            {
                (*eit)->beginAnimation();
            }
        }

        return true;
      }
    }
    return false;
}

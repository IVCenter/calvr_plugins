#include "OssimPlanet.h"

using namespace cvr;
using namespace osg;

ossimPlanet* OssimPlanet::planet;
OssimPlanet* OssimPlanet::oplanet;

const double earthRadiusMM = osg::WGS_84_RADIUS_EQUATOR * 1000.0;
static const string FILES("Plugin.OssimPlanet.Kml");

// Constructor
OssimPlanet::OssimPlanet()
{

}

osg::Vec3d OssimPlanet::convertUtm84ToLatLon(int zone, char hemi, double northing, double easting)
{
     // translate to lon lat
     ossimGpt result;
     //ossimUtmpt test(36, 'N', 733463.783, 3396594.585, ossimDatumFactory::instance()->wgs84());
     ossimUtmpt point(zone, hemi, northing, easting, ossimDatumFactory::instance()->wgs84());
     point.convertToGround(result);
     return osg::Vec3d(result.lond(), result.latd(), result.heightMSL());
}

bool OssimPlanet::addModel(osg::Node* node, double lat, double lon, osg::Vec3 scale, double height, double heading, double pitch, double roll)
{
   // will only execute if the plugin has been loaded
   if(node == NULL || planet == NULL)
       return false;

   osg::Vec3d llh(lat,lon,height);
   llh[2] += planet->model()->getGeoidOffset(lat,
                                          lon);
   osg::Matrixd localToWorld;
   planet->model()->lsrMatrix(llh, localToWorld);

   // add similar as a kml file
   osg::Vec3d tempLlh(lat,
                      lon,
                      height + planet->model()->getHeightAboveEllipsoid(lat,
                                                   lon));

   osg::MatrixTransform* localToWorldTransform = new osg::MatrixTransform();

   planet->model()->lsrMatrix(tempLlh, localToWorld, heading);
   localToWorld = (osg::Matrixd::scale(scale)*
                   osg::Matrixd::rotate(pitch, osg::Vec3d(1.0, 0.0, 0.0))*
                   osg::Matrixd::rotate(roll, osg::Vec3d(0.0, 1.0, 0.0)))*localToWorld;
   localToWorldTransform->setMatrix(localToWorld);

   osg::MatrixTransform* modelTransform = new osg::MatrixTransform;
   double scale2 = (1.0/planet->model()->getNormalizationScale());
   modelTransform->setMatrix(osg::Matrixd::scale((scale2),
                                                 (scale2),
                                                 (scale2)));
   modelTransform->addChild(node);
   modelTransform->getOrCreateStateSet()->setMode(GL_RESCALE_NORMAL,osg::StateAttribute::ON);

   localToWorldTransform->addChild(modelTransform);
   ossimSetNonPowerOfTwoTextureVisitor nv;
   localToWorldTransform->accept(nv);
   planet->addChild(localToWorldTransform);

   return true;
}

OssimPlanet* OssimPlanet::instance()
{
    return OssimPlanet::oplanet;
}

// intialize
bool OssimPlanet::init()
{

   SceneManager::instance()->setDepthPartitionActive(true);

   // static setting
   if(!OssimPlanet::oplanet)
       oplanet = new OssimPlanet();

   ossimInit::instance()->initialize();
   //ossimPreferences::instance()->loadPreferences(ossimFilename("/home/covise/.ossim/ossim_preferences"));
   osg::ref_ptr<ossimPlanetGrid> grid = new ossimPlanetAdjustableCubeGrid(ossimPlanetAdjustableCubeGrid::MEDIUM_CAP);

   ossimKeywordlist kwl;
   kwl.addFile(ConfigManager::getEntry("Plugin.OssimPlanet.ConfigFile").c_str());
   osg::ref_ptr<ossimPlanetTextureLayer> groupLayer = ossimPlanetTextureLayerRegistry::instance()->createLayer(kwl.toString());

   ossimPlanetTerrainGeometryTechnique* technique = new ossimPlanetTerrainGeometryTechnique();

   planet = new ossimPlanet();

   ossimPlanetTerrain::CullAmountType cullAmount = ossimPlanetTerrain::HIGH_CULL;
   ossimPlanetTerrain::SplitMergeSpeedType splitMergeSpeed = ossimPlanetTerrain::MEDIUM_SPEED;
   ossimPlanetTerrain::ElevationDensityType elevationDensity = ossimPlanetTerrain::MEDIUM_ELEVATION_DENSITY;
   ossimPlanetTerrain::TextureDensityType textureDensity = ossimPlanetTerrain::MEDIUM_TEXTURE_DENSITY;
   double minTimeToCompilePerFrame = .003;

   osg::ref_ptr<ossimPlanetTerrain> terrain = new ossimPlanetTerrain(grid.get());
   terrain->setPrecompileEnabledFlag(false);
   terrain->setTerrainTechnique(technique);
   terrain->setCullAmountType(cullAmount);
   terrain->setSplitMergeSpeedType(splitMergeSpeed);
   terrain->setTextureDensityType(textureDensity);
   terrain->setElevationDensityType(elevationDensity);
   terrain->setElevationExaggeration(1.0);
   terrain->setMinimumTimeToCompilePerFrameInSeconds(minTimeToCompilePerFrame);
   terrain->setElevationMemoryCache(new ossimPlanetMemoryImageCache);
   terrain->elevationCache()->setMinMaxCacheSizeInMegaBytes(128, 256);

   // add texture  
   if( groupLayer.valid() )
   {
      terrain->setTextureLayer(0, groupLayer.get());
   }

   terrain->initElevation();
   terrain->setElevationEnabledFlag(true);

   std::vector<ossimFilename> kmlFiles;
   osg::ref_ptr<ossimPlanetKmlLayer> kmlLayer;


   // read in kml from configuration file
   std::vector<std::string> tagList;
   ConfigManager::getChildren(FILES, tagList);

   for(int i = 0; i < tagList.size(); i++)
   {
        std::string tag = FILES + "." + tagList[i];
	kmlFiles.push_back(ossimFilename(ConfigManager::getEntry("value", tag, "")));
   }

   if(kmlFiles.size() > 0)
   {
      kmlLayer = new ossimPlanetKmlLayer();
      planet->addChild(kmlLayer.get());
      ossim_uint32 idx = 0;
      for(idx = 0; idx < kmlFiles.size(); ++idx)
      {
         kmlLayer->addKml(kmlFiles[idx]);
      }
   }
   
      
   // set ephemeris
   ephemeris = NULL;
   
   //set object transform and ignore root transform
   osg::Matrix objects = PluginHelper::getObjectMatrix();
   objects.setTrans(0.0, earthRadiusMM * 3.0, 0.0);  // scales the planet
   PluginHelper::setObjectMatrix(objects);    // moves the planet; matrix should be a pure translation and rotation
   PluginHelper::setObjectScale(earthRadiusMM); // this matrix should be a pure scale
   
   planet->addChild(terrain.get());
   
   // enable cloud cover
   cloudCover();

   //osg::Group *models = new osg::Group();
   //readArtifactsFile(ConfigManager::getEntry("Plugin.ArtifactVis.DataBase"));
   //displayArtifacts(models);
   //addModel(models, 30.628039, 35.491239, osg::Vec3(0.06, 0.06, 0.06), 20.0, 0.0, 90.0, 135.03); 
   //addModel(models, 30.628039, 35.491239, osg::Vec3(0.06, 0.06, 0.06), 10.0, 0.0, 90.0, 135.0); 

   SceneManager::instance()->getObjectsRoot()->addChild(planet);

   return true;
}

void OssimPlanet::getObjectIntersection(osg::Node *root, osg::Vec3& wPointerStart, osg::Vec3& wPointerEnd, IsectInfo& isect)
{
        // Compute intersections of viewing ray with objects:
	osgUtil::IntersectVisitor iv;
	osg::ref_ptr<osg::LineSegment> testSegment = new osg::LineSegment();
	testSegment->set(wPointerStart, wPointerEnd);
	iv.addLineSegment(testSegment.get());
	iv.setTraversalMask(2);
	
	// Traverse the whole scenegraph.
	// Non-Interactive objects must have been marked with setNodeMask(~2):     
	root->accept(iv);
	isect.found = false;
	if (iv.hits())
	{
	    osgUtil::IntersectVisitor::HitList& hitList = iv.getHitList(testSegment.get());
	    if(!hitList.empty())
	    {
	         isect.point     = hitList.front().getWorldIntersectPoint();
	         isect.normal    = hitList.front().getWorldIntersectNormal();
	         isect.geode     = hitList.front()._geode.get();
	         isect.found     = true;

		 //osg::Vec3d llh;
		 //osg::Vec3d wpt = isect.point * PluginHelper::getWorldToObjectTransform();
                 //planet->model()->inverse(wpt, llh);
		 //printf("Long lat and height is %f %f %f\n", llh.x(), llh.y(), llh.z());
	    }
	}
}

void OssimPlanet::cloudCover()
{

   osg::Camera* theEphemerisCamera = NULL;
 
   ossimFilename sunTextureFile = "";
   ossimFilename moonTextureFile = "";
   //ossim_float64 visibility = 1000000000.0;
   ossim_float64 visibility = 1000.0;
   ossim_float64 fogNear = 10.0;
   ossim_int32 cloudCoverage = 20;
   ossim_float64 cloudSharpness = .95;
   ossim_float64 cloudAltitude = 20000;

   ossimLocalTm date;
   date.setYear(2011);
   date.setMonth(1);
   date.setDay(1);
   date.setHour(3);
   //date.now();

   ossimPlanet* tempPlanet = new ossimPlanet;
   tempPlanet->setComputeIntersectionFlag(false);

   ephemeris = new ossimPlanetEphemeris(ossimPlanetEphemeris::MOON_LIGHT | ossimPlanetEphemeris::SKY | ossimPlanetEphemeris::AMBIENT_LIGHT | ossimPlanetEphemeris::SUN_LIGHT | ossimPlanetEphemeris::FOG);
   theEphemerisCamera = new osg::Camera;

   tempPlanet->addChild(ephemeris);
   SceneManager::instance()->getObjectsRoot()->addChild(tempPlanet);

   ephemeris->setRoot(SceneManager::instance()->getObjectsRoot());

   theEphemerisCamera->setProjectionResizePolicy(CVRViewer::instance()->getCamera()->getProjectionResizePolicy());
   theEphemerisCamera->setClearColor(CVRViewer::instance()->getCamera()->getClearColor());
   theEphemerisCamera->setRenderOrder(osg::Camera::PRE_RENDER);
   theEphemerisCamera->setRenderTargetImplementation( CVRViewer::instance()->getCamera()->getRenderTargetImplementation() );
   theEphemerisCamera->setClearMask(GL_COLOR_BUFFER_BIT);

   CVRViewer::instance()->getCamera()->setClearMask(CVRViewer::instance()->getCamera()->getClearMask() & ~GL_COLOR_BUFFER_BIT);

   if(CVRViewer::instance()->getCamera()->getViewport())
   {
         theEphemerisCamera->setViewport(new osg::Viewport(*CVRViewer::instance()->getCamera()->getViewport()));
   }
   else
   {
         theEphemerisCamera->setViewport(new osg::Viewport());
   }

   osg::Matrix theCurrentViewMatrix = CVRViewer::instance()->getCamera()->getViewMatrix() * PluginHelper::getWorldToObjectTransform();
   osg::Matrix theCurrentViewMatrixInverse = theCurrentViewMatrix.inverse(theCurrentViewMatrix);
   theEphemerisCamera->setViewMatrix(theCurrentViewMatrixInverse);

   CVRViewer::instance()->addSlave(theEphemerisCamera, false);
   ephemeris->setCamera(theEphemerisCamera);
   //theEphemerisCamera->setEventCallback(new ossimPlanetTraverseCallback());
   //theEphemerisCamera->setUpdateCallback(new ossimPlanetTraverseCallback());
   //theEphemerisCamera->setCullCallback(new ossimPlanetTraverseCallback());

   ephemeris->setDate(date);
   ephemeris->setApplySimulationTimeOffsetFlag(true);
   ephemeris->setSunTextureFromFile(sunTextureFile);
   ephemeris->setMoonTextureFromFile(moonTextureFile);
   ephemeris->setMoonScale(osg::Vec3d(1.0, 1.0, 1.0));
   ephemeris->setGlobalAmbientLight(osg::Vec3d(0.1, 0.1, 0.1));
   ephemeris->setVisibility(visibility);
   ephemeris->setFogNear(fogNear);
   ephemeris->setFogMode(ossimPlanetEphemeris::LINEAR);

   ephemeris->setNumberOfCloudLayers(1);
   ephemeris->cloudLayer(0)->computeMesh(cloudAltitude, 128, 128, 0);
   ephemeris->cloudLayer(0)->updateTexture(0, cloudCoverage, cloudSharpness);
   //ephemeris->cloudLayer(0)->setSpeedPerHour(1000, OSSIM_MILES);
   ephemeris->cloudLayer(0)->setSpeedPerHour(10000, OSSIM_MILES);
   ephemeris->cloudLayer(0)->setScale(3);
   ephemeris->cloudLayer(0)->setMaxAltitudeToShowClouds(cloudAltitude*2.0);
}

// this is called if the plugin is removed at runtime
OssimPlanet::~OssimPlanet()
{
   fprintf(stderr,"OssimPlanet::~OssimPlanet\n");
}


void OssimPlanet::preFrame()
{
/*
    osg::Matrix theCurrentViewMatrix = CVRViewer::instance()->getCamera()->getViewMatrix() * PluginHelper::getWorldToObjectTransform() * planetRescale;
    osg::Matrix theCurrentViewMatrixInverse = theCurrentViewMatrix.inverse(theCurrentViewMatrix);
    theEphemerisCamera->setViewMatrix(theCurrentViewMatrixInverse);
*/

    osg::Matrix originToPlanetObjectSpace = PluginHelper::getWorldToObjectTransform();
    // distance to surface in planet units (default without using intersection with surface)
    double distanceToSurface = originToPlanetObjectSpace.getTrans().length() - 1.0;
    
    IsectInfo itest;
    osg::Vec3 originObject(0.0,0.0,0.0);
    osg::Vec3 endObject = PluginHelper::getObjectToWorldTransform().getTrans();
    getObjectIntersection(SceneManager::instance()->getScene(), originObject, endObject, itest);

    if(itest.found)
    {
	if(ComController::instance()->isMaster())
	{
	    distanceToSurface = (originToPlanetObjectSpace.getTrans() - (itest.point * PluginHelper::getWorldToObjectTransform())).length(); 
	    ComController::instance()->sendSlaves(&distanceToSurface,sizeof(double));
	}
	else
	{
	    ComController::instance()->readMaster(&distanceToSurface,sizeof(double));
	}
    }

    double minNavScale = 20.0;
    double maxNavScale = 2000000.0;
    double minDistance = 0.0;   
    double maxDistance = 0.3;  //ratio is multiplied by the earth radius
    double ratio =  (distanceToSurface - minDistance) / (maxDistance - minDistance);

    if(distanceToSurface <= minDistance)
    {
        ratio = 0.0;
    }
    else if(distanceToSurface >= maxDistance)
    {
        ratio = 1.0;
    }

    double scaleFactor = minNavScale + (ratio * (maxNavScale - minNavScale));
    Navigation::instance()->setScale(scaleFactor);
}
CVRPLUGIN(OssimPlanet)

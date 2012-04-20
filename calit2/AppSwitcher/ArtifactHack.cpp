#include "ArtifactHack.h"
#include "PointCloud.h"
#include "Plane.h"

#include <cstdio>
#include <cmath>
#include <string>
using std::printf;
using std::fabs;
using std::fmod;
using std::sqrt;
using std::string;

#include "planesite.h"

#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/LOD>
#include <osg/Quat>
#include <osg/PositionAttitudeTransform>
using namespace osg;

#include <cvrConfig/ConfigManager.h>

#include <osgEarth/Map>
#include <osgEarth/MapNode>

#include <cvrUtil/LocalToWorldVisitor.h>
#include <cvrKernel/SceneManager.h>
#include <cvrMenu/SubMenu.h>
using namespace cvr;

CVRPLUGIN(ArtifactHack)

extern Flock the_flock;
extern double fly_near;
extern double fly_far;

static string cloudpath;

static Node* cloud; 
static ref_ptr<MatrixTransform> on_the_earth;
static ref_ptr<MatrixTransform> cloud_scale;
static ref_ptr<MatrixTransform> rotation;
static ref_ptr<LOD> lod; 
static osgEarth::Map* map;

static double digscale;
static double dignear, digfar;

static double lat = 32.88211;
static double lon = -117.234271;
static double height = 10000.0;

/* // san diego, generic
static double lat = 32.73;
static double lon = -117.17;
static double height = 10000.0;
*/

/*
// jordan
static double lat = 30.628039;
static double lon = 35.419239;
static double height = 0.0;
*/

static double dest_lat = 32.73;
static double dest_lon = 0.0;
static double dest_height = 0.0;

double rotX = 0.0;
double rotY = 0.0;
double rotZ = 0.0;

void recompute_matrix()
{
    Matrixd a, b, c;
    a.makeRotate(rotX, Vec3(1,0,0));
    b.makeRotate(rotY, Vec3(0,1,0));
    c.makeRotate(rotZ, Vec3(0,0,1));

    rotation->setMatrix(b*a*c);

    
    Matrixd scaleMat;
    scaleMat.makeScale(digscale, digscale, digscale);
    cloud_scale->setMatrix(scaleMat);

    double convert = 1000000/(2*digscale);

    lod->setRange(0, 0, dignear * convert);
    lod->setRange(1, dignear * convert, digfar * convert);
    lod->setRange(2, digfar * convert, FLT_MAX);

}


osg::Matrixd earthMove(MatrixTransform* mat,
                       double lat, double lon, double height)
{
    osg::Matrixd output;

    //printf("Map = %p\n", ::map);

    double x, y, z;

    map->getProfile() -> getSRS()
                      -> getEllipsoid()
                      -> computeLocalToWorldTransformFromLatLongHeight(
                            osg::DegreesToRadians(lat),
                            osg::DegreesToRadians(lon),
                            height,
                            output);
    mat->setMatrix(output);
    return output;
}

double distanceTo(Node* node)
{
    Vec3d it = getLocalToWorldMatrix(node).getTrans();
    Vec3d pointer = PluginHelper::getHandMat().getTrans();
    
    Vec3d d = it - pointer;

    double dx2 = d.x() * d.x();
    double dy2 = d.y() * d.y();
    double dz2 = d.z() * d.z();

    return sqrt(dx2 + dy2 + dz2);
}

//---------------------------------------------------------------------
// Menu Code
//---------------------------------------------------------------------

DistanceMenu::DistanceMenu(string name, double distance) : SubMenu(name, name)
{
    _partial_name = name;
    setDistance(distance);
}

DistanceMenu::~DistanceMenu() {}

void DistanceMenu::setDistance(double distance)
{
    this->distance = distance;
    //printf("distance = %.3f\n", distance);
    char tmp[100];
    snprintf(tmp, 100, "%.3f km", distance);
    _name = _partial_name + string(tmp);
    setDirty(true);
}

//---------------------------------------------------------------------

void ArtifactHack::update_rangemenu()
{
    double flydist = distanceTo(the_flock.getMeasureNode()) / 1000000.0;
    if(flydist < fly_near)
    {
        for(int i = 0; i < menu_root->getNumChildren(); i++)
            if(menu_root->getChild(i) == menu_flyerrange)
                break;
        // need to add it
        menu_root->addItem(menu_flyerrange);
    }
    else if(flydist > fly_near + 100)
    {
        menu_root->removeItem(menu_flyerrange);
    }
}

void ArtifactHack::makeMenu()
{
    menu_root = new cvr::SubMenu("AppSwitcher","AppSwitcher");
    PluginHelper::addRootMenuItem(menu_root);

    menu_artifactText= new MenuText("Test...");
    menu_root->addItem(menu_artifactText);

    menu_artifact = new DistanceMenu("Artifact", 0);
    //menu_root->addItem(menu_artifact);
    
    menu_flyerText = new MenuText("...");
    menu_root->addItem(menu_flyerText);

    menu_flyerrange = new MenuRangeValue("Flyer Speed", 0, 1, 0.1, 0.1);
    
    update_rangemenu();

}


ArtifactHack::~ArtifactHack()
{
    // nothing special to do, for now
}

void loadConfig()
{
    string config_rotx = ConfigManager::getEntry("rotx", "Plugin.AppSwitcher.digsite","0"); 
    string config_roty = ConfigManager::getEntry("roty", "Plugin.AppSwitcher.digsite","0");
    string config_rotz = ConfigManager::getEntry("rotz", "Plugin.AppSwitcher.digsite","0");

    string config_lat    = ConfigManager::getEntry("lat", "Plugin.AppSwitcher.digsite", "0");
    string config_lon    = ConfigManager::getEntry("lon", "Plugin.AppSwitcher.digsite", "0");
    string config_height = ConfigManager::getEntry("height", "Plugin.AppSwitcher.digsite", "0");

    string config_scale = ConfigManager::getEntry("scale", "Plugin.AppSwitcher.digsite","0");

    rotX = atof(config_rotx.c_str());
    rotY = atof(config_roty.c_str());
    rotZ = atof(config_rotz.c_str());

    lat    = atof(config_lat.c_str());
    lon    = atof(config_lon.c_str());
    height = atof(config_height.c_str());

    digscale = atof(config_scale.c_str());



    string config_dignear = ConfigManager::getEntry("threshNear", "Plugin.AppSwitcher.digsite", "0");
    string config_digfar  = ConfigManager::getEntry("threshFar" , "Plugin.AppSwitcher.digsite", "0");
    dignear = atof(config_dignear.c_str());
    digfar  = atof(config_digfar.c_str());

    cloudpath = ConfigManager::getEntry("value", "Plugin.AppSwitcher.cloudpath", "");
}

bool ArtifactHack::init()
{
    printf("AppSwitcher  init...\n");


    makeMenu();


    on_the_earth = new MatrixTransform();
    cloud_scale  = new MatrixTransform();
    lod = new LOD();
    rotation = new MatrixTransform();

    Matrixd scaleMat;
    //float scale = 100000;
    float scale = digscale;
    scaleMat.makeScale(scale, scale, scale);
    cloud_scale->setMatrix(scaleMat);
    on_the_earth->addChild(rotation);
    rotation->addChild(cloud_scale);


    loadConfig();

    osgEarth::MapNode* mapNode = osgEarth::MapNode::findMapNode( 
                SceneManager::instance()->getObjectsRoot() 
    );

    
    if(mapNode)
    {
        map = mapNode->getMap();


        printf("Map = %p\n", map);

        /*
        double lat = 32.73;
        double lon = -117.17;
        double height = 0.0;
        */

        /*        
        osg::Matrixd output;


        map->getProfile() -> getSRS()
                          -> getEllipsoid()
                          -> computeLocalToWorldTransformFromLatLongHeight(
                                osg::DegreesToRadians(lat),
                                osg::DegreesToRadians(lon),
                                height,
                                output);
        on_the_earth->setMatrix(output);
        */
        
        double torad = 3.1415926535 / 180.0;

        //Quat q = Quat(0,0,1, -46) * Quat(1,0,0, 21) * Quat(0,1,0, -113);
        //Quat q = Quat(0,0,1, -46 * torad) * Quat(1,0,0, 21 * torad) * Quat(0,1,0, -113 * torad);
        
        /*
        Matrix roll, tilt, heading;
        roll.makeRotate( -46 * torad, Vec3(0,1,0));
        tilt.makeRotate(  21 * torad, Vec3(1,0,0));
        heading.makeRotate(-113 * torad, Vec3(0,0,1));

        //m.setRotate(q);
        rotation->setMatrix(roll * tilt * heading);
        */

        /*
            rotX = 1.400;
            rotY = -0.200;
            rotZ = -3.400;
            recompute_matrix();
        */


        earthMove(on_the_earth, lat, lon, height);

    }
    else
    {
        printf("Could not access map node!\n");
    }

    printf("Loading point cloud (this make take a few moments...)\n");
    //cloud = readPointCloud("/home/cmcfarla/CalVR-calvr_plugins-23ee176/calit2/ArtifactVis/ArchInterface/Model/KIS2.ply"); 
    cloud = readPointCloud(cloudpath.c_str());
    //cloud = makePlane();

    if(!cloud)
        fprintf(stderr, "Failed to load point cloud geometry.\n");
    else
        printf("Point cloud load complete.\n");

    if(cloud && map)
    {
        cloud_scale->addChild(cloud);
        lod->addChild(on_the_earth, 0, 1);
        SceneManager::instance()
            -> getObjectsRoot()
            -> addChild(lod);
    }

    flyer_start();
    recompute_matrix();
    return true;
}

/*
bool edgeCheckUpdate(int* prev, int new_val)
{
    bool flag = false;
    if(prev != new_val)
        flag = true;

    *prev = new_val;
    return flag;
}
*/


void ArtifactHack::preFrame()
{
    static double theta = 0.0;

    update_rangemenu();
    the_flock.modify_speed(menu_flyerrange->getValue());

    //printf("%.3f kilometers away\n", distanceTo(cloud)/1000000.0);
   
    flyer_step();

    char tmp[100];
    const char* category = "";
    double digdist = distanceTo(cloud) / 1000000.0;
    if(digdist < dignear)
        category = "(near)";
    else if(digdist < digfar)
        category = "(medium)";
    else
        category = "(far)";


    snprintf(tmp, 100, "Dig Site: %.3f km %s", digdist, category);
    menu_artifactText->setText(string(tmp));
    
    
    menu_artifact->setDistance(distanceTo(cloud)/1000000.0);




// ----------------------------
    category = "";
    double flydist = distanceTo(the_flock.getMeasureNode()) / 1000000.0;
    if(flydist < fly_near)
        category = "(near)";
    else if(flydist < fly_far)
        category = "(medium)";
    else
        category = "(far)";

    snprintf(tmp, 100, "Planes: %.3f km %s", flydist, category);
    menu_flyerText->setText(string(tmp));

// ----------------------------



    /*
    Vec3f dest(dest_lat, dest_lon, dest_height);
    Vec3f here(lat, lon, height);

    Vec3f delta = dest - here;
    delta.normalize();

    delta *= 0.1;

    lat += delta.x();
    lon += delta.y();
    height += delta.z();

    earthMove(on_the_earth, lat, lon, height);


    theta += 0.1;

    Matrixd mat;
    mat.makeRotate(theta, Vec3f(0,0,1));
    */
    //rotation->setMatrix(mat);
    
}

bool ArtifactHack::processEvent(cvr::InteractionEvent* event)
{
    KeyboardInteractionEvent* kie = event->asKeyboardEvent();
    if(kie)
    {
        switch(kie->getKey())
        {
            case 'a':
            case 'A':
                rotX -= 0.1;
                break;

            case 'd':
            case 'D':
                rotX += 0.1;
                break;

            case 'w':
            case 'W':
                rotY += 0.1;
                break;
            case 's':
            case 'S':
                rotY -= 0.1;
                break;

            case 'q':
            case 'Q':
                rotZ -= 0.1;
                break;
            case 'e':
            case 'E':
                rotZ += 0.1;
                break;

            case 't':
            case 'T':
                printf("rotX = %.3f\n", rotX);
                printf("rotY = %.3f\n", rotY);
                printf("rotZ = %.3f\n", rotZ);
                break;

        }

        recompute_matrix();
        return true;
    }

    return false;
}





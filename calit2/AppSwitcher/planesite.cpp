#include "planesite.h"
#include "Plane.h"

#include <cvrKernel/SceneManager.h>
#include <cvrConfig/ConfigManager.h>
#include <osg/Quat>
#include <string>
using std::string;
using std::vector;
using namespace osg;
using cvr::ConfigManager;

double fly_near, fly_far;


#define TURN (2.0*3.1415926535)

extern osg::Matrixd earthMove(MatrixTransform* mat,
                       double lat, double lon, double height);

Flyer::Flyer(double theta)
{
    node = makePlane();

    translate = new MatrixTransform();
    rotate    = new MatrixTransform();
    rotate_local = new MatrixTransform();
    scale     = new MatrixTransform();

    rotate->addChild(translate);
    translate->addChild(rotate_local);
    rotate_local->addChild(scale);
    scale->addChild(node);

    Matrix rot_tmp;
    rot_tmp.makeRotate(theta, Vec3(0,0,1));
    rotate->setMatrix(rot_tmp);

    Matrix trans_tmp;
    trans_tmp.makeTranslate(Vec3(100 * 1000,0,0));
    translate->setMatrix(trans_tmp);

    
    Matrix rlocal_tmp;
    rlocal_tmp.makeRotate(-TURN/4,Vec3(0,0,1));
    rotate_local->setMatrix(rlocal_tmp);
    

    Matrix scale_tmp;
    scale_tmp.makeScale(5000,5000,5000);
    scale->setMatrix(scale_tmp);
}

Node* Flyer::getNode()
{
    return rotate;
}

// --------------------------------------------------------------------

void Flock::init(double lat, double lon, double height)
{
    on_the_earth = new MatrixTransform();
    carousel = new MatrixTransform();
    rotation_speed= 0.17 / 60.0; //apx 10 degrees per second?
    theta = 0;

    earthMove(on_the_earth, lat, lon, height);
    on_the_earth->addChild(carousel);

    mylod = new LOD();
    //lod->addChild(on_the_earth, 0, fly_far / (2*5000));
    mylod->addChild(on_the_earth, 0, 1);
    mylod->setRange(0, 0, fly_far * 10000000.0/(2.0*5000));

    
    placeholder = new Group();
    carousel->addChild(placeholder);
    
}

void Flock::update()
{
    theta += rotation_speed;
    theta = fmod(theta, 2.0*3.1415926535);
    Matrixd rotation;
    rotation.makeRotate(theta, Vec3(0,0,1));
    carousel->setMatrix(rotation);

}

void Flock::add(Flyer* b)
{
    flyers.push_back(b);
    carousel->addChild(b->getNode());
}

Node* Flock::getNode()
{
    return mylod;
    //return on_the_earth;
}

Node* Flock::getMeasureNode()
{
    return placeholder; 
}

void Flock::modify_speed(double zero_to_one)
{
    double top_speed = 10 * 0.17 / 60.0;
    rotation_speed = zero_to_one * top_speed;
}

// --------------------------------------------------------------------

Flock the_flock;


void flyer_start()
{
    string lat_str = ConfigManager::getEntry("lat", "Plugin.AppSwitcher.flyer", "0");
    string lon_str = ConfigManager::getEntry("lon", "Plugin.AppSwitcher.flyer", "0");
    string h_str   = ConfigManager::getEntry("height", "Plugin.AppSwitcher.flyer", "0");

    double lat, lon, h;
    lat = atof(lat_str.c_str());
    lon = atof(lon_str.c_str());
    h   = atof(h_str.c_str());


    string near_str = ConfigManager::getEntry("threshNear", "Plugin.AppSwitcher.flyer", "0");
    string far_str  = ConfigManager::getEntry("threshFar", "Plugin.AppSwitcher.flyer", "0");
    fly_near = atof(near_str.c_str());
    fly_far  = atof(far_str.c_str());


    
    the_flock.init(lat, lon, h);

    Flyer *b1, *b2, *b3, *b4;
    b1 = new Flyer(0 * TURN); 
    b2 = new Flyer(0.25 * TURN);
    b3 = new Flyer(0.5 * TURN);
    b4 = new Flyer(0.75 * TURN);
    
    the_flock.add(b1);
    the_flock.add(b2);
    the_flock.add(b3);
    the_flock.add(b4);

    the_flock.mylod->computeBound();


    cvr::SceneManager::instance()
        ->getObjectsRoot()
        ->addChild(the_flock.getNode());
}

void flyer_step()
{
    the_flock.update();
}


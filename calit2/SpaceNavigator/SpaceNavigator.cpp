#include "SpaceNavigator.h"

#include <kernel/PluginHelper.h>
#include <osg/Vec4>
#include <osg/Matrix>
#include <osgUtil/SceneView>
#include <osg/Camera>
#include <config/ConfigManager.h>
#include <kernel/ComController.h>

using namespace std;
using namespace osg;
using namespace cvr;

CVRPLUGIN(SpaceNavigator)

SpaceNavigator::SpaceNavigator()
{
}

SpaceNavigator::~SpaceNavigator()
{
}

bool SpaceNavigator::init()
{
    bool status = false;
    if(ComController::instance()->isMaster())
    {
	if(spnav_open()==-1) 
	{
	    cerr << "SpaceNavigator: Failed to connect to the space navigator daemon" << endl;
	}
	else
	{
	    status = true;
	}
	ComController::instance()->sendSlaves((char *)&status, sizeof(bool));
    }
    else
    {
	ComController::instance()->readMaster((char *)&status, sizeof(bool));
    }

    transMult = ConfigManager::getFloat("Plugin.SpaceNavigator.TransMult", 1.0);
    rotMult = ConfigManager::getFloat("Plugin.SpaceNavigator.RotMult", 1.0);

    transcale = -0.05 * transMult;
    rotscale = -0.000009 * rotMult;

   return status;
}

void SpaceNavigator::preFrame()
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
    }
    else
    {
	ComController::instance()->readMaster((char *)finalmat.ptr(), sizeof(double[16]));
    }

    PluginHelper::setObjectMatrix(finalmat);


}


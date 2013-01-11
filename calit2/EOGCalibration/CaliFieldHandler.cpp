/***************************************************************
* File Name: CaliFieldHandler.cpp
*
* Description: 
*
* Written by ZHANG Lelin on Sept 20, 2010
*
***************************************************************/
#include "CaliFieldHandler.h"

using namespace std;
using namespace osg;


float CaliFieldHandler::RIGHT_RANGE_MIN(0.f);
float CaliFieldHandler::LEFT_RANGE_MIN(0.f);
float CaliFieldHandler::UPWARD_RANGE_MIN(0.f);
float CaliFieldHandler::DOWNWARD_RANGE_MIN(0.f);
float CaliFieldHandler::DEPTH_RANGE_MIN(0.5f);

float CaliFieldHandler::RIGHT_RANGE_MAX(M_PI/9.f);
float CaliFieldHandler::LEFT_RANGE_MAX(M_PI/9.f);
float CaliFieldHandler::UPWARD_RANGE_MAX(M_PI/9.f);
float CaliFieldHandler::DOWNWARD_RANGE_MAX(M_PI/9.f);
float CaliFieldHandler::DEPTH_RANGE_MAX(5.0f);

float CaliFieldHandler::PHI_RES(M_PI/45.f); 
float CaliFieldHandler::THETA_RES(M_PI/45.f);
float CaliFieldHandler::RAD_RES(0.5);


/***************************************************************
*  Constructor: CaliFieldHandler()
***************************************************************/
CaliFieldHandler::CaliFieldHandler(MatrixTransform *rootViewerTrans): mFlagVisible(false),
			mSwitch(NULL), mWireframeGeode(NULL), mWireframeGeometry(NULL)
{
    gPhiMin = M_PI/2.f - RIGHT_RANGE_MAX;
    gPhiMax = M_PI/2.f + LEFT_RANGE_MAX;

    gThetaMin = M_PI/2.f - UPWARD_RANGE_MAX;
    gThetaMax = M_PI/2.f + DOWNWARD_RANGE_MAX;

    gRadMin = DEPTH_RANGE_MIN;
    gRadMax = DEPTH_RANGE_MAX;  

    gNumPhiSample = (int)((gPhiMax - gPhiMin) / PHI_RES) + 1;
    gNumThetaSample = (int)((gThetaMax - gThetaMin) / PHI_RES) + 1;
    gNumRadSample = (int)((gRadMax - gRadMin) / RAD_RES) + 1;

    gPhiSampleMin = 0;		gPhiSampleMax = gNumPhiSample - 1;
    gThetaSampleMin = 0;	gThetaSampleMax = gNumThetaSample - 1;
    gRadSampleMin = 0;		gRadSampleMax = gNumRadSample - 1;

    initWireFrames(rootViewerTrans);
}


/***************************************************************
*  Function: setVisible()
***************************************************************/
void CaliFieldHandler::setVisible(bool flag)
{
    mFlagVisible = flag;

    if (!mSwitch) return;
    if (flag) mSwitch->setAllChildrenOn();
    else mSwitch->setAllChildrenOff();
}


/***************************************************************
*  Function: initWireFrames()
***************************************************************/
void CaliFieldHandler::initWireFrames(MatrixTransform *rootViewerTrans)
{
    /* setup wire frame in calibration field */
    mSwitch = new Switch();
    mWireframeGeode = new Geode();
    mWireframeGeometry = new Geometry();
    Vec3Array* vertices = new Vec3Array;
    Vec3Array* normals = new Vec3Array;
    mSwitch->setAllChildrenOff();

    /* create vertex primitive sets */
    float rad, phi, theta;
    for (int i = 0; i < gNumRadSample; i++)
    {
        rad = gRadMin + RAD_RES * i;;
        for (int j = 0; j < gNumPhiSample; j++)
        {
            phi = gPhiMin + PHI_RES * j;
            for (int k = 0; k < gNumThetaSample; k++)
            {
                theta = gThetaMin + THETA_RES * k;

                Vec3 pos;
                sphericToCartesian(phi, theta, rad, pos);
                vertices->push_back(pos);
                normals->push_back(Vec3(0, 0, 1));

                /* put boxes to eliminate edge flashing effects */
                Box *box = new Box(pos, 0.01);
                Drawable *boxDrawable = new ShapeDrawable(box);
                mWireframeGeode->addDrawable(boxDrawable); 
            }
        }
    }
    mWireframeGeometry->setVertexArray(vertices);
    mWireframeGeometry->setNormalArray(normals);

    /* create edge primitive sets */
    DrawElementsUInt* edges = createEdgePrimitiveSet();
    mWireframeGeometry->addPrimitiveSet(edges);

    mWireframeGeode->addDrawable(mWireframeGeometry); 
    mSwitch->addChild(mWireframeGeode);
    rootViewerTrans->addChild(mSwitch);

    /* apply field textures */
    Material* material = new Material;
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.f,1.f,0.f, 1.f));
    StateSet* fieldStateSet = new StateSet();
    fieldStateSet->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
    mWireframeGeode->setStateSet(fieldStateSet);
}


/***************************************************************
*  Function: updateWireFrames()
***************************************************************/
void CaliFieldHandler::updateWireFrames(const float left, const float right, 
			const float up, const float down, const float depth)
{
    /* remove existing wire frame set */
    int numPrimitiveSet = mWireframeGeometry->getNumPrimitiveSets();
    if (numPrimitiveSet > 0) mWireframeGeometry->removePrimitiveSet(0, numPrimitiveSet);

    /* update parameters */
    gPhiSampleMin = (gNumPhiSample - 1) / 2 - (int)((right / PHI_RES) + 0.001f);
    gPhiSampleMax = (gNumPhiSample - 1) / 2 + (int)((left / PHI_RES) + 0.001f); 
    gThetaSampleMin = (gNumThetaSample - 1) / 2 - (int)((up / THETA_RES) + 0.001f);
    gThetaSampleMax = (gNumThetaSample - 1) / 2 + (int)((down / THETA_RES) + 0.001f);
    gRadSampleMin = 0;
    gRadSampleMax = (int)((depth - DEPTH_RANGE_MIN) / RAD_RES + 0.001f);

    /* create edge primitive sets */
    DrawElementsUInt* edges = createEdgePrimitiveSet();
    mWireframeGeometry->addPrimitiveSet(edges);
}


/***************************************************************
*  Function: createEdgePrimitiveSet()
***************************************************************/
DrawElementsUInt *CaliFieldHandler::createEdgePrimitiveSet()
{
    DrawElementsUInt* edges = new DrawElementsUInt(PrimitiveSet::LINES, 0); 

    /* set vertex connections */
    int quad_rad = gNumPhiSample * gNumThetaSample;
    for (int i = gRadSampleMin; i < gRadSampleMax; i++)
    {
        for (int j = gPhiSampleMin; j < gPhiSampleMax + 1; j++)
        {
            for (int k = gThetaSampleMin; k < gThetaSampleMax + 1; k++)
            {
                int idx = i * quad_rad + j * gNumThetaSample + k;
                edges->push_back(idx);
                edges->push_back(idx+quad_rad);
            }
        }
    }
    for (int i = gRadSampleMin; i < gRadSampleMax + 1; i++)
    {
        for (int j = gPhiSampleMin; j < gPhiSampleMax; j++)
        {
            for (int k = gThetaSampleMin; k < gThetaSampleMax + 1; k++)
            {
                int idx = i * quad_rad + j * gNumThetaSample + k;
                edges->push_back(idx);
                edges->push_back(idx+gNumThetaSample);
            }
        }
    }
    for (int i = gRadSampleMin; i < gRadSampleMax + 1; i++)
    {
        for (int j = gPhiSampleMin; j < gPhiSampleMax + 1; j++)
        {
            for (int k = gThetaSampleMin; k < gThetaSampleMax; k++)
            {
                int idx = i * quad_rad + j * gNumThetaSample + k;
                edges->push_back(idx);
                edges->push_back(idx+1);
            }
        }
    }

    return edges;
}


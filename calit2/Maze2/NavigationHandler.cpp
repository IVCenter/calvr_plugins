/***************************************************************
* File Name: NavigationHandler.cpp
*
* Description: Implementation of button based navigation control
*
* Written by ZHANG Lelin on May 18, 2011
*
***************************************************************/
#include "NavigationHandler.h"


using namespace osg;
using namespace std;
using namespace cvr;


float NavigationHandler::gMovSpeedUnit(0.04f);
float NavigationHandler::gRotSpeedUnit(0.8f * M_PI / 180.f);

// Constructor
NavigationHandler::NavigationHandler(): mFlagEnabled(false), mButtonType(STILL)
{
    mScale = 1.f;
    mMovSpeed = gMovSpeedUnit;
    mRotSpeed = gRotSpeedUnit;

    mViewDir = Vec3(0, 1, 0);
    mViewPos = Vec3(0, 0, 0);
}


/***************************************************************
* Function: updateNaviStates()
***************************************************************/
void NavigationHandler::updateNaviStates(const float &scale, const osg::Vec3 &viewDir, const osg::Vec3 &viewPos)
{
    mScale = scale;
    mViewDir = viewDir;
    mViewPos = viewPos;
}


/***************************************************************
* Function: updateButtonStates()
***************************************************************/
void NavigationHandler::updateButtonStates()
{
    if (!mFlagEnabled) return;

    /* get navigation button responses */
    float spinX = PluginHelper::getValuator(0, 0);
    float spinY = PluginHelper::getValuator(0, 1);

    if (spinX > 2.f || spinX < -2.f || spinY > 2.f || spinY < -2.f) return;

    mButtonType = STILL;
    if (spinX > 0.1)
    {
		mButtonType = (ButtonType) (((unsigned char) mButtonType) | ((unsigned char) RIGHT));
		mRotSpeed = gRotSpeedUnit * spinX;
    }
    else if (spinX < -0.1)
    {
		mButtonType = (ButtonType) (((unsigned char) mButtonType) | ((unsigned char) LEFT));
		mRotSpeed = gRotSpeedUnit * (-spinX);
    }
    if (spinY > 0.1)
    {
		mButtonType = (ButtonType) (((unsigned char) mButtonType) | ((unsigned char) UP));
		mMovSpeed = gMovSpeedUnit * (-spinY);
    }
    else if (spinY < -0.1)
    {
		mButtonType = (ButtonType) (((unsigned char) mButtonType) | ((unsigned char) DOWN));
		mMovSpeed = gMovSpeedUnit * spinY;
    }
}


/***************************************************************
* Function: updateKeys()
***************************************************************/
void NavigationHandler::updateKeys(const int &keySym, bool pressFlag)
{
    if (!mFlagEnabled) return;

    /* key press event: enable navi states */
    if (pressFlag)
    {
		if (keySym == 65361) mButtonType = (ButtonType) (((unsigned char) mButtonType) | ((unsigned char) LEFT));
		else if (keySym == 65363) mButtonType = (ButtonType) (((unsigned char) mButtonType) | ((unsigned char) RIGHT));
		if (keySym == 65362) mButtonType = (ButtonType) (((unsigned char) mButtonType) | ((unsigned char) UP));
		else if (keySym == 65364) mButtonType = (ButtonType) (((unsigned char) mButtonType) | ((unsigned char) DOWN));
    }

    /* key release event: disable navi states */
    else
    {
		if (keySym == 65361) mButtonType = (ButtonType) (((unsigned char) mButtonType) & (0xff - (unsigned char) LEFT));
		else if (keySym == 65363) mButtonType = (ButtonType) (((unsigned char) mButtonType) & (0xff - (unsigned char) RIGHT));
		if (keySym == 65362) mButtonType = (ButtonType) (((unsigned char) mButtonType) & (0xff - (unsigned char) UP));
		else if (keySym == 65364) mButtonType = (ButtonType) (((unsigned char) mButtonType) & (0xff - (unsigned char) DOWN));
    }
}


/***************************************************************
* Function: updateXformMat()
***************************************************************/
void NavigationHandler::updateXformMat()
{
    // const float mMovSpeed = 0.04f;
    // const float mRotSpeed = 0.8f * M_PI / 180.f;
    Matrix leftRotMat, rightRotMat, fwdOffsetMat, bwdOffsetMat;
	Matrix curXMat = PluginHelper::getObjectMatrix();

// cerr << "mButtonType = " << (int) mButtonType << endl;

    /* modify xform matrix */
    if ((int)(((unsigned char) mButtonType) & ((unsigned char) LEFT)) > 0)
    {
        leftRotMat.makeRotate(-mRotSpeed, Vec3(0, 0, 1));
		curXMat = curXMat * leftRotMat;
        PluginHelper::setObjectMatrix(curXMat);

cerr << "moving left \n";
    }
    else if ((int)(((unsigned char) mButtonType) & ((unsigned char) RIGHT)) > 0)
    {
        rightRotMat.makeRotate(mRotSpeed, Vec3(0, 0, 1));
		curXMat = curXMat * rightRotMat;
        PluginHelper::setObjectMatrix(curXMat);

cerr << "moving right \n";
    }

    if ((int)(((unsigned char) mButtonType) & ((unsigned char) UP)) > 0)
    {
        fwdOffsetMat.makeTranslate((Vec3(0, 1, 0) * (-mMovSpeed)) * mScale);
		curXMat = curXMat * fwdOffsetMat;
        PluginHelper::setObjectMatrix(curXMat);

cerr << "moving up \n";
    }
    else if ((int)(((unsigned char) mButtonType) & ((unsigned char) DOWN)) > 0)
    {
        bwdOffsetMat.makeTranslate((Vec3(0, 1, 0) * mMovSpeed) * mScale);
		curXMat = bwdOffsetMat * bwdOffsetMat;
        PluginHelper::setObjectMatrix(curXMat);

cerr << "moving down \n";
    }
}


















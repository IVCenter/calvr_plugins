/***************************************************************
* File Name: CalibrationController.cpp
*
* Description: 
*
* Written by ZHANG Lelin on July 21, 2010
*
***************************************************************/
#include "CalibrationController.h"

using namespace std;
using namespace osg;


const float CalibrationController::gNoseOffset(-0.01f);

/***************************************************************
*  Constructor: CalibrationController()
***************************************************************/
CalibrationController::CalibrationController(Group *rootGroup, const string &datadir): mCaliFlag(false), mPlaybackFlag(false),
	mLeftRange(M_PI / 9), mRightRange(M_PI / 9), mUpwardRange(M_PI / 9), mDownwardRange(M_PI / 9), 
	mMinDepthRange(0.5), mMaxDepthRange(5), mHorFreq(1), mVerFreq(1), mDepFreq(1),
	mFieldRight(Vec3(1, 0, 0)), mFieldFront(Vec3(0, 1, 0)), mFieldUp(Vec3(0, 0, 1)), mFieldPos(Vec3(0, 0, 0))
{
    mTimer = 0.0;
    mLastAppear = 0.0; 
    mAppearInterval = cvr::ConfigManager::getFloat("value", "Plugin.EOGCalibration.AppearTestInterval", 5);
    mPhi = 0;
    mTheta = 0;
    mRad = 0;

    mViewerAlignmentTrans = new MatrixTransform();
    rootGroup->addChild(mViewerAlignmentTrans);

    /* apply downwards offset towards nose's position */
    mNoseOffsetTrans = new MatrixTransform();
    mViewerAlignmentTrans->addChild(mNoseOffsetTrans);
    Matrixd noseoffsetMat;
    noseoffsetMat.makeTranslate(Vec3(0, 0, gNoseOffset));
    mNoseOffsetTrans->setMatrix(noseoffsetMat);

	BallHandler::setDataDir(datadir);
    mCaliBallHandler = new CaliBallHandler(mNoseOffsetTrans);
	mPlaybackBallHandler = new PlaybackBallHandler(mNoseOffsetTrans);
    mCaliFieldHandler = new CaliFieldHandler(mNoseOffsetTrans);

    mDataDir = datadir;
}


/***************************************************************
*  Function: startCalibration()
***************************************************************/
void CalibrationController::startCalibration()
{
    mCaliFlag = true;

    /* create an instance log file that takes record of all parameters */
    ofstream outFile;
    string filename = mDataDir + "/EOGTestLog.txt";
    outFile.open(filename.c_str(), ios::app);
    if (!outFile) {
        cout << "CaveCAD::CalibrationController: Unable to open file " << filename << endl;
        return;
    }

    /* get current local time */
    time_t rawtime;
    time(&rawtime);

    outFile << endl;
    outFile << "Current local time: " << ctime(&rawtime) << endl;
    outFile << "Horizontal Range: " << mLeftRange << " " << mRightRange << endl;
    outFile << "Vertical Range: " << mDownwardRange << " " << mUpwardRange << endl;
    outFile << "Depth Range: " << mMinDepthRange << " " << mMaxDepthRange << endl;
    outFile << "Frequency: [" << mHorFreq << " " << mVerFreq << " " << mDepFreq << " ]" << endl;
    outFile << endl;
    outFile << "mFieldRight = " << mFieldRight.x() << " " << mFieldRight.y() << " " << mFieldRight.z() << endl;
    outFile << "mFieldFront = " << mFieldFront.x() << " " << mFieldFront.y() << " " << mFieldFront.z() << endl;
    outFile << "mFieldUp = " << mFieldUp.x() << " " << mFieldUp.y() << " " << mFieldUp.z() << endl;
    outFile << "mFieldPos = " << mFieldPos.x() << " " << mFieldPos.y() << " " << mFieldPos.z() << endl;
    outFile << endl;
    outFile.close();
}


/***************************************************************
*  Function: stopCalibration()
***************************************************************/
void CalibrationController::stopCalibration()
{
    mCaliFlag = false;
    mTimer = 0.0;
}


/***************************************************************
*  Function: startPlayback()
***************************************************************/
void CalibrationController::startPlayback()
{
	mPlaybackFlag = true;
	mPlaybackBallHandler->resertVirtualTimer();
}


/***************************************************************
*  Function: stopPlayback()
***************************************************************/
void CalibrationController::stopPlayback()
{
	mPlaybackFlag = false;
	mPlaybackBallHandler->resertVirtualTimer();
}

void CalibrationController::setAppearFlag(bool flag)
{
    mAppearFlag = flag;
}

/***************************************************************
*  Function: setCaliBallVisible()
***************************************************************/
void CalibrationController::setCaliBallVisible(bool flag)
{
    mCaliBallHandler->setVisible(flag);
}


/***************************************************************
*  Function: isCaliBallVisible()
***************************************************************/
bool CalibrationController::isCaliBallVisible()
{
    return mCaliBallHandler->isVisible();
}


/***************************************************************
*  Function: setPlaybackBallVisible()
***************************************************************/
void CalibrationController::setPlaybackBallVisible(bool flag)
{
    mPlaybackBallHandler->setVisible(flag);
}


/***************************************************************
*  Function: isPlaybackBallVisible()
***************************************************************/
bool CalibrationController::isPlaybackBallVisible()
{
    return mPlaybackBallHandler->isVisible();
}


/***************************************************************
*  Function: setCaliFieldVisible()
***************************************************************/
void CalibrationController::setCaliFieldVisible(bool flag)
{
    mCaliFieldHandler->setVisible(flag);
}


/***************************************************************
*  Function: isCaliFieldVisible()
***************************************************************/
bool CalibrationController::isCaliFieldVisible()
{
    return mCaliFieldHandler->isVisible();
}


/***************************************************************
*  Function: resetCaliBall()
*
*  Place the calibration ball to default position in the field
*
***************************************************************/
void CalibrationController::resetCaliBall()
{
    mTimer = 0;
}


/***************************************************************
*  Function: resetCaliField()
*
*  Align the calibration field with viewer's front direction
*
***************************************************************/
void CalibrationController::resetCaliField(const Matrixf &invBaseMat)
{
    Matrixf scaleMat = Matrixf( 1400.f, 0.f, 0.f, 0.f, 
                                0.f, 1400.f, 0.f, 0.f, 
                                0.f, 0.f, 1400.f, 0.f, 
                                0.f, 0.f, 0.f,    1.f);
    mViewerAlignmentTrans->setMatrix(scaleMat * invBaseMat);

    /* update field position & orientations */
    mFieldRight = Vec3(mViewMat(0, 0), mViewMat(0, 1), mViewMat(0, 2));
    mFieldFront = Vec3(mViewMat(1, 0), mViewMat(1, 1), mViewMat(1, 2));
    mFieldUp = Vec3(mViewMat(2, 0), mViewMat(2, 1), mViewMat(2, 2));
    mFieldPos = Vec3(mViewMat(3, 0), mViewMat(3, 1), mViewMat(3, 2)) / 1000.f + Vec3(0, 0, gNoseOffset);
    
    std::string filename;
    bool isFound;

    filename = cvr::ConfigManager::getEntry("Plugin.Maze2.CalibrationFieldFile", "", &isFound);
    if (!isFound)
        return;

    std::string data = "";
    char buf[1024];

    sprintf(buf, "right %f %f %f\nfront %f %f %f\nup %f %f %f\nposition %f %f %f\n",
        mFieldRight[0], mFieldRight[1], mFieldRight[2],
        mFieldFront[0], mFieldFront[1], mFieldFront[2],
        mFieldUp[0],    mFieldUp[1],    mFieldUp[2],
        mFieldPos[0],   mFieldPos[1],   mFieldPos[2]);
    
    data.append(buf);
    std::ofstream file;
    file.open(filename.c_str());
    file << data;
    file.flush();
    file.close();
    std::cout << buf << std::endl;
}


/***************************************************************
*  Function: updateCaliTime()
***************************************************************/
void CalibrationController::updateCaliTime(const double &frameDuration)
{
    if (!mCaliFlag) return;
    mTimer += frameDuration;
}


/***************************************************************
*  Function: updatePlaybackTime()
***************************************************************/
void CalibrationController::updatePlaybackTime(const double &frameDuration)
{
	if (!mPlaybackFlag) return;
	mPlaybackBallHandler->updatePlaybackTime(frameDuration);
}


/***************************************************************
*  Function: updateCaliBallPos()
***************************************************************/
void CalibrationController::updateCaliBallPos(float &phi, float &theta, float &rad)
{
    /* calculate phase parameters */
    if (!mAppearFlag)
    {
        phi = 	0.5 * (mLeftRange - mRightRange) + M_PI / 2 + 
            0.5 * (mRightRange + mLeftRange) * sin(2 * M_PI * mHorFreq * mTimer);
        theta = 	0.5 * (mDownwardRange - mUpwardRange) + M_PI / 2 +  
            0.5 * (mUpwardRange + mDownwardRange) * sin(2 * M_PI * mVerFreq * mTimer);
        rad = 	0.5 * (mMinDepthRange + mMaxDepthRange) + 
            0.5 * (mMaxDepthRange - mMinDepthRange) * sin(2 * M_PI * mDepFreq * mTimer);

        mCaliBallHandler->updateCaliBall(phi, theta, rad);
    }

    else if (mAppearFlag && mTimer - mLastAppear > mAppearInterval)
    {
        mLastAppear = mTimer;
        // center + random offset within range
        float offset = (double) rand() / (RAND_MAX);
        if (rand() % 2 == 0)
            offset *= -1;
        phi = 0.5 * (mLeftRange - mRightRange) + M_PI / 2 + 
            0.5 * (mRightRange + mLeftRange) * offset;

        offset = (double) rand() / (RAND_MAX);
        if (rand() % 2 == 0)
            offset *= -1;
        theta = 	0.5 * (mDownwardRange - mUpwardRange) + M_PI / 2 +  
            0.5 * (mUpwardRange + mDownwardRange) * offset;

        offset = (double) rand() / (RAND_MAX);
        if (rand() % 2 == 0)
            offset *= -1;
        rad = 	0.5 * (mMinDepthRange + mMaxDepthRange) + 
            0.5 * (mMaxDepthRange - mMinDepthRange) * offset;


        mPhi = phi;
        mTheta = theta;
        mRad = rad;
    }
    else
    {
        phi = mPhi; 
        theta = mTheta;
        rad = mRad;
        mCaliBallHandler->updateCaliBall(phi, theta, rad);
    }
}


/***************************************************************
*  Function: updatePlaybackBallPos()
***************************************************************/
void CalibrationController::updatePlaybackBallPos()
{
	mPlaybackBallHandler->updatePlaybackBallPos();
}


/***************************************************************
*  Function: updateCaliParam()
***************************************************************/
void CalibrationController::updateCaliParam(const CaliFieldParameter& typ, const float &val)
{
    /* update parameters */
    if (typ == LEFT_RANGE) mLeftRange = val * M_PI / 180.f;
    else if (typ == RIGHT_RANGE) mRightRange = val * M_PI / 180.f;
    else if (typ == UPWARD_RANGE) mUpwardRange = val * M_PI / 180.f;
    else if (typ == DOWNWARD_RANGE) mDownwardRange = val * M_PI / 180.f;
    else if (typ == DEPTH_RANGE) mMaxDepthRange = val;
    else if (typ == HORIZONTAL_FREQ) mHorFreq = val;
    else if (typ == VERTICAL_FREQ) mVerFreq = val;
    else if (typ == DEPTH_FREQ) mDepFreq = val;
    else return;

    /* update geometry object handlers */
    if (typ >= LEFT_RANGE && typ <= DEPTH_RANGE)
    {
        mCaliFieldHandler->updateWireFrames(mLeftRange, mRightRange, mUpwardRange, mDownwardRange, mMaxDepthRange);
    }
}


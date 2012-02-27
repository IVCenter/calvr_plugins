/***************************************************************
* File Name: BallHandler.cpp
*
* Description: 
*
* Written by ZHANG Lelin on Sept 20, 2010
*
***************************************************************/
#include "BallHandler.h"

using namespace std;
using namespace osg;


float BallHandler::BOUNDING_BALL_SIZE(0.025f);
float BallHandler::CENTER_BALL_SIZE(0.005f);

float PlaybackBallHandler::VISUAL_POLE_LENGTH(5.0f);
float PlaybackBallHandler::VISUAL_POLE_RADIUS(0.05f);

string BallHandler::gDataDir("");

/***************************************************************
*  Constructor: BallHandler()
***************************************************************/
BallHandler::BallHandler(): mFlagVisible(false),
			mBoundingBallGeode(NULL), mCenterBallGeode(NULL)
{
}


/***************************************************************
*  Constructor: CaliBallHandler()
***************************************************************/
CaliBallHandler::CaliBallHandler(MatrixTransform *rootViewerTrans)
{
	Vec4 exteriorBallColor = Vec4(1.0f, 1.0f, 0.0f, 1.0f);
    initCaliBallGeometry(rootViewerTrans, exteriorBallColor);
}


/***************************************************************
*  Constructor: PlaybackBallHandler()
***************************************************************/
PlaybackBallHandler::PlaybackBallHandler(MatrixTransform *rootViewerTrans)
{
	/* import playback data */
	mVirtualTimer = 0.0;
	mStartVirtualTime = 0.0;
	mMaxVirtualTime = 0.0;
	mPlaybackItr = -1;
	importPlaybackFile(gDataDir + "/EOGClient/EOGPlayback.dat");

	/* create original calibration ball */
	Vec4 exteriorBallColor = Vec4(0.0f, 1.0f, 0.0f, 1.0f);
	initCaliBallGeometry(rootViewerTrans, exteriorBallColor);

	/* create extra 'ghost' ball for prediction */
	mGhostBallTrans = new MatrixTransform();
	mGhostBallTrans->addChild(mBoundingBallGeode);
    mGhostBallTrans->addChild(mCenterBallGeode);
	rootViewerTrans->addChild(mGhostBallTrans);

	mHeadSwitch = new Switch();
	mHeadTrans = new MatrixTransform();
	mPoleTrans = new MatrixTransform();
	mEyeBallTrans = new MatrixTransform();

	/* load eye balls geometry from existing VRML file */
	mEyeBallNode = osgDB::readNodeFile(gDataDir + "/EOGClient/EyeBall.WRL");
	mEyeBallTrans->addChild(mEyeBallNode);
	mHeadTrans->addChild(mEyeBallTrans);

	Matrixf eyeBallScaleMat;
    eyeBallScaleMat.makeScale(Vec3(0.02f, 0.02f, 0.02f));
	mEyeBallTrans->setMatrix(eyeBallScaleMat);

	/* create cone shaped geometries for 'mPoleGeode'*/
	mPoleGeode = new Geode();
    Cone *poleShape = new Cone(Vec3(0, VISUAL_POLE_LENGTH * 0.75, 0), VISUAL_POLE_RADIUS, VISUAL_POLE_LENGTH);
	poleShape->setRotation(Quat(M_PI * 0.5f, Vec3(1, 0, 0)));
    Drawable *poleDrawable = new ShapeDrawable(poleShape);
    mPoleGeode->addDrawable(poleDrawable);
	mPoleTrans->addChild(mPoleGeode);

	/* apply pole texture */
    Material* poleMaterial = new Material;
    poleMaterial->setDiffuse(Material::FRONT_AND_BACK, exteriorBallColor);
    poleMaterial->setAlpha(Material::FRONT_AND_BACK, 0.2f);
    StateSet* poleMaterialStateSet = new StateSet();
    poleMaterialStateSet->setAttributeAndModes(poleMaterial, StateAttribute::OVERRIDE | StateAttribute::ON);
    poleMaterialStateSet->setMode(GL_BLEND, StateAttribute::OVERRIDE | osg::StateAttribute::ON );
    poleMaterialStateSet->setRenderingHint(StateSet::TRANSPARENT_BIN);
    mPoleGeode->setStateSet(poleMaterialStateSet);

	mHeadTrans->addChild(mPoleTrans);
	mHeadSwitch->addChild(mHeadTrans);
	mHeadSwitch->setAllChildrenOff();

	rootViewerTrans->addChild(mHeadSwitch);
}


/***************************************************************
*  Function: setVisible()
***************************************************************/
void BallHandler::setVisible(bool flag)
{
    mFlagVisible = flag;
    if (flag) mBallSwitch->setAllChildrenOn();
    else mBallSwitch->setAllChildrenOff();
}


/***************************************************************
*  Function: setVisible()
***************************************************************/
void PlaybackBallHandler::setVisible(bool flag)
{
	mFlagVisible = flag;
    if (flag)
	{
		mBallSwitch->setAllChildrenOn();
		mHeadSwitch->setAllChildrenOn();
	}
    else
	{
		mBallSwitch->setAllChildrenOff();
		mHeadSwitch->setAllChildrenOff();
	}
}


/***************************************************************
*  Function: initCaliBallGeometry()
***************************************************************/
void BallHandler::initCaliBallGeometry(MatrixTransform *rootViewerTrans, const Vec4 &exteriorBallColor)
{
    /* create calibration ball */
    mBallSwitch = new Switch();
    mBallTrans = new MatrixTransform();
    rootViewerTrans->addChild(mBallSwitch);
    mBallSwitch->addChild(mBallTrans);
    mBallSwitch->setAllChildrenOff();

    mBoundingBallGeode = new Geode();
    Sphere *bndSphere = new Sphere(Vec3(), BOUNDING_BALL_SIZE);
    Drawable *bndSphereDrawable = new ShapeDrawable(bndSphere);
    mBoundingBallGeode->addDrawable(bndSphereDrawable);

    mCenterBallGeode = new Geode();
    Sphere *ctrSphere = new Sphere(Vec3(), CENTER_BALL_SIZE); 
    Drawable *ctrSphereDrawable = new ShapeDrawable(ctrSphere);
    mCenterBallGeode->addDrawable(ctrSphereDrawable);

    mBallTrans->addChild(mBoundingBallGeode);
    mBallTrans->addChild(mCenterBallGeode);    

    /* apply calibration ball textures */
    Material* bndSphereMaterial = new Material;
    bndSphereMaterial->setDiffuse(Material::FRONT_AND_BACK, exteriorBallColor);
    bndSphereMaterial->setAlpha(Material::FRONT_AND_BACK, 0.4f);
    StateSet* bndSphereStateSet = new StateSet();
    bndSphereStateSet->setAttributeAndModes(bndSphereMaterial, StateAttribute::OVERRIDE | StateAttribute::ON);
    bndSphereStateSet->setMode(GL_BLEND, StateAttribute::OVERRIDE | osg::StateAttribute::ON );
    bndSphereStateSet->setRenderingHint(StateSet::TRANSPARENT_BIN);
    mBoundingBallGeode->setStateSet(bndSphereStateSet);

    Material* ctrSphereMaterial = new Material;
    ctrSphereMaterial->setDiffuse(Material::FRONT_AND_BACK, Vec4(1.f,1.f,1.f, 1.f));
    ctrSphereMaterial->setAlpha(Material::FRONT_AND_BACK, 1.0f);
    StateSet* ctrSphereStateSet = new StateSet();
    ctrSphereStateSet->setAttributeAndModes(ctrSphereMaterial, StateAttribute::OVERRIDE | StateAttribute::ON);
    ctrSphereStateSet->setMode(GL_BLEND, StateAttribute::OVERRIDE | osg::StateAttribute::ON );
    mCenterBallGeode->setStateSet(ctrSphereStateSet);
}


/***************************************************************
*  Function: updateCaliBall()
***************************************************************/
void BallHandler::updateCaliBall(const float &phi, const float &theta, const float &rad)
{
    Vec3 pos;
    sphericToCartetion(phi, theta, rad, pos);

    Matrixf ballTransMat;
    ballTransMat.makeTranslate(pos);
    mBallTrans->setMatrix(ballTransMat);
}


/***************************************************************
*  Function: importPlaybackFile()
***************************************************************/
void PlaybackBallHandler::importPlaybackFile(const string &filename)
{
	ifstream inFile;
    inFile.open(filename.c_str());
    if (!inFile) {
        cout << "PlaybackBallHandler::Unable to import playback file " << filename << endl;
        return;
    }

	double ts;
	Vec3 headPos, caliBallPos, predBallPos;
    while (inFile >> ts)
    {
		inFile >> headPos.x();			inFile >> headPos.y();			inFile >> headPos.z();
		inFile >> caliBallPos.x();		inFile >> caliBallPos.y();		inFile >> caliBallPos.z();
		inFile >> predBallPos.x();		inFile >> predBallPos.y();		inFile >> predBallPos.z();

		PlaybackEntry *entry = new PlaybackEntry;
		entry->mTS = ts;
		entry->mHeadPos = headPos;
		entry->mCaliBallPos = caliBallPos;
		entry->mPredBallPos = predBallPos;
		mEntryVector.push_back(entry); 
	}

	inFile.close();

	/* calculate time span of the playback data */
	const int numEntries = mEntryVector.size();
	if (numEntries > 0)
	{
		mStartVirtualTime = mEntryVector[0]->mTS;
		mMaxVirtualTime = mEntryVector[numEntries - 1]->mTS - mStartVirtualTime;
	}
}


/***************************************************************
*  Function: updatePlaybackTime()
***************************************************************/
void PlaybackBallHandler::updatePlaybackTime(const double &frameDuration)
{
	mVirtualTimer += frameDuration;
	if (mVirtualTimer >= mMaxVirtualTime)
	{
		mVirtualTimer = 0.0;
		mPlaybackItr = -1;
	}
	else
	{
		if (mEntryVector.size() == 0) return;
		while (mEntryVector[mPlaybackItr + 1]->mTS < mVirtualTimer + mStartVirtualTime)
		{
			if (++mPlaybackItr >= mEntryVector.size() - 1) break;
		}
	}
}


/***************************************************************
*  Function: updatePlaybackBallPos()
***************************************************************/
void PlaybackBallHandler::updatePlaybackBallPos()
{
	if (mEntryVector.size() == 0) return;

	/* interpolate between entry 'mPlaybackItr' and 'mPlaybackItr + 1' */
	if (mPlaybackItr == -1) mPlaybackItr++;
	else if (mPlaybackItr + 1 >= mEntryVector.size()) mPlaybackItr = 0;

	const PlaybackEntry *frontEntry = mEntryVector[mPlaybackItr];
	const PlaybackEntry *backEntry = mEntryVector[mPlaybackItr + 1];

	const double diff = backEntry->mTS - frontEntry->mTS;
	if (diff > 0.0)
	{
		const double frontCoef = (backEntry->mTS - (mVirtualTimer + mStartVirtualTime)) / diff;
		const double backCoef = ((mVirtualTimer + mStartVirtualTime) - frontEntry->mTS) / diff;

		Vec3 headPos = frontEntry->mHeadPos * frontCoef + backEntry->mHeadPos * backCoef;
		Vec3 caliBallPos = frontEntry->mCaliBallPos * frontCoef + backEntry->mCaliBallPos * backCoef;
		Vec3 predBallPos = frontEntry->mPredBallPos * frontCoef + backEntry->mPredBallPos * backCoef;

		/* apply current head position translation */
		Matrixf headTransMat;
		headTransMat.makeTranslate(headPos);
		mHeadTrans->setMatrix(headTransMat);

		/* apply current stimuli calibration position */
		Matrixf caliTransMat;
    	caliTransMat.makeTranslate(caliBallPos);
    	mBallTrans->setMatrix(caliTransMat);

		/* apply current predictive ball position */
		Matrixf predTransMat;
    	predTransMat.makeTranslate(predBallPos);
    	mGhostBallTrans->setMatrix(predTransMat);

		/* set matrix rotations to pole transform */
		Matrixf poleRotMat;
		poleRotMat.makeRotate(Vec3(0, 1, 0), predBallPos - headPos);
		mPoleTrans->setMatrix(poleRotMat);
	}
}


/***************************************************************
*  Function: getPlaybackTimeLabel()
***************************************************************/
const string PlaybackBallHandler::getPlaybackTimeLabel()
{
	char timerStr[64];
	sprintf(timerStr, "Time = %3.3f s", mVirtualTimer);
	string label = string(timerStr);
	return label;
}


















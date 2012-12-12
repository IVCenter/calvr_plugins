#ifndef _ShapeHelper_
#define _ShapeHelper_

#include <osg/MatrixTransform>
#include <osg/Vec3d>
#include <osg/Vec4>
#include <osg/Group>

#include "vvtokenizer.h"
#include "../shapes/BasicShape.h"
#include "../shapes/TriangleShape.h"
#include "../shapes/CircleShape.h"
#include "../shapes/RectShape.h"

#include <string>
#include <vector>
#include <math.h>
#include <limits>

#include "TrackerTree.h"

using namespace std;
using namespace osg;


class ShapeHelper  
{
public:        	
	ShapeHelper(Group *);
	ShapeHelper();


	virtual ~ShapeHelper();

	// returns a shape randomly generated
	BasicShape * genRandomShape();


	// determines whether to add or update a shape to current geode
	// this method is the ideal way to handle most data packets
	void processData(char*);
	
	void setObjectRoot(Group *);

	Group * getGroup();


	bool processedAll;
	bool debug;

protected:

	// getting parameters from the data
	double getDoubleParam(string);
	int getIntParam(string);
	char* getCharParam(string);
	double getDoubleParamC(char*);
	int getIntParamC(char*);
	char* getCharParamC(char*);	

	vvTokenizer * tok;


	// setting up the more complex parameters
	Vec3d getCenter();
	Vec3d getP1();
	Vec3d getP2();
	Vec3d getP3();
	Vec3d getP4();

	Vec4d getC1();
	Vec4d getC2();
	Vec4d getC3();

	// instead of separate gen/update, we're going to group by shape
	void handleCircle(int);
	void handleTriangle(int);
	void handleRect(int);

	// the group/scene we're managing
	Group * group;

 //Tree to store geodes of the group
 TrackerTree* tree;

	double random();
	double random(double,double);

	// other handlers
	int shapeCount;	
	bool genAll;

	int updateIndex;	
};

#endif

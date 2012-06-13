#include "ShapeHelper.h"

#include <fcntl.h>
#include <iostream>
#include <cstring>

#include <osg/Matrix>
#include <osg/CullFace>
#include <osgDB/ReadFile>

#include <cvrKernel/SceneManager.h>
#include <cvrKernel/SceneObject.h>
#include <cvrKernel/PluginHelper.h>

using namespace osg;
using namespace std;

#define MAXNUMSHAPE 3
#define RENDERSCALE 1

ShapeHelper::ShapeHelper(Group * _gr) 
{
	group = _gr;
	
	tree = new TrackerTree();
	
	tok = new vvTokenizer();
	tok->setEOLisSignificant(true);
	tok->setCaseConversion(vvTokenizer::VV_LOWER);
	tok->setParseNumbers(true);
	tok->setWhitespaceCharacter('=');
	shapeCount = 0;
	processedAll = false;
	genAll = false;
	debug = false;	
	updateIndex = 0;
}

ShapeHelper::ShapeHelper() 
{	
	group = new Group();
	
	tree = new TrackerTree();
	
	tok = new vvTokenizer();
	tok->setEOLisSignificant(true);
	tok->setCaseConversion(vvTokenizer::VV_LOWER);
	tok->setParseNumbers(true);
	tok->setWhitespaceCharacter('=');
	shapeCount = 0;
	processedAll = false;
	genAll = false;
	debug = false;	
	updateIndex = 0;
}



ShapeHelper::~ShapeHelper()
{
	delete tok;
}


void ShapeHelper::processData(char* data) {
	tok->newData(data);
	vvTokenizer::TokenType ttype;
	ttype = tok->nextToken();
	string pch = tok->sval;


	// i really need to implement easier ways to debug with pd
	if (shapeCount == MAXNUMSHAPE)// && debug) 
	{
		genAll = true;
		processedAll = true;
		updateIndex = 0;
	}	
	// keeping track of the updateIndex, so it doesnt go out of bound
	if (updateIndex > MAXNUMSHAPE-1)
	{
		updateIndex = 0;
	}


	// this is for finding a certain geode by comment
	// the protocol should be something like find=circle with red center
	if (0 == pch.compare("find"))
	{
		tok->nextToken();
		string query = tok->sval;

		// TODO add tree functions here
    TrackerNode* temp = tree->get(query,tree->root);

		// set pch to the according shape so the ifs down there can process
    
		// set the updateIndex, which is the positionInRoot return by tree node
		updateIndex = temp->positionInRoot;
	}


	if (0 == pch.compare("circle"))
	{
		// if we finished generating, we tell handle update instead
		if (genAll){			
			handleCircle(updateIndex);
			updateIndex++;
		}
		// otherwise generate a new one
		else 
		{
			handleCircle(-1);
		}

	}
	else if (0 == pch.compare("triangle"))
	{
		// update
		if (genAll)
		{
			handleTriangle(updateIndex);
			updateIndex++;
		}
		// generate
		else
		{
			handleTriangle(-1);
		}
	}
	else if (0 == pch.compare("rect"))
	{
		// update
		if (genAll)
		{
			handleRect(updateIndex);
			updateIndex++;
		}
		// generate
		else
		{
			handleRect(-1);
		}
	}
	else if (0 == pch.compare("done"))
	{
		processedAll = true;
		genAll = true;
	}

}



double ShapeHelper::getDoubleParamC(char* param) {
	vvTokenizer::TokenType ttype;
	ttype = tok->nextToken();
	while (ttype != vvTokenizer::VV_EOL) {
		if (0 == strcmp(tok->sval, param)) {
			ttype = tok->nextToken();
			tok->reset();
			return tok->nval;
		} else {
			ttype = tok->nextToken();
		}
	}
	tok->reset();
	return numeric_limits<double>::quiet_NaN();
}


int ShapeHelper::getIntParamC(char* param) {

	vvTokenizer::TokenType ttype;

	ttype = tok->nextToken();
	while (ttype != vvTokenizer::VV_EOL) {
		if (0 == strcmp(tok->sval, param)) {
			ttype = tok->nextToken();
			tok->reset();
			return tok->nval;
		} else {
			ttype = tok->nextToken();
		}

	}
	tok->reset();
	return numeric_limits<double>::quiet_NaN();

}


char* ShapeHelper::getCharParamC(char* param) {
	vvTokenizer::TokenType ttype;

	ttype = tok->nextToken();
	while (ttype != vvTokenizer::VV_EOL) {
		if (0 == strcmp(tok->sval, param)) {
			ttype = tok->nextToken();
			tok->reset();
			return tok->sval;
		} else {
			ttype = tok->nextToken();
		}

	}
	tok->reset();
	return NULL;
}


double ShapeHelper::getDoubleParam(string param) {
	return getDoubleParamC(&param[0]);
}

int ShapeHelper::getIntParam(string param) {

	return getIntParamC(&param[0]);

}


char* ShapeHelper::getCharParam(string param) {

	return getCharParamC(&param[0]);
}



//returns a random number between 0 and 1
double ShapeHelper::random() {
	return rand() / double(RAND_MAX);
}

// returns a random number between min and max
double ShapeHelper::random(double min, double max) {
	return (max - min) * random() + min;
}

void ShapeHelper::setObjectRoot(Group * _gr)
{
	group = _gr;
}


Vec3d ShapeHelper::getCenter()
{
	Vec3d center;

	double	cx = getDoubleParam("cx");
	double	cy = getDoubleParam("cy");
	double	cz = getDoubleParam("cz");

	if (!isnan(cx) && !isnan(cy) && !isnan(cz)) 
	{
		center = Vec3d(cx * RENDERSCALE, cy * RENDERSCALE, cz * RENDERSCALE);
	}

	return center;
}


Vec3d ShapeHelper::getP1()
{
	Vec3d p1;
	double	p1x = getDoubleParam("p1x");
	double	p1y = getDoubleParam("p1y");
	double	p1z = getDoubleParam("p1z");

	if (!isnan(p1x) && !isnan(p1y) && !isnan(p1z)) 
	{
		p1 = Vec3d(p1x * RENDERSCALE, p1y * RENDERSCALE, p1z * RENDERSCALE);
	}

	return p1;
}

Vec3d ShapeHelper::getP2()
{
	Vec3d p2;
	double	p2x = getDoubleParam("p2x");
	double	p2y = getDoubleParam("p2y");
	double	p2z = getDoubleParam("p2z");

	if (!isnan(p2x) && !isnan(p2y) && !isnan(p2z)) 
	{
		p2 = Vec3d(p2x * RENDERSCALE, p2y * RENDERSCALE, p2z * RENDERSCALE);
	}

	return p2;
}

Vec3d ShapeHelper::getP3()
{
	Vec3d p3;

	double 	p3x = getDoubleParam("p3x");
	double	p3y = getDoubleParam("p3y");
	double	p3z = getDoubleParam("p3z");


	if (!isnan(p3x) && !isnan(p3y) && !isnan(p3z))
	{
		p3 = Vec3d(p3x * RENDERSCALE, p3y * RENDERSCALE, p3z * RENDERSCALE);
	}

	return p3;
}

Vec3d ShapeHelper::getP4()
{
	Vec3d p4;

	double	p4x = getDoubleParam("p4x");
	double	p4y = getDoubleParam("p4y");
	double	p4z = getDoubleParam("p4z");

	if (!isnan(p4x) && !isnan(p4y) && !isnan(p4z)) 
	{
		p4 = Vec3d(p4x * RENDERSCALE, p4y * RENDERSCALE, p4z * RENDERSCALE);
	}

	return p4;
}


// get color1 from incoming data
Vec4d ShapeHelper::getC1()
{
	Vec4d c1;

	double	c1r = getDoubleParam("c1r");
	double	c1g = getDoubleParam("c1g");
	double	c1b = getDoubleParam("c1b");
	double	c1a = getDoubleParam("c1a");

	if (!isnan(c1r) && !isnan(c1g) && !isnan(c1b) && !isnan(c1a)) {
		c1 = Vec4d(c1r, c1g, c1b, c1a);
	}

	return c1;
}

Vec4d ShapeHelper::getC2()
{
	Vec4d c2;

	double	c2r = getDoubleParam("c2r");
	double	c2g = getDoubleParam("c2g");
	double	c2b = getDoubleParam("c2b");
	double	c2a = getDoubleParam("c2a");

	if (!isnan(c2r) && !isnan(c2g) && !isnan(c2b) && !isnan(c2a)) {
		c2 = Vec4d(c2r, c2g, c2b, c2a);
	}

	return c2;
}

Vec4d ShapeHelper::getC3()
{
	Vec4d c3;

	double	c3r = getDoubleParam("c3r");
	double	c3g = getDoubleParam("c3g");
	double	c3b = getDoubleParam("c3b");
	double	c3a = getDoubleParam("c3a");

	if (!isnan(c3r) && !isnan(c3g) && !isnan(c3b) && !isnan(c3a)) {
		c3 = Vec4d(c3r, c3g, c3b, c3a);
	}

	return c3;
}

Group * ShapeHelper::getGroup()
{
	return group;
}

void ShapeHelper::handleCircle(int positionInGroup)
{
	int pIG = -1;
	Geode* geode;
	Drawable * dr;
	CircleShape * circle;

	char * comment = getCharParam("comment");

	if (comment == NULL) {
		string empty = "";
		comment = &empty[0];
	}

	// if we're not creating a new geometry, we're going to get the geode at position pIG
	if (positionInGroup != -1)
	{
		pIG = positionInGroup;
		geode = dynamic_cast<Geode*> (group->getChild(pIG));
		dr = dynamic_cast<Drawable*> (geode->getDrawable(0));
		circle = (CircleShape*) dr;
	}
	// otherwise, we'll make a new geode
	else 
	{
		geode = new Geode();
	}


	// get radius parsed from the incoming data
	double radius = getDoubleParam("radius");	

	if (!isnan(radius))
	{
		radius = radius * RENDERSCALE;
		circle->setRadius(radius);
	}
	// if we're generating, give it a default value
	else if (pIG == -1)
	{
		radius = 100;
	}

	int tess = getIntParam("tess");

	if (tess == 0)
	{
		tess = 10;
	}

	Vec3d center = getCenter();

	// double check if center is initialized, can getCenter() return an unintialized Vec3d? or will it throw an error already?
	if (center.valid())
	{
		if (pIG != -1)
		{
			circle->setCenter(center);
		}
	} else
	{
		// if we're generating a new circle and there isn't a center specified
		if (pIG == -1)
		{
			center = Vec3d(0,0,0);
		}
	}

	// check colors too
	Vec4d c1 = getC1();

	if (c1.valid())
	{	
		if (pIG != -1)
		{
			circle->setColor(c1);
		}
	}
	else
	{
		// we'll give it a nice lavender color
		if (pIG == -1)
		{
			c1 = Vec4d((195/255),(149/255),(237/255),0.8);
		}
	}

	Vec4d c2 = getC2();

	if (c2.valid())
	{	
		if (pIG != -1)
		{
			circle->setGradient(c2);
		}
	}
	else
	{
		// default black
		if (pIG == -1)
		{
			c2 = Vec4d(0,0,0,0.8);
		}
	}


	if (pIG != -1)
	{
		circle->updateAll();
	}
	else 
	{
		circle = new CircleShape(comment,center,radius,c1,c2,tess);
		geode->addDrawable(circle);
		group->addChild(geode);
			
		shapeCount++;

		// TODO add comment and geode to tracking tree
		tree->root = tree->insert(comment,pIG,geode,tree->root);
	}

}

// for now we're only taking care of triangles generated witha center, 
// individual points will be handled later on
void ShapeHelper::handleTriangle(int positionInGroup)
{
	int pIG = -1;
	Geode* geode;
	Drawable * dr;
	TriangleShape * tri;

	char * comment = getCharParam("comment");

	if (comment == NULL) {
		string empty = "";
		comment = &empty[0];
	}

	// if we're not creating a new geometry, we're going to get the geode at position pIG
	if (positionInGroup != -1)
	{
		pIG = positionInGroup;
		geode = dynamic_cast<Geode*> (group->getChild(pIG));
		dr = dynamic_cast<Drawable*> (geode->getDrawable(0));
		tri = (TriangleShape*) dr;
	}
	// otherwise, we'll make a new geode
	else 
	{
		geode = new Geode();
	}


	// get radius parsed from the incoming data
	double len = getDoubleParam("length");

	if (!isnan(len))
	{
		len = len * RENDERSCALE;
		tri->setLength(len);
	}
	// if we're generating, give it a default value
	else if (pIG == -1)
	{
		len = 100;
	}


	Vec3d center = getCenter();

	// double check if center is initialized, can getCenter() return an unintialized Vec3d? or will it throw an error already?
	if (center.valid())
	{
		if (pIG != -1)
		{
			tri->setCenter(center);
		}
	} else
	{
		// if we're generating a new circle and there isn't a center specified
		if (pIG == -1)
		{
			center = Vec3d(0,0,0);
		}
	}

	// check colors too
	Vec4d c1 = getC1();

	if (c1.valid())
	{	
		if (pIG != -1)
		{
			tri->setColor(c1);
		}
	}
	else
	{
		// we'll give it a nice lavender color
		if (pIG == -1)
		{
			c1 = Vec4d(62/255, 180/255, 137/255,0.8);
		}
	}

	Vec4d c2 = getC2();

	if (c2.valid())
	{	
		if (pIG != -1)
		{
			tri->setGradient(c2);
		}
	}
	else
	{
		// default black
		if (pIG == -1)
		{
			c2 = Vec4d(0,0,0,0.8);
		}
	}


	if (pIG != -1)
	{
		tri->updateAll();
	}
	else 
	{
		tri = new TriangleShape(comment,center,len,c1,c2);
		geode->addDrawable(tri);
		group->addChild(geode);

		shapeCount++;

		// TODO add comment and geode to tracking tree
		tree->root = tree->insert(comment,pIG,geode,tree->root);
	}


}

// likewise, we're only handling rects with a center point
void ShapeHelper::handleRect(int positionInGroup)
{
	int pIG = -1;
	Geode* geode;
	Drawable * dr;
	RectShape * rect;

	char * comment = getCharParam("comment");

	if (comment == NULL) {
		string empty = "";
		comment = &empty[0];
	}

	// if we're not creating a new geometry, we're going to get the geode at position pIG
	if (positionInGroup != -1)
	{
		pIG = positionInGroup;
		geode = dynamic_cast<Geode*> (group->getChild(pIG));
		dr = dynamic_cast<Drawable*> (geode->getDrawable(0));
		rect = (RectShape*) dr;
	}
	// otherwise, we'll make a new geode
	else 
	{
		geode = new Geode();
	}



	double	height = getDoubleParam("height");
	double	wid = getDoubleParam("width");
	if (!isnan(height))
	{
		height = height * RENDERSCALE;
		rect->setHeight(height);
	}
	else if (pIG == -1)
	{
		height = 100;
	}
	if (!isnan(wid))
	{
		wid = wid * RENDERSCALE;
		rect->setWidth(wid);
	}
	else if (pIG == -1)
	{
		wid = 100;
	}

	Vec3d center = getCenter();

	// double check if center is initialized, can getCenter() return an unintialized Vec3d? or will it throw an error already?
	if (center.valid())
	{
		if (pIG != -1)
		{
			rect->setCenter(center);
		}
	} else
	{
		// if we're generating a new circle and there isn't a center specified
		if (pIG == -1)
		{
			center = Vec3d(0,0,0);
		}
	}

	// check colors too
	Vec4d c1 = getC1();

	if (c1.valid())
	{	
		if (pIG != -1)
		{
			rect->setColor(c1);
		}
	}
	else
	{
		// we'll give it a nice "jonquil" color
		if (pIG == -1)
		{
			c1 = Vec4d(250/255, 218/255, 94/255, 0.8);
		}
	}

	Vec4d c2 = getC2();

	if (c2.valid())
	{	
		if (pIG != -1)
		{
			rect->setGradient(c2);
		}
	}
	else
	{
		// default black
		if (pIG == -1)
		{
			c2 = Vec4d(0,0,0,0.8);
		}
	}


	if (pIG != -1)
	{
		rect->updateAll();
	}
	else 
	{
		rect = new RectShape(comment,center,wid,height,c1,c2);
		geode->addDrawable(rect);
		group->addChild(geode);

		shapeCount++;

		// TODO add comment and geode to tracking tree
		tree->root = tree->insert(comment,pIG,geode,tree->root);
	}
}

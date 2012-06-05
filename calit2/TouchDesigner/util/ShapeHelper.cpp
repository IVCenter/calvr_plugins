#include "ShapeHelper.h"

#include <fcntl.h>
#include <iostream>
#include <cstring>

#include <osg/Matrix>
#include <osg/CullFace>
#include <osgDB/ReadFile>

using namespace osg;
using namespace std;

#define DEBUGSHAPECOUNT 2
#define RENDERSCALE 1

int updateIndex = 0;

ShapeHelper::ShapeHelper(Geode * _geo) {
	geode = _geo;
	tok = new vvTokenizer();
	tok->setEOLisSignificant(true);
	tok->setCaseConversion(vvTokenizer::VV_LOWER);
	tok->setParseNumbers(true);
	tok->setWhitespaceCharacter('=');
	shapeCount = 0;
	processedAll = false;
	debugOn = false;
}

ShapeHelper::ShapeHelper() {
	geode = new Geode();
	tok = new vvTokenizer();
	tok->setEOLisSignificant(true);
	tok->setCaseConversion(vvTokenizer::VV_LOWER);
	tok->setParseNumbers(true);
	tok->setWhitespaceCharacter('=');
	shapeCount = 0;
	processedAll = false;
	debugOn = false;
}

ShapeHelper::~ShapeHelper() {
	delete tok;
}

void ShapeHelper::processData(char* data) {
	tok->newData(data);
	vvTokenizer::TokenType ttype;
	ttype = tok->nextToken();
	string pch = tok->sval;

	/*	if (ttype == vvTokenizer::VV_NUMBER)
	 {
	 int shapeId = tok->nval;
	 Drawable * db = geode->getDrawable(shapeId);            
	 Geometry * geo = db->asGeometry();
	 updateShape((BasicShape*)geo);		
	 }
	 
	 */
	//	else 

	if (0 == pch.compare("circle") && genAll) {
		updateCircle( updateIndex);
		updateIndex++;

		if (updateIndex >= DEBUGSHAPECOUNT) {
			updateIndex = 0;
			processedAll = true;
		}
		geode->dirtyBound();
	} else if (0 == pch.compare("circle")) {

		geode->addDrawable(genCircle());
		shapeCount++;

		if (shapeCount >= DEBUGSHAPECOUNT) {
			genAll = true;
		}
	} else if (0 == pch.compare("trianglec") && genAll) {
		updateTriangleC( updateIndex);

		updateIndex++;

		if (updateIndex >= DEBUGSHAPECOUNT) {
			updateIndex = 0;
			processedAll = true;
		}
		geode->dirtyBound();

	} else if (0 == pch.compare("trianglec")) {
		shapeCount++;
		geode->addDrawable(genTriangleC());
	} else if (0 == pch.compare("trianglep")) {
		shapeCount++;
		geode->addDrawable(genTriangleP());
	} else if (0 == pch.compare("rectc")) {
		shapeCount++;
		geode->addDrawable(genRectC());
	} else if (0 == pch.compare("rectp")) {
		shapeCount++;
		geode->addDrawable(genRectP());
	}
	// triangle with center given
	else if (0 == pch.compare("updatetc")) {
		updateTriangleC();
	}
	// triangle with points given
	else if (0 == pch.compare("updatetp")) {
		updateTriangleP();
	}
	// rectangle with center	
	else if (0 == pch.compare("updaterc")) {
		updateRectC();
	}
	// rectangle with points
	else if (0 == pch.compare("updaterp")) {
		updateRectP();
	}
	// circle
	else if (0 == pch.compare("updatec")) {
		updateCircle();
	} else if (0 == pch.compare("done")) {
		processedAll = true;
	} else {
		//		cerr << "command not recognized " << endl;
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

// generates a circle within incoming data packet
CircleShape* ShapeHelper::genCircle() {

	CircleShape * circle;

	double cx = getDoubleParam("cx") * RENDERSCALE;
	double cy = getDoubleParam("cy") * RENDERSCALE;
	double cz = getDoubleParam("cz") * RENDERSCALE;
	double radius = getDoubleParam("radius") * RENDERSCALE;
	double c1r = getDoubleParam("c1r");
	double c1g = getDoubleParam("c1g");
	double c1b = getDoubleParam("c1b");
	double c1a = getDoubleParam("c1a");
	double c2r = getDoubleParam("c2r");
	double c2g = getDoubleParam("c2g");
	double c2b = getDoubleParam("c2b");
	double c2a = getDoubleParam("c2a");
	int tess = getIntParam("tess");
	char * comment = getCharParam("comment");

	Vec3d center(cx, cy, cz);

	// cerr << "generating shape # " << shapeCount << endl;

	// if no tesselation specified, use 10 as default
	if (tess == 0) {
		tess = 10;
	}

	if (comment == NULL) {
		string empty = "";
		comment = &empty[0];
	}

	if (!isnan(c1r) && !isnan(c1g) && !isnan(c1b) && !isnan(c1a)) {
		Vec4d c1(c1r, c1g, c1b, c1a);

		if (!isnan(c2r) && !isnan(c2g) && !isnan(c2b) && !isnan(c2a)) {
			Vec4d c2(c2r, c2g, c2b, c2a);
			circle = new CircleShape(comment, center, radius, c1, c2, tess);
			circle->setId(shapeCount);
			return circle;
		}

		circle = new CircleShape(comment, center, radius, c1, tess);
		circle->setId(shapeCount);
		return circle;
	}

	circle = new CircleShape(comment, center, radius);
	circle->setId(shapeCount);
	return circle;
}

// generates a triangle with a center given
TriangleShape * ShapeHelper::genTriangleC() {
	TriangleShape * tr;
	int cx = getIntParam("cx") * RENDERSCALE;
	int cy = getIntParam("cy") * RENDERSCALE;
	int cz = getIntParam("cz") * RENDERSCALE;
	int height = getIntParam("length") * RENDERSCALE;
	double c1r = getDoubleParam("c1r");
	double c1g = getDoubleParam("c1g");
	double c1b = getDoubleParam("c1b");
	double c1a = getDoubleParam("c1a");
	double c2r = getDoubleParam("c2r");
	double c2g = getDoubleParam("c2g");
	double c2b = getDoubleParam("c2b");
	double c2a = getDoubleParam("c2a");
	char * comment = getCharParam("comment");

	Vec3d center(cx, cy, cz);

	cerr << "generating shape # " << shapeCount << endl;

	if (comment == NULL) {
		string empty = "";
		comment = &empty[0];
	}

	if (!isnan(c1r) && !isnan(c1g) && !isnan(c1b) && !isnan(c1a)) {
		Vec4d c1(c1r, c1g, c1b, c1a);

		if (!isnan(c2r) && !isnan(c2g) && !isnan(c2b) && !isnan(c2a)) {
			Vec4d c2(c2r, c2g, c2b, c2a);
			tr = new TriangleShape(comment, center, height, c1, c2);
			tr->setId(shapeCount);
			return tr;
		}

		tr = new TriangleShape(comment, center, height, c1);
		tr->setId(shapeCount);
		return tr;

	}

	tr = new TriangleShape(comment, center, height);
	tr->setId(shapeCount);
	return tr;

}

// generates a triangle with 3 points given
TriangleShape * ShapeHelper::genTriangleP() {
	TriangleShape * tr;
	int p1x = getIntParam("p1x") * RENDERSCALE;
	int p1y = getIntParam("p1y") * RENDERSCALE;
	int p1z = getIntParam("p1z") * RENDERSCALE;
	int p2x = getIntParam("p2x") * RENDERSCALE;
	int p2y = getIntParam("p2y") * RENDERSCALE;
	int p2z = getIntParam("p2z") * RENDERSCALE;
	int p3x = getIntParam("p3x") * RENDERSCALE;
	int p3y = getIntParam("p3y") * RENDERSCALE;
	int p3z = getIntParam("p3z") * RENDERSCALE;
	double c1r = getDoubleParam("c1r");
	double c1g = getDoubleParam("c1g");
	double c1b = getDoubleParam("c1b");
	double c1a = getDoubleParam("c1a");
	double c2r = getDoubleParam("c2r");
	double c2g = getDoubleParam("c2g");
	double c2b = getDoubleParam("c2b");
	double c2a = getDoubleParam("c2a");
	double c3r = getDoubleParam("c3r");
	double c3g = getDoubleParam("c3g");
	double c3b = getDoubleParam("c3b");
	double c3a = getDoubleParam("c3a");
	char * comment = getCharParam("comment");

	Vec3d p1(p1x, p1y, p1z);
	Vec3d p2(p2x, p2y, p2z);
	Vec3d p3(p3x, p3y, p3z);

	cerr << "generating shape # " << shapeCount << endl;

	if (comment == NULL) {
		string empty = "";
		comment = &empty[0];
	}

	if (!isnan(c1r) && !isnan(c1g) && !isnan(c1b) && !isnan(c1a)) {
		Vec4d c1(c1r, c1g, c1b, c1a);

		if (!isnan(c2r) && !isnan(c2g) && !isnan(c2b) && !isnan(c2a)) {
			Vec4d c2(c2r, c2g, c2b, c2a);

			if (!isnan(c3r) && !isnan(c3g) && !isnan(c3b) && !isnan(c3a)) {
				Vec4d c3(c3r, c3g, c3b, c3a);
				tr = new TriangleShape(comment, p1, p2, p3, c1, c2, c3);
				tr->setId(shapeCount);
				return tr;
			}
			tr = new TriangleShape(comment, p1, p2, p3, c1, c2);
			return tr;
		}

		tr = new TriangleShape(comment, p1, p2, p3, c1);
		tr->setId(shapeCount);
		return tr;

	}

	tr = new TriangleShape(comment, p1, p2, p3);
	tr->setId(shapeCount);
	return tr;

}

// generates a rectangle with center given
RectShape * ShapeHelper::genRectC() {
	RectShape * rect;
	int cx = getIntParam("cx") * RENDERSCALE;
	int cy = getIntParam("cy") * RENDERSCALE;
	int cz = getIntParam("cz") * RENDERSCALE;
	int height = getIntParam("height") * RENDERSCALE;
	int wid = getIntParam("width") * RENDERSCALE;
	double c1r = getDoubleParam("c1r");
	double c1g = getDoubleParam("c1g");
	double c1b = getDoubleParam("c1b");
	double c1a = getDoubleParam("c1a");
	double c2r = getDoubleParam("c2r");
	double c2g = getDoubleParam("c2g");
	double c2b = getDoubleParam("c2b");
	double c2a = getDoubleParam("c2a");
	char * comment = getCharParam("comment");

	Vec3d center(cx, cy, cz);

	cerr << "generating shape # " << shapeCount << endl;

	if (comment == NULL) {
		string empty = "";
		comment = &empty[0];
	}

	if (!isnan(c1r) && !isnan(c1g) && !isnan(c1b) && !isnan(c1a)) {
		Vec4d c1(c1r, c1g, c1b, c1a);

		if (!isnan(c2r) && !isnan(c2g) && !isnan(c2b) && !isnan(c2a)) {
			Vec4d c2(c2r, c2g, c2b, c2a);
			rect = new RectShape(comment, center, wid, height, c1, c2);
			rect->setId(shapeCount);
			return rect;
		}

		rect = new RectShape(comment, center, wid, height, c1);
		rect->setId(shapeCount);
		return rect;

	}

	rect = new RectShape(comment, center, wid, height);
	rect->setId(shapeCount);
	return rect;

}

//generates a rectangle with 4 points given
RectShape * ShapeHelper::genRectP() {
	RectShape * rect;
	int p1x = getIntParam("p1x") * RENDERSCALE;
	int p1y = getIntParam("p1y") * RENDERSCALE;
	int p1z = getIntParam("p1z") * RENDERSCALE;
	int p2x = getIntParam("p2x") * RENDERSCALE;
	int p2y = getIntParam("p2y") * RENDERSCALE;
	int p2z = getIntParam("p2z") * RENDERSCALE;
	int p3x = getIntParam("p3x") * RENDERSCALE;
	int p3y = getIntParam("p3y") * RENDERSCALE;
	int p3z = getIntParam("p3z") * RENDERSCALE;
	int p4x = getIntParam("p4x") * RENDERSCALE;
	int p4y = getIntParam("p4y") * RENDERSCALE;
	int p4z = getIntParam("p4z") * RENDERSCALE;
	double c1r = getDoubleParam("c1r");
	double c1g = getDoubleParam("c1g");
	double c1b = getDoubleParam("c1b");
	double c1a = getDoubleParam("c1a");
	double c2r = getDoubleParam("c2r");
	double c2g = getDoubleParam("c2g");
	double c2b = getDoubleParam("c2b");
	double c2a = getDoubleParam("c2a");
	char * comment = getCharParam("comment");

	Vec3d p1(p1x, p1y, p1z);
	Vec3d p2(p2x, p2y, p2z);
	Vec3d p3(p3x, p3y, p3z);
	Vec3d p4(p4x, p4y, p4z);

	cerr << "generating shape # " << shapeCount << endl;

	if (comment == NULL) {
		string empty = "";
		comment = &empty[0];
	}

	if (!isnan(c1r) && !isnan(c1g) && !isnan(c1b) && !isnan(c1a)) {
		Vec4d c1(c1r, c1g, c1b, c1a);

		if (!isnan(c2r) && !isnan(c2g) && !isnan(c2b) && !isnan(c2a)) {
			Vec4d c2(c2r, c2g, c2b, c2a);
			rect = new RectShape(comment, p1, p2, p3, p4, c1, c2);
			rect->setId(shapeCount);
			return rect;
		}

		rect = new RectShape(comment, p1, p2, p3, p4, c1);
		rect->setId(shapeCount);
		return rect;

	}

	rect = new RectShape(comment, p1, p2, p3, p4);
	rect->setId(shapeCount);
	return rect;

}

//returns a random number between 0 and 1
double ShapeHelper::random() {
	return rand() / double(RAND_MAX);
}

// returns a random number between min and max
double ShapeHelper::random(double min, double max) {
	return (max - min) * random() + min;
}

/**
 *	circle updating functions
 */
void ShapeHelper::updateCircle() {
	int id = getIntParam("id");
	Drawable * db = geode->getDrawable(id);
	Geometry * geo = db->asGeometry();
	updateCircle((CircleShape*) geo);
}

void ShapeHelper::updateCircle(int _id) {
	Drawable * db = geode->getDrawable(_id);
	Geometry * geo = db->asGeometry();
	updateCircle((CircleShape*) geo);
}

void ShapeHelper::updateCircle(CircleShape* geo) {

	double cx = getDoubleParam("cx") * RENDERSCALE;
	double cy = getDoubleParam("cy") * RENDERSCALE;
	double cz = getDoubleParam("cz") * RENDERSCALE;
	double radius = getDoubleParam("radius") * RENDERSCALE;
	double c1r = getDoubleParam("c1r");
	double c1g = getDoubleParam("c1g");
	double c1b = getDoubleParam("c1b");
	double c1a = getDoubleParam("c1a");
	double c2r = getDoubleParam("c2r");
	double c2g = getDoubleParam("c2g");
	double c2b = getDoubleParam("c2b");
	double c2a = getDoubleParam("c2a");

	Vec3d center(cx, cy, cz);

	//	cerr << "updating shape # " << geo->getId() << endl;

	if (!isnan(c1r) && !isnan(c1g) && !isnan(c1b) && !isnan(c1a)) {
		Vec4d c1(c1r, c1g, c1b, c1a);
		((CircleShape*) geo)->setColor(c1);
	}

	if (!isnan(c2r) && !isnan(c2g) && !isnan(c2b) && !isnan(c2a)) {
		Vec4d c2(c2r, c2g, c2b, c2a);
		((CircleShape*) geo)->setGradient(c2);
	}

	if (radius != 0) {
		((CircleShape*) geo)->setRadius(radius);
	}

	((CircleShape*) geo)->setCenter(center);
	((CircleShape*) geo)->updateAll();

}

/**
 *	rectangle with center updating functions
 */
void ShapeHelper::updateRectC() {
	int id = getIntParam("id");
	Drawable * db = geode->getDrawable(id);
	Geometry * geo = db->asGeometry();
	updateRectC((RectShape*) geo);
}

void ShapeHelper::updateRectC(RectShape* geo) {
	int cx = getIntParam("cx");
	int cy = getIntParam("cy");
	int cz = getIntParam("cz");
	int height = getIntParam("height");
	int wid = getIntParam("width");
	double c1r = getDoubleParam("c1r");
	double c1g = getDoubleParam("c1g");
	double c1b = getDoubleParam("c1b");
	double c1a = getDoubleParam("c1a");
	double c2r = getDoubleParam("c2r");
	double c2g = getDoubleParam("c2g");
	double c2b = getDoubleParam("c2b");
	double c2a = getDoubleParam("c2a");
	char * comment = getCharParam("comment");

	Vec3d center(cx, cy, cz);

	cerr << "updating shape # " << geo->getId() << endl;

	if (!isnan(c1r) && !isnan(c1g) && !isnan(c1b) && !isnan(c1a)) {
		Vec4d c1(c1r, c1g, c1b, c1a);
		((RectShape*) geo)->setColor1(c1);
	}

	if (!isnan(c2r) && !isnan(c2g) && !isnan(c2b) && !isnan(c2a)) {
		Vec4d c2(c2r, c2g, c2b, c2a);
		((RectShape*) geo)->setColor2(c2);
	}

	if (height > 0) {
		((RectShape*) geo)->setHeight(height);
	}

	if (wid > 0) {
		((RectShape*) geo)->setWidth(wid);
	}

	((RectShape*) geo)->setCenter(center);

	((RectShape*) geo)->updateAll();
}

/**
 *	triangle with center updating functions
 */
void ShapeHelper::updateTriangleC() {
	int id = getIntParam("id");
	updateTriangleC(id);
}

void ShapeHelper::updateTriangleC(int _id) {
	Drawable * db = geode->getDrawable(_id);
	Geometry * geo = db->asGeometry();
	updateTriangleC((TriangleShape*) geo);
}

void ShapeHelper::updateTriangleC(TriangleShape* geo) {
	int cx = getIntParam("cx");
	int cy = getIntParam("cy");
	int cz = getIntParam("cz");
	int len = getIntParam("length");
	double c1r = getDoubleParam("c1r");
	double c1g = getDoubleParam("c1g");
	double c1b = getDoubleParam("c1b");
	double c1a = getDoubleParam("c1a");
	double c2r = getDoubleParam("c2r");
	double c2g = getDoubleParam("c2g");
	double c2b = getDoubleParam("c2b");
	double c2a = getDoubleParam("c2a");
	char * comment = getCharParam("comment");

	Vec3d center(cx, cy, cz);

	cerr << "updating shape # " << geo->getId() << endl;

	if (!isnan(c1r) && !isnan(c1g) && !isnan(c1b) && !isnan(c1a)) {
		Vec4d c1(c1r, c1g, c1b, c1a);
		((TriangleShape*) geo)->setColor1(c1);
	}

	if (!isnan(c2r) && !isnan(c2g) && !isnan(c2b) && !isnan(c2a)) {
		Vec4d c2(c2r, c2g, c2b, c2a);
		((TriangleShape*) geo)->setColor2(c2);
	}

	if (len > 0) {
		((TriangleShape*) geo)->setLength(len);
	}

	((TriangleShape*) geo)->setCenter(center);

	((TriangleShape*) geo)->updateAll();
}

/**
 *	triangle with points updating functions
 */
void ShapeHelper::updateTriangleP() {
	int id = getIntParam("id");
	Drawable * db = geode->getDrawable(id);
	Geometry * geo = db->asGeometry();
	updateTriangleP((TriangleShape*) geo);
}

void ShapeHelper::updateTriangleP(TriangleShape* geo) {
	int p1x = getIntParam("p1x");
	int p1y = getIntParam("p1y");
	int p1z = getIntParam("p1z");
	int p2x = getIntParam("p2x");
	int p2y = getIntParam("p2y");
	int p2z = getIntParam("p2z");
	int p3x = getIntParam("p3x");
	int p3y = getIntParam("p3y");
	int p3z = getIntParam("p3z");
	double c1r = getDoubleParam("c1r");
	double c1g = getDoubleParam("c1g");
	double c1b = getDoubleParam("c1b");
	double c1a = getDoubleParam("c1a");
	double c2r = getDoubleParam("c2r");
	double c2g = getDoubleParam("c2g");
	double c2b = getDoubleParam("c2b");
	double c2a = getDoubleParam("c2a");
	double c3r = getDoubleParam("c3r");
	double c3g = getDoubleParam("c3g");
	double c3b = getDoubleParam("c3b");
	double c3a = getDoubleParam("c3a");
	char * comment = getCharParam("comment");

	Vec3d p1(p1x, p1y, p1z);
	Vec3d p2(p2x, p2y, p2z);
	Vec3d p3(p3x, p3y, p3z);

	((TriangleShape*) geo)->setPoints(p1, p2, p3);

	cerr << "updating shape # " << geo->getId() << endl;

	if (!isnan(c1r) && !isnan(c1g) && !isnan(c1b) && !isnan(c1a)) {
		Vec4d c1(c1r, c1g, c1b, c1a);
		((TriangleShape*) geo)->setColor1(c1);

	}

	if (!isnan(c2r) && !isnan(c2g) && !isnan(c2b) && !isnan(c2a)) {
		Vec4d c2(c2r, c2g, c2b, c2a);
		((TriangleShape*) geo)->setColor2(c2);

	}

	if (!isnan(c3r) && !isnan(c3g) && !isnan(c3b) && !isnan(c3a)) {
		Vec4d c3(c3r, c3g, c3b, c3a);
		((TriangleShape*) geo)->setColor3(c3);
	}

	((TriangleShape*) geo)->updateAll();
}

/**nt
 *	rectangle with points updating functions
 */
void ShapeHelper::updateRectP() {
	int id = getIntParam("id");
	Drawable * db = geode->getDrawable(id);
	Geometry * geo = db->asGeometry();
	updateRectP((RectShape*) geo);
}

void ShapeHelper::updateRectP(RectShape* geo) {
	int p1x = getIntParam("p1x");
	int p1y = getIntParam("p1y");
	int p1z = getIntParam("p1z");
	int p2x = getIntParam("p2x");
	int p2y = getIntParam("p2y");
	int p2z = getIntParam("p2z");
	int p3x = getIntParam("p3x");
	int p3y = getIntParam("p3y");
	int p3z = getIntParam("p3z");
	int p4x = getIntParam("p4x");
	int p4y = getIntParam("p4y");
	int p4z = getIntParam("p4z");
	double c1r = getDoubleParam("c1r");
	double c1g = getDoubleParam("c1g");
	double c1b = getDoubleParam("c1b");
	double c1a = getDoubleParam("c1a");
	double c2r = getDoubleParam("c2r");
	double c2g = getDoubleParam("c2g");
	double c2b = getDoubleParam("c2b");
	double c2a = getDoubleParam("c2a");
	char * comment = getCharParam("comment");

	Vec3d p1(p1x, p1y, p1z);
	Vec3d p2(p2x, p2y, p2z);
	Vec3d p3(p3x, p3y, p3z);
	Vec3d p4(p4x, p4y, p4z);

	((RectShape*) geo)->setPoints(p1, p2, p3, p4);

	cerr << "updating shape # " << geo->getId() << endl;

	if (!isnan(c1r) && !isnan(c1g) && !isnan(c1b) && !isnan(c1a)) {
		Vec4d c1(c1r, c1g, c1b, c1a);
		((RectShape*) geo)->setColor1(c1);
	}

	if (!isnan(c2r) && !isnan(c2g) && !isnan(c2b) && !isnan(c2a)) {
		Vec4d c2(c2r, c2g, c2b, c2a);
		((RectShape*) geo)->setColor2(c2);
	}

	((RectShape*) geo)->updateAll();

}

Geode * ShapeHelper::getGeode() {
	return geode;
}

// lets 
void ShapeHelper::updateShape(int _uType) {

	// circles/shared
	double cx = 0;
	double cy = 0;
	double cz = 0;
	double radius = 100;
	int tess = 10;

	// tri
	double len = 100;

	//rect
	double height = 100;
	double wid = 100;

	//shared
	double c1r = 1.0;
	double c1g = 1.0;
	double c1b = 1.0;
	double c1a = 0.8;
	double c2r = 0.0;
	double c2g = 0.0;
	double c2b = 0.0;
	double c2a = 0.8;
	double c3r = 1.0;
	double c3g = 1.0;
	double c3b = 1.0;
	double c3a = 0.8;

	char * comment = NULL;

	//tri
	double p1x = 0;
	double p1y = 0;
	double p1z = 0;
	double p2x = 0;
	double p2y = 0;
	double p2z = 0;
	double p3x = 0;
	double p3y = 0;
	double p3z = 0;

	//rect
	double p4x = 0;
	double p4y = 0;
	double p4z = 0;

	// position vectors
	Vec3d center(0,0,0);
	Vec3d p1(0,0,0);
	Vec3d p2(0,0,0);
	Vec3d p3(0,0,0);
	Vec3d p4(0,0,0);

	// color vectorsnt
	Vec4d c1(0,0,0,1);
	Vec4d c2(0,0,0,1);
	Vec4d c3(0,0,0,1);

	// fetch the colors and comments
	c1r = getDoubleParam("c1r");
	c1g = getDoubleParam("c1g");
	c1b = getDoubleParam("c1b");
	c1a = getDoubleParam("c1a");
	c2r = getDoubleParam("c2r");
	c2g = getDoubleParam("c2g");
	c2b = getDoubleParam("c2b");
	c2a = getDoubleParam("c2a");
	c3r = getDoubleParam("c3r");
	c3g = getDoubleParam("c3g");
	c3b = getDoubleParam("c3b");
	c3a = getDoubleParam("c3a");
	
	
	comment = getCharParam("comment");

	if (!isnan(c1r) && !isnan(c1g) && !isnan(c1b) && !isnan(c1a)) {
		c1 = Vec4d(c1r, c1g, c1b, c1a);
	}

	if (!isnan(c2r) && !isnan(c2g) && !isnan(c2b) && !isnan(c2a)) {
		c2 = Vec4d(c2r, c2g, c2b, c2a);
	}
	
	if (!isnan(c3r) && !isnan(c3g) && !isnan(c3b) && !isnan(c3a)) {
		c3 = Vec4d(c3r, c3g, c3b, c3a);
	}

	if (comment == NULL) {
		string empty = "";
		comment = &empty[0];
	}
	
	
	

	// determine whether its a shape with center point specified or vertices specifed, odd number = center
	if (_uType % 2 == 1) {
		cx = getDoubleParam("cx") * RENDERSCALE;
		cy = getDoubleParam("cy") * RENDERSCALE;
		cz = getDoubleParam("cz") * RENDERSCALE;

		center = Vec3d(cx, cy, cz);

		// if it's a circle, we get radius and tess
		if (_uType == 5) {
			radius = getDoubleParam("radius") * RENDERSCALE;
			tess = getIntParam("tess") * RENDERSCALE;

		}
		// rectangles, get width and height
		else if (_uType == 3) {
			height = getDoubleParam("height");
			wid = getDoubleParam("width");
		}
		// triangles, get length
		else if (_uType == 1) {
			len = getDoubleParam("length");
		}

	} else {
		p1x = getDoubleParam("p1x") * RENDERSCALE;
		p1y = getDoubleParam("p1y") * RENDERSCALE;
		p1z = getDoubleParam("p1z") * RENDERSCALE;
		p2x = getDoubleParam("p2x") * RENDERSCALE;
		p2y = getDoubleParam("p2y") * RENDERSCALE;
		p2z = getDoubleParam("p2z") * RENDERSCALE;
		p3x = getDoubleParam("p3x") * RENDERSCALE;
		p3y = getDoubleParam("p3y") * RENDERSCALE;
		p3z = getDoubleParam("p3z") * RENDERSCALE;

		p1 = Vec3d(p1x, p1y, p1z);
		p2 = Vec3d(p2x, p2y, p2z);
		p3 = Vec3d(p3x, p3y, p3z);

		p4x = getDoubleParam("p4x") * RENDERSCALE;
		p4y = getDoubleParam("p4y") * RENDERSCALE;
		p4z = getDoubleParam("p4z") * RENDERSCALE;

		p4 = Vec3d(p4x, p4y, p4z);

	}

}

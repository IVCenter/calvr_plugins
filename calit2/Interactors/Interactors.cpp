#include "Interactors.h"

#include <cvrKernel/InteractionManager.h>
#include <cvrConfig/ConfigManager.h>

#include <osg/PolygonMode>
#include <osg/LineWidth>
#include <osgDB/ReadFile>

#include <string>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

CVRPLUGIN(Interactors)

using namespace cvr;

// For debbuging
void print(osg::Vec3 a)
{
	std::cout << "x = " << a.x() << " y = " << a.y() << " z = " << a.z() << std::endl;
}

Interactors::Interactors()
{
    _activePlane = -1;
    _activePart = -1;
	_moving = false;
	_firstTime = true;
}

Interactors::~Interactors()
{
}

bool Interactors::init()
{
    _clipPlaneMenu = new SubMenu("Interactors","Interactors");
	_planesNode = new osg::Group();
//	_bbg = new osg::Geode();
//	_lastBox = new osg::ShapeDrawable();
//	_bbg->addDrawable(_lastBox);
//	_planesNode->addChild(_bbg);
	PluginHelper::getObjectsRoot()->addChild(_planesNode);
    for(int i = 0; i < MAX_CLIP_PLANES; i++)
	{
		_planeList.push_back(new PlaneInfo(i, _planesNode));

		std::stringstream ss;
		ss << "Place Plane " << i;
		MenuCheckbox * cb = new MenuCheckbox(ss.str(),false);
		cb->setCallback(this);
		_placeList.push_back(cb);
		_clipPlaneMenu->addItem(cb);

		std::stringstream ss2;
		ss2 << "Enable Plane " << i;
		cb = new MenuCheckbox(ss2.str(),false);
		cb->setCallback(this);
		_enableList.push_back(cb);
		_clipPlaneMenu->addItem(cb);
	}
	_arrowTexture = new osg::Texture2D();
	// protect from being optimized away as static state:
	_arrowTexture->setDataVariance(osg::Object::DYNAMIC); 

        std::string iconpath = ConfigManager::getEntry("Plugin.Interactors.DoubleArrow");
	osg::Image* arrow = osgDB::readImageFile(iconpath);
	if (arrow)
		_arrowTexture->setImage(arrow);
	else
		std::cerr << "Interactors: couldn't find texture." << std::endl;

	PluginHelper::addRootMenuItem(_clipPlaneMenu);

	//	osg::LineWidth* linewidth = new osg::LineWidth();
//	linewidth->setWidth(20.0f);
//	_bbg->getOrCreateStateSet()->setAttributeAndModes(linewidth, osg::StateAttribute::ON); 

	/*
//  Big QUAD on xy plane 
	osg::Vec3Array* verts = new osg::Vec3Array;
	grid = new osg::Geometry;
	verts->push_back(osg::Vec3(-5000,5000,0));
	verts->push_back(osg::Vec3(-5000,-5000,0));
	verts->push_back(osg::Vec3(5000,-5000,0));
	verts->push_back(osg::Vec3(5000,5000,0));
	grid->setVertexArray( verts );
	grid->addPrimitiveSet(new osg::DrawArrays( GL_QUADS, 0, 4) );
	osg::PolygonMode* polygonMode = new osg::PolygonMode();	 
	// Wireframe for outer quad
	polygonMode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::FILL);
 	osg::StateSet* stateSet = grid->getOrCreateStateSet();
	stateSet->setAttributeAndModes(polygonMode, osg::StateAttribute::ON);
	stateSet->setMode( GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED );
	stateSet->setTextureMode( 0, GL_TEXTURE_2D, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED );
//	_bbg->addDrawable(grid);
*/
	return true;
}

void Interactors::menuCallback(MenuItem * item)
{
	for(int i = 0; i < _placeList.size(); i++)
	{
		if(item == _placeList[i])
		{
			if(_placeList[i]->getValue())
			{
				if(_activePlane >= 0)
				{
					_placeList[i]->setValue(false);
				}
				setSelectedGeometry(i,PLANE);
				_moving = true;				

				// Reset plane position
				osg::Matrix hTr;
				osg::Matrix h2w = PluginHelper::getHandMat(0);	
				osg::Matrix w2o = PluginHelper::getWorldToObjectTransform();
				hTr.makeTranslate(osg::Vec3(0.,4000.,0.));
				osg::Matrix h2o = hTr * h2w * w2o;
				_planeList[i]->local2o->setMatrix(h2o);

				if(!_enableList[i]->getValue())
				{
					_enableList[i]->setValue(true);
					menuCallback(_enableList[i]);
				}
				//osg::StateSet * stateset = PluginHelper::getObjectTransform()->getOrCreateStateSet();

				//stateset->setAttributeAndModes(_planeList[i],osg::StateAttribute::ON);
			}
			else
			{
				deselectGeometry();
				_moving = false;
			}
			return;
		}
	}

	for(int i = 0; i < _enableList.size(); i++)
	{
		if(item == _enableList[i])
		{
			_planeList[i]->isEnabled = _enableList[i]->getValue();
			//osg::StateSet * stateset = PluginHelper::getObjectTransform()->getOrCreateStateSet();

			if(_enableList[i]->getValue())
			{
				//std::cerr << "Enable plane " << i << std::endl;
				//stateset->setAttributeAndModes(_planeList[i],osg::StateAttribute::ON);
				PluginHelper::getObjectsRoot()->addClipPlane(_planeList[i]->clipPlane.get());
			}
			else
			{
				if(i == _activePlane)
				{
					_placeList[i]->setValue(false);
					menuCallback(_placeList[i]);
				}
				//std::cerr << "Disable plane " << i << std::endl;
				PluginHelper::getObjectsRoot()->removeClipPlane(_planeList[i]->clipPlane.get());
				//stateset->setAttributeAndModes(_planeList[i],osg::StateAttribute::OFF);
			}

			return;
		}
	}
}

bool Interactors::processEvent(InteractionEvent * event)
{
	if(event->getEventType() == TRACKED_BUTTON_INTER_EVENT)
		buttonEvent(event->getInteraction(), event->asTrackedButtonEvent()->getButton(), 0, PluginHelper::getHandMat());
	return false;
}

void Interactors::deselectGeometry(int part)
{
	osg::Vec4Array* c = new osg::Vec4Array;
	osg::Vec4 selected_color(0.,1.,0.,1.);
	osg::Vec4 unselected_color(1.,1.,1.,1.);
	
	if(_activePlane < 0)
		return;

	if(part < 0)
		part = _activePart;

	if(part == PLANE)
	{
		osg::Vec4Array* c = new osg::Vec4Array;
		c->push_back(unselected_color);
		_planeList[_activePlane]->quad->setColorArray(c);
	}
	else if(part == ARROW)
	{
		osg::Vec4Array* c = new osg::Vec4Array;
		c->push_back(unselected_color);
		_planeList[_activePlane]->arrow->setColorArray(c);
	}
	else if(part == WIRE)
	{
		osg::Vec4Array* c = new osg::Vec4Array;
		c->push_back(unselected_color);
		_planeList[_activePlane]->toggleWire->setColorArray(c);
	}

	_activePlane = -1;
	_activePart = -1;
}

void Interactors::setSelectedGeometry(int planeIndex, int part)
{
	osg::Vec4Array* c = new osg::Vec4Array;
	osg::Vec4 selected_color(0.,1.,0.,1.);
	osg::Vec4 unselected_color(1.,1.,1.,1.);

	deselectGeometry();
	if(planeIndex < 0 ) return;

	_activePart = part;
	if(part == PLANE)
	{
		_activePlane = planeIndex;
		if(_planeList[planeIndex]->quad != NULL)
		{
			osg::Vec4Array* c = new osg::Vec4Array;
			c->push_back(selected_color);
			_planeList[planeIndex]->quad->setColorArray(c);
		}
	}
	else if(part == ARROW)
	{
		_activePlane = planeIndex;
		if(_planeList[planeIndex]->arrow != NULL)
		{
			osg::Vec4Array* c = new osg::Vec4Array;
			c->push_back(selected_color);
			_planeList[planeIndex]->arrow->setColorArray(c);
		}
	}
	else if(part == WIRE)
	{
		_activePlane = planeIndex;
		if(_planeList[planeIndex]->toggleWire != NULL)
		{
			osg::Vec4Array* c = new osg::Vec4Array;
			c->push_back(selected_color);
			_planeList[planeIndex]->toggleWire->setColorArray(c);
		}
	}

}

bool Interactors::buttonEvent(int type, int button, int hand, const osg::Matrix & mat)
{
	if(hand != 0 || button != 0) return false;
	if(type == BUTTON_DOWN && _activePart == WIRE)
	{
		PlaneInfo* p =_planeList[_activePlane];
		int i,j;
	    i = p->wireIndex;
		j = p->planeSwitch->getNumChildren();

		if(i>=j)
			p->planeSwitch->setAllChildrenOff();
		else if(i==0)
			p->planeSwitch->setValue(0,true);
		else
		{
			p->planeSwitch->setValue(i-1,false);
			p->planeSwitch->setValue(i,true);
		}
		p->wireIndex = (i+1)%(j+1);
		return true;
	}
	if(type == BUTTON_DOWN && _activePlane >= 0 && !_moving)
	{
		_moving = true;
		_firstTime = true;
		return true;
	}
	if(type == BUTTON_DOWN && _activePlane >= 0 && _moving)
	{
		_moving = false;
		_firstTime = true;
		_placeList[_activePlane]->setValue(false);
		menuCallback(_placeList[_activePlane]);
		return true;
	}
	return false;
}

// function adapted from osgWorks library:
// static bool buildWirePlaneData
bool Interactors::buildPlane( osg::Matrix l2o, int planeIndex )
{
	_planeList[planeIndex]->local2o->setMatrix(l2o);
	_planeList[planeIndex]->quad = new osg::Geometry();
	_planeList[planeIndex]->cross = new osg::Geometry();
	_planeList[planeIndex]->wireframe = new osg::Geometry();
	osg::Vec3Array* verts;
	osg::Geometry* geom;
	osg::Geode* geode;

	// Changing only these vectors will change the size of the geometry
	osg::Vec3 u(1000.,0.,0.), v(0.,0.,1000.);	

	_planeList[planeIndex]->lPoint = osg::Vec3(0.,0.,0.);
	_planeList[planeIndex]->lNormal = osg::Vec3(0.,-1.,0.);
	// The y axis offset is to prevent a plane from clipping off its own geometry
	osg::Vec3 corner(-u.x()/2, -1., -v.z()/2);
	
	// Building plane wireframe
	geom = _planeList[planeIndex]->wireframe;
	geode = _planeList[planeIndex]->wireframeG;
   	verts = new osg::Vec3Array;
	osg::Vec2s subdivisions(10,10);

	osg::Vec3 end( corner + v );
	int idx;
	for( idx=0; idx <= subdivisions.x(); idx++ )
	{
		const float percent( (float)idx / (float)(subdivisions.x()) );
		const osg::Vec3 strut( u * percent );
		verts->push_back( corner+strut );
		verts->push_back( end+strut );
	}
	end.set( corner + u );
	for( idx=0; idx <= subdivisions.y(); idx++ )
	{
		const float percent( (float)idx / (float)(subdivisions.y()) );
		const osg::Vec3 strut( v * percent );
		verts->push_back( corner+strut );
		verts->push_back( end+strut );
	}

	geom->setVertexArray( verts );
	geom->setColorBinding( osg::Geometry::BIND_OVERALL );

	geom->addPrimitiveSet(new osg::DrawArrays( GL_LINES, 0, verts->getNumElements()) );

	osg::PolygonMode* polygonMode = new osg::PolygonMode();
	polygonMode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
	// Disable lighting and texture mapping for wire primitives.
	osg::StateSet* stateSet = geom->getOrCreateStateSet();
	stateSet->setAttributeAndModes(polygonMode, osg::StateAttribute::ON);
	stateSet->setMode( GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED );
	stateSet->setTextureMode( 0, GL_TEXTURE_2D, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED );
	
	geode->addDrawable(geom);

	// Building plane's central cross
	verts = new osg::Vec3Array;
	geom = _planeList[planeIndex]->cross;
	geode = _planeList[planeIndex]->crossG;

	verts->push_back(corner + (u/2));
	verts->push_back(corner + (u/2) + v);
	verts->push_back(corner + (v/2));
	verts->push_back(corner + (v/2) + u);

	geom->setVertexArray( verts );
	geom->setColorBinding( osg::Geometry::BIND_OVERALL );

	geom->addPrimitiveSet(new osg::DrawArrays( GL_LINES, 0, verts->getNumElements()) );

	polygonMode = new osg::PolygonMode();
	polygonMode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
	// Disable lighting and texture mapping for wire primitives.
	stateSet = geom->getOrCreateStateSet();
	stateSet->setAttributeAndModes(polygonMode, osg::StateAttribute::ON);
	stateSet->setMode( GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED );
	stateSet->setTextureMode( 0, GL_TEXTURE_2D, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED );
	
	geode->addDrawable(geom);


	// Building plane's outer QUAD
	verts = new osg::Vec3Array;
	geom = _planeList[planeIndex]->quad;
	geode = _planeList[planeIndex]->quadG;

	verts->push_back(corner + v + u); 
	verts->push_back(corner + v);
	verts->push_back(corner);
	verts->push_back(corner + u);	

	geom->addPrimitiveSet(new osg::DrawArrays( GL_QUADS, 0, 4) );

	geom->setVertexArray( verts );
	geom->setColorBinding( osg::Geometry::BIND_OVERALL );

	geom->addPrimitiveSet(new osg::DrawArrays( GL_LINES, 0, verts->getNumElements()) );

	polygonMode = new osg::PolygonMode();
	polygonMode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
	// Disable lighting and texture mapping for wire primitives.
	stateSet = geom->getOrCreateStateSet();
	stateSet->setAttributeAndModes(polygonMode, osg::StateAttribute::ON);
	stateSet->setMode( GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED );
	stateSet->setTextureMode( 0, GL_TEXTURE_2D, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED );
	
	geode->addDrawable(geom);

	_planeList[planeIndex]->planeSwitch->setAllChildrenOff();
	_planeList[planeIndex]->planeSwitch->setValue(0,true);

	// Done building plane, now build arrow

	osg::Geode* arrowGeode = _planeList[planeIndex]->arrowGeode;
	_planeList[planeIndex]->arrow = new osg::Geometry();
	geom = _planeList[planeIndex]->arrow;
	verts = new osg::Vec3Array;
	
	verts->push_back( corner+v+u + osg::Vec3(10,-1,0));
	verts->push_back( corner+v+u + osg::Vec3(10,-1,-70));
	verts->push_back( corner+v+u + osg::Vec3(80,-1,-70));
	verts->push_back( corner+v+u + osg::Vec3(80,-1,0));

	osg::Vec2Array* texcoords = new osg::Vec2Array(4);
	(*texcoords)[0].set(0.0f,0.0f); 
	(*texcoords)[1].set(0.0f,1.0f);
	(*texcoords)[2].set(1.0f,1.0f); 
	(*texcoords)[3].set(1.0f,0.0f);
	geom->setTexCoordArray(0,texcoords);

	geom->setVertexArray( verts );
	geom->setColorBinding( osg::Geometry::BIND_PER_PRIMITIVE_SET );
	osg::Vec4Array* c = new osg::Vec4Array;
	osg::Vec4 unselected_color(1.,1.,1.,1.);
	c->push_back(unselected_color);
	geom->setColorArray(c);

	geom->addPrimitiveSet(new osg::DrawArrays( GL_QUADS, 0, 4) );

	polygonMode = new osg::PolygonMode();	 
	polygonMode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::FILL);

	// Disable lighting and texture mapping for wire primitives.
	stateSet = geom->getOrCreateStateSet();
	stateSet->setAttributeAndModes(polygonMode, osg::StateAttribute::ON);
	stateSet->setMode( GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED );
	stateSet->setTextureMode( 0, GL_TEXTURE_2D, osg::StateAttribute::ON);
	stateSet->setTextureAttributeAndModes(0,_arrowTexture,osg::StateAttribute::ON);

	arrowGeode->addDrawable(geom);


	// Done building arrow. Build wire icon

	geode = _planeList[planeIndex]->toggleWireGeode;
	_planeList[planeIndex]->toggleWire = new osg::Geometry();
	geom = _planeList[planeIndex]->toggleWire;
	verts = new osg::Vec3Array;
	
	verts->push_back( corner+v+u + osg::Vec3(10,-1,-80));
	verts->push_back( corner+v+u + osg::Vec3(10,-1,-150));
	verts->push_back( corner+v+u + osg::Vec3(80,-1,-150));
	verts->push_back( corner+v+u + osg::Vec3(80,-1,-80));
	
	verts->push_back( corner+v+u + osg::Vec3(10,-1,-115));
	verts->push_back( corner+v+u + osg::Vec3(80,-1,-115));
	verts->push_back( corner+v+u + osg::Vec3(45,-1,-80));
	verts->push_back( corner+v+u + osg::Vec3(45,-1,-150));

	geom->setVertexArray( verts );
	geom->setColorBinding( osg::Geometry::BIND_OVERALL );
	c = new osg::Vec4Array;
	c->push_back(unselected_color);
	geom->setColorArray(c);

	geom->addPrimitiveSet(new osg::DrawArrays( GL_QUADS, 0, 4) );
	geom->addPrimitiveSet(new osg::DrawArrays( GL_LINES, 4, 4) );

	polygonMode = new osg::PolygonMode();	 
	polygonMode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);

	stateSet = geom->getOrCreateStateSet();
	stateSet->setAttributeAndModes(polygonMode, osg::StateAttribute::ON);
	stateSet->setMode( GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::PROTECTED );
	stateSet->setTextureMode( 0, GL_TEXTURE_2D, osg::StateAttribute::OFF);

	geode->addDrawable(geom);

	if(_activePlane == planeIndex)
		setSelectedGeometry(_activePlane, _activePart);

	return( true );
}

void Interactors::movePlane(osg::Matrix last_h2o, osg::Matrix h2o, int planeIndex)
{
	PlaneInfo* p = _planeList[planeIndex];	
	osg::Matrix w2o = PluginHelper::getWorldToObjectTransform();
	osg::Matrix oDiff = osg::Matrix::inverse(last_h2o) * h2o;	
	
	p->local2o->postMult(oDiff);
	p->clipPlane->setClipPlane(osg::Plane(p->getONormal(), p->getOPoint()));
}

void Interactors::resizePlane(osg::Vec3 oLast_point, osg::Vec3 oPoint, int planeIndex)
{
	osg::Matrix o2l, ls, s;
	o2l = _planeList[planeIndex]->local2o->getInverseMatrix();
	s.makeScale(oPoint * o2l);
	ls.makeScale(oLast_point * o2l);
	_planeList[planeIndex]->local2o->preMult(s*osg::Matrix::inverse(ls));
}

void Interactors::preFrame()
{
	osg::Matrix h2w = PluginHelper::getHandMat(0);	
	osg::Matrix w2o = PluginHelper::getWorldToObjectTransform();
	osg::Vec3 oPointer;

	if(_activePlane >= 0 && _moving) 	
	{		
		osg::Matrix hTr;
		hTr.makeTranslate(osg::Vec3(0.,4000.,0.));
		osg::Matrix h2o = hTr * h2w * w2o;

		if(_activePart == PLANE) // A plane is being placed
		{
			if(_planeList[_activePlane]->quad == NULL)
			{
				buildPlane(h2o, _activePlane);
				_lastMat = h2o;
			}
			if(_firstTime)
			{
				_firstTime = false;
				_lastMat = h2o;
				return;
			}
			movePlane(_lastMat, h2o, _activePlane);
			_lastMat = h2o;
		}
		else if(_activePart == ARROW)
		{
			osg::Vec3 p = h2o.getTrans() - _planeList[_activePlane]->getOPoint();
			if(_firstTime)
			{
				_lastPoint = p;
				_firstTime = false;
			}
			else
			{
				resizePlane(_lastPoint, p, _activePlane);
				_lastPoint = p;
			}
		}
	}
	else // Check if the wand is intersecting any geometry
	{
		osg::Vec3 pointerStart = h2w.getTrans() * w2o;
		oPointer.set(0.0f, 10000.0f, 0.0f);
		oPointer = oPointer * h2w;
		oPointer = oPointer * w2o;

		std::vector<IsectInfo> isecvec;
		isecvec = getObjectIntersection(_planesNode, pointerStart, oPointer);
		std::vector<IsectInfo>::iterator it;

		for(it = isecvec.begin(); it < isecvec.end(); it++)
			if(it->found)
				for(int i = 0; i < _planeList.size(); i++)
				{
					if(it->geode == _planeList[i]->quadG)
					{
						setSelectedGeometry(i,PLANE);
						return;
					}
					else if(it->geode == _planeList[i]->arrowGeode)
					{
						setSelectedGeometry(i,ARROW);
						return;
					}
					else if(it->geode == _planeList[i]->toggleWireGeode)
					{
						setSelectedGeometry(i,WIRE);
						return;
					}
				}
		deselectGeometry();
	}
}

void Interactors::message(int type, char*& data, bool colaborative)
{
	if(type == GET_ALL_WPLANES)
	{
		/* Sample usage
		 *	osg::Matrix w2o = PluginHelper::getWorldToObjectTransform();
		 *	std::vector<osg::Vec3> * planes;
		 *	planes = (std::vector<osg::Vec3>*)malloc(sizeof(std::vector<osg::Vec3>));
		 *	message(GET_ALL_WPLANES, reinterpret_cast<char*&>(planes), false);
		 *	std::vector<osg::Vec3> pl = *planes;
         *
		 *	for(int i=0;i<pl.size();i+=2) 
		 *		_planeList[i/2]->clipPlane->setClipPlane(osg::Plane(pl[i]*w2o,pl[i+1]*w2o));
         *
		 *	delete planes;
		 */
		std::vector<osg::Vec3> * planes;
		planes = new std::vector<osg::Vec3>();
		for(int i=0;i<_planeList.size();i++)	
		{
			if(_planeList[i]->isEnabled)
			{
				planes->push_back(osg::Vec3(_planeList[i]->getWNormal()));
				planes->push_back(osg::Vec3(_planeList[i]->getWPoint()));
			}
		}
		memcpy(data,planes,sizeof(*planes));
		free(planes);
	}
	/*
	else if(type == GET_WPOINT)
	{
		osg::Vec3 a = _planeList[_activePlane]->getWPoint();
		memcpy((void*)(data), (void*)(&a), sizeof(a));
	}
	else if(type == GET_WNORMAL)
	{
		osg::Vec3 a = _planeList[_activePlane]->getWNormal();
		memcpy((void*)(data), (void*)(&a), sizeof(a));
	}
	else if(type == GET_OPOINT)
	{
		osg::Vec3 a = _planeList[_activePlane]->getOPoint();
		memcpy((void*)(data), (void*)(&a), sizeof(a));
	}
	else if(type == GET_ONORMAL)
	{
		osg::Vec3 a = _planeList[_activePlane]->getONormal();
		memcpy(data, (void*)(&a), sizeof(a));
	}
*/	
}

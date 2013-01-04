#ifndef CLIP_PLANE_PLUGIN_H
#define CLIP_PLANE_PLUGIN_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrKernel/PluginHelper.h>

#include <osg/Geode>
#include <osg/Texture2D>

#include <osg/ClipPlane>
#include <osg/PositionAttitudeTransform>
#include <osg/MatrixTransform>

#include <vector>

#define MAX_CLIP_PLANES 6

class Interactors : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
		enum{ GET_ALL_WPLANES/*, GET_WPOINT, GET_WNORMAL, GET_ONORMAL, GET_OPOINT*/ };
        Interactors();
        ~Interactors();

        bool init();

        void menuCallback(cvr::MenuItem * item);

        bool buttonEvent(int type, int button, int hand, const osg::Matrix & mat);

        void preFrame();
		
		bool processEvent(cvr::InteractionEvent*);

		void setSelectedGeometry(int planeIndex, int part);
		void deselectGeometry(int part=-1337);
		void message(int type, char*& data, bool colaborative);
 		void resizePlane(osg::Vec3, osg::Vec3, int planeIndex);
 		//void resizePlane(osg::Vec3, int planeIndex);

    protected:

		class PlaneInfo
		{	
			public:			
			bool isEnabled;
			osg::MatrixTransform* scale;
			// The plane that will actually clip
			osg::ref_ptr<osg::ClipPlane> clipPlane;
			// The plane portion wireframe drawn
			osg::Geode* quadG;
			osg::Geode* crossG;
			osg::Geode* wireframeG;
			osg::Geometry* quad;	
			osg::Geometry* cross;
			osg::Geometry* wireframe;
			// Plane structure used to set clipPlane
			osg::Plane plane;
			// Geode where the geometry lies 
			osg::Switch* planeSwitch;
			osg::Geode* arrowGeode;
			osg::Geode* toggleWireGeode;
			osg::Geometry* arrow;
			osg::Geometry* toggleWire;
			int wireIndex;
			// Local space to Object space matrix transform
			osg::MatrixTransform* local2o;
			// Plane normal in local coordinates 
			// (I recommend using the get functions instead)
			osg::Vec3 lNormal;
			osg::Matrix _scale;
			// A point in the plane in local coordinates
			osg::Vec3 lPoint;
			osg::Matrix _s;
			// Plane normal in object coordinates
			osg::Vec3 getONormal()
			{ 	
				return this->lNormal * this->local2o->getMatrix() - this->getOPoint();
			}
			// Plane normal in world coordinates
			osg::Vec3 getWNormal()
			{
				return this->getONormal() * cvr::PluginHelper::getObjectToWorldTransform();
			}
			// A point in the plane in object coordinates
			osg::Vec3 getOPoint()
			{
				return this->lPoint * this->local2o->getMatrix();
			}
			// A point in the plane in world coordinates
			osg::Vec3 getWPoint()
			{
				return this->getOPoint() * cvr::PluginHelper::getObjectToWorldTransform();
			}
			PlaneInfo(int clipPlaneNum, osg::Group* _node){
				wireIndex = 1;
				isEnabled = false;
				clipPlane = new osg::ClipPlane(clipPlaneNum); 
				quad = NULL;				
				local2o = new osg::MatrixTransform();
				wireframeG = new osg::Geode();
				quadG = new osg::Geode();
				crossG = new osg::Geode();
				planeSwitch = new osg::Switch();
				planeSwitch->addChild(wireframeG);
				planeSwitch->addChild(crossG);
				arrowGeode = new osg::Geode();
				toggleWireGeode = new osg::Geode();
				scale = new osg::MatrixTransform(osg::Matrix::scale(osg::Vec3(1.0f,1.0f,1.0f)));
				local2o->addChild(quadG);
				local2o->addChild(planeSwitch);
				local2o->addChild(arrowGeode);
				local2o->addChild(toggleWireGeode);
				_node->addChild(local2o);
			};
		};
		enum{ PLANE, ARROW, WIRE };
		std::vector<PlaneInfo*> _planeList;
        cvr::SubMenu * _clipPlaneMenu;
        std::vector<cvr::MenuCheckbox *> _placeList;
        std::vector<cvr::MenuCheckbox *> _enableList;
		osg::Texture2D* _arrowTexture;

//		osg::Geometry * beam;
//		osg::Geode* _bbg;
//		osg::Geometry* grid;
		osg::Matrix _lastMat;
		osg::Vec3 _lastPoint;
		osg::Vec3 _firstPoint;
		bool _firstTime;
		osg::Group* _planesNode;
        int _activePlane;
		int _activePart;
		bool _moving;

		bool buildPlane( const osg::Matrix m, int planeIndex );
		void movePlane(osg::Matrix last_h2o, osg::Matrix h2o, int planeIndex);

};

#endif
	//---- For Debugging -----
	/* Bounding Box	
	osg::BoundingBox bb = _planeList[planeIndex]->geode->getBoundingBox();
	_bbg->removeDrawable(_lastBox);
	_lastBox = new osg::ShapeDrawable(new osg::Box(osg::Vec3(
					(bb.xMax() + bb.xMin())/2, 
					(bb.yMax() + bb.yMin())/2, 
					(bb.zMax() + bb.zMin())/2)
				, bb.xMax() - bb.xMin()  
				, bb.yMax() - bb.yMin()  
				, bb.zMax() - bb.zMin()));	
	_bbg->addDrawable(_lastBox);
	*/
	/* Bounding Sphere
	osg::BoundingSphere bs = _planeList[planeIndex]->geode->getBound();
	_bbg->removeDrawable(_lastBox);
	_lastBox = new osg::ShapeDrawable(new osg::Sphere(bs.center(), bs.radius()));
	_bbg->addDrawable(_lastBox);
	*/
	
/*  Big QUAD on xy plane 
	osg::Vec3Array* verts = new osg::Vec3Array;
	grid = new osg::Geometry;
	verts->push_back(osg::Vec3(-10000,10000,0));
	verts->push_back(osg::Vec3(-10000,-10000,0));
	verts->push_back(osg::Vec3(10000,-10000,0));
	verts->push_back(osg::Vec3(10000,10000,0));
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
/*//	Drawing the plane normal
	_bbg->removeDrawable(beam);
	osg::Vec3 sp = p->point * l2o;// * o2w;
	osg::Vec3 ep = p->normal*1000*l2o;// * o2w;
	beam = new osg::Geometry;
	osg::ref_ptr<osg::Vec3Array> points = new osg::Vec3Array;
	points->push_back(sp);
	points->push_back(ep);
	osg::ref_ptr<osg::Vec4Array> color = new osg::Vec4Array;
	color->push_back(osg::Vec4(1.0,0.0,0.0,1.0));
	beam->setVertexArray(points.get());
	beam->setColorArray(color.get());
	beam->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE);
	beam->addPrimitiveSet(new osg::DrawArrays(GL_LINES,0,2));
		std::cerr<<"hudisoahf"<<std::endl;
	_bbg->addDrawable(beam);
*/

	//--------------------------

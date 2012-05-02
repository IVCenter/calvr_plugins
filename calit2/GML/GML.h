#ifndef _GML_H
#define _GML_H

#include <queue>
#include <vector>

// CVR
#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/ScreenBase.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/Navigation.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/FileHandler.h>


// OSG
#include <osg/Group>
#include <osg/Vec3>
#include <osgDB/ReadFile>
#include <osgDB/FileUtils>


// IGRAPH
#include <igraph/igraph.h>
#include <igraph/types.h>
#include <igraph/attributes.h>

#include "Lerpable.h"

class Edge;

class Node
{
    public:
	Node();
	~Node();
	void setName(std::string);
	std::string getName();
	void addEdge(Edge*);
	std::vector<Edge* > getEdges() { return _edges; };
	int getNumOfActiveEdges();
	std::pair< Node*, Edge* > getFirstActiveNode();
	void resetEdges();
	float getX() { return _x; };
	float getY() { return _y; };
	float getZ() { return _z; };
	float getW() { return _w; };
	float getH() { return _h; };
	void setX(float x) { _x = x; };
	void setY(float y) { _y = y; };
	void setZ(float z) { _z = z; };
	void setW(float w) { _w = w; };
	void setH(float h) { _h = h; };
	float getR() { return _r; };
	float getG() { return _g; };
	float getB() { return _b; };
	void setR(float r) { _r = r; };
	void setG(float g) { _g = g; };
	void setB(float b) { _b = b; };
	Lerpable<osg::Vec3> * getLerp() { return lerp; };
	void setLastPoint(osg::Vec3 last) { lastPoint = last; };
	osg::Vec3 getLastPoint() { return lastPoint; };
	void saveLastPoint() { lastPoint = point; };
	osg::Vec3 getPoint() { return point; };
	void savePoint(float x, float y, float z) { point.set(x, y, z); };
	void setMouseOver() { mouseOver = true; };
	bool getMouseOver() { return mouseOver; };
	void resetMouseOver() {mouseOver = false; };
	bool isSeen() { return seen; };
	void setSeen() { seen = true; };
	void resetSeen() { seen = false; };
	void setIndex(std::string ind) { index = ind; };
	std::string  getIndex() { return index; };

	void setIndice(int ind) { _indice = ind; };
	int getIndice() { return _indice; }; 
	int getColorStartIndex() { return _colorStartIndex; };
	void setColorStartIndex(int index) { _colorStartIndex = index; };

    private:
        std::string _label;
	int _indice;
	int _colorStartIndex;
        float _x;
        float _y;
        float _z;
        float _w;
        float _h;
        float _r;
        float _g;
        float _b;
	Lerpable<osg::Vec3> * lerp;
	osg::Vec3 lastPoint;
	osg::Vec3 point;
	std::vector<Edge* > _edges;
	bool mouseOver;
	bool seen;
	std::string index;
};

class Edge 
{
    public:
	Edge();
	Edge(bool);
	~Edge();
	bool hasBeenSeen() { return _seen; };
	void reset() { _seen = false; };  // resets seen param to false
	Node* getSource() { return _source; };
	Node* getTarget() { return _target; };
	void setSource(Node* source) { _source = source; };
	void setTarget(Node* target) { _target = target; };
	void setR(float r) { _r = r; };
	void setG(float g) { _g = g; };
	void setB(float b) { _b = b; };
	void setW(float w) { _w = w; };
	float getR() { return _r; };
	float getG() { return _g; };
	float getB() { return _b; };
	float getW() { return _w; };
	void visited() { _seen = true; };
	std::string getName() { return _label; };
	void setName(std::string label) { _label = label; };
	bool isFalseEdge() { return _falseEdge; };
	bool isSeen() { return _seen; };
	void setIndex(int ind) { index = ind; };
	int  getIndex() { return index; };
	void setSourceArrow(bool source) { _sourceArrow = source; };
	void setTargetArrow(bool target) { _targetArrow = target; };
	bool getSourceArrow() { return _sourceArrow; };
	bool getTargetArrow() { return _targetArrow; };


    private:
	std::string _label;
	float _r;
	float _g;
	float _b;
	float _w;
	Node* _source;//The source node
	Node* _target;//The target node
	bool _seen;
	bool _falseEdge;
	bool _sourceArrow;
	bool _targetArrow;

	/*Additional Member Attributes*/
	int index;
};

struct DataParams
{
	float xmin; 
	float xmax; 
	float ymin; 
	float ymax; 
	float lastradius; 
	float lastaddition;	
};

class GML : public cvr::MenuCallback, public cvr::CVRPlugin, public cvr::FileLoadCallback
{
  private:
    osg::Group* group;
    osg::Geode* geode;
    Node* activeNode;
    Node* mouseOver;
    std::map<std::string, Node* > nodes;
    bool pointsMoving;
    int numEdges;

    osg::Uniform* objectScale;
    osg::Uniform* world2object;
    osg::Uniform* nodeScale;
    osg::Uniform* edgeScale;

    // used for updating
    std::vector<Node* > *nodelist;

    cvr::SubMenu* GMLLayoutMenuItem;
       
    //RowMenu* GMLLayoutMenu;
    cvr::MenuCheckbox* showArrows;
    cvr::MenuCheckbox* showLoops;
    cvr::MenuButton* forceDirectedLayout;

    //void createModel(Node*);
    void createGraphics();

    // plugin data that will be updated to stop destroying and reloading of scenegraph
    osg::Vec3Array * verticesP;
    osg::Vec3Array * verticesE;
    osg::Vec4Array * colorsNodes;
    osg::Vec4Array * colorsEdges;
    osg::Geometry * nodeGeom;
    osg::Geometry * lineGeom;
 
    void initializeVertices();
    void updateVertices();
    void updateColor(Node * node, bool);

    // new breadth first create model traversal
    void makeLocations(Node* rootnode);
    void makeForceDirectedLayoutLocations();

    void createText(std::string, osg::Vec3, float);
    float xmin, ymin, zmin, xmax, ymax, zmax;
    void resetNodes(); // called reset on all edges
    void intersectionTesting();
    void clearGraph();
    void updateMovement();
    void movePoints();
    void createConnectivity();

  public:
    GML();
    virtual ~GML();
    bool init();
    void menuCallback(cvr::MenuItem * item);
    bool processEvent(cvr::InteractionEvent * event);
    virtual bool loadFile(std::string file);
    void preFrame();
};
#endif

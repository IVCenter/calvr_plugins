#ifndef _ARTIFACTVIS2_
#define _ARTIFACTVIS2_

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/TabbedDialogPanel.h>
#include <cvrMenu/DialogPanel.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuTextButtonSet.h>
#include <cvrMenu/MenuList.h>

#include <osg/Material>
#include <osg/MatrixTransform>
#include <osgText/Text>

#include <string>
#include <vector>
#include <map>

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <X11/Xlib.h>
#include <spnav.h>
#include <vrpn_Tracker.h>
#include <quat.h>

static double position[24][3];
static double orientation[24][4];
static vrpn_Tracker_Remote *remote;
static void VRPN_CALLBACK tracker_handler(void *userdata, const vrpn_TRACKERCB t);



class Artifact
{
  public:
    std::string dc;
    std::vector<std::string> fields;
    std::vector<std::string> values;
    double pos[3];
    osg::Drawable * drawable;
    osgText::Text * label;
    osg::Vec3 modelPos;
    bool visible;
    bool selected;
    bool showLabel;
    double distToCam;
    Artifact()
    {
      pos[0] = 0.0;
      pos[1] = 0.0;
      pos[2] = 0.0;
      showLabel = true;
    }
};

class ArtifactVis2 : public cvr::MenuCallback, public cvr::CVRPlugin
{
    public:
          ArtifactVis2();
        virtual ~ArtifactVis2();
        class compare
        {
          public:
            bool operator() (const Artifact* a1, const Artifact* a2)
            {
                return ((*a1).distToCam < (*a2).distToCam);
            }
        };
        void message(int type, char* data);

        bool init();

        bool processEvent(cvr::InteractionEvent * event);
        bool statusSpnav;
        void menuCallback(cvr::MenuItem * item);
        void preFrame();

        void setDCVisibleStatus(std::string dc, bool status);
        void updateVisibleStatus();
        std::string parseDate(std::string date);
        static ArtifactVis2 * instance();
        std::vector<osg::Vec3> getArtifactsPos();
        float selectArtifactSelected();
        osg::Matrix getSelectMatrix();
        void setSelectMatrix(osg::Matrix & mat);
        bool _selectActive;

        int _activeArtifact;

        void testSelected();

        //Kinect Test

        osg::Geode * skeletonGeode;

        // Position and orientation of each joint

        double eulerOrientation[3];


    protected:

        static ArtifactVis2 * _artifactvis2;
        int _testA;

        struct Locus
        {
            std::vector<std::string> fields;
            std::vector<std::string> values;
            std::string id;
            std::string name;
            osg::Geometry* geom;
            osg::Geometry* tgeom;
            osg::ref_ptr<osg::Geode> fill_geode;
            osg::ref_ptr<osg::Geode> line_geode;
            osg::ref_ptr<osg::Geode > top_geode;
            osg::Geode* text_geode;
            bool visible;
            std::vector<osg::Vec3d> coordsTop;
            std::vector<osg::Vec3d> coordsBot;
            osgText::Text * label;
        };
        struct QueryGroup
        {
            std::string name;
            std::string query;
            bool sf;
            std::vector<Artifact *> artifacts;
            std::vector<Locus*> loci;
            bool active;
            std::string timestamp;
            std::string kmlPath;
            osg::ref_ptr<osg::MatrixTransform> sphereRoot;
            bool updated;
            osg::Vec3d center;
        };
        struct Table
        {
            std::string name;
            cvr::SubMenu * queryMenu;
            cvr::MenuButton * genQuery;
            cvr::MenuButton * clearConditions;
            cvr::MenuButton * saveQuery;
            cvr::SubMenu * viewQuery;
            cvr::MenuButton * addOR;
            cvr::MenuButton * removeOR;
            cvr::SubMenu * conditions;
            std::vector<cvr::SubMenu *> querySubMenu;
            std::vector<cvr::SubMenu *> querySubMenuSlider;
            std::vector<cvr::MenuTextButtonSet *> queryOptions;
            std::vector<cvr::MenuCheckbox *> querySlider;
            std::vector<std::vector<std::string> > sliderEntry;
            std::vector<cvr::MenuList *> queryOptionsSlider;
            cvr::MenuText * query_view;
            std::string current_query;
        };
        osg::Node * _models[676];
        bool _modelLoaded[676];
        cvr::SubMenu * _modelDisplayMenu;
        std::vector<cvr::MenuCheckbox *> _showModelCB;
	std::vector<cvr::MenuButton *> _reloadModel;
        cvr::SubMenu * _pcDisplayMenu;
        std::vector<cvr::MenuCheckbox *> _showPCCB;
	std::vector<cvr::MenuButton *> _reloadPC;
        cvr::SubMenu *_avMenu;
        cvr::SubMenu *_displayMenu;
        cvr::SubMenu * _artifactDisplayMenu;
        std::vector<cvr::SubMenu *> _queryOptionMenu;
        std::vector<cvr::MenuCheckbox *> _queryOption;
        std::vector<cvr::SubMenu *> _showQueryInfo;
        std::vector<cvr::MenuCheckbox *> _queryDynamicUpdate;
        std::vector<cvr::MenuText *> _queryInfo;
        std::vector<cvr::MenuButton *> _eraseQuery;
        std::vector<cvr::MenuButton *> _centerQuery;
	std::vector<cvr::MenuCheckbox *> _toggleLabel; //new
        cvr::SubMenu * _locusDisplayMenu;
        cvr::MenuTextButtonSet * _locusDisplayMode;
        cvr::MenuCheckbox * _selectArtifactCB;
	cvr::MenuCheckbox * _scaleBar; //new
        cvr::MenuCheckbox * _selectCB;
        cvr::SubMenu * _tablesMenu;
        osg::MatrixTransform * _root;

        cvr::TabbedDialogPanel * _artifactPanel;
        cvr::DialogPanel * _selectionStatsPanel;

        std::string _picFolder;

        std::vector<std::string> _dcList;
        std::map<std::string,bool> _dcVisibleMap;

        std::vector<Table*> _tables;
        std::vector<QueryGroup*> _query;

        osg::Vec3 _selectStart;
        osg::Vec3 _selectCurrent;

        osg::Material * _defaultMaterial;

        osg::Vec4 _colors[729];

        //osg::LOD * _my_own_root;
  	std::vector<osg::Vec3Array * > _coordsPC;
        std::vector<osg::Vec4Array * > _colorsPC;
        std::vector<osg::Vec3d> _avgOffset;
        std::vector<osg::Node* > _modelSFileNode;
        std::vector<osg::ref_ptr<osg::MatrixTransform> > _siteRoot;
        std::vector<osg::ref_ptr<osg::MatrixTransform> > _pcRoot;
        std::vector<osg::Vec3d> _sitePos;
        std::vector<osg::Vec3d> _siteScale;
        std::vector<osg::Vec3d> _siteRot;
        std::vector<osg::Vec3d> _pcPos;
        std::vector<osg::Vec3d> _pcScale;
        std::vector<osg::Vec3d> _pcRot;
        std::vector<int> _pcFactor; //New
        osg::ref_ptr<osg::MatrixTransform> _selectBox;
        osg::ref_ptr<osg::MatrixTransform> _selectMark;
        osg::ref_ptr<osg::MatrixTransform> _selectModelLoad; //New Add
        osg::Node* _modelFileNode;
        double _selectRotx;
        double _selectRoty;
        double _selectRotz;
        double _snum;
        osg::ref_ptr<osg::MatrixTransform> _scaleBarModel; //New Add
        osg::Vec3 _modelartPos; //New Add
        double _xRotMouse; //New Add
        double _yRotMouse; //New Add
        //float _LODmaxRange;
        float _sphereRadius;


        bool _ossim;
        bool _osgearth;

        std::string getTimeModified(std::string file);
        int dc2Int(std::string dc);
	void loadScaleBar(osg::Vec3d start); //new
        void setActiveArtifact(int art, int query);
        void readQuery(QueryGroup* query);
        std::vector<std::string> getSelectedQueries();
        void listArtifacts();
        void displayArtifacts(QueryGroup * query);
        osg::Drawable* createObject(std::string dc, float tessellation, osg::Vec3d & pos);
        void clearConditions(Table* t);
        void readPointCloud(int index);
        void readSiteFile(int index);
        void readLocusFile(QueryGroup * query);
        void setupSiteMenu();
	void reloadSite(int index); //New
        void setupLocusMenu();
        void setupQuerySelectMenu();
        void setupTablesMenu();
        void setupQueryMenu(Table * table);
        void updateSelect();
        std::string getCurrentQuery(Table * t);
        bool modelExists(const char * filename);
        void loadModels();
        void rotateModel(double rx, double ry, double rz);

        //Space Navigator
        float transMult, rotMult;
        float transcale, rotscale;

};

#endif

#ifndef _MVSIM_H_
#define _MVSIM_H_

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/Screens/ScreenMVMaster.h>
#include <cvrKernel/Screens/ScreenMVSimulator.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/PopupMenu.h>

#include <osg/Group>

class MVSim : public cvr::CVRPlugin, public cvr::MenuCallback
{
    public:
        MVSim();
        ~MVSim();

        bool init();
        void menuCallback(cvr::MenuItem * item);
        void preFrame();

    protected:
        cvr::ScreenMVSimulator * _screenMVSim;
        osg::Matrix * head0;
        osg::Matrix * head1;

        osg::ref_ptr<osg::MatrixTransform> viewTransform0;
        osg::ref_ptr<osg::MatrixTransform> viewTransform1;

        void stepEvent();
        void saveCurrentHeadMatrices();
        void loadHeadMatrices();

        bool _run;
        float _delay;
        int _event;
        cvr::SubMenu * mvsMenu;
        cvr::MenuButton * startSim;
        cvr::MenuButton * stopSim;
        cvr::MenuButton * resetSim;
        cvr::MenuButton * stepSim;
        cvr::MenuRangeValue * delaySim;
        cvr::SubMenu * sceneMenu;
        cvr::MenuButton * scene1;
        cvr::SubMenu * setHeadMenu;
        std::map<cvr::SubMenu *, osg::Matrix *> headMats;
        cvr::MenuButton * saveHeads;
        cvr::MenuButton * loadHeads;
        cvr::MenuCheckbox * showDiagramBox;

        osg::ref_ptr<osg::Switch> scene1switch;
        osg::ref_ptr<osg::Group> diagram;
        osg::ref_ptr<osg::MatrixTransform> cone0;
        osg::ref_ptr<osg::MatrixTransform> cone1;

        /**
          * @brief Set diagram as shown or not
          * @param show Whether or not to show the diagram
          */
        void showDiagram(bool show);
        /**
          * @brief Creates the geometry for the cave as needed and attaches it to the diagram node
          * @param masterScreen pointer to the screen(mvmaster) that the diagram is to be rendered on
          */
        void setupCaveDiagram(cvr::ScreenMVMaster * masterScreen);
};

#endif

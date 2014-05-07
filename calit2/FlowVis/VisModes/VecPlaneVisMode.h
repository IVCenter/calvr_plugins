#ifndef VEC_PLANE_VIS_MODE_H
#define VEC_PLANE_VIS_MODE_H

#include "VisMode.h"

#include <map>
#include <pthread.h>

#include <GL/gl.h>

class VecPlaneVisMode : public VisMode
{
    public:
        VecPlaneVisMode();
        virtual ~VecPlaneVisMode();

        virtual void initContext(int context);
        virtual void uinitContext(int context);
        virtual void draw(int context);

    protected:
        std::map<int,bool> _shaderInitMap;
        pthread_mutex_t _shaderInitLock;

        std::map<int,GLuint> _normalProgram;

        std::map<int,GLuint> _normalFloatProgram;
        std::map<int,GLint> _normalFloatMinUni;
        std::map<int,GLint> _normalFloatMaxUni;

        std::map<int,GLuint> _normalIntProgram;
        std::map<int,GLint> _normalIntMinUni;
        std::map<int,GLint> _normalIntMaxUni;

        std::map<int,GLuint> _normalVecProgram;
        std::map<int,GLint> _normalVecMinUni;
        std::map<int,GLint> _normalVecMaxUni;

        std::map<int,GLuint> _vecPlaneProgram;
        std::map<int,GLint> _vecPlaneMinUni;
        std::map<int,GLint> _vecPlaneMaxUni;
        std::map<int,GLint> _vecPlanePointUni;
        std::map<int,GLint> _vecPlaneNormalUni;
        std::map<int,GLint> _vecPlaneUpUni;
        std::map<int,GLint> _vecPlaneRightUni;
        std::map<int,GLint> _vecPlaneUpNormUni;
        std::map<int,GLint> _vecPlaneRightNormUni;
        std::map<int,GLint> _vecPlaneBasisLengthUni;
};

#endif

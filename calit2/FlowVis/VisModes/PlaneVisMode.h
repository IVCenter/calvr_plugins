#ifndef PLANE_VIS_MODE_H
#define PLANE_VIS_MODE_H

#include "VisMode.h"

#include <map>

#ifndef WIN32
#include <pthread.h>
#else
#include "../pthread_win.h"
#endif

#include <GL/gl.h>

class PlaneVisMode : public VisMode
{
    public:
        PlaneVisMode();
        virtual ~PlaneVisMode();

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

        std::map<int,GLuint> _planeFloatProgram;
        std::map<int,GLint> _planeFloatMinUni;
        std::map<int,GLint> _planeFloatMaxUni;
        std::map<int,GLint> _planeFloatPointUni;
        std::map<int,GLint> _planeFloatNormalUni;
        std::map<int,GLint> _planeFloatAlphaUni;

        std::map<int,GLuint> _planeVecProgram;
        std::map<int,GLint> _planeVecMinUni;
        std::map<int,GLint> _planeVecMaxUni;
        std::map<int,GLint> _planeVecPointUni;
        std::map<int,GLint> _planeVecNormalUni;
        std::map<int,GLint> _planeVecAlphaUni;
};

#endif

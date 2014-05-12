#ifndef SEP_ATT_LINE_VIS_MODE_H
#define SEP_ATT_LINE_VIS_MODE_H

#include "VisMode.h"

#include <map>

#ifndef WIN32
#include <pthread.h>
#else
#include "../pthread_win.h"
#endif

#include <GL/gl.h>

class SepAttLineVisMode : public VisMode
{
    public:
        SepAttLineVisMode();
        virtual ~SepAttLineVisMode();

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
};

#endif

#ifndef PLUGIN_CUBE_DRAWABLE_H
#define PLUGIN_CUBE_DRAWABLE_H

#include <cvrUtil/glesDrawable.h>
#include <unordered_map>
#include <cvrUtil/ARCoreManager.h>
#include <glm/glm.hpp>
namespace {
    enum Face {
        FRONT = 0, BACK, LEFT, RIGHT, UP, BOTTOM
    };
}
class cameraHelper{
public:
    static glm::vec3 CameraPosition(){
        auto eye_pos = cvr::ARCoreManager::instance()->getCameraPose();
        return glm::vec3(eye_pos[4], eye_pos[5], eye_pos[6]);
    }
    static glm::vec3 CameraPosition_ModelCoordinates(glm::mat4 modelMat){
        glm::mat4 inv_model = glm::inverse(modelMat);
        auto eye_pos = cvr::ARCoreManager::instance()->getCameraPose();
        glm::vec4 eye_model = inv_model * glm::vec4(eye_pos[4], eye_pos[5], eye_pos[6], 1.0f);
        float inv_w = 1.0f / eye_model.w;
        return glm::vec3(eye_model.x * inv_w, eye_model.y * inv_w, eye_model.z * inv_w);
    }
    static glm::vec3 CameraViewDirection(){
        osg::Vec3d eye, center, up;
        cvr::ARCoreManager::instance()->getViewMatrix()->getLookAt(eye, center, up);
        return glm::vec3(center.x() - eye.x(),
                         center.y() - eye.y(),
                         center.z() - eye.z());
    }
    static glm::vec3 CameraViewCenter(){
        osg::Vec3d eye, center, up;
        cvr::ARCoreManager::instance()->getViewMatrix()->getLookAt(eye, center, up);
        return glm::vec3(center[0], center[1], center[2]);
    }

};
typedef std::pair<glm::vec3, int> Polygon;
typedef std::unordered_map<Face, std::vector<int>> PolygonMap;

class cubeDrawable: public cvr::glesDrawable {
protected:
    const int MAX_VERTEX_NUM = 15;
    const int MAX_INDICE_NUM = 90;
    const int VAO_DATA_LEN = 6;

    const GLfloat sVertex[48] = {//World					//Color
            -0.5f,-0.5f,0.5f, 0.0f,0.0f,1.0f,		//x0, y0, z1, //	//v0
            0.5f,-0.5f,0.5f, 1.0f,0.0f,1.0f,		//x1,y0,z1, //	//v1
            0.5f,0.5f,0.5f,	1.0f,1.0f,1.0f,		//x1, y1, z1,//	//v2
            -0.5f,0.5f,0.5f,0.0f,1.0f,1.0f,		//x0,y1,z1, //	//v3
            -0.5f,-0.5f,-0.5f,0.0f,0.0f,0.0f,	//x0,y0,z0,//	//v4
            0.5f,-0.5f,-0.5f,1.0f,0.0f,0.0f,		//x1,y0,z0,//	//v5
            0.5f,0.5f,-0.5f,1.0f,1.0f,0.0f,	//x1,y1,z0, //	//v6
            -0.5f,0.5f,-0.5f,0.0f,1.0f,0.0f,		//x0,y1,z0//	//v7
    };
    const GLuint sIndices[36] = { 0,1,2,0,2,3,	//front
                                  4,6,7,4,5,6,	//back
                                  4,0,3,4,3,7,	//left
                                  1,5,6,1,6,2,	//right
                                  3,2,6,3,6,7,	//top
                                  4,5,1,4,1,0,	//bottom
    };
    GLfloat* vertices_;
    GLuint* indices_;

    GLuint _VAO, _VBO[2], _EBO;
    GLuint _shader_program;
    int indices_num_, vertices_num_;

    glm::mat4 _modelMat = glm::mat4(1.0f);

    //cutting
    bool is_cutting = true, is_in_deeper = false;
    float cutting_length;
    glm::vec3 last_cutting_norm = glm::vec3(FLT_MAX), start_cutting;
    std::vector<Polygon> polygon;
    PolygonMap polygon_map;

    void restore_original_cube();
    void updateCuttingPlane(glm::vec3 p, glm::vec3 p_norm);
    void updateGeometry(std::vector<Polygon> polygon, PolygonMap polygon_map, std::vector<int> rpoints);

public:
    cubeDrawable();
    void setCuttingPlane(float percent = .0f);

    void Initialization();
    void updateOnFrame();
    void drawImplementation(osg::RenderInfo&) const;
};

#endif

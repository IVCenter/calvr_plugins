#include "cubeDrawable.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GLES3/gl3.h>

#include <cvrUtil/AndroidHelper.h>
#include <cvrUtil/AndroidStdio.h>

#include "mathHelper.h"
using namespace glm;

void cubeDrawable::restore_original_cube(){
    indices_num_ = 36;
    vertices_num_ = 8;
    memcpy(vertices_, sVertex, sizeof(GLfloat) * VAO_DATA_LEN * vertices_num_);
    memcpy(indices_, sIndices, sizeof(GLuint) * indices_num_);
}
void cubeDrawable::setCuttingPlane(float percent) {
    is_in_deeper = percent>0.5f;
    polygon.clear();
    polygon_map.clear();

    //get the nearest - farest vertex
    //s1. trans eye position back to model-coord

    glm::vec3 eye_model3 = cameraHelper::CameraPosition_ModelCoordinates(_modelMat);

    vec3 vdir_model = -glm::normalize(eye_model3);

//    cvr::ARCoreManager::instance()->getViewMa
    if(!glm::any(glm::equal(cameraHelper::CameraViewDirection(), last_cutting_norm))){

        //update cutting nearest - fartest position
        vec3 nearest_plane_intersection;
        if(getNearestRayAABBIntersection(eye_model3, vdir_model,
                                         vec3(-0.5), vec3(0.5),
                                         nearest_plane_intersection)){
            float hdist = glm::distance(cameraHelper::CameraViewCenter(),
                    nearest_plane_intersection);
            hdist*=1.3f;
            start_cutting = -hdist * vdir_model;
            cutting_length = 2.0f * hdist;
            last_cutting_norm = cameraHelper::CameraPosition();
            is_cutting = true;
        } else{
            is_cutting = false;
            LOGE("=====is not cutting, eye pos: %f, %f, %f, dir %f %f %f", eye_model3.x, eye_model3.y, eye_model3.z, vdir_model.x, vdir_model.y, vdir_model.z);

        }
    }
    if(!is_cutting) return;
    vec3 pop_model = start_cutting + percent * cutting_length * vdir_model;
    updateCuttingPlane(pop_model, vdir_model);
}
//p and p norm should be in model space
void cubeDrawable::updateCuttingPlane(glm::vec3 p, glm::vec3 p_norm){
    //view_dir and p_norm should be on the same side
    vec3 view_model_dir = glm::inverse(mat3(_modelMat)) * cameraHelper::CameraViewDirection();
    if(dot(p_norm, view_model_dir) < 0)
        p_norm = -p_norm;

    getIntersectionPolygon(p, p_norm, vec3(-0.5f), vec3(0.5f), polygon, polygon_map);
    if(polygon.empty()){
        vec3 eye_model = cameraHelper::CameraPosition_ModelCoordinates(_modelMat);
        if(is_in_deeper){
            vertices_num_ = 0; indices_num_ =0;
        }else{
            restore_original_cube();
        }
        updateOnFrame();
        return;
    }
    //Test 8 points which should be remove
    std::vector<int> rpoints;
    std::vector<vec3> rpoints_values;

    for(int i=0;i<8;i++){
        vec3 vertex = vec3(sVertex[VAO_DATA_LEN*i], sVertex[VAO_DATA_LEN*i+1],sVertex[VAO_DATA_LEN*i+2]);//point to test
        if(dot(vertex - p, p_norm) >= 0) {
            rpoints.push_back(i);
            rpoints_values.push_back(vec3(sVertex[VAO_DATA_LEN * i], sVertex[VAO_DATA_LEN * i+1],sVertex[VAO_DATA_LEN * i+2]));
        }

    }
    std::deque<int> polygon_to_be_erased;
    //check if new points overlap with original vertices
    for(int idx = polygon.size();idx >-1; idx--){
        auto got = std::find(rpoints_values.begin(), rpoints_values.end(), polygon[idx].first);
        if(got != rpoints_values.end()){
            polygon_to_be_erased.push_front(idx);
            for(auto face= polygon_map.begin(); face!=polygon_map.end(); face++){
                auto tmpVec = face->second;
                auto gotid = std::find(tmpVec.begin(), tmpVec.end(), idx);
                if(gotid!=tmpVec.end()){
                    int cid = std::distance(tmpVec.begin(), gotid);
                    tmpVec.erase(tmpVec.begin() + cid);
                    face->second = tmpVec;
                }
            }
        }
    }
    updateGeometry(polygon, polygon_map, rpoints);
    updateOnFrame();
}

//update vertices, indices, and corresponding numbers
void cubeDrawable::updateGeometry(std::vector<Polygon> polygon, PolygonMap polygon_map, std::vector<int> rpoints){
//    delete(vertices_);
//    delete(indices_);
    //create new vertices
    int rp_num = rpoints.size();
    vertices_num_ = rp_num + polygon.size();
    std::vector<int>id_map(vertices_num_, 0);
//    int id_map[vertices_num_] = {0};

    GLfloat * c_vertices_ = new GLfloat[VAO_DATA_LEN*vertices_num_];
    int nidx = 0;
    for(auto id:rpoints){
//        c_vertices_[VAO_DATA_LEN*nidx] = sVertex[id*VAO_DATA_LEN];c_vertices_[VAO_DATA_LEN*nidx+1] = sVertex[id*3+1];c_vertices_[3*nidx+2] = sVertex[id*3+2];
        memcpy(c_vertices_+VAO_DATA_LEN*nidx, sVertex+VAO_DATA_LEN*id, VAO_DATA_LEN*sizeof(GL_FLOAT));
        id_map[nidx] = id; nidx++;
    }

    memcpy(vertices_, c_vertices_, VAO_DATA_LEN * vertices_num_*sizeof(GLfloat));

    //points are copied
    //faces

    std::vector<GLuint> c_indices;
    std::set<int>rp_set(rpoints.begin(), rpoints.end());
    for(int i=0; i<6; i++){
//        auto uset = getIntersection(rp_set, );
        std::set<int> uset;
        std::set<int> face_set(sIndices+i*6, sIndices+(i+1)*6);
        set_intersection(rp_set.begin(),rp_set.end(),face_set.begin(),face_set.end(),
                         std::inserter(uset,uset.begin()));

        if(uset.empty()) continue;
        auto got = polygon_map.find((Face)i);
        if(got == polygon_map.end()) {
            //c_indices.insert(c_indices.end(),sIndices+i*6,sIndices+(i+1)*6 );
            int start_pos = i*6, cid;
            for(int j=0; j< 6;j++){
                cid = std::distance(id_map.begin(), std::find(id_map.begin(), id_map.begin()+rp_num, sIndices[start_pos+j]));
                c_indices.push_back(cid);
            }
        }
        else{
            //re order the points
            //for new points: get id -> vec3
            std::vector<Polygon>face_points;
            for(auto id:got->second){
                int cid = std::distance(id_map.begin(), std::find(id_map.begin()+rp_num, id_map.end(), id));
                face_points.push_back(std::make_pair(vec3(vertices_[VAO_DATA_LEN*cid], vertices_[VAO_DATA_LEN*cid+1], vertices_[VAO_DATA_LEN*cid+2]), cid));
            }
            for(auto id:uset){
                int cid = std::distance(id_map.begin(), std::find(id_map.begin(), id_map.begin()+rp_num, id));
                face_points.push_back(std::make_pair(vec3(vertices_[VAO_DATA_LEN*cid], vertices_[VAO_DATA_LEN*cid+1], vertices_[VAO_DATA_LEN*cid+2]), cid));
            }

            vec3 origin = face_points[0].first;
            vec3 p_norm = getPlaneNormal(face_points[0].first, face_points[1].first, face_points[2].first);
            std::sort(face_points.begin(), face_points.end(), [&](const Polygon& p1, const Polygon& p2)->bool{
                vec3 cross_v = glm::cross(p1.first - origin, p2.first - origin);
                return dot(cross_v, p_norm)<0;
            });


            //face points are sorted counter_clock_wise
            for(int i=0; i<face_points.size()-2; i++){
                GLuint indice[] = {(GLuint)face_points[0].second, (GLuint)face_points[i+1].second, (GLuint)face_points[i+2].second};
                //check if two points are the same
//                vec3 p0 = face_points[0].first,
//                     p1 = face_points[i+1].first,
//                     p2 = face_points[i+2].first;
//                if(glm::all(glm::equal(p0, p1)) || glm::all(glm::equal(p0, p2)) ||glm::all(glm::equal(p1, p2))) continue;
                c_indices.insert(c_indices.end(), indice, indice+3);
            }
        }
    }

    int npid[polygon.size()];
    for(int i=0;i<polygon.size();i++)
        npid[i] = std::distance(id_map.begin(), std::find(id_map.begin()+rp_num, id_map.end(), polygon[i].second));
    //generate face that all from new points'
    for(int i=0; i<polygon.size()-2; i++){
        GLuint indice[] = {(GLuint)npid[0], (GLuint)npid[i+1], (GLuint)npid[i+2]};
        c_indices.insert(c_indices.end(), indice, indice+3);
    }

//    indices_ = memcpy(c_indices.begin(), );
    indices_num_ = c_indices.size();
    memcpy(indices_, c_indices.data(), indices_num_ * sizeof(GLuint));
    delete (c_vertices_);
}
void cubeDrawable::Initialization() {
    cvr::glesDrawable::Initialization();
    _shader_program = cvr::assetLoader::instance()->
            createGLShaderProgramFromFile(
            "shaders/raycastVolume.vert",
            "shaders/raycastVolume.frag");

    if(!_shader_program)
        LOGE("=====failed to create shader ===");

    vertices_ = new GLfloat[MAX_VERTEX_NUM * VAO_DATA_LEN];
    indices_ = new GLuint[MAX_INDICE_NUM];

    restore_original_cube();

    glGenVertexArrays(1, &_VAO);
    glBindVertexArray(_VAO);

    glGenBuffers(1, _VBO);
    glBindBuffer(GL_ARRAY_BUFFER, _VBO[0]);
    glBufferData(GL_ARRAY_BUFFER, VAO_DATA_LEN*MAX_VERTEX_NUM* sizeof(GL_FLOAT), nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VAO_DATA_LEN * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VAO_DATA_LEN * sizeof(float), (void*)(3 * sizeof(float)));

    glGenBuffers(1, &_EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, MAX_INDICE_NUM * sizeof(GL_UNSIGNED_INT), nullptr, GL_DYNAMIC_DRAW);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    setCuttingPlane();

}

void cubeDrawable::drawImplementation(osg::RenderInfo &) const {
    glUniformMatrix4fv(
            glGetUniformLocation(_shader_program, "uModelMat"),
            1,
            GL_FALSE,
            &_modelMat[0][0]);

    glUniformMatrix4fv(
            glGetUniformLocation(_shader_program, "uViewMat"),
            1,
            GL_FALSE,
            cvr::ARCoreManager::instance()->getViewMatrix()->ptr());

    glUniformMatrix4fv(
            glGetUniformLocation(_shader_program, "uProjMat"),
            1,
            GL_FALSE,
            cvr::ARCoreManager::instance()->getProjMatrix()->ptr());

    glBindVertexArray(_VAO);

    glDrawElements(GL_TRIANGLES, indices_num_, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void cubeDrawable::updateOnFrame() {
    glBindBuffer(GL_ARRAY_BUFFER, _VBO[0]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, VAO_DATA_LEN * vertices_num_  *sizeof(GL_FLOAT), vertices_);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EBO);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, indices_num_  *sizeof(GL_UNSIGNED_INT), indices_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
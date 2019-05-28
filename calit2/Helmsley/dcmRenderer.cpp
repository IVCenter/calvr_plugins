#include "dcmRenderer.h"
#include <GLES3/gl3.h>
#include <cvrUtil/AndroidHelper.h>
#include <GLES3/gl32.h>
#include "Color.h"
void dcmRenderer::assemble_texture_3d(){
    glGenTextures(1, &_volume_tex_id);
    // bind 3D texture target
    glBindTexture(GL_TEXTURE_3D, _volume_tex_id);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    // pixel transfer happens here from client to OpenGL server
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, (int)volume_size.x, (int)volume_size.y, (int)volume_size.z, 0, GL_RED, GL_UNSIGNED_BYTE, cvr::ARCoreManager::dcm_img_data);
    delete []cvr::ARCoreManager::dcm_img_data;
}
void dcmRenderer::create_trans_texture(){
    glActiveTexture(GL_TEXTURE1);
    //create texture object
    glGenTextures(1, &_trans_tex_id);
    glBindTexture(GL_TEXTURE_2D, _trans_tex_id);
    //bind current texture object and set the data
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 32, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, transfer_color);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void dcmRenderer::Initialization(){
    volume_size = glm::vec3(cvr::ARCoreManager::img_width,
            cvr::ARCoreManager::img_height,
            cvr::ARCoreManager::img_nums);
    cubeDrawable::Initialization();
    assemble_texture_3d();
    create_trans_texture();
}
void dcmRenderer::updateOnFrame(){
    cubeDrawable::updateOnFrame();

}
void dcmRenderer::drawImplementation(osg::RenderInfo& info) const{
    cubeDrawable::drawImplementation(info);

    glUseProgram(_shader_program);

    glActiveTexture(GL_TEXTURE0);
    glUniform1i(glGetUniformLocation(_shader_program, "uSampler_tex"), 0);
    glBindTexture(GL_TEXTURE_3D, _volume_tex_id);

    glActiveTexture(GL_TEXTURE1);
    glUniform1i(glGetUniformLocation(_shader_program, "uSampler_trans"), 1);
    glBindTexture(GL_TEXTURE_2D, _trans_tex_id);


    glUniform1f(glGetUniformLocation(_shader_program, "sample_step_inverse"), adjustParam[0]);
    glUniform1f(glGetUniformLocation(_shader_program, "val_threshold"),adjustParam[1]);
    glUniform1f(glGetUniformLocation(_shader_program, "brightness"), adjustParam[2]);

    glUniform1i(glGetUniformLocation(_shader_program, "u_use_color_transfer"), use_color_tranfer);
    glUniform1i(glGetUniformLocation(_shader_program, "u_use_ligting"), use_lighting);
    glUniform1i(glGetUniformLocation(_shader_program, "u_use_interpolation"), use_interpolation);
    glUniform1i(glGetUniformLocation(_shader_program, "u_draw_naive"), use_simple_cube);

    glUniform1f(glGetUniformLocation(_shader_program, "volumex"), volume_size.x);
    glUniform1f(glGetUniformLocation(_shader_program, "volumey"), volume_size.y);
    glUniform1f(glGetUniformLocation(_shader_program, "volumez"), volume_size.z);

    float lightIa[3] = { 0.8,0.8,0.8 };
    float lightId[3] = { 0.7,0.7,0.7 };
    float lightIs[3] = { 0.5,0.5,0.5 };

    glUniform3fv(glGetUniformLocation(_shader_program, "Light.Ia"), 1, lightIa);
    glUniform3fv(glGetUniformLocation(_shader_program, "Light.Id"), 1, lightId);
    glUniform3fv(glGetUniformLocation(_shader_program, "Light.Is"), 1, lightIs);
    glUniform1f(glGetUniformLocation(_shader_program, "shiness"), 32.0f);

    glUseProgram(0);

}
#version 120
#extension GL_EXT_geometry_shader4: enable

varying out float sphere_radius;
varying out vec3 vertex_light_position;
varying out vec4 eye_position; 
uniform float objectScale;
uniform float pointSize;

void
main(void)
{
    sphere_radius = gl_FrontColorIn[0].a * objectScale * 2.0 * pointSize;
    float halfsize = sphere_radius * 0.5;
    gl_TexCoord[0] = gl_TexCoordIn[0][0];
    gl_FrontColor = gl_FrontColorIn[0];
    gl_FrontColor.a = 1.0;

    vertex_light_position = normalize(gl_LightSource[0].position.xyz);
    eye_position = gl_PositionIn[0];

    gl_TexCoord[0].st = vec2(-1.0,-1.0);
    gl_Position = gl_PositionIn[0];
    gl_Position.xy += vec2(-halfsize, -halfsize);
    gl_Position = gl_ProjectionMatrix * gl_Position;
    EmitVertex();
    
    gl_TexCoord[0].st = vec2(-1.0,1.0);
    gl_Position = gl_PositionIn[0];
    gl_Position.xy += vec2(-halfsize, halfsize);
    gl_Position = gl_ProjectionMatrix * gl_Position;
    EmitVertex();
    
    gl_TexCoord[0].st = vec2(1.0,-1.0);
    gl_Position = gl_PositionIn[0];
    gl_Position.xy += vec2(halfsize, -halfsize);
    gl_Position = gl_ProjectionMatrix * gl_Position;
    EmitVertex();
    
    gl_TexCoord[0].st = vec2(1.0,1.0);
    gl_Position = gl_PositionIn[0];
    gl_Position.xy += vec2(halfsize, halfsize);
    gl_Position = gl_ProjectionMatrix * gl_Position;
    EmitVertex();

    EndPrimitive();
}

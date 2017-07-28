#version 150 compatibility

out DataIn {
    vec2 texCoord;
    vec4 color;
} v_out;

void main(void)
{

    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    
    v_out.texCoord = gl_MultiTexCoord0.xy;
    v_out.color = gl_Color;
}

#version 150 compatibility
#extension GL_ARB_gpu_shader5 : enable

uniform sampler2D glyphTexture;

in DataOut {
    vec2 texCoord;
    vec3 color;
} f_in;

void main(void)
{
    gl_FragColor = vec4(f_in.color, 1.0) * texture(glyphTexture, f_in.texCoord).aaaa;
    //gl_FragColor = f_color * texture(glyphTexture, f_texCoord).aaaa;
}

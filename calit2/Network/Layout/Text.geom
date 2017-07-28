#version 150 compatibility
#extension GL_EXT_geometry_shader4 : enable
#extension GL_ARB_gpu_shader5 : enable

layout(triangles) in;
layout (triangle_strip, max_vertices=3) out;

uniform vec3 colorTable[12];
uniform float minWeight;

in DataIn {
   vec2 texCoord;
   vec4 color;
} v_in[];

out DataOut {
    vec2 texCoord;
    vec3 color;
} f_out;

void
main(void)
{
    // use to quickly disable text rendering
    if( v_in[0].color.b > minWeight )
    {
        for(int i = 0; i < 3; i++)
        {
	        f_out.texCoord = v_in[i].texCoord; 
	        f_out.color = colorTable[int(v_in[i].color.w)]; 

	        gl_Position = gl_in[i].gl_Position;
	        EmitVertex();
        }
	EndPrimitive();
    }
}

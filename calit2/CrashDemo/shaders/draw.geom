#version 120 
#extension GL_EXT_geometry_shader4 : enable 

varying in vec3 ls[3];
//varying in vec3 norm[3];
//varying out float light;
 
void main() 
{ 
    //vec3 normal = normalize(gl_NormalMatrix * vec3(1.0,0.0,0.0));
    vec3 normal = normalize(cross(gl_TexCoordIn[1][0].xyz - gl_TexCoordIn[0][0].xyz, gl_TexCoordIn[2][0].xyz - gl_TexCoordIn[0][0].xyz));
    //vec3 normal = normalize(gl_NormalMatrix * norm[0]);
    vec3 n1 = normalize(gl_NormalMatrix * normal);
    vec3 n2 = normalize(gl_NormalMatrix * -normal);
    for(int i = 0; i < gl_VerticesIn; ++i) 
    {
	//vec3 normal = normalize(gl_NormalMatrix * norm[i]);
	//vec3 n1 = normalize(gl_NormalMatrix * normal);
	//vec3 n2 = normalize(gl_NormalMatrix * -normal);
	float diffuse = max(0.0, dot(ls[i], n1));
	float diffuse2 = max(0.0, dot(ls[i], n2));
	gl_FrontColor = gl_FrontColorIn[i];
	gl_FrontColor.g = clamp(diffuse+diffuse2, 0.0, 1.0);
	//light = clamp(diffuse+diffuse2, 0.0, 1.0);
	//light = i * 0.3;
	gl_Position = gl_PositionIn[i]; 
	EmitVertex(); 
    }
}

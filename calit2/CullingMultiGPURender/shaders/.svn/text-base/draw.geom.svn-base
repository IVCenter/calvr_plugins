#version 120 
#extension GL_EXT_geometry_shader4 : enable 

/*
 * Geometry shader to calculate lighting from the triangle vertices
 */

varying in vec3 ls[3];
 
void main() 
{
    // use gl_Vertex from two points passed from vert shader to compute the normal of the triangle 
    vec3 normal = normalize(cross(gl_TexCoordIn[1][0].xyz - gl_TexCoordIn[0][0].xyz, gl_TexCoordIn[2][0].xyz - gl_TexCoordIn[0][0].xyz));

    // create camera space normals for two sided lighting
    vec3 n1 = normalize(gl_NormalMatrix * normal);
    vec3 n2 = normalize(gl_NormalMatrix * -normal);

    for(int i = 0; i < gl_VerticesIn; ++i) 
    {
	// calculate diffuse component for each side
	float diffuse = max(0.0, dot(ls[i], n1));
	float diffuse2 = max(0.0, dot(ls[i], n2));

	gl_FrontColor = gl_FrontColorIn[i];

	// set lighting multiplier into green output channel
	gl_FrontColor.g = clamp(diffuse+diffuse2, 0.0, 1.0);

	gl_Position = gl_PositionIn[i];
	
	// output vertex 
	EmitVertex(); 
    }
}

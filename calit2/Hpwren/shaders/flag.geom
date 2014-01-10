#version 120
#extension GL_EXT_geometry_shader4: enable

uniform int numSegments;
uniform float Phase;
uniform float Frequency;
uniform float Amplitude;

varying out float lightIntensity;

void generateHeight(vec4 point)
{
	float angle = (point.x + Phase) * Frequency;
	point.y = sin(angle) * Amplitude;

	vec3 normal = normalize(vec3(-Amplitude*Frequency*cos(angle), 1.0, 0.0));

	// Transform normal and position to eye space (for fragment shader)
	vec3 tnorm = normalize( vec3( gl_NormalMatrix * normal ) );

	vec3 lightPos = normalize(gl_LightSource[0].position.xyz);

	point = gl_ModelViewMatrix * point;
	lightIntensity = dot( normalize(lightPos - point.xyz), tnorm );
	lightIntensity = abs( lightIntensity );
	lightIntensity *= 1.5;

	gl_Position = gl_ProjectionMatrix * point;
	EmitVertex();
}

void generateFlag()
{
        float subLength = (gl_PositionIn[0].x - gl_PositionIn[2].x) / numSegments;

	generateHeight(gl_PositionIn[0]);	

	generateHeight(gl_PositionIn[1]);

	vec4 pos1 = gl_PositionIn[0];
	vec4 pos2 = gl_PositionIn[1];

	// create triangle strips for the flag and also compute normals
        for(int i = 1; i < numSegments; i++)
        {
		pos1.x = subLength * i;
        	generateHeight(pos1);

		pos2.x = subLength * i;
        	generateHeight(pos2);
        }

	EndPrimitive();
}


 
void main(void)
{
	generateFlag();
}

#version 460

out vec4 FragColor;

in vs_out {
	vec2 uv;
	vec3 vertPos;
	vec3 norm;
	vec3 fragPos;
} i;

uniform vec3 lightPos;
//uniform mat4 objToWorld;

vec3 CalcPointLight(vec3 lightColor, vec3 lightPos, float constant, float linear, float quadratic);
vec3 ColonTexture();
float hash1(uint n);

void main() {
    //vec3 pLight = vec3(objToWorld * vec4(lightPos, 1.0));
    float numLights = 1;
	vec3 lightColor = vec3(1.0);

    vec3 result = vec3(0.0);
    for(int i = 0; i < numLights; ++i) {
        result += CalcPointLight(lightColor, lightPos, 0.75f, 0.0003f, 0.000001f);
    }
        
    FragColor = vec4(result, 1.0);
}

vec3 CalcPointLight(vec3 lightColor, vec3 lightPos, float constant, float linear, float quadratic) {
   //MVP
	vec3 viewPos = vec3(0.0);

	//Material
	vec3 matAmbient = vec3(0.25, 0.15, 0.15);
	vec3 matDiffuse = vec3(1.0, 0.42, 0.42);
	vec3 matSpecular = vec3(0.75, 0.4, 0.4);
    float shininess = 30.0;

	// ambient
    vec3 ambient = lightColor * matAmbient;
  	
    // diffuse 
    vec3 norm = normalize(i.norm);
    vec3 lightDir = normalize(i.fragPos - lightPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = lightColor * (diff * ColonTexture());
    
    // specular
    vec3 viewDir = normalize(viewPos - i.fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = lightColor * (spec * matSpecular);  

    // attenuation
    float lightDist    = length(lightPos - i.fragPos);
    float attenuation = 1.0 / (constant + linear * lightDist + 
  			     quadratic * (lightDist * lightDist));  

    // combine results
    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;

    return (ambient + diffuse + specular);
}

vec3 ColonTexture() {
	vec3 ratio = normalize(vec3(1, 0.42, 0.42));

    float rand = hash1(uint(i.vertPos.x * i.vertPos.y * i.vertPos.z));
    float scale = ((rand + 1) / 2);

    vec3 result = ratio * scale;

    return result;
}

float hash1( uint n ) {
	n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    return float( n & uvec3(0x7fffffffU))/float(0x7fffffff);
}

